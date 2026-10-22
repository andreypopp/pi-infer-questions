"""The packed Neural Engine taggers of `uv run convert-ane`: the Markdown pipeline, its parity
against torch (`uv run parity-ane`) and the golden fixture of the native port (`uv run golden-ane`).
"""

import argparse
import json
import logging
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ane import COREML_ANE, INPUTS, build
from corpus import load_units
from evaluate import decode, evaluate, project_gold
from extract import analyse, with_trace
from jsonl import DATA, file_digest, read_json, read_jsonl, sha256, write_json, write_text
from label import LABELS, gold_of, lined_context, load_labelled, load_prompt
from model import CHECKPOINT, TASKS, load, load_tokenizer, require_tags, torch_taggers, untyped
from normalise import normalise
from synth import GENERATIONS
from tagging import KINDS, MAX_TOKENS, OPTION_TRAILING, encode, spans_of, tags_for
from units import ITEM, category, units

# external: coremltools warns on import that the installed torch is newer than the tested one
logging.getLogger("coremltools").setLevel(logging.ERROR)

import coremltools as ct  # noqa: E402
from coremltools.models import CompiledMLModel  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = DATA / "golden-pi.json"
PARITY = DATA / "parity-ane.json"
QUOTAS = (("options", 50), ("questions", 70), ("gated", 50), ("plain", 30))
CANDIDATES = 20000
MAX_DROP = 0.02
GATED = ("question_f1", "question_f1_overlap", "option_f1", "fpr")
REPORTED = ("kind_f1_yes_no", "kind_f1_open", "kind_f1_choice")


@dataclass(frozen=True)
class Runtime:
    tokenizer: object
    settings: dict
    models: dict


def require_checkpoint(settings: dict, directory: Path, checkpoint: Path) -> None:
    """Refuse a pack, or an ONNX directory, that was not converted from the checkpoint."""
    for task in TASKS:
        if settings["digests"]["checkpoint"][task] != file_digest(
            checkpoint / task / "model.safetensors"
        ):
            raise ValueError(f"{directory} was converted from another checkpoint than {checkpoint}")


def load_runtime(directory: Path, checkpoint: Path) -> Runtime:
    """The tokenizer and the compiled models of the directory: on the Neural Engine, or on the CPU
    for an fp32 pack, which the Neural Engine cannot run. The pack must come from the checkpoint."""
    settings = read_json(directory / "settings.json")
    require_checkpoint(settings, directory, checkpoint)
    compute = (
        ct.ComputeUnit.CPU_AND_NE if settings["precision"] == "float16" else ct.ComputeUnit.CPU_ONLY
    )
    models = {
        task: CompiledMLModel(str(directory / f"{task}.mlmodelc"), compute_units=compute)
        for task in TASKS
    }
    require_tags(settings["tags"], directory)
    return Runtime(AutoTokenizer.from_pretrained(directory), settings, models)


def pack_greedy(lengths: list[int]) -> list[list[int]]:
    """Consecutive units in packs of at most MAX_TOKENS tokens."""
    packs: list[list[int]] = []
    used = MAX_TOKENS
    for i, n in enumerate(lengths):
        if used + n > MAX_TOKENS:
            packs.append([])
            used = 0
        packs[-1].append(i)
        used += n
    return packs


def tag(runtime: Runtime, task: str, texts: list[str], trailing: str = ""):
    """The spans of each text (typed for the question task), and the tokens and packs that made
    them."""
    encoded = [encode(runtime.tokenizer, t) for t in texts]
    packs = pack_greedy([len(ids) for ids, _ in encoded])
    spans: list = [None] * len(texts)
    for pack in packs:
        ids = np.full((1, MAX_TOKENS), runtime.settings["pad_token_id"], dtype=np.int32)
        position, segment = (np.zeros((1, MAX_TOKENS), dtype=np.int32) for _ in range(2))
        offset, placed = 0, []
        for k, i in enumerate(pack):
            n = len(encoded[i][0])
            ids[0, offset : offset + n] = encoded[i][0]
            position[0, offset : offset + n] = np.arange(n)
            segment[0, offset : offset + n] = k + 1
            placed.append((i, offset, n))
            offset += n
        inputs = dict(zip(INPUTS, (ids, position, segment), strict=True))
        logits = np.asarray(runtime.models[task].predict(inputs)["logits"])[0, :, 0, :]
        for i, start, n in placed:
            tags = logits[:, start : start + n].argmax(0).tolist()
            found = spans_of(texts[i], encoded[i][1], tags, trailing)
            spans[i] = found if task == "question" else untyped(found)
    tokens = [{"ids": ids, "offsets": [list(o) for o in offsets]} for ids, offsets in encoded]
    return spans, {"tokens": tokens, "packs": packs}


def agreement(tokenizer, texts: list[str], left: list, right: list) -> float:
    """The share of tokens with the same tag when the decoded spans are tagged again."""
    same = total = 0
    for text, a, b in zip(texts, left, right, strict=True):
        _, offsets = encode(tokenizer, text)
        tags_a = tags_for(offsets, [(*q, KINDS.index(k)) for q, k, _ in a])
        tags_b = tags_for(offsets, [(*q, KINDS.index(k)) for q, k, _ in b])
        same += sum(x == y for x, y in zip(tags_a, tags_b, strict=True))
        total += len(offsets)
    return same / total


def different_inputs(left: dict, right: dict, inputs: dict[str, list[str]]) -> dict:
    """For each input set, how many texts the two tagger sets decode differently."""
    counts = {}
    for name, items in inputs.items():
        task, trailing = ("question", "") if name == "question" else ("option", OPTION_TRAILING)
        counts[name] = {
            "inputs": len(items),
            "different": sum(
                a != b
                for a, b in zip(
                    left[task](items, trailing), right[task](items, trailing), strict=True
                )
            ),
        }
    return counts


def paths(description: str, argv: list[str] | None, pack: Path = COREML_ANE) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, default=CHECKPOINT, help="checkpoint directory")
    parser.add_argument("--pack", type=Path, default=pack, help="converted models directory")
    return parser.parse_args(argv)


def run_parity(
    model: Path, exact, shipped: dict, gate: str, output: Path, extra=lambda texts: {}
) -> int:
    """The shipped taggers against torch on the dev units. Prints counts and deltas only.
    Conversion, a hard gate: the fp32 conversion (`exact`, which converts into the scratch
    directory that it gets) decodes every question input and every option input exactly like torch.
    Quantization, a limit: the shipped taggers may not lose more than MAX_DROP of question F1,
    overlap F1 and end-to-end option F1, nor gain more than MAX_DROP of FPR; the drop of the kind
    F1 and the differing units are reported only."""
    rows, _ = load_units()
    records = load_labelled(LABELS, rows, load_prompt())[0]["dev"]
    texts = [r["lined"] for r in records]
    tokenizer = load_tokenizer(model)
    torch_side = memo(torch_taggers(tokenizer, load(model), cpu=True))
    held = [project_gold(tokenizer, t, gold_of(r)) for t, r in zip(texts, records, strict=True)]
    inputs = {
        "question": texts,
        "option_predicted": [
            t[a:b]
            for t, found in zip(texts, torch_side["question"](texts), strict=True)
            for a, b, k in found
            if KINDS[k] == "choice"
        ],
        "option_gold": [
            t[a:b]
            for t, unit in zip(texts, held, strict=True)
            for (a, b), _, kind, _ in filter(None, unit)
            if kind == "choice"
        ],
    }
    with tempfile.TemporaryDirectory() as scratch:
        conversion = different_inputs(torch_side, exact(Path(scratch)), inputs)
    torch_all = evaluate(tokenizer, torch_side, records)["all"]
    shipped_all = evaluate(tokenizer, shipped, records)["all"]
    drops = {m: torch_all[m] - shipped_all[m] for m in GATED}
    drops["fpr"] = -drops["fpr"]
    failing = [m for m, drop in drops.items() if drop > MAX_DROP]
    gate_failing = [n for n, d in conversion.items() if d["different"]]
    decoded = {
        name: decode(t, texts, [lined_context(r) for r in records])
        for name, t in (("torch", torch_side), ("shipped", shipped))
    }
    result = {
        "units": len(texts),
        gate: {"different": conversion, "outside_limit": gate_failing},
        "quantization": {
            "limit": MAX_DROP,
            "drops": drops,
            "outside_limit": failing,
            "reported": {m: torch_all[m] - shipped_all[m] for m in REPORTED},
            "inputs_with_different_decoding": different_inputs(torch_side, shipped, inputs),
            "units_with_different_pipeline_decoding": sum(
                a != b for a, b in zip(decoded["torch"], decoded["shipped"], strict=True)
            ),
            "question_tag_agreement": agreement(
                tokenizer, texts, decoded["torch"], decoded["shipped"]
            ),
        },
        **extra(texts),
        "torch": torch_all,
        "shipped": shipped_all,
    }
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("torch", "shipped")}))
    return 0 if not (failing or gate_failing) else 1


def parity(argv: list[str] | None = None) -> int:
    """The packed taggers against torch on the dev units (`run_parity`); the conversion gate is
    the packing."""
    args = paths(parity.__doc__, argv)

    def exact(scratch: Path) -> dict:
        build(args.model, scratch / "fp32", quantize=False)
        return spans_only(packed_taggers(load_runtime(scratch / "fp32", args.model)))

    shipped = spans_only(packed_taggers(load_runtime(args.pack, args.model)))
    return run_parity(args.model, exact, shipped, "packing", PARITY)


def memo(taggers: dict) -> dict:
    """The taggers, each text decoded once."""
    cache: dict = {}

    def cached(task: str):
        def tagger(texts: list[str], trailing: str = "") -> list:
            new = [t for t in dict.fromkeys(texts) if (task, trailing, t) not in cache]
            for text, spans in zip(new, taggers[task](new, trailing), strict=True):
                cache[task, trailing, text] = spans
            return [cache[task, trailing, t] for t in texts]

        return tagger

    return {task: cached(task) for task in taggers}


def packed_taggers(runtime: Runtime) -> dict:
    """For each task, a function from texts to their spans and the trace of the packs."""
    return {task: lambda t, tr="", task=task: tag(runtime, task, t, tr) for task in TASKS}


def spans_only(traced: dict) -> dict:
    """Taggers that return their spans only, each text decoded once."""
    return memo({task: lambda t, tr="", f=f: f(t, tr)[0] for task, f in traced.items()})


def synthetic_messages() -> list[str]:
    """The generated messages of data/synth, in the order of their hashes. They hold no real
    text."""
    seen = {
        message
        for generation in read_jsonl(GENERATIONS)
        if generation["status"] == "ok"
        for message in generation["messages"]
    }
    return sorted((t for t in seen if 20 <= len(t) <= 6000), key=sha256)


def shape(markdown: str) -> str:
    """What the text promises, whatever a model finds in it: no unit that a model reads, units
    with a decision word only, a choice (a list or "or" in a unit with a "?"), or questions."""
    found = [u.text for u in units(*normalise(markdown)) if category(u.text) != "other"]
    if not found:
        return "plain"
    if not any("?" in t for t in found):
        return "gated"
    if any(
        sum(bool(ITEM.match(line)) for line in t.split("\n")) >= 2 or " or " in t
        for t in found
        if "?" in t
    ):
        return "options"
    return "questions"


def run_golden(model: Path, taggers: dict, settings: dict, output: Path) -> int:
    """The fixture of a native port, made by `taggers` (which return a trace) from the Markdown
    fixtures and whole synthetic messages. The cases depend on the texts only (`shape`). Prints
    counts only."""
    cases = [
        {"name": f"fixture:{p.name}", **analyse(taggers, text := p.read_text()), "markdown": text}
        for p in sorted([*FIXTURES.glob("*.txt"), *FIXTURES.glob("*.md")])
    ]
    quotas = dict(QUOTAS)
    for text in synthetic_messages()[:CANDIDATES]:
        group = shape(text)
        if quotas[group] > 0:
            quotas[group] -= 1
            cases.append(
                {"name": f"synthetic:{sha256(text)}", "markdown": text} | analyse(taggers, text)
            )
        if not any(quotas.values()):
            break
    synthetic = [c for c in cases if c["name"].startswith("synthetic")]
    reference = with_trace(torch_taggers(load_tokenizer(model), load(model)))
    different = sum(
        [(q["range"], q["kind"]) for q in analyse(reference, c["markdown"])["questions"]]
        != [(q["range"], q["kind"]) for q in c["questions"]]
        for c in cases
    )
    header = {
        "about": "Offsets are code points. Per case: markdown; units (all units of the normalised "
        "message: text, origin = Markdown offset of each unit character, context = length of the "
        "heading or bold line that starts the unit, with its line break (0 for none), gated = "
        "goes to the model); question_tokens (units = unit index of each text, tokens = ids and "
        "offsets in the text with [0, 0] for special tokens, packs = positions in units, in the "
        "packs or batches of the models); questions (unit, span in the unit, range and text in the "
        "Markdown, kind, options with ranges, option_spans in the question); option_tokens "
        "(questions = question index of each text, tokens, packs = positions in questions; only "
        "the questions of kind choice). Units and tokens are the units with their line breaks.",
        "settings": settings,
        "digest": sha256(json.dumps([c["markdown"] for c in cases], ensure_ascii=False)),
    }
    lines = ",\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases)
    head = json.dumps(header, ensure_ascii=False)[:-1]
    write_text(output, f'{head}, "cases": [\n{lines}\n]}}\n')
    print(
        json.dumps(
            {
                "cases": len(cases),
                "by_shape": dict(Counter(shape(c["markdown"]) for c in synthetic)),
                "with_options": sum(any(q["options"] for q in c["questions"]) for c in cases),
                "questions": sum(len(c["questions"]) for c in cases),
                "differ_from_torch": different,
                "bytes": output.stat().st_size,
            }
        )
    )
    return 0


def golden(argv: list[str] | None = None) -> int:
    """The fixture of the native port: `data/golden-pi.json`, made by the packed Core ML models
    (`run_golden`)."""
    args = paths(golden.__doc__, argv)
    runtime = load_runtime(args.pack, args.model)
    return run_golden(args.model, packed_taggers(runtime), runtime.settings, GOLDEN)
