"""What the Core ML and ONNX packs share: settings.json, the convert command, the tagging driver,
the parity against torch (`uv run parity-ane`, `uv run parity-onnx`) and the golden fixture of the
native port (`uv run golden-ane`, `uv run golden-onnx`). It does not import a runtime."""

import argparse
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer

from corpus import load_units
from evaluate import decode, evaluate, held_of
from extract import analyse, ask, option_texts
from jsonl import file_digest, read_json, sha256, write_json, write_text
from label import LABELS, lined_context, load_labelled, load_prompt
from model import CHECKPOINT, TASKS, load, load_tokenizer, require_tags, torch_taggers, untyped
from normalise import normalise
from synth import messages
from tagging import (
    CLOSING,
    KINDS,
    MAX_TOKENS,
    OPENING,
    OPTION_TRAILING,
    TAG_NAMES,
    encode,
    spans_of,
    tags_for,
)
from units import DECISION, ITEM, category, units

FIXTURES = Path(__file__).parent / "fixtures"
QUOTAS = (("options", 50), ("questions", 70), ("gated", 50), ("plain", 30))
CANDIDATES = 20000
MAX_DROP = 0.02
GATED = ("question_f1", "question_f1_overlap", "option_f1", "fpr")
REPORTED = ("kind_f1_yes_no", "kind_f1_open", "kind_f1_choice")


@dataclass(frozen=True)
class Runtime:
    """The loaded pack: `group` splits the token counts of the texts into the units of one model
    call, and `predict` gives the tags of the token ids of one group."""

    tokenizer: object
    settings: dict
    models: dict
    group: Callable[[list[int]], list[list[int]]]
    predict: Callable[[object, int, list[list[int]]], list[list[int]]]


def source_digest() -> str:
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return sha256("".join(f"{p.name}:{sha256(p.read_text())}\n" for p in sources))


def common_settings(model: Path, tokenizer) -> dict:
    """The part of settings.json that both packs hold."""
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "tags": TAG_NAMES,
        "max_tokens": MAX_TOKENS,
        "trim": {"closing": CLOSING, "opening": OPENING, "option_trailing": OPTION_TRAILING},
        "decision": {"pattern": DECISION.pattern, "flags": "i"},
        "digests": {
            "source": source_digest(),
            "checkpoint": {task: file_digest(model / task / "model.safetensors") for task in TASKS},
        },
    }


def convert_main(name: str, description: str, output: Path, build: Callable, argv) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=output)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"{name}: {args.output} exists", file=sys.stderr)
        return 1
    build(args.model, args.output, quantize=True)
    return 0


def open_runtime(
    directory: Path, checkpoint: Path, models: Callable[[dict], dict], group: Callable, predict
) -> Runtime:
    """The tokenizer and the models (loaded from the settings) of a converted directory, which
    must come from the checkpoint and have the tags of this code."""
    settings = read_json(directory / "settings.json")
    digests = settings["digests"]["checkpoint"]
    if any(digests[task] != file_digest(checkpoint / task / "model.safetensors") for task in TASKS):
        raise ValueError(f"{directory} was converted from another checkpoint than {checkpoint}")
    require_tags(settings["tags"], directory)
    return Runtime(
        AutoTokenizer.from_pretrained(directory), settings, models(settings), group, predict
    )


def tag(runtime: Runtime, task: str, texts: list[str], trailing: str = ""):
    """The spans of each text, typed for the question task."""
    encoded = [encode(runtime.tokenizer, t) for t in texts]
    pad = runtime.settings["pad_token_id"]
    spans: list = [None] * len(texts)
    for members in runtime.group([len(ids) for ids, _ in encoded]):
        tags = runtime.predict(runtime.models[task], pad, [encoded[i][0] for i in members])
        for i, row in zip(members, tags, strict=True):
            found = spans_of(texts[i], encoded[i][1], row, trailing)
            spans[i] = found if task == "question" else untyped(found)
    return spans


def trace(runtime: Runtime, texts: list[str]) -> dict:
    """The tokens of the texts, and the groups (positions in the texts) of the model calls."""
    encoded = [encode(runtime.tokenizer, t) for t in texts]
    return {
        "tokens": [{"ids": ids, "offsets": [list(o) for o in offsets]} for ids, offsets in encoded],
        "packs": runtime.group([len(ids) for ids, _ in encoded]),
    }


def pack_taggers(runtime: Runtime) -> dict:
    return {task: partial(tag, runtime, task) for task in TASKS}


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


def paths(description: str, argv: list[str] | None, pack: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, default=CHECKPOINT, help="checkpoint directory")
    parser.add_argument("--pack", type=Path, default=pack, help="converted models directory")
    return parser.parse_args(argv)


def run_parity(
    args: argparse.Namespace,
    build: Callable,
    load_runtime: Callable,
    gate: str,
    output: Path,
    extra=lambda texts: {},
) -> int:
    """The shipped pack (`args.pack`) against torch on the dev units. Prints counts and deltas
    only. Conversion, a hard gate: the fp32 conversion (`build` into a scratch directory) decodes
    every question input and every option input exactly like torch.
    Quantization, a limit: the shipped taggers may not lose more than MAX_DROP of question F1,
    overlap F1 and end-to-end option F1, nor gain more than MAX_DROP of FPR; the drop of the kind
    F1 and the differing units are reported only."""
    model = args.model
    shipped = memo(pack_taggers(load_runtime(args.pack, model)))
    rows, _ = load_units()
    records = load_labelled(LABELS, rows, load_prompt())[0]["dev"]
    texts = [r["lined"] for r in records]
    tokenizer = load_tokenizer(model)
    torch_side = memo(torch_taggers(tokenizer, load(model), cpu=True))
    held = held_of(tokenizer, records)
    inputs = {
        "question": texts,
        "option_predicted": option_texts(texts, ask(torch_side, texts, [0] * len(texts))),
        "option_gold": option_texts(
            texts, [[(span, kind) for span, _, kind, _ in filter(None, unit)] for unit in held]
        ),
    }
    with tempfile.TemporaryDirectory() as scratch:
        build(model, Path(scratch) / "fp32", quantize=False)
        exact = memo(pack_taggers(load_runtime(Path(scratch) / "fp32", model)))
        conversion = different_inputs(torch_side, exact, inputs)
    torch_all = evaluate(tokenizer, torch_side, records, held)["all"]
    shipped_all = evaluate(tokenizer, shipped, records, held)["all"]
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


def run_golden(model: Path, runtime: Runtime, output: Path) -> int:
    """The fixture of a native port, made by the pack of `runtime` from the Markdown fixtures and
    whole synthetic messages. The cases depend on the texts only (`shape`). Prints counts only."""
    taggers, traced = pack_taggers(runtime), partial(trace, runtime)
    cases = [
        {
            "name": f"fixture:{p.name}",
            **analyse(taggers, text := p.read_text(), traced),
            "markdown": text,
        }
        for p in sorted([*FIXTURES.glob("*.txt"), *FIXTURES.glob("*.md")])
    ]
    quotas = dict(QUOTAS)
    for text in messages()[:CANDIDATES]:
        group = shape(text)
        if quotas[group] > 0:
            quotas[group] -= 1
            cases.append(
                {"name": f"synthetic:{sha256(text)}", "markdown": text}
                | analyse(taggers, text, traced)
            )
        if not any(quotas.values()):
            break
    synthetic = [c for c in cases if c["name"].startswith("synthetic")]
    reference = torch_taggers(load_tokenizer(model), load(model))
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
        "settings": runtime.settings,
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
