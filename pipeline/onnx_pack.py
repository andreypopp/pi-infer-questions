"""The ONNX taggers for the CPU of Linux: `uv run convert-onnx`, their parity against torch
(`uv run parity-onnx`) and the golden fixture of the Linux build (`uv run golden-onnx`).

Each task is one model with the inputs input_ids and attention_mask, both (batch, length), and the
logits (batch, length, tags). The batch and the length are dynamic: the truncation at MAX_TOKENS
stays in the pipeline. The weights are int8 (dynamic quantization); the activations are quantized
at run time, per tensor, so a result depends on the batch that holds the unit. The batches are a
function of the token counts (`batches`); the native build reads the same limits from settings.json.
The output directory is self-contained, like the one of `convert-ane`.
"""

import argparse
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoTokenizer

from ane import source_digest
from ane_pack import (
    Runtime,
    paths,
    require_checkpoint,
    run_golden,
    run_parity,
    spans_only,
)
from jsonl import DATA, file_digest, read_json, write_json
from model import CHECKPOINT, TASKS, load, load_tokenizer, require_tags, untyped
from tagging import (
    CLOSING,
    MAX_TOKENS,
    OPENING,
    OPTION_TRAILING,
    TAG_NAMES,
    encode,
    spans_of,
)
from units import DECISION

ONNX = DATA / "onnx"
ARCH = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x64"}[platform.machine()]
GOLDEN_ONNX = DATA / f"golden-onnx-{ARCH}.json"
PARITY_ONNX = DATA / f"parity-onnx-{ARCH}.json"
INPUTS = ("input_ids", "attention_mask")
BATCH_ROWS = 32
BATCH_TOKENS = 2048
OPSET = 17


class Tagger(nn.Module):
    def __init__(self, tagger) -> None:
        super().__init__()
        self.tagger = tagger

    def forward(self, input_ids, attention_mask):
        return self.tagger(input_ids=input_ids, attention_mask=attention_mask).logits


def export(tagger, path: Path) -> None:
    ids = torch.randint(1000, 2000, (2, 12))
    torch.onnx.export(
        Tagger(tagger).eval(),
        (ids, torch.ones_like(ids)),
        str(path),
        input_names=list(INPUTS),
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "length"} for name in (*INPUTS, "logits")},
        opset_version=OPSET,
        dynamo=False,
    )


def build(model: Path, output: Path, quantize: bool) -> None:
    """The ONNX taggers of the checkpoint in `output`, with the tokenizer and settings.json."""
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        for task, tagger in load(model).items():
            if not quantize:
                export(tagger, output / f"{task}.onnx")
                continue
            full = Path(scratch) / f"{task}.onnx"
            export(tagger, full)
            quantize_dynamic(full, output / f"{task}.onnx", weight_type=QuantType.QInt8)
    tokenizer = load_tokenizer(model)
    tokenizer.save_pretrained(output)
    write_json(
        output / "settings.json",
        {
            "pad_token_id": tokenizer.pad_token_id,
            "tags": TAG_NAMES,
            "max_tokens": MAX_TOKENS,
            "batch": {"rows": BATCH_ROWS, "tokens": BATCH_TOKENS},
            "trim": {"closing": CLOSING, "opening": OPENING, "option_trailing": OPTION_TRAILING},
            "decision": {"pattern": DECISION.pattern, "flags": "i"},
            "weights": "int8 dynamic" if quantize else "float32",
            "opset": OPSET,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "digests": {
                "source": source_digest(),
                "checkpoint": {
                    task: file_digest(model / task / "model.safetensors") for task in TASKS
                },
                "models": {task: file_digest(output / f"{task}.onnx") for task in TASKS},
                "tokenizer": file_digest(output / "tokenizer.json"),
            },
        },
    )


def batches(lengths: list[int]) -> list[list[int]]:
    """Unit indexes in batches of similar length: the units by length, then a new batch when one
    would hold more than BATCH_ROWS rows or BATCH_TOKENS padded tokens."""
    found: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        if (
            found
            and len(found[-1]) < BATCH_ROWS
            and (len(found[-1]) + 1) * lengths[i] <= BATCH_TOKENS
        ):
            found[-1].append(i)
        else:
            found.append([i])
    return found


def load_runtime(directory: Path, checkpoint: Path, threads: int = 0) -> Runtime:
    """The tokenizer and the sessions of the directory, on the CPU."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    models = {
        task: ort.InferenceSession(
            str(directory / f"{task}.onnx"), options, providers=["CPUExecutionProvider"]
        )
        for task in TASKS
    }
    settings = read_json(directory / "settings.json")
    require_checkpoint(settings, directory, checkpoint)
    require_tags(settings["tags"], directory)
    return Runtime(AutoTokenizer.from_pretrained(directory), settings, models)


def tag(runtime: Runtime, task: str, texts: list[str], trailing: str = ""):
    """The spans of each text (typed for the question task), and the tokens and batches that made
    them."""
    encoded = [encode(runtime.tokenizer, t) for t in texts]
    rows = batches([len(ids) for ids, _ in encoded])
    spans: list = [None] * len(texts)
    for row in rows:
        width = max(len(encoded[i][0]) for i in row)
        ids = np.full((len(row), width), runtime.settings["pad_token_id"], dtype=np.int64)
        mask = np.zeros((len(row), width), dtype=np.int64)
        for r, i in enumerate(row):
            n = len(encoded[i][0])
            ids[r, :n], mask[r, :n] = encoded[i][0], 1
        logits = runtime.models[task].run(["logits"], dict(zip(INPUTS, (ids, mask), strict=True)))[
            0
        ]
        tags = logits.argmax(-1)
        for r, i in enumerate(row):
            found = spans_of(
                texts[i], encoded[i][1], tags[r, : len(encoded[i][0])].tolist(), trailing
            )
            spans[i] = found if task == "question" else untyped(found)
    tokens = [{"ids": ids, "offsets": [list(o) for o in offsets]} for ids, offsets in encoded]
    return spans, {"tokens": tokens, "packs": rows}


def onnx_taggers(runtime: Runtime) -> dict:
    return {task: lambda t, tr="", task=task: tag(runtime, task, t, tr) for task in TASKS}


def latency(directory: Path, checkpoint: Path, texts: list[str], threads: int) -> dict:
    """Milliseconds to tag one unit alone with the question model, after one warm-up."""
    runtime = load_runtime(directory, checkpoint, threads)
    tag(runtime, "question", texts[:1])
    times = []
    for text in texts:
        start = time.perf_counter()
        tag(runtime, "question", [text])
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "mean": statistics.fmean(times),
        "median": times[len(times) // 2],
        "p95": times[int(len(times) * 0.95)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ONNX)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"convert-onnx: {args.output} exists", file=sys.stderr)
        return 1
    build(args.model, args.output, quantize=True)
    return 0


def parity(argv: list[str] | None = None) -> int:
    """The ONNX taggers against torch on the dev units (`ane_pack.run_parity`); the conversion gate
    is the export. The latency is of the CPU of this machine, whatever it is."""
    args = paths(parity.__doc__, argv, ONNX)

    def exact(scratch: Path) -> dict:
        build(args.model, scratch / "fp32", quantize=False)
        return spans_only(onnx_taggers(load_runtime(scratch / "fp32", args.model)))

    def timed(texts: list[str]) -> dict:
        return {
            "latency_ms_per_unit": {
                "about": "question model, one unit per run, on the CPU of this machine",
                "threads_1": latency(args.pack, args.model, texts, 1),
                "threads_default": latency(args.pack, args.model, texts, 0),
            }
        }

    shipped = spans_only(onnx_taggers(load_runtime(args.pack, args.model)))
    return run_parity(args.model, exact, shipped, "export", PARITY_ONNX, timed)


def golden(argv: list[str] | None = None) -> int:
    """The fixture of the Linux build, `data/golden-onnx-<arch>.json` for the architecture of this
    machine (the int8 results depend on it), made by the int8 ONNX models (`run_golden`)."""
    args = paths(golden.__doc__, argv, ONNX)
    runtime = load_runtime(args.pack, args.model)
    return run_golden(args.model, onnx_taggers(runtime), runtime.settings, GOLDEN_ONNX)
