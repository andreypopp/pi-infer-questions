"""The ONNX taggers for the CPU of Linux: `uv run convert-onnx`, their parity against torch
(`uv run parity-onnx`) and the golden fixture of the Linux build (`uv run golden-onnx`).

Each task is one model with the inputs input_ids and attention_mask, both (batch, length), and the
logits (batch, length, tags). The batch and the length are dynamic: the truncation at MAX_TOKENS
stays in the pipeline. The weights are int8 (dynamic quantization); the activations are quantized
at run time, per tensor, so a result depends on the batch that holds the unit. The batches are a
function of the token counts (`batches`); the native build reads the same limits from settings.json.
The output directory is self-contained, like the one of `convert-ane`.
"""

import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from jsonl import DATA, file_digest, write_json
from model import TASKS, load, load_tokenizer
from pack import (
    Runtime,
    common_settings,
    convert_main,
    open_runtime,
    paths,
    run_golden,
    run_parity,
    tag,
)

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
    common = common_settings(model, tokenizer)
    write_json(
        output / "settings.json",
        {
            **common,
            "batch": {"rows": BATCH_ROWS, "tokens": BATCH_TOKENS},
            "weights": "int8 dynamic" if quantize else "float32",
            "opset": OPSET,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "digests": {
                **common["digests"],
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


def predict(model, pad: int, ids: list[list[int]]) -> list[list[int]]:
    """The tags of the units, which are the rows of one batch."""
    width = max(map(len, ids))
    tokens = np.full((len(ids), width), pad, dtype=np.int64)
    mask = np.zeros((len(ids), width), dtype=np.int64)
    for r, unit in enumerate(ids):
        tokens[r, : len(unit)], mask[r, : len(unit)] = unit, 1
    tags = model.run(["logits"], dict(zip(INPUTS, (tokens, mask), strict=True)))[0].argmax(-1)
    return [tags[r, : len(unit)].tolist() for r, unit in enumerate(ids)]


def load_runtime(directory: Path, checkpoint: Path, threads: int = 0) -> Runtime:
    """The sessions of the directory, on the CPU."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads

    def models(settings: dict) -> dict:
        return {
            task: ort.InferenceSession(
                str(directory / f"{task}.onnx"), options, providers=["CPUExecutionProvider"]
            )
            for task in TASKS
        }

    return open_runtime(directory, checkpoint, models, batches, predict)


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
    return convert_main("convert-onnx", __doc__, ONNX, build, argv)


def parity(argv: list[str] | None = None) -> int:
    """The ONNX taggers against torch on the dev units (`pack.run_parity`); the conversion gate
    is the export. The latency is of the CPU of this machine, whatever it is."""
    args = paths(parity.__doc__, argv, ONNX)

    def timed(texts: list[str]) -> dict:
        return {
            "latency_ms_per_unit": {
                "about": "question model, one unit per run, on the CPU of this machine",
                "threads_1": latency(args.pack, args.model, texts, 1),
                "threads_default": latency(args.pack, args.model, texts, 0),
            }
        }

    return run_parity(args, build, load_runtime, "export", PARITY_ONNX, timed)


def golden(argv: list[str] | None = None) -> int:
    """The fixture of the Linux build, `data/golden-onnx-<arch>.json` for the architecture of this
    machine (the int8 results depend on it), made by the int8 ONNX models (`pack.run_golden`)."""
    args = paths(golden.__doc__, argv, ONNX)
    return run_golden(args.model, load_runtime(args.pack, args.model), GOLDEN_ONNX)
