"""The packed Neural Engine taggers of `uv run convert-ane`: their parity against torch
(`uv run parity-ane`) and the golden fixture of the native port (`uv run golden-ane`).
"""

import logging
from pathlib import Path

import numpy as np

from ane import COREML_ANE, INPUTS, build
from jsonl import DATA
from model import TASKS
from pack import Runtime, open_runtime, paths, run_golden, run_parity
from tagging import MAX_TOKENS

# external: coremltools warns on import that the installed torch is newer than the tested one
logging.getLogger("coremltools").setLevel(logging.ERROR)

import coremltools as ct  # noqa: E402
from coremltools.models import CompiledMLModel  # noqa: E402

GOLDEN = DATA / "golden-pi.json"
PARITY = DATA / "parity-ane.json"


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


def predict(model, pad: int, ids: list[list[int]]) -> list[list[list[float]]]:
    """The logits of the tokens of the units, which share one sequence."""
    tokens = np.full((1, MAX_TOKENS), pad, dtype=np.int32)
    position, segment = (np.zeros((1, MAX_TOKENS), dtype=np.int32) for _ in range(2))
    offset, placed = 0, []
    for k, unit in enumerate(ids):
        n = len(unit)
        tokens[0, offset : offset + n] = unit
        position[0, offset : offset + n] = np.arange(n)
        segment[0, offset : offset + n] = k + 1
        placed.append((offset, n))
        offset += n
    inputs = dict(zip(INPUTS, (tokens, position, segment), strict=True))
    logits = np.asarray(model.predict(inputs)["logits"])[0, :, 0, :]
    return [logits[:, start : start + n].T.tolist() for start, n in placed]


def load_runtime(directory: Path, checkpoint: Path) -> Runtime:
    """The compiled models of the directory: on the Neural Engine, or on the CPU for an fp32 pack,
    which the Neural Engine cannot run."""

    def models(settings: dict) -> dict:
        compute = (
            ct.ComputeUnit.CPU_AND_NE
            if settings["precision"] == "float16"
            else ct.ComputeUnit.CPU_ONLY
        )
        return {
            task: CompiledMLModel(str(directory / f"{task}.mlmodelc"), compute_units=compute)
            for task in TASKS
        }

    return open_runtime(directory, checkpoint, models, pack_greedy, predict)


def parity(argv: list[str] | None = None) -> int:
    """The packed taggers against torch on the dev units (`pack.run_parity`); the conversion gate
    is the packing."""
    args = paths(parity.__doc__, argv, COREML_ANE)

    return run_parity(args, build, load_runtime, "packing", PARITY)


def golden(argv: list[str] | None = None) -> int:
    """The fixture of the native port: `data/golden-pi.json`, made by the packed Core ML models
    (`pack.run_golden`)."""
    args = paths(golden.__doc__, argv, COREML_ANE)
    return run_golden(args.model, load_runtime(args.pack, args.model), GOLDEN)
