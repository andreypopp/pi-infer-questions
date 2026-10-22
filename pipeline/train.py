"""Train the question tagger and the option tagger, and select a seed on the real dev split.

`uv run train` trains the seeds on the labelled train units and keeps the best on the dev units.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from corpus import load_units
from evaluate import evaluate, held_of
from label import LABELS, gold_of, load_labelled, load_prompt
from model import (
    CHECKPOINT,
    batch_tensors,
    device,
    load,
    load_tokenizer,
    new_model,
    save,
    torch_taggers,
)
from synth import TRAIN_BATCHES, labelled_units
from tagging import KINDS, Typed, encode, tags_for

LEARNING_RATE = 1e-4
EPOCHS = 10
SEEDS = (0, 1, 2)
CLIP = 1.0
BATCH = 16
WARMUP = 0.1
MAX_FPR = 0.05


def prepare(tokenizer, data: list[tuple[str, list[Typed]]]) -> list[tuple[list[int], list[int]]]:
    """(token ids, tags) of (text, spans) examples, shortest first. A text over MAX_TOKENS tokens
    is cut there, and the spans that end after the cut are left out."""
    encoded = []
    for text, spans in data:
        ids, offsets = encode(tokenizer, text)
        seen = max(end for _, end in offsets)
        encoded.append((ids, tags_for(offsets, [s for s in spans if s[1] <= seen])))
    return sorted(encoded, key=lambda item: len(item[0]))


def fit(tokenizer, task: str, encoded: list, seed: int, epochs: int):
    """Train a tagger on prepared examples. The batches hold texts of similar length; their order
    changes with each epoch."""
    random.seed(seed)
    torch.manual_seed(seed)
    hardware = device()
    batches = [encoded[first : first + BATCH] for first in range(0, len(encoded), BATCH)]
    model = new_model(task).to(hardware).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps = epochs * len(batches)
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, int(WARMUP * steps)), steps)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(batches)
        for batch in batches:
            tensors = batch_tensors(
                tokenizer, [i for i, _ in batch], hardware, [t for _, t in batch]
            )
            model(**tensors).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    return model.eval()


def examples(tokenizer, records: list[dict]) -> dict[str, list]:
    """The question tagger reads each unit and tags the questions with their kinds; the option
    tagger reads each choice."""
    gold = [(r["lined"], gold_of(r)) for r in records]
    return {
        "question": prepare(
            tokenizer, [(t, [(a, b, KINDS.index(kind)) for (a, b), kind, _ in g]) for t, g in gold]
        ),
        "option": prepare(
            tokenizer,
            [
                (t[a:b], [(x, y, 0) for x, y in options])
                for t, g in gold
                for (a, b), kind, options in g
                if kind == "choice"
            ],
        ),
    }


def select(scores: dict[int, dict]) -> int | None:
    """Of the seeds with a dev FPR under MAX_FPR, the best mean of question F1 and end-to-end
    option F1."""
    ok = {seed: s["all"] for seed, s in scores.items() if s["all"]["fpr"] < MAX_FPR}
    return max(ok, key=lambda s: (ok[s]["question_f1"] + ok[s]["option_f1"]) / 2, default=None)


def run(
    tokenizer,
    train_records: list[dict],
    dev_records: list[dict],
    out: Path,
    seeds: list[int],
    epochs: int,
) -> tuple[int | None, dict[int, dict]]:
    """Train each seed under out/seed<N>, score it on dev, and copy the selected one to
    out/selected. Returns the selected seed and the dev scores."""
    scores = {}
    encoded = examples(tokenizer, train_records)
    held = held_of(tokenizer, dev_records)
    for seed in seeds:
        taggers = {task: fit(tokenizer, task, data, seed, epochs) for task, data in encoded.items()}
        save(taggers, tokenizer, out / f"seed{seed}")
        scores[seed] = evaluate(tokenizer, torch_taggers(tokenizer, taggers), dev_records, held)
    chosen = select(scores)
    shutil.rmtree(out / "selected", ignore_errors=True)
    if chosen is not None:
        shutil.copytree(out / f"seed{chosen}", out / "selected")
    return chosen, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--root", type=Path, default=CHECKPOINT.parent, help="run directory")
    parser.add_argument(
        "--synth", nargs="+", choices=TRAIN_BATCHES, help="add synthetic batches to train"
    )
    parser.add_argument("--eval", action="store_true", help="score root/selected on eval")
    args = parser.parse_args(argv)
    rows, _ = load_units()
    labelled, skipped = load_labelled(LABELS, rows, load_prompt())
    print(f"train: skipped records {skipped}", file=sys.stderr)
    tokenizer = load_tokenizer()
    train_records = labelled["train"]
    if args.synth:
        train_records = train_records + labelled_units(args.synth)
    print(f"train: {len(train_records)} training units", file=sys.stderr)
    if args.eval:
        scores = evaluate(
            tokenizer,
            torch_taggers(tokenizer, load(args.root / "selected")),
            labelled["eval"],
            held_of(tokenizer, labelled["eval"]),
        )
    else:
        chosen, scores = run(
            tokenizer,
            train_records,
            labelled["dev"],
            args.root,
            args.seeds,
            args.epochs,
        )
        scores = {"selected": chosen, "dev": scores, "skipped": skipped}
        if chosen is None:
            print(f"train: no seed has a dev FPR under {MAX_FPR}", file=sys.stderr)
    print(json.dumps(scores, indent=1))
    return 0 if args.eval or chosen is not None else 1


if __name__ == "__main__":
    sys.exit(main())
