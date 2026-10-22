"""Dry-run report: unit counts per split and the projected cost of each labelling set.

Counts only, never text: `uv run report`.
"""

import argparse
import json
import sys
from collections import Counter

from corpus import (
    SPLITS,
    Row,
    labelling_rows,
    load_units,
    samples,
    uniform_order,
)
from deepseek import price
from jsonl import DATA, read_jsonl, write_json, write_jsonl
from label import LABELS, load_prompt
from model import load_tokenizer
from tagging import MAX_TOKENS, token_count
from units import category

REPORT = DATA / "reports" / "dry-run.json"
STEPS = (1000, 2000, 5000, 10000, 20000)
CALL_OVERHEAD = 10  # role and template tokens around a unit
ANSWER_EMPTY = 10  # {"questions": []}
ANSWER_PER_UNIT_TOKEN = 1.3  # a copy of the question and the options
ANSWER_JSON = 30  # the JSON around the copies
REASONING = {"question": 300, "other": 100}  # thinking tokens of a call
CORRECTION_RATE = 0.1  # share of calls that need a second turn
CORRECTION_COST = 2  # a second turn costs about this many calls


def call_cost(system: int, tokens: int, kind: str) -> float:
    asks = kind == "question"
    answer = int(ANSWER_PER_UNIT_TOKEN * tokens) + ANSWER_JSON if asks else ANSWER_EMPTY
    output = REASONING["question" if asks else "other"] + answer
    cost = price(system, tokens + CALL_OVERHEAD, output, peak=False)
    return cost * (1 + CORRECTION_COST * CORRECTION_RATE)


def set_cost(rows: list[Row], tokens: dict[str, int], system: int) -> dict[str, float | int]:
    """Calls and off-peak dollars of labelling rows; the system prompt is cached after the first
    call."""
    total = sum(
        samples(row) * call_cost(system, tokens[row.text], category(row.text)) for row in rows
    )
    first_call = price(0, system, 0, peak=False) - price(system, 0, 0, peak=False)
    return {
        "units": len(rows),
        "calls": sum(samples(row) for row in rows),
        "dollars": round(total + first_call, 4),
    }


def dry_run(rows: list[Row], tokens: dict[str, int], system: int) -> dict:
    counts = {s: Counter(category(r.text) for r in rows if r.split == s) for s in SPLITS}
    long = {s: sum(tokens[r.text] > MAX_TOKENS for r in rows if r.split == s) for s in SPLITS}
    question = [r for r in rows if category(r.text) == "question"]
    pool = [r for r in rows if category(r.text) == "pool"]
    others = uniform_order(r for r in rows if category(r.text) == "other")
    sets = {"questions": question, "questions+pool": question + pool}
    for step in STEPS:
        if step < len(others):
            sets[f"questions+pool+others{step}"] = question + pool + others[:step]
    sets["questions+pool+all_others"] = question + pool + others
    return {
        "units": {s: {**counts[s], "total": sum(counts[s].values())} for s in SPLITS},
        "over_max_tokens": {
            s: {"units": long[s], "share": round(long[s] / max(1, sum(counts[s].values())), 4)}
            for s in SPLITS
        },
        "system_prompt_tokens": system,
        "projected_cost": {name: set_cost(chosen, tokens, system) for name, chosen in sets.items()},
    }


OTHER_SAMPLES = {"train": 2000, "eval": 1000}
OTHERS = DATA / "reports" / "other-questions.jsonl"


def other_questions(labels: list[dict], rows: list[Row]) -> dict:
    """Write the units of the two "other" samples that got a non-empty label, and return the
    counts of each sample. The samples are the first units of each split in the uniform order."""
    by_text = {r["text"]: r for r in labels}
    others = uniform_order(r for r in rows if category(r.text) == "other")
    counts, found = {}, []
    for split, size in OTHER_SAMPLES.items():
        sample = [r for r in others if r.split == split][:size]
        records = [by_text[r.text] for r in sample if r.text in by_text]
        asked = [r for r in records if r["questions"]]
        counts[split] = {
            "sample": size,
            "labelled": Counter(r["status"] for r in records),
            "with_questions": len(asked),
            "rate": round(len(asked) / max(1, size), 4),
        }
        found += [{"split": split, **r} for r in asked]
    write_jsonl(OTHERS, found)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--others", action="store_true", help="report the 'other' samples")
    if parser.parse_args().others:
        rows, _ = load_units()
        print(json.dumps(other_questions(read_jsonl(LABELS), rows)))
        return 0
    everything, dedup = load_units()
    rows = labelling_rows(everything)
    tokenizer = load_tokenizer()
    tokens = {r.text: token_count(tokenizer, r.text) for r in rows}
    system = token_count(tokenizer, load_prompt().text)
    result = {"dedup": dedup, "without_letters": len(everything) - len(rows)}
    write_json(REPORT, result | dry_run(rows, tokens, system))
    print(REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
