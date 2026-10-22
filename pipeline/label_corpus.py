"""Label the corpus units: `uv run label-corpus --take question:eval=4 --take pool=2`."""

import argparse
import sys

from corpus import (
    SPLITS,
    Row,
    labelling_rows,
    load_units,
    samples,
    uniform_order,
)
from deepseek import open_session
from label import LABELS, load_prompt, run
from units import category

CATEGORIES = ("question", "pool", "other")


def take(rows: list[Row], spec: str) -> list[Row]:
    """CATEGORY[:SPLIT]=N|all: the first N units of the category in uniform order, so that a
    larger N holds the smaller one."""
    selector, count = spec.split("=")
    name, _, split = selector.partition(":")
    if name not in CATEGORIES or split and split not in SPLITS:
        raise ValueError(f"{spec}: the category is one of {CATEGORIES}, the split one of {SPLITS}")
    chosen = [r for r in uniform_order(rows) if category(r.text) == name and split in ("", r.split)]
    return chosen if count == "all" else chosen[: int(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", action="append", required=True, metavar="CATEGORY[:SPLIT]=N")
    parser.add_argument("--limit", type=int, help="at most this many units in all")
    parser.add_argument("--peak", action="store_true", help="allow a call in DeepSeek peak hours")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    rows, _ = load_units()
    chosen = list(dict.fromkeys(r for spec in args.take for r in take(labelling_rows(rows), spec)))
    chosen = chosen[: args.limit]
    with open_session("label-corpus", args.peak) as session:
        if session is None:
            return 1
        before = session.ledger.total
        counts = run(session, load_prompt(), chosen, LABELS, args.workers)
        calls = sum(samples(r) for r in chosen)
        print(
            f"label-corpus: {len(chosen)} units, up to {calls} calls, "
            f"{counts}, this run ${session.ledger.total - before:.4f}, "
            f"ledger ${session.ledger.total:.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
