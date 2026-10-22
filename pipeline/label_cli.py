"""Label one text with one DeepSeek call for each unit: `uv run label [FILE]`."""

import argparse
import getpass
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus import has_letters, redacted
from deepseek import CACHE, BudgetExceeded, PeakHours, open_session
from label import WORKERS, Dropped, label_unit, load_prompt
from normalise import normalise
from units import units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="UTF-8 text; stdin if absent")
    parser.add_argument("--no-cache", action="store_true", help="always make a fresh call")
    parser.add_argument("--peak", action="store_true", help="allow a call in DeepSeek peak hours")
    args = parser.parse_args(argv)
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    prompt = load_prompt()
    found = [redacted(u, getpass.getuser()) for u in units(*normalise(text))]
    found = [(line, context) for line, _, context in found if has_letters(line)]
    code = 0
    with (
        tempfile.TemporaryDirectory() as fresh,
        open_session("label", args.peak, Path(fresh) if args.no_cache else CACHE) as session,
    ):
        if session is None:
            return 1
        before = session.ledger.total
        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for (line, _), (result, _) in zip(
                    found,
                    pool.map(lambda item: label_unit(session, prompt, *item, "sample0"), found),
                    strict=True,
                ):
                    print(line)
                    if isinstance(result, Dropped):
                        print(f"  dropped: {result.reason}")
                    for q in [] if isinstance(result, Dropped) else result:
                        print(f"  [{q.kind}] {q.question}")
                        for option in q.options:
                            print(f"    - {option}")
        except PeakHours:
            print("label: DeepSeek peak hours; pass --peak to call anyway", file=sys.stderr)
            code = 1
        except BudgetExceeded:
            print("label: the budget is spent", file=sys.stderr)
            code = 1
        print(
            f"label: prompt {prompt.digest}, this run ${session.ledger.total - before:.4f}, "
            f"ledger ${session.ledger.total:.4f}",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
