"""Print the questions of a text and their options: `uv run extract [FILE]`."""

import argparse
import json
import sys
from pathlib import Path

from model import CHECKPOINT, load, load_tokenizer, torch_taggers
from normalise import normalise
from tagging import KINDS, MAX_TOKENS, OPTION_TRAILING, after_context, message_range, token_count
from units import category, units


def with_trace(taggers: dict) -> dict:
    """Taggers that return their spans and a trace, the form of `analyse`: torch has none."""
    return {
        task: lambda texts, trailing="", f=f: (f(texts, trailing), None)
        for task, f in taggers.items()
    }


def analyse(taggers: dict, markdown: str) -> dict:
    """The questions of a Markdown message, with the intermediate data of every step: the units
    (gated or not), the question spans of the gated ones, the options of the choice questions,
    and the ranges in the message. The models read the units with their line breaks."""
    found = units(*normalise(markdown))
    gated = [i for i, u in enumerate(found) if category(u.text) != "other"]
    spans, question_trace = taggers["question"]([found[i].text for i in gated])
    spans = [after_context(pairs, found[i].context) for i, pairs in zip(gated, spans, strict=True)]
    asked = [(gated[n], a, b, KINDS[k]) for n, pairs in enumerate(spans) for a, b, k in pairs]
    choices = [n for n, (_, _, _, kind) in enumerate(asked) if kind == "choice"]
    texts = [found[i].text[a:b] for i, a, b, _ in (asked[n] for n in choices)]
    options, option_trace = taggers["option"](texts, OPTION_TRAILING)
    inner_of = dict(zip(choices, options, strict=True))
    questions = []
    for n, (i, a, b, kind) in enumerate(asked):
        origin, inner = found[i].origin, inner_of.get(n, [])
        span = message_range(origin, markdown, a, b)
        ranges = [message_range(origin, markdown, a + x, a + y) for x, y in inner]
        questions.append(
            {
                "unit": i,
                "span": [a, b],
                "range": list(span),
                "text": markdown[span[0] : span[1]],
                "kind": kind,
                "options": [{"range": list(r), "text": markdown[r[0] : r[1]]} for r in ranges],
                "option_spans": [list(s) for s in inner],
            }
        )
    return {
        "units": [
            {"text": u.text, "origin": u.origin, "context": u.context, "gated": n in gated}
            for n, u in enumerate(found)
        ],
        "question_tokens": question_trace and {"units": gated, **question_trace},
        "questions": questions,
        "option_tokens": option_trace and {"questions": choices, **option_trace},
    }


def extract(tokenizer, taggers: dict, text: str) -> tuple[list[dict], int]:
    """Questions in text order with their kinds, and how many units were cut at the token limit.
    Each question is a substring of text, as are its options."""
    found = analyse(with_trace(taggers), text)
    cut = sum(token_count(tokenizer, u["text"]) > MAX_TOKENS for u in found["units"] if u["gated"])
    return [
        {
            "question": q["text"],
            "kind": q["kind"],
            "options": [o["text"] for o in q["options"]],
            "range": q["range"],
            "option_ranges": [o["range"] for o in q["options"]],
        }
        for q in found["questions"]
    ], cut


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="UTF-8 text; stdin if absent")
    parser.add_argument("--ranges", action="store_true", help="add character offsets in the input")
    parser.add_argument("--model", type=Path, help="checkpoint directory")
    args = parser.parse_args(argv)
    raw = args.file.read_bytes() if args.file else sys.stdin.buffer.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print("extract: input is not UTF-8", file=sys.stderr)
        return 1
    path = args.model or CHECKPOINT
    if not (path / "question").is_dir():
        print(f"extract: no model in {path}; train first", file=sys.stderr)
        return 1
    tokenizer = load_tokenizer(path)
    taggers = torch_taggers(tokenizer, load(path))
    questions, cut = extract(tokenizer, taggers, text)
    if cut:
        print(f"extract: {cut} unit(s) cut at {MAX_TOKENS} tokens", file=sys.stderr)
    if not args.ranges:
        questions = [
            {"question": q["question"], "kind": q["kind"], "options": q["options"]}
            for q in questions
        ]
    print(json.dumps({"questions": questions}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
