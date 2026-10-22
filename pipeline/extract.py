"""Print the questions of a text and their options: `uv run extract [FILE]`."""

import argparse
import json
import sys
from pathlib import Path

from corpus import has_letters
from model import CHECKPOINT, load, load_tokenizer, torch_taggers
from normalise import normalise
from tagging import KINDS, MAX_TOKENS, OPTION_TRAILING, after_context, message_range, token_count
from units import category, units


def ask(taggers: dict, texts: list[str], contexts: list[int]) -> list[list[tuple]]:
    """(span, kind) of each question with a letter in each text, after the first `context`
    characters."""
    return [
        [((a, b), KINDS[k]) for a, b, k in after_context(found, context) if has_letters(text[a:b])]
        for text, found, context in zip(texts, taggers["question"](texts), contexts, strict=True)
    ]


def option_texts(texts: list[str], asked: list[list[tuple]]) -> list[str]:
    """The choice questions of the texts: the only texts that the option tagger reads."""
    return [
        t[a:b]
        for t, unit in zip(texts, asked, strict=True)
        for (a, b), kind, *_ in unit
        if kind == "choice"
    ]


def attach_options(taggers: dict, texts: list[str], asked: list[list[tuple]]) -> list[list[tuple]]:
    """(span, kind, option spans in the span) of each question in `asked`."""
    options = iter(taggers["option"](option_texts(texts, asked), OPTION_TRAILING))
    return [
        [(span, kind, tuple(next(options)) if kind == "choice" else ()) for span, kind, *_ in unit]
        for unit in asked
    ]


def analyse(taggers: dict, markdown: str, trace=None) -> dict:
    """The questions of a Markdown message, with the intermediate data of every step: the units
    (gated or not), the question spans of the gated ones, the options of the choice questions,
    and the ranges in the message. The models read the units with their line breaks. `trace`
    gives the tokens of the texts that a model reads."""
    found = units(*normalise(markdown))
    gated = [i for i, u in enumerate(found) if category(u.text) != "other"]
    texts = [found[i].text for i in gated]
    asked = ask(taggers, texts, [found[i].context for i in gated])
    options = attach_options(taggers, texts, asked)
    flat = [(i, *question) for i, unit in zip(gated, options, strict=True) for question in unit]
    choices = [n for n, (_, _, kind, _) in enumerate(flat) if kind == "choice"]
    questions = []
    for i, (a, b), kind, inner in flat:
        origin = found[i].origin
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
        "question_tokens": trace and {"units": gated, **trace(texts)},
        "questions": questions,
        "option_tokens": trace and {"questions": choices, **trace(option_texts(texts, asked))},
    }


def extract(tokenizer, taggers: dict, text: str) -> tuple[list[dict], int]:
    """Questions in text order with their kinds, and how many units were cut at the token limit.
    Each question is a substring of text, as are its options."""
    found = analyse(taggers, text)
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
