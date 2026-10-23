"""Questions as whole sentences: the sentences of a unit, the snap of a span to them, and the
decoder that tags each sentence from the mean of its token probabilities."""

import math
from functools import reduce
from operator import add

from tagging import KINDS, Span, Typed, sentence_end
from units import ITEM

Gold = list[tuple[Span, str, tuple[Span, ...]]]  # (question span, kind, option spans in the span)


def segments(text: str) -> list[Span]:
    """The sentences of each line. A line end ends a sentence, and a list marker stays in the
    sentence after it."""
    found, position = [], 0
    for line in text.split("\n"):
        end = position + len(line)
        marker = ITEM.match(line)
        i = position + (marker.end() if marker else 0)
        start = position
        while i < end:
            while i < end and text[i].isspace():
                i += 1
            if i >= end:
                break
            stop = sentence_end(text, i, end)
            found.append((start if start == position else i, stop))
            start = i = stop
        position = end + 1
    return [(a, b) for a, b in found if text[a:b].strip()]


def trim(text: str, a: int, b: int) -> Span:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def snap_gold(text: str, gold: Gold) -> Gold:
    """The questions from the first sentence that they overlap to the last one. Questions with the
    same range become one, with the first kind and all the options."""
    sentences = segments(text)
    found: dict[Span, tuple[str, set[Span]]] = {}
    for (a, b), kind, options in gold:
        hit = [s for s in sentences if s[0] < b and a < s[1]]
        span = trim(text, hit[0][0], hit[-1][1]) if hit else (a, b)
        _, inside = found.setdefault(span, (kind, set()))
        inside.update((a + x, a + y) for x, y in options)
    return [
        (span, kind, tuple(sorted((x - span[0], y - span[0]) for x, y in options)))
        for span, (kind, options) in found.items()
    ]


def softmax(row: list[float]) -> list[float]:
    top = max(row)
    exps = [math.exp(x - top) for x in row]
    total = reduce(add, exps)
    return [x / total for x in exps]


def decode(text: str, offsets: list[Span], logits: list[list[float]]) -> list[Typed]:
    """A sentence is a question sentence when the probability of its tokens that is inside a kind
    is above that of O (mean over its tokens); its kind is the likeliest. It starts a question when
    its first token is likelier a B than an I, or when no question sentence is before it. The kind
    of a question is the likeliest over all its sentences."""
    spans: list[list] = []
    open_question = False
    for a, b in segments(text):
        rows = [softmax(logits[i]) for i, (s, e) in enumerate(offsets) if e > s and a <= s < b]
        if not rows:
            open_question = False
            continue
        mean = [reduce(add, column) / len(rows) for column in zip(*rows, strict=True)]
        inside = [mean[1 + 2 * k] + mean[2 + 2 * k] for k in range(len(KINDS))]
        if mean[0] >= max(inside):
            open_question = False
            continue
        kind = inside.index(max(inside))
        first = rows[0]
        if open_question and first[2 * kind + 1] <= first[2 * kind + 2]:
            spans[-1][1] = b
            spans[-1][2] = [x + y for x, y in zip(spans[-1][2], inside, strict=True)]
        else:
            spans.append([a, b, inside])
        open_question = True
    return [(*trim(text, a, b), inside.index(max(inside))) for a, b, inside in spans]
