"""Units: the paragraphs and list items of a normalised message."""

import re
from dataclasses import dataclass

from normalise import Mapped, lines_of

ITEM = re.compile(r"\s*(?:[-*+•]|\d+[.)]|[A-Za-z][.)])\s")
CLOSERS = ")]\"'”"
DECISION = re.compile(
    r"\b(?:which|prefer|choose|choice|options?|alternatively|either|decide|decisions?|pick|"
    r"would you like|do you want|let me know|your call|up to you|should i|shall i|"
    r"want me to|confirm|approve|go ahead|proceed|recommend\w*|tell me|"
    r"you(?:['’]d)? (?:want|like|prefer|agree)|you['’]d rather|rather|unless you|"
    r"say (?:the word|if|when|so)|whether|waiting (?:on|for) you|go-ahead|ways? (?:to|out)|"
    r"two ways|the question is|suggest\w*|propos\w*|objections?|if you object|sign[- ]off|"
    r"green light|happy to|questions?|still open|open (?:questions?|points?|items?|asks?)|"
    r"what['’]s open|for you|pending)\b",
    re.I,
)


@dataclass(frozen=True)
class Unit(Mapped):
    """context: the length of the heading or bold line at the start, with its line break; a span
    never starts in it."""

    context: int


def paragraphs(mapped: Mapped) -> list[list[int]]:
    """Runs of non-blank lines, as line numbers."""
    runs: list[list[int]] = [[]]
    for number, (start, end) in enumerate(lines_of(mapped)):
        if mapped.text[start:end].strip():
            runs[-1].append(number)
        elif runs[-1]:
            runs.append([])
    return [run for run in runs if run]


def marker_kind(line: str) -> str:
    first = line.lstrip()[0]
    return "number" if first.isdigit() else "letter" if first.isalpha() else "bullet"


def segments(unit: Mapped) -> list[Mapped]:
    """A unit with a top-level list item that holds a "?" splits: the text before the list, each
    such item with its nested items, and each run of other top-level items are units. The lined
    text has no indentation, so the top-level items are those with the marker kind of the first
    item."""
    lines = unit.text.split("\n")
    starts = [sum(len(line) + 1 for line in lines[:n]) for n in range(len(lines))]
    items = [n for n, line in enumerate(lines) if ITEM.match(line)]
    if not items:
        return [unit]
    top = [n for n in items if marker_kind(lines[n]) == marker_kind(lines[items[0]])]
    ends = [next((n for n in items if n > t), len(lines)) for t in top]
    asks = ["?" in "\n".join(lines[t:e]) for t, e in zip(top, ends, strict=True)]
    if not any(asks):
        return [unit]
    cuts = [0] if top[0] > 0 else []
    cuts += [t for t, ask, before in zip(top, asks, [False, *asks], strict=False) if ask or before]
    cuts = sorted(set(cuts) | {top[0]})
    bounds = [*cuts, len(lines)]
    out = []
    for first, last in zip(bounds, bounds[1:], strict=False):
        end = starts[last] - 1 if last < len(lines) else len(unit.text)
        piece = unit.slice(starts[first], end)
        out.append(piece.slice(0, len(piece.text.rstrip())))
    return out


def units(mapped: Mapped, headings: set[int]) -> list[Unit]:
    """A list paragraph joins the unit before it when that unit ends with ? or : (before closing
    brackets and quotes), holds a list item or starts with a heading line. A unit keeps its lines
    (`Mapped.lined`); its collapsed text is the one that the gate and the labels read."""
    bounds = lines_of(mapped)
    groups: list[list[int]] = []
    for run in paragraphs(mapped):
        if groups and ITEM.match(mapped.text[slice(*bounds[run[0]])]):
            before = [mapped.text[slice(*bounds[n])] for n in groups[-1]]
            if (
                groups[-1][0] in headings
                or before[-1].rstrip().rstrip(CLOSERS).endswith(("?", ":"))
                or any(map(ITEM.match, before))
            ):
                groups[-1] += run
                continue
        groups.append(run)
    found = []
    for group in groups:
        lined = mapped.slice(bounds[group[0]][0], bounds[group[-1]][1]).lined()
        pieces = segments(lined)
        context = 0
        if group[0] in headings:
            if len(pieces) > 1:
                pieces = pieces[1:]
            else:
                head = lined.text.index("\n")
                context = len(lined.text) - len(lined.text[head:].lstrip("\n"))
        found += [Unit(p.text, p.origin, context) for p in pieces]
    return found


def category(text: str) -> str:
    """ "question" for a unit with a "?", "pool" for one with a decision word, else "other". Only
    the first two categories are labelled, trained on and read by the model. Line breaks count as
    spaces."""
    text = " ".join(text.split())
    if "?" in text:
        return "question"
    return "pool" if DECISION.search(text) else "other"
