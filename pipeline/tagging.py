"""BIO tags over tokens, and the character spans they stand for. A tag has a kind: BEGIN and INSIDE
of kind k are BEGIN + 2k and INSIDE + 2k. The question tagger has the three KINDS, the option
tagger one."""

import re
import unicodedata

OUT, BEGIN, INSIDE = 0, 1, 2
KINDS = ("yes_no", "open", "choice")
TAG_NAMES = {
    "question": ["O", *(f"{part}-{kind}" for kind in KINDS for part in ("B", "I"))],
    "option": ["O", "B", "I"],
}
IGNORE = -100
MAX_TOKENS = 512
OPTION_TRAILING = ",.?;:"
CODE_SPAN = re.compile(r"(`+)[^`\n]+?\1")
CLOSING = {")": "(", "]": "[", "”": "“", '"': '"', "'": "'"}
OPENING = {"(": ")", "[": "]", "“": "”", '"': '"', "'": "'"}

Span = tuple[int, int]
Typed = tuple[int, int, int]  # start, end and the index of the kind


def encode(tokenizer, text: str) -> tuple[list[int], list[Span]]:
    """Token ids and character offsets, at most MAX_TOKENS, with the special tokens at (0, 0).
    An offset never has a space at its edge."""
    encoded = tokenizer(text, truncation=True, max_length=MAX_TOKENS, return_offsets_mapping=True)
    offsets = []
    for start, end in encoded["offset_mapping"]:
        while start < end and text[start].isspace():
            start += 1
        offsets.append((start, end) if start < end else (0, 0))
    return encoded["input_ids"], offsets


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def tags_for(offsets: list[Span], spans: list[Typed]) -> list[int]:
    """BEGIN on the first token of each span, INSIDE on the rest, IGNORE on special tokens."""
    tags = [IGNORE if end == 0 else OUT for _, end in offsets]
    for first, last, kind in spans:
        inside = [i for i, (s, e) in enumerate(offsets) if e > s and s < last and e > first]
        for position, i in enumerate(inside):
            tags[i] = (BEGIN if position == 0 else INSIDE) + 2 * kind
    return tags


def unbalanced(part: str, char: str, mate: str) -> bool:
    """char is a bracket or quote of part that its mate does not match."""
    if char == mate:
        return part.count(char) % 2 == 1
    return part.count(char) > part.count(mate)


def trim(text: str, span: Span) -> Span:
    """A token can hold a bracket or a quote after a "?". Cut a closing one at the end of the span
    that has no opener inside it, and an opening one at the start that has no closer."""
    start, end = span
    while end - start > 1 and text[end - 1] in CLOSING:
        if not unbalanced(text[start:end], text[end - 1], CLOSING[text[end - 1]]):
            break
        end -= 1
    while end - start > 1 and text[start] in OPENING:
        if not unbalanced(text[start:end], text[start], OPENING[text[start]]):
            break
        start += 1
    return start, end


def balance_ticks(text: str, span: Span) -> Span:
    """A span with an odd number of backticks has lost half of a code span at an edge: extend the
    end to a closing backtick that follows within a word, else the start to one right before it,
    else cut the backtick at an edge."""
    start, end = span
    if text[start:end].count("`") % 2 == 0:
        return span
    closer = text.find("`", end)
    if closer != -1 and closer - end <= 8 and not any(c.isspace() for c in text[end:closer]):
        return start, closer + 1
    if start > 0 and text[start - 1] == "`":
        return start - 1, end
    if text[start] == "`":
        return start + 1, end
    return (start, end - 1) if text[end - 1] == "`" else span


def message_range(origin: list[int], message: str, start: int, end: int) -> Span:
    """The range of the unit span [start, end) in the message, grown to whole code spans: Markdown
    normalisation drops the backticks of code spans from the unit, so a span can end inside one
    or next to it."""
    a, b = origin[start], origin[end - 1] + 1
    for code in CODE_SPAN.finditer(message):
        if code.start() < b and a < code.end():
            a, b = min(a, code.start()), max(b, code.end())
    return a, b


def cut_trailing(text: str, span: Span, trailing: str) -> Span:
    """The span without the characters of `trailing` at its end. A token can hold a comma or a
    full stop with the word before it."""
    start, end = span
    while end - start > 1 and text[end - 1] in trailing:
        end -= 1
    return start, end


def with_marks(text: str, end: int) -> int:
    """The end moved over the combining marks that follow it: the offset of a token covers only
    the first scalar of a character that normalisation composed."""
    while end < len(text) and unicodedata.category(text[end]).startswith("M"):
        end += 1
    return end


def spans_of(text: str, offsets: list[Span], tags: list[int], trailing: str = "") -> list[Typed]:
    """BEGIN INSIDE* runs of one kind as character spans, without brackets, quotes and `trailing`
    characters that the tokens added at the edges; a span with nothing left, or only `trailing`
    characters, is dropped. An INSIDE after no run, or after a run of another kind, starts
    nothing."""
    found: list[Typed] = []
    start = last = None
    kind = 0

    def close() -> None:
        span = balance_ticks(text, trim(text, (start, with_marks(text, last))))
        a, b = trim(text, cut_trailing(text, span, trailing))
        if text[a:b].strip(trailing):
            found.append((a, b, kind))

    for (s, e), tag in zip(offsets, tags, strict=True):
        if e == 0:
            continue
        if tag > OUT and start is not None and tag % 2 == 0 and (tag - 1) // 2 == kind:
            last = e
            continue
        if start is not None:
            close()
        start, last = (s, e) if tag > OUT and tag % 2 == 1 else (None, None)
        kind = (tag - 1) // 2
    if start is not None:
        close()
    return found


def after_context(spans: list[Typed], context: int) -> list[Typed]:
    """The spans without the first `context` characters of the text: a span that ends there is
    dropped, one that starts there starts after it."""
    return [(max(a, context), b, k) for a, b, k in spans if b > context]


def project(tokenizer, text: str, spans: list[Span], trailing: str = "") -> list[Span]:
    """The spans as the tokens can hold them: what the tags of the spans decode to."""
    _, offsets = encode(tokenizer, text)
    typed = spans_of(text, offsets, tags_for(offsets, [(a, b, 0) for a, b in spans]), trailing)
    return [(a, b) for a, b, _ in typed]
