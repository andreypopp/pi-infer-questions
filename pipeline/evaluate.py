"""Scores of the taggers on labelled units: question and option F1, kinds, exact match, false
positives."""

from label import gold_of, lined_context
from tagging import (
    KINDS,
    MAX_TOKENS,
    OPTION_TRAILING,
    Span,
    after_context,
    project,
    token_count,
)
from units import ITEM

Unit = list[
    tuple[Span, str, tuple[Span, ...]]
]  # (question span, kind, option spans in the question)


def project_gold(
    tokenizer, text: str, gold: Unit
) -> list[tuple[Span, Span, str, tuple[Span, ...]] | None]:
    """For each gold question: its span in the text, its projected span, its kind and its projected
    options, or None when the tokens cut it off."""
    out = []
    for span, kind, options in gold:
        question = project(tokenizer, text, [span])
        if not question:
            out.append(None)
            continue
        first, last = question[0]
        inside = [
            (span[0] + x - first, span[0] + y - first)
            for x, y in options
            if first <= span[0] + x and span[0] + y <= last
        ]
        kept = tuple(project(tokenizer, text[first:last], inside, OPTION_TRAILING))
        out.append((span, question[0], kind, kept))
    return out


def changed(gold: Unit, held: list) -> int:
    """How many gold questions and options the tokens change or lose."""
    count = 0
    for (span, _, options), item in zip(gold, held, strict=True):
        if item is None:
            count += 1 + len(options)
            continue
        wanted = {(span[0] + x, span[0] + y) for x, y in options}
        kept = {(item[1][0] + x, item[1][0] + y) for x, y in item[3]}
        count += (item[1] != span) + len(wanted - kept)
    return count


def decode(taggers: dict, texts: list[str], contexts: list[int]) -> list[Unit]:
    """The questions of each text with their kinds. The option tagger reads the choices only."""
    spans = [
        after_context(found, context)
        for found, context in zip(taggers["question"](texts), contexts, strict=True)
    ]
    asked = [
        t[a:b]
        for t, found in zip(texts, spans, strict=True)
        for a, b, k in found
        if KINDS[k] == "choice"
    ]
    options = iter(taggers["option"](asked, OPTION_TRAILING))
    return [
        [
            ((a, b), KINDS[k], tuple(next(options)) if KINDS[k] == "choice" else ())
            for a, b, k in found
        ]
        for found in spans
    ]


def f1(found: set, wanted: set) -> float:
    hit = len(found & wanted)
    return 2 * hit / (len(found) + len(wanted)) if found or wanted else 1.0


def questions_of(units: list[Unit]) -> set:
    return {(n, q) for n, unit in enumerate(units) for q, _, _ in unit}


def options_of(units: list[Unit]) -> set:
    return {(n, q, o) for n, unit in enumerate(units) for q, _, options in unit for o in options}


def kinded(units: list[Unit], kind: str) -> set:
    return {(n, q) for n, unit in enumerate(units) for q, k, _ in unit if k == kind}


def overlap_hits(found: list[Span], wanted: list[Span]) -> int:
    """How many found spans overlap a wanted span that no other found span matched."""
    used: set[int] = set()
    for a, b in sorted(found):
        for j, (c, d) in enumerate(sorted(wanted)):
            if j not in used and a < d and c < b:
                used.add(j)
                break
    return len(used)


def overlap_scores(pred: list[Unit], gold: list[Unit]) -> dict[str, float]:
    hits = sum(
        overlap_hits([q for q, _, _ in p], [q for q, _, _ in g])
        for p, g in zip(pred, gold, strict=True)
    )
    found, wanted = sum(map(len, pred)), sum(map(len, gold))
    return {
        "question_f1_overlap": 2 * hits / (found + wanted) if found + wanted else 1.0,
        "overlap_recall": hits / max(1, wanted),
        "overlap_precision": hits / max(1, found),
    }


def choice_hit(pred: list[Unit], gold: list[Unit]) -> float:
    """The share of gold choices that a predicted choice overlaps with at least half of the gold
    options."""
    wanted = hits = 0
    for p, g in zip(pred, gold, strict=True):
        for (c, d), kind, options in g:
            if kind == "choice":
                wanted += 1
                hits += any(
                    k == "choice" and a < d and c < b and len(found) >= (len(options) + 1) // 2
                    for (a, b), k, found in p
                )
    return hits / max(1, wanted)


def ceiling(originals: list[Unit], held: list) -> dict[str, float]:
    """F1 of the gold as the tokens hold it against the gold itself, for questions and options:
    the best that a perfect tagger can score on exact spans."""
    wanted_q = questions_of(originals)
    wanted_o = options_of(originals)
    got_q, got_o = set(), set()
    for n, (unit, items) in enumerate(zip(originals, held, strict=True)):
        for (q, _, _), item in zip(unit, items, strict=True):
            if item:
                got_q.add((n, q) if item[1] == q else (n, item[1]))
                got_o |= {(n, q, o) for o in item[3]}
    return {
        "ceiling_question_f1": f1(got_q, wanted_q),
        "ceiling_option_f1": f1(got_o, wanted_o),
    }


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def part(gold: list[Unit], pred: list[Unit], inner: list[Unit]) -> dict[str, float]:
    matched = questions_of(pred) & questions_of(gold)
    kind_of = {(n, q): k for n, unit in enumerate(gold) for q, k, _ in unit}
    return {
        "units": len(gold),
        "question_f1": f1(questions_of(pred), questions_of(gold)),
        "kind_accuracy": sum(
            (n, q) in matched and kind_of[n, q] == k
            for n, unit in enumerate(pred)
            for q, k, _ in unit
        )
        / max(1, len(matched)),
        **{f"kind_f1_{k}": f1(kinded(pred, k), kinded(gold, k)) for k in KINDS},
        **overlap_scores(pred, gold),
        "choice_hit": choice_hit(pred, gold),
        "fpr": rate([bool(p) for p, g in zip(pred, gold, strict=True) if not g]),
        "option_f1": f1(options_of(pred), options_of(gold)),
        "option_f1_in_gold": f1(options_of(inner), options_of(gold)),
        "exact": sum(p == g for p, g in zip(pred, gold, strict=True)) / max(1, len(gold)),
    }


def markers(text: str) -> list[int]:
    """The offsets of the lines that start with a list marker."""
    return [
        sum(map(len, text.split("\n")[:n])) + n
        for n, line in enumerate(text.split("\n"))
        if ITEM.match(line)
    ]


def is_option_list(text: str, gold: Unit) -> bool:
    items = markers(text)
    return len(items) >= 2 and any(
        kind == "choice" and a <= items[0] and b > items[-1] for (a, b), kind, _ in gold
    )


def evaluate(tokenizer, taggers: dict, records: list[dict]) -> dict[str, dict]:
    """Scores of all units, of the units with 2 or more questions, of those with a choice and of
    the three kinds of list unit, against the gold as the tokens hold it. The "all" part also has
    the false positive rates, the number of gold spans that the tokens change and the units over
    the token limit."""
    texts = [r["lined"] for r in records]
    originals = [gold_of(r) for r in records]
    held = [project_gold(tokenizer, t, g) for t, g in zip(texts, originals, strict=True)]
    gold = [[(q, kind, options) for _, q, kind, options in filter(None, unit)] for unit in held]
    pred = decode(taggers, texts, [lined_context(r) for r in records])
    choices = [
        t[a:b]
        for t, unit in zip(texts, gold, strict=True)
        for (a, b), kind, _ in unit
        if kind == "choice"
    ]
    options = iter(taggers["option"](choices, OPTION_TRAILING))
    inner = [
        [(q, kind, tuple(next(options)) if kind == "choice" else ()) for q, kind, _ in unit]
        for unit in gold
    ]
    parts = {
        "all": lambda t, g: True,
        "multi": lambda t, g: len(g) >= 2,
        "choice": lambda t, g: any(kind == "choice" for _, kind, _ in g),
        "option_lists": is_option_list,
        "question_lists": lambda t, g: bool(ITEM.match(t)) and "?" in t,
        "decision_lists": lambda t, g: len(markers(t)) >= 2 and "?" not in t,
    }
    out = {}
    for name, keep in parts.items():
        chosen = [i for i, (t, g) in enumerate(zip(texts, gold, strict=True)) if keep(t, g)]
        out[name] = part(*([x[i] for i in chosen] for x in (gold, pred, inner)))
    empty = [(t, bool(p)) for t, g, p in zip(texts, originals, pred, strict=True) if not g]
    asked, plain = (
        [
            {
                (n, q)
                for n, unit in enumerate(units)
                for q, _, _ in unit
                if ("?" in texts[n][q[0] : q[1]]) == marked
            }
            for units in (gold, pred)
        ]
        for marked in (True, False)
    )
    out["all"] |= {
        "proposals_gold": len(plain[0]),
        "proposals_found": len(plain[1]),
        "proposal_recall": len(plain[0] & plain[1]) / max(1, len(plain[0])),
        "proposal_precision": len(plain[0] & plain[1]) / max(1, len(plain[1])),
        "recall_question_mark": len(asked[0] & asked[1]) / max(1, len(asked[0])),
        "fpr": rate([found for _, found in empty]),
        "fpr_question_mark": rate([found for t, found in empty if "?" in t]),
        "fpr_plain": rate([found for t, found in empty if "?" not in t]),
        **ceiling(originals, held),
        "gold_changed_by_tokens": sum(map(changed, originals, held)),
        "over_max_tokens": sum(token_count(tokenizer, t) > MAX_TOKENS for t in texts),
    }
    return out
