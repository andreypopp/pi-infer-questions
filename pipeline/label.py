"""Label units with one DeepSeek call each: validation, one correction turn, sample agreement."""

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from openai import APIError

from corpus import SPLITS, Row
from deepseek import STOP_SPEND, BudgetExceeded, PeakHours, Session, complete
from jsonl import DATA, append_jsonl, read_jsonl, sha256
from tagging import KINDS, Span
from units import ITEM, category

LABELS = DATA / "labels.jsonl"
PROMPT = Path(__file__).parent / "unit_prompt.md"
MIN_OPTION = 3
WORKERS = 128


@dataclass(frozen=True)
class Question:
    question: str
    kind: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class Dropped:
    reason: str  # "length" or "invalid"


@dataclass(frozen=True)
class Prompt:
    text: str
    digest: str


def current(record: dict, prompt: Prompt) -> bool:
    return record["prompt"] == prompt.digest


def load_prompt(path: Path = PROMPT) -> Prompt:
    text = path.read_text()
    return Prompt(text, sha256(text))


def occurrences(text: str, part: str) -> list[int]:
    """Where part occurs in text, also where occurrences overlap."""
    found, start = [], text.find(part)
    while start >= 0:
        found.append(start)
        start = text.find(part, start + 1)
    return found


def option_errors(number: int, question: str, options: list[str]) -> list[str]:
    errors, spans = [], []
    for option in options:
        where = occurrences(question, option)
        if len(option) < MIN_OPTION:
            errors.append(f"question {number}: an option is shorter than {MIN_OPTION} characters")
        elif option != option.strip() or option.endswith(".") or ITEM.match(option + " "):
            errors.append(
                f"question {number}: an option has an edge space, a full stop or a marker"
            )
        elif len(where) != 1:
            errors.append(f"question {number}: an option does not occur once in the question")
        else:
            spans.append((where[0], where[0] + len(option)))
    if len(set(options)) != len(options):
        errors.append(f"question {number}: duplicate options")
    spans.sort()
    if any(after[0] < before[1] for before, after in zip(spans, spans[1:], strict=False)):
        errors.append(f"question {number}: options overlap")
    return errors


def parse(unit: str, content: str, context: int) -> tuple[list[Question], list[str]]:
    """The questions of an answer, and the errors of the answer. The errors never quote it. A
    question never starts in the first `context` characters of the unit."""
    try:
        answer = json.loads(content)
        items = answer["questions"]
        if set(answer) != {"questions"} or not isinstance(items, list):
            raise ValueError
        questions = [
            Question(i["question"], i["kind"], tuple(i["options"]))
            for i in items
            if set(i) == {"question", "kind", "options"}
        ]
        if len(questions) != len(items) or not all(
            isinstance(q.question, str) and all(isinstance(o, str) for o in q.options)
            for q in questions
        ):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return [], ["the answer does not have the shape of the schema"]
    errors, end = [], 0
    for number, q in enumerate(questions, 1):
        if q.kind not in KINDS:
            errors.append(f"question {number}: the kind is not choice, yes_no or open")
        elif (q.kind == "choice") != bool(q.options):
            errors.append(f"question {number}: only a choice has options, and it has them")
        if q.question != q.question.strip():
            errors.append(f"question {number}: the text has an edge space")
            continue
        where = occurrences(unit, q.question) if q.question else []
        if len(where) != 1:
            errors.append(f"question {number}: the text does not occur exactly once in the unit")
            continue
        if where[0] < context:
            errors.append(
                f"question {number}: starts in the heading or bold line before the list, which "
                "is context and never part of a question"
            )
        if where[0] < end:
            errors.append(f"question {number}: not in text order, or overlaps the one before")
        end = max(end, where[0] + len(q.question))
        errors += option_errors(number, q.question, list(q.options))
    return ([] if errors else questions), errors


def label_unit(
    session: Session, prompt: Prompt, unit: str, context: int, nonce: str
) -> tuple[list[Question] | Dropped, bool]:
    """The questions of a unit, or why it was dropped; and whether a correction turn was needed."""
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": f"Text: {unit}"},
    ]
    for turn in range(2):
        content, finish = complete(session, messages, nonce)
        if finish == "length":
            return Dropped("length"), turn == 1
        questions, errors = parse(unit, content, context)
        if not errors:
            return questions, turn == 1
        correction = "Errors in your answer:\n" + "\n".join(errors) + "\nAnswer again, as JSON."
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": correction},
        ]
    return Dropped("invalid"), True


def canonical(questions: list[Question]) -> str:
    return json.dumps(
        [[q.question, q.kind, sorted(q.options, key=q.question.index)] for q in questions]
    )


def agreed(results: list[list[Question] | Dropped]) -> list[Question] | None:
    """The label that at least 2 of 3 samples give, else None."""
    votes = Counter(canonical(r) for r in results if not isinstance(r, Dropped))
    if not votes or votes.most_common(1)[0][1] < 2:
        return None
    best = votes.most_common(1)[0][0]
    return next(r for r in results if not isinstance(r, Dropped) and canonical(r) == best)


def label_row(session: Session, prompt: Prompt, row: Row, samples: int) -> dict:
    """The record of a row. Its status is labelled, length, invalid or disagree."""
    outcomes = [
        label_unit(session, prompt, row.text, row.context, f"sample{n}") for n in range(samples)
    ]
    results = [result for result, _ in outcomes]
    label = results[0] if len(results) == 1 else agreed(results)
    drops = [r.reason for r in results if isinstance(r, Dropped)]
    if isinstance(label, list):
        status = "labelled"
    elif isinstance(label, Dropped):
        status = label.reason
    else:
        status = "length" if "length" in drops else "disagree"
    questions = label if status == "labelled" else []
    return {
        "text": row.text,
        "status": status,
        "questions": [
            {"question": q.question, "kind": q.kind, "options": list(q.options)} for q in questions
        ],
        "prompt": prompt.digest,
        "corrections": sum(corrected for _, corrected in outcomes),
        "length_stops": drops.count("length"),
    }


def lined_context(record: dict) -> int:
    """The length of the record's context in its lined text: the position of the first character
    after it."""
    text, lined = record["text"], record["lined"]
    if not record["context"]:
        return 0
    places = [i for i, char in enumerate(lined) if not char.isspace()]
    return places[sum(not char.isspace() for char in text[: record["context"]])]


def gold_of(record: dict) -> list[tuple[Span, str, tuple[Span, ...]]]:
    """(question span, kind, option spans in the question) of a labelled record, in text order,
    as spans of the record's lined text. The lined text has the characters of the text, so the
    n-th character that is no white space is the same in both."""
    text, lined = record["text"], record["lined"]
    places = [i for i, char in enumerate(lined) if not char.isspace()]
    before = list(accumulate((not char.isspace() for char in text), initial=0))

    def lined_span(start: int, end: int) -> Span:
        return places[before[start]], places[before[end] - 1] + 1

    found = []
    for q in record["questions"]:
        start = text.index(q["question"])
        a, b = lined_span(start, start + len(q["question"]))
        options = sorted(
            lined_span(start + q["question"].index(o), start + q["question"].index(o) + len(o))
            for o in q["options"]
        )
        found.append(((a, b), q["kind"], tuple((x - a, y - a) for x, y in options)))
    return found


def load_labelled(
    path: Path, rows: list[Row], prompt: Prompt
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """The labelled records by split, and how many records were skipped. The split is the one of
    the frozen units. A record of a text that is no unit now, or of another prompt, is skipped. So
    is a dev or eval record without line breaks; a train one reads its unit on one line."""
    split_of = {row.text: row.split for row in rows}
    lined_of = {row.text: row.lined for row in rows}
    context_of = {row.text: row.context for row in rows}
    labelled: dict[str, list[dict]] = {split: [] for split in SPLITS}
    skipped = {"not_a_unit": 0, "other_prompt": 0, "no_lines": 0}
    latest: dict[str, dict] = {}
    for record in read_jsonl(path):
        if record["text"] not in split_of:
            skipped["not_a_unit"] += 1
        elif not current(record, prompt):
            skipped["other_prompt"] += 1
        elif record["status"] == "labelled" and category(record["text"]) != "other":
            latest[record["text"]] = record
    for text, record in latest.items():
        split, lined = split_of[text], lined_of[text]
        if lined is None and split != "train":
            skipped["no_lines"] += 1
        else:
            labelled[split].append(record | {"lined": lined or text, "context": context_of[text]})
    return labelled, skipped


def run(
    session: Session,
    prompt: Prompt,
    rows: list[Row],
    out: Path,
    workers: int,
    samples: int,
) -> dict[str, int]:
    """Label the rows that `out` does not have yet, appending a record for each, from `samples`
    calls each. A budget, spend or peak stop ends the run, and an API error leaves its row for a
    rerun. Returns the count of each status."""
    done = {r["text"] for r in read_jsonl(out) if current(r, prompt)}
    stop = threading.Event()
    errors: list[str] = []

    def work(row: Row) -> None:
        if session.ledger.total > STOP_SPEND:
            stop.set()
        if stop.is_set():
            return
        try:
            record = label_row(session, prompt, row, samples)
        except (BudgetExceeded, PeakHours):
            stop.set()
            return
        except APIError:
            errors.append("error")
            return
        append_jsonl(out, record)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, [row for row in rows if row.text not in done]))
    records = [r for r in read_jsonl(out) if current(r, prompt)]
    return dict(Counter(r["status"] for r in records)) | {
        "error": len(errors),
        "stopped": int(stop.is_set()),
    }
