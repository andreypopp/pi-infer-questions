"""The units of the transcripts: the frozen session split, deduplication and the labelling sets."""

import argparse
import getpass
import json
import random
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from jsonl import DATA, read_json, read_jsonl, sha256, write_json, write_jsonl
from normalise import normalise
from redact import redact
from transcripts import ROOTS, messages
from units import Unit, units

SPLIT = DATA / "split.json"
MESSAGES = DATA / "messages.jsonl"
UNITS = DATA / "units.jsonl"
UNITS_META = DATA / "units.json"
SEED = 0
SHARES = (("eval", 0.15), ("dev", 0.10))
SPLITS = ("eval", "dev", "train")
HELD_OUT_SAMPLES = 3


@dataclass(frozen=True)
class Row:
    """text is the unit on one line, which the gate and the labels read; lined is the unit with its
    line breaks, which the models read. Both have the same characters but the white space. None
    means that the line breaks are unknown: the unit was not found again in the transcripts.
    context is the length of the heading or bold line at the start of text, with its space."""

    text: str
    session: str
    split: str
    lined: str | None
    context: int


def digest(items: Iterable[str]) -> str:
    return sha256("\n".join(sorted(items)))


def assign(sessions: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(sessions))
    random.Random(SEED).shuffle(ordered)
    split, start = {}, 0
    for name, share in SHARES:
        count = round(share * len(ordered))
        split |= dict.fromkeys(ordered[start : start + count], name)
        start += count
    return split | dict.fromkeys(ordered[start:], "train")


def freeze_split(sessions: Iterable[str]) -> None:
    split = assign(sessions)
    digests = {name: digest(s for s, n in split.items() if n == name) for name in SPLITS}
    write_json(SPLIT, {"seed": SEED, "digests": digests, "sessions": split})


def load_split() -> dict[str, str]:
    frozen = read_json(SPLIT)
    for name, expected in frozen["digests"].items():
        if digest(s for s, n in frozen["sessions"].items() if n == name) != expected:
            raise ValueError(f"{SPLIT} was edited: the {name} digest differs")
    return frozen["sessions"]


def redacted(unit: Unit, user: str) -> tuple[str, str, int]:
    """(text, lined, context) of a unit, redacted. A redaction that changes the white space of the
    text leaves the unit on one line."""
    text, lined = redact(unit.collapse().text, user=user), redact(unit.text, user=user)
    context = 0
    if unit.context:
        heading = redact(unit.slice(0, unit.context).collapse().text, user=user)
        if not text.startswith(heading + " "):
            raise ValueError("the redaction of a unit crosses its heading")
        context = len(heading) + 1
    return text, lined if " ".join(lined.split()) == text else text, context


def archive() -> list[dict]:
    """The assistant replies of data/messages.jsonl, after it took in the replies of the live
    transcripts that it does not hold yet. A deleted transcript keeps its replies."""
    kept = read_jsonl(MESSAGES)
    known = {(m["session"], m["text"]) for m in kept}
    new = {
        (session, text): {"session": session, "text": text}
        for source, root in ROOTS.items()
        for session, text in messages(source, root)
        if (session, text) not in known
    }
    if new:
        kept += new.values()
        write_jsonl(MESSAGES, kept)
    return kept


def scan(user: str) -> list[tuple[str, str, str, int]]:
    """(session, text, lined, context) of every unit of every archived reply."""
    return [
        (reply["session"], *redacted(unit, user))
        for reply in archive()
        for unit in units(*normalise(reply["text"]))
    ]


def deduplicate(
    rows: list[tuple[str, str, str, int]], split_of: dict[str, str]
) -> tuple[list[Row], dict[str, int]]:
    """One row for each unit text. A text in several splits stays in eval, else in dev."""
    assigned = [
        Row(text, session, split_of[session], lined, context)
        for session, text, lined, context in rows
        if session in split_of
    ]
    best: dict[str, Row] = {}
    shared: set[str] = set()
    for row in assigned:
        kept = best.setdefault(row.text, row)
        if row.split != kept.split:
            shared.add(row.text)
            if SPLITS.index(row.split) < SPLITS.index(kept.split):
                best[row.text] = row
    return list(best.values()), {
        "units": len(rows),
        "unassigned_session": len(rows) - len(assigned),
        "duplicates": len(assigned) - len(best),
        "in_several_splits": len(shared),
    }


def units_digest(rows: Iterable[Row]) -> str:
    return digest(f"{row.split}\t{row.text}" for row in rows)


def build(user: str) -> dict[str, int]:
    """Write the units once, with their digest and the dedup counts. The first call freezes the
    session split too."""
    if UNITS.exists():
        raise FileExistsError(f"{UNITS} is frozen")
    rows = scan(user)
    if not SPLIT.exists():
        freeze_split(session for session, *_ in rows)
    found, counts = deduplicate(rows, load_split())
    write_jsonl(UNITS, (asdict(r) for r in found))
    write_json(UNITS_META, {"digest": units_digest(found), "dedup": counts})
    return counts


def main() -> int:
    """Build the frozen units from the archived pi and Claude Code replies: `uv run corpus`."""
    argparse.ArgumentParser(description=main.__doc__).parse_args()
    try:
        counts = build(getpass.getuser())
    except FileExistsError as error:
        print(f"corpus: {error}", file=sys.stderr)
        return 1
    print(json.dumps(counts))
    return 0


def load_units() -> tuple[list[Row], dict[str, int]]:
    """The frozen units and the dedup counts. The unit texts of the three splits are disjoint."""
    rows = [Row(**item) for item in read_jsonl(UNITS)]
    meta = read_json(UNITS_META)
    if units_digest(rows) != meta["digest"]:
        raise ValueError(f"{UNITS} was edited: its digest differs")
    if len({row.text for row in rows}) != len(rows):
        raise ValueError(f"{UNITS} has a unit text in two places")
    return rows, meta["dedup"]


def uniform_order(rows: Iterable[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: sha256(row.text))


def samples(row: Row) -> int:
    return HELD_OUT_SAMPLES if row.split in ("dev", "eval") else 1


def has_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def labelling_rows(rows: list[Row]) -> list[Row]:
    """The units that any labelling set can hold: those with a letter."""
    return [row for row in rows if has_letters(row.text)]
