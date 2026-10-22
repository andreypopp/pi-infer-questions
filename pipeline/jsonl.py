"""Private atomic writes, JSON and JSON lines files, and the hash that names things."""

import hashlib
import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def short_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def sha256(text: str) -> str:
    return short_digest(text.encode())


def file_digest(path: Path) -> str:
    return short_digest(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    """Replace the file whole, with mode 0600; a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """The records of a file; a missing file has none."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, items: Iterable[object]) -> None:
    write_text(path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))


def append_jsonl(path: Path, item: object) -> None:
    line = (json.dumps(item, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)
