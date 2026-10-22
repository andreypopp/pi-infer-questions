"""The assistant text blocks of pi and Claude Code transcripts."""

import json
from collections.abc import Iterator
from pathlib import Path

ROOTS = {
    "pi": Path.home() / ".pi" / "agent" / "sessions",
    "claude": Path.home() / ".claude" / "projects",
}


def messages(source: str, root: Path) -> Iterator[tuple[str, str]]:
    """(session, text) of each assistant text block under root, in file and line order. A
    session is "<source>:<id>"."""
    for path in sorted(root.rglob("*.jsonl")):
        session = path.stem
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                if '"assistant"' not in raw and '"session' not in raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "session" and isinstance(item.get("id"), str):
                    session = item["id"]
                if isinstance(item.get("sessionId"), str):
                    session = item["sessionId"]
                message = item.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                    continue
                if source == "pi":
                    assistant = item.get("type") == "message" and message.get("role") == "assistant"
                else:
                    assistant = item.get("type") == "assistant"
                for block in message["content"] if assistant else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            yield f"{source}:{session}", text
