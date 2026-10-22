"""DeepSeek flash client: spend ledger with a limit, response cache, retries, peak guard."""

import fcntl
import getpass
import json
import os
import random
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from jsonl import DATA, append_jsonl, read_jsonl, sha256, write_text
from redact import redact

LEDGER = DATA / "ledger.jsonl"
CACHE = DATA / "cache"
MODEL = "deepseek-flash"
EFFORT = "low"
PRICES = {  # per 1M tokens, (off-peak, peak): https://api-docs.deepseek.com/quick_start/pricing
    "hit": (0.003, 0.006),
    "miss": (0.15, 0.3),
    "output": (0.6, 1.2),
}
MAX_OUTPUT = 8000
PEAK_HOURS = ((1, 4), (6, 10))
PEAK_MARGIN = timedelta(minutes=15)
LIMIT = 50.0
STOP_SPEND = 47.0  # a run starts no new unit above this total
MAX_ATTEMPTS = 5
NOT_BILLED = {429, 500, 502, 503}


def now() -> datetime:
    return datetime.now(UTC)


def is_peak(moment: datetime) -> bool:
    return moment.weekday() < 5 and any(a <= moment.hour < b for a, b in PEAK_HOURS)


def near_peak(moment: datetime) -> bool:
    """Peak now, or within the margin: a request that starts now may end in peak."""
    return is_peak(moment) or is_peak(moment + PEAK_MARGIN)


class PeakHours(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class LedgerBusy(Exception):
    pass


def price(hit: int, miss: int, output: int, *, peak: bool) -> float:
    i = int(peak)
    return (hit * PRICES["hit"][i] + miss * PRICES["miss"][i] + output * PRICES["output"][i]) / 1e6


def usage_fields(usage: object, *, peak: bool) -> dict[str, float]:
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    output = int(getattr(usage, "completion_tokens", 0) or 0)
    hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0) or max(0, prompt - hit)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "hit": hit,
        "miss": miss,
        "output": output,
        "reasoning": int(getattr(details, "reasoning_tokens", 0) or 0),
        "peak": peak,
        "cost": price(hit, miss, output, peak=peak),
    }


def worst_cost(messages: list[dict[str, str]]) -> float:
    """Every UTF-8 byte is at most one token."""
    tokens = sum(len(m["content"].encode()) + 20 for m in messages) + 500
    return price(0, tokens, MAX_OUTPUT, peak=True)


class Ledger:
    """Spend of this pipeline. A reservation is written before a call; one without a settle
    entry counts at its reserved maximum. One process holds the ledger at a time."""

    def __init__(self, path: Path) -> None:
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.guard = open(path.with_name(path.name + ".lock"), "w")
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LedgerBusy(f"another process holds {path}") from None
        held: dict[str, float] = {}
        self.total = 0.0
        for item in read_jsonl(path):
            if "reserve" in item:
                held[item["id"]] = item["reserve"]
            else:
                held.pop(item["id"], None)
                self.total += item["cost"]
        self.total += sum(held.values())

    def reserve(self, worst: float) -> str:
        with self.lock:
            if self.total + worst > LIMIT:
                raise BudgetExceeded
            reservation = uuid.uuid4().hex
            append_jsonl(self.path, {"id": reservation, "reserve": worst})
            self.total += worst
            return reservation

    def settle(
        self, reservation: str, worst: float, fields: dict[str, float], at: datetime
    ) -> None:
        with self.lock:
            stamp = at.isoformat(timespec="seconds")
            append_jsonl(self.path, {"id": reservation, "at": stamp, **fields})
            self.total += fields["cost"] - worst

    def close(self) -> None:
        self.guard.close()


@dataclass(frozen=True)
class Session:
    """What a call needs: the client, the ledger, the cache directory and the peak policy."""

    client: OpenAI
    ledger: Ledger
    cache: Path
    user: str
    allow_peak: bool


def send(session: Session, body: dict[str, object]) -> tuple[str, str]:
    worst = worst_cost(body["messages"])  # type: ignore[arg-type]
    attempt = 0
    while True:
        if near_peak(now()) and not session.allow_peak:
            raise PeakHours
        peak = is_peak(now())
        reservation = session.ledger.reserve(worst)
        try:
            response = session.client.chat.completions.create(**body)
        except APIStatusError as error:
            if error.status_code not in NOT_BILLED:
                raise
            session.ledger.settle(reservation, worst, {"cost": 0.0}, now())
            failure: Exception = error
        except (APIConnectionError, APITimeoutError) as error:
            failure = error  # the server may have billed it: the reservation stays at its maximum
        else:
            usage = getattr(response, "usage", None)
            fields = usage_fields(usage, peak=peak) if usage else {"cost": worst}
            session.ledger.settle(reservation, worst, fields, now())
            choice = response.choices[0]
            return choice.message.content or "", str(choice.finish_reason or "stop")
        attempt += 1
        if attempt == MAX_ATTEMPTS:
            raise failure
        time.sleep(min(30.0, float(2**attempt)) + random.random())


def complete(session: Session, messages: list[dict[str, str]], nonce: str) -> tuple[str, str]:
    """(content, finish_reason). A user message that redaction would change is refused. A cached
    answer costs nothing, and a damaged cache file is a miss. A request that would start within
    the margin before peak hours raises PeakHours before it is sent."""
    for message in messages:
        if (
            message["role"] == "user"
            and redact(message["content"], user=session.user) != message["content"]
        ):
            raise ValueError("a message is not redacted")
    body = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT,
        "reasoning_effort": EFFORT,
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    path = session.cache / f"{sha256(json.dumps([body, nonce], sort_keys=True))}.json"
    try:
        item = json.loads(path.read_text())
        return item["content"], item["finish_reason"]
    except (OSError, ValueError, KeyError):
        pass
    content, finish = send(session, body)
    write_text(path, json.dumps({"content": content, "finish_reason": finish}))
    return content, finish


@contextmanager
def open_session(name: str, peak: bool, cache: Path = CACHE) -> Iterator[Session | None]:
    """A session on the ledger of this pipeline, or None after a message to stderr: no API key, or
    another process holds the ledger. The ledger closes when the block ends."""
    client = open_client()
    if client is None:
        yield None
        return
    try:
        ledger = Ledger(LEDGER)
    except LedgerBusy as error:
        print(f"{name}: {error}", file=sys.stderr)
        yield None
        return
    try:
        yield Session(client, ledger, cache, getpass.getuser(), peak)
    finally:
        ledger.close()


def open_client() -> OpenAI | None:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return None
    return OpenAI(api_key=key, base_url="https://api.deepseek.com", max_retries=0, timeout=900)
