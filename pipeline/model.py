"""The Ettin encoder as a BIO token tagger: the question model and the option model."""

from functools import partial
from pathlib import Path

import torch
from huggingface_hub import logging as hub_logging
from transformers import AutoModelForTokenClassification, AutoTokenizer, logging

import sentences
from jsonl import DATA
from tagging import TAG_NAMES, Span, Typed, encode, spans_of

logging.set_verbosity_error()
logging.disable_progress_bar()
hub_logging.set_verbosity_error()
MODEL = "jhu-clsp/ettin-encoder-32m"
REVISION = "1b8ba06455dd44f80fc9c1ca9e22806157a57379"
TASKS = ("question", "option")
CHECKPOINT = DATA / "model" / "selected"
BATCH = 32
CPU_TOKENS = 5000  # measured: the GPU costs 100 ms to start and saves 20 ms per 1000 tokens


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pretrained(loader, source, **options):
    """The cached files; the hub only when they are not cached yet."""
    try:
        return loader.from_pretrained(source, local_files_only=True, **options)
    except OSError:
        return loader.from_pretrained(source, **options)


def load_tokenizer(path: Path | None = None):
    if path is None:
        return pretrained(AutoTokenizer, MODEL, revision=REVISION)
    return AutoTokenizer.from_pretrained(path / TASKS[0])


def new_model(task: str):
    return pretrained(
        AutoModelForTokenClassification, MODEL, revision=REVISION, num_labels=len(TAG_NAMES[task])
    )


def save(taggers: dict, tokenizer, path: Path) -> None:
    for task, model in taggers.items():
        model.save_pretrained(path / task)
        tokenizer.save_pretrained(path / task)


def require_tags(found: dict[str, int | list[str]], where: Path) -> None:
    """Refuse a model whose tags are not the ones of this code: a tag count for each task of a
    checkpoint, the tag names of a Core ML settings.json."""
    wanted = {task: len(names) for task, names in TAG_NAMES.items()}
    if found not in (wanted, TAG_NAMES):
        raise ValueError(f"{where}: the model has tags {found}, this code has {TAG_NAMES}")


def load(path: Path) -> dict:
    models = {
        task: AutoModelForTokenClassification.from_pretrained(path / task).eval() for task in TASKS
    }
    require_tags({task: model.config.num_labels for task, model in models.items()}, path)
    return models


def batch_tensors(tokenizer, ids: list[list[int]], hardware, labels=None) -> dict:
    """Padded input_ids and attention_mask, and padded labels when given."""
    width = max(map(len, ids))
    batch = {
        "input_ids": [i + [tokenizer.pad_token_id] * (width - len(i)) for i in ids],
        "attention_mask": [[1] * len(i) + [0] * (width - len(i)) for i in ids],
    }
    if labels is not None:
        batch["labels"] = [t + [-100] * (width - len(t)) for t in labels]
    return {key: torch.tensor(value, device=hardware) for key, value in batch.items()}


def torch_taggers(tokenizer, models: dict, cpu: bool = False) -> dict:
    """For each task, a function from texts (and trailing characters to cut) to their spans: the
    question tagger gives typed spans, the option tagger plain ones. `cpu` keeps the models off
    the GPU, for an exact comparison with another backend."""
    return {
        task: partial(predict, tokenizer, model, task == "question", cpu=cpu)
        for task, model in models.items()
    }


def untyped(spans: list[Typed]) -> list[Span]:
    return [(a, b) for a, b, _ in spans]


def predict(
    tokenizer,
    model,
    typed: bool,
    texts: list[str],
    trailing: str = "",
    cpu: bool = False,
) -> list[list[Typed]] | list[list[Span]]:
    """The tagged character spans of each text."""
    encoded = [encode(tokenizer, text) for text in texts]
    hardware = torch.device("cpu")
    if not cpu and sum(len(ids) for ids, _ in encoded) >= CPU_TOKENS:
        hardware = device()
    model.to(hardware).eval()
    out: list = [[] for _ in texts]
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i][0]))
    for first in range(0, len(order), BATCH):
        batch = order[first : first + BATCH]
        with torch.no_grad():
            output = model(**batch_tensors(tokenizer, [encoded[i][0] for i in batch], hardware))
        if typed:
            scores = output.logits.cpu().tolist()
        else:
            tags = output.logits.argmax(-1).cpu().tolist()
        for row, i in enumerate(batch):
            ids, offsets = encoded[i]
            if typed:
                found = sentences.decode(texts[i], offsets, scores[row][: len(ids)])
            else:
                found = spans_of(texts[i], offsets, tags[row][: len(ids)], trailing)
            out[i] = found if typed else untyped(found)
    return out
