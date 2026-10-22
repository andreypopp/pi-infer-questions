"""Smoke tests of `extract`: the model test skips until a checkpoint exists."""

from pathlib import Path

import pytest

from extract import extract, main
from model import CHECKPOINT, load, load_tokenizer, torch_taggers
from tagging import balance_ticks

FIXTURES = [
    p.read_text(encoding="utf-8")
    for p in sorted(Path(__file__).parent.parent.glob("fixtures/*.txt"))
] + [""]


@pytest.mark.skipif(not (CHECKPOINT / "question").is_dir(), reason="no trained checkpoint yet")
def test_questions_and_options_are_exact_substrings_in_text_order():
    tokenizer, taggers = load_tokenizer(CHECKPOINT), load(CHECKPOINT)
    taggers = torch_taggers(tokenizer, taggers)
    plan = "or a plan in `plans/` for it"
    assert balance_ticks(plan, (3, 20)) == (3, 21)
    assert balance_ticks("x `y z", (0, 3)) == (0, 2)
    assert balance_ticks(plan, (14, 21)) == (13, 21)
    for text in FIXTURES:
        found, _ = extract(tokenizer, taggers, text)
        assert [q["range"][0] for q in found] == sorted(q["range"][0] for q in found)
        for q in found:
            start, end = q["range"]
            assert text[start:end] == q["question"]
            for option, (a, b) in zip(q["options"], q["option_ranges"], strict=True):
                assert text[a:b] == option and start <= a and b <= end


def test_invalid_utf8_is_refused(tmp_path, capsys):
    path = tmp_path / "in.txt"
    path.write_bytes(b"\xff\xfe")
    assert main([str(path)]) == 1
    assert "not UTF-8" in capsys.readouterr().err
