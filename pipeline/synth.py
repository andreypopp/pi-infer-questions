"""Synthetic assistant messages for training: `uv run synth --calls 10` generates, runs the units
pipeline and labels, `uv run synth --report` prints the statistics.

One DeepSeek call writes a few messages for a sampled scenario; the call is a session. Files are in
data/synth/: generations.jsonl and labels.jsonl (the format of data/labels.jsonl). The units are
derived from the generations, train split only. Nothing in data/units.jsonl or data/labels.jsonl
changes.
"""

import argparse
import getpass
import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import APIError

from corpus import Row, has_letters, load_units, redacted
from deepseek import BudgetExceeded, PeakHours, complete, open_session
from jsonl import DATA, append_jsonl, read_jsonl
from label import Prompt, current, load_prompt, run
from normalise import normalise
from units import category, units

SYNTH = DATA / "synth"
GENERATIONS = SYNTH / "generations.jsonl"
LABELS = SYNTH / "labels.jsonl"
PROMPT = Path(__file__).parent / "synth_prompt.md"
NEGATIVE_SHARES = {
    "base": 0.2,
    "negatives": 0.7,
    "narration": 0.85,
    "options": 0.15,
    "lists": 0.4,
    "lists-check": 0.4,
}
BATCHES = (*NEGATIVE_SHARES, "sonnet")  # "sonnet" is stored data: read and labelled, not generated
TRAIN_BATCHES = tuple(b for b in BATCHES if b != "lists-check")  # lists-check is held out
LIST_BATCHES = ("lists", "lists-check")
WORKERS = 24

PROJECTS = (
    "a web app (React frontend, Node API)",
    "a command-line tool in Rust",
    "a compiler for a small language",
    "an iOS app in SwiftUI",
    "a data pipeline with Airflow and Spark",
    "infrastructure as code with Terraform",
    "a machine learning training repository in PyTorch",
    "a 2D game in Godot",
    "a documentation site built with a static site generator",
    "operating system tooling (shell scripts, launchd, systemd)",
    "a Python library published to PyPI",
    "a Go microservice with a Postgres database",
    "an Android app in Kotlin",
    "a monorepo with a CI pipeline in GitHub Actions",
    "a database migration tool",
    "a browser extension",
)
SITUATIONS = (
    "it finished a change and wants the user's decision on the next step",
    "it found a bug while working on something else",
    "it is blocked by a design choice that only the user can make",
    "it finished a code review and has findings",
    "tests are failing and it has several ways to proceed",
    "it proposes a plan before it starts a larger task",
    "a migration ran and it needs a decision about the leftovers",
    "benchmark results came in and it must choose between approaches",
    "a dependency upgrade conflicts with another package",
    "it found existing code that overlaps with the request",
    "the work is ready to commit, push or release",
    "it needs a credential, a config value or access it does not have",
    "the requirements are ambiguous in a few places",
    "it found cleanup candidates and asks which to remove",
)
LAYOUTS = {
    "one short paragraph": 1,
    "several paragraphs": 1,
    "a numbered list of questions": 1,
    "a bulleted list of questions": 1,
    "a question followed by numbered choices": 1,
    "a question followed by bulleted choices": 1,
    "the choices come before the question": 2,
    'bold labels such as "**Plan:**" and "**Question:**"': 1,
    "questions mixed into a report of what it did": 1,
    "a report with a code block, then the questions": 1,
}
POSITIVE = {  # pattern: (weight, description)
    "several in a row": (
        3,
        "several questions in a row in one paragraph or list item, "
        'e.g. "What does X do? Is it A or B?"',
    ),
    "inline alternatives": (
        1.5,
        'inline alternatives in a question: "to A, to B, or to both?"',
    ),
    "short yes/no items": (4, "short yes/no questions as separate list items"),
    "leading clause": (2, 'a question that starts with a clause: "For the X, which … ?"'),
    "one option": (
        2,
        'a question with one option: "Is dropping the cache acceptable, or too risky?"',
    ),
    "open": (4, 'an open question with no finite answer: "What should the name be?"'),
    "request": (
        4,
        'a request without a question mark: "Let me know which …", "Tell me if …"',
    ),
    "proposal recommend": (
        1.5,
        'a proposal with no question mark: "I recommend X. Say the word and I\'ll do it."',
    ),
    "proposal suggestion": (
        1.5,
        'a proposal with no question mark: "My suggestion: X. I wait for your go-ahead."',
    ),
    "proposal plan": (
        1.5,
        'a proposal with no question mark: "Plan: X, then Y. Once you confirm, I\'ll start."',
    ),
    "proposal unless": (
        1.5,
        "an action that happens unless the user objects, with no question mark: "
        '"Next I\'ll delete the stale branches, unless you object."',
    ),
    "proposal propose": (
        1.5,
        'a proposal with no question mark: "I propose X. Happy to do Y if you prefer."',
    ),
    "choice": (2, 'a question that offers two to four named options: "Should I A, B or C?"'),
    "yes/no": (1.5, "a plain yes/no question"),
}
NEGATIVE = {
    "rhetorical": "rhetorical questions that the agent answers itself in the next sentence "
    '("Why does it fail? Because …"); no question for the user',
    "quoted": "question marks inside code, logs, test output, quoted user text or commit "
    "messages; no question for the user",
    "urls": 'question marks in URLs, query strings, regexes and file names ("/colou?r/", '
    '"api?page=2"); no question for the user',
    "steps": "lists of steps or findings that are not choices; no question for the user",
    "headings": 'headings or status lines that contain a question mark ("**Why it failed?**" '
    "followed by the answer); no question for the user",
    "decided": 'a report of options that were already decided ("I chose A over B because …"); '
    "no question for the user",
    "acting": 'an action the agent takes now ("I\'ll push the fix now", "Next I run the '
    'tests"); no question for the user',
    "advice": 'a recommendation in a report that waits for nothing ("I recommend this pattern '
    'for the other services too"); no question for the user',
    "narration": 'decision words in plain narration ("I\'d rather keep …", "either way it '
    'works", "whether it passes is up to CI"); no question for the user',
}
NARRATION = {
    "confirming": 'the agent announces a check it does now, with no question: "Let me confirm '
    'that the fix closes it.", "Confirming which ones, then fixing:", "I\'ll verify whether the '
    'flag is read.", "Let me see whether the build passes."; no question for the user',
    "proceeding": 'the agent moves on: "Proceeding with the migration.", "Let\'s proceed to write '
    'the edits.", "Now let me check the other call sites."; no question for the user',
    "decision": 'the agent reports what it decided: "Decision: drop the fallback.", "I decided '
    'to keep the flag.", "I chose the smaller patch."; no question for the user',
    "report advice": "a recommendation inside a finished report that waits for no answer "
    '("In hindsight I\'d add a retry here; the report covers it."); no question for the user',
    "progress": 'tool-progress lines on one line each: "Reading config.go…", "Adding a test for '
    'the empty case…", "Confirming the count…", "Running the suite again:"; '
    "no question for the user",
    "before list": "a line that ends with a colon and introduces a list of findings, checks or "
    'steps, as in "Confirming which ones, then fixing:"; no question for the user',
    "identifiers": "short headings or list items made of code identifiers, flags or file names "
    '("1. `parse_args`", "- `--dry-run` flag", "### Decision"); no question for the user',
}
CONTROLS = {
    "should i": (
        'a real question that uses the same words: "Should I proceed?", "Shall I '
        'continue with the migration?"'
    ),
    "confirm": 'a real question: "Can you confirm the port?", "Do you confirm the flag stays?"',
    "decide": 'a real choice: "Decide whether to keep the flag or drop it: which do you prefer?"',
    "whether": 'a real question: "Should I check whether the cache is stale, or skip it?"',
}
NARRATION_LAYOUTS = {
    "one short paragraph": 2,
    "several paragraphs": 2,
    "a bulleted list of findings, no questions": 1,
    "a report with a code block, then a closing line": 1,
    "one line of progress, a colon, then a list": 2,
    "a short heading or a list of code identifiers": 1,
}
OPTION_LISTS = {
    "lowercase inline": (
        3,
        'a question and its options in one line, with lowercase letters: "Which season do you '
        'like most? a) Spring b) Summer c) Autumn d) Winter"',
    ),
    "parenthesis inline": (
        2,
        'a question and its options in one line, with letters in parentheses: "Which format? '
        '(a) JSON (b) YAML (c) TOML"',
    ),
    "letters per line": (
        3,
        "a question, then one option per line with the markers a) b) c) or A) B) C) or A. B. C.",
    ),
    "numbers per line": (
        3,
        "a question, then one option per line with the markers 1) 2) 3) or 1. 2. 3.",
    ),
    "numbers inline": (
        2,
        'a question and its options in one line with numbers: "Where should it run? 1) locally '
        '2) in CI 3) on staging"',
    ),
    "bold label": (
        2,
        'a numbered question with a bold label, and lettered options on the next line: "1. '
        '**Scope.** Which files may I change? a) only src/ b) also lib/ c) everything"',
    ),
}
OPTION_LISTS_NEGATIVE = {
    "lettered steps": 'steps in a lettered list "a) … b) … c) …", inline or one per line; no '
    "question for the user",
    "lettered findings": 'findings or changes in a list with the markers "(a)", "A)" or "1)"; '
    "no question for the user",
}
OPTION_LAYOUTS = {
    "one short paragraph": 2,
    "a short report of what it did, then the question with its options": 3,
    "several questions in a row, each with its own options": 2,
    "a report with a code block, then the question with its options": 1,
}
LIST_LAYOUTS = {
    "a short report, then a list under a lead-in line that ends with a colon": 3,
    "a short report, then a list under a Markdown heading such as '### Open questions' with "
    "no lead-in sentence": 2,
    "a short report, then a list under a bold line such as '**Decisions:**'": 2,
    "a list under a lead-in, and nothing else in the message": 1,
    "a list under a lead-in, then one closing sentence": 1,
}
LIST_PATTERNS = {
    "decision list": (
        4,
        'a lead-in that announces decisions ("Open questions, with my recommendations:", '
        '"Decisions:", "Still open:", "Waiting on you:", "Two things for you:"), then 2 to 12 '
        "numbered or bulleted items. Each item states one decision with a recommendation and a "
        'reason ("1. The empty reply: count it as the report, with a notice. I agree with the '
        'plan."). No item has a question mark',
    ),
    "decision list alternatives": (
        3,
        "a lead-in that announces decisions or open points, then 2 to 8 items. Some items name "
        'two or three alternatives, with or without a recommendation ("the release: wait for '
        'phase 1, or ship now; I recommend waiting"), some state one recommendation, some ask '
        'for a fact or a name ("the name of the new flag"), some are status items that wait '
        "for nobody (running, in flight, done). No item has a question mark",
    ),
    "option list": (
        4,
        'a lead-in that asks one question or announces one decision ("Which do you prefer?", '
        '"Two options:", "How should I proceed?"), then 2 to 12 items that are the alternatives '
        'of that decision, with or without a recommended item ("(my recommendation)"). The '
        "items are not questions",
    ),
    "question list": (
        3,
        "a lead-in sentence, then 3 to 12 numbered or bulleted items, each one question that "
        'ends with a question mark; some items hold two sentences that form one choice ("Should '
        'I keep the cache? Or drop it?")',
    ),
    "go-ahead list": (
        2,
        'one go-ahead for a whole batch: a lead-in such as "Still waiting on your go-ahead for '
        'the batch:" or "I need your approval for these:", then 2 to 8 items that are the parts '
        "of that batch. The items are not questions",
    ),
}
LISTS_NEGATIVE = {
    "status list": 'a list under "Still running:", "In flight:" or "Done:" whose items are '
    "status; no question for the user",
    "finished report": "a finished report under a lead-in that has a decision word "
    '("Decisions made:", "The safe fixes, which need no design decision:", "Decisions '
    'recorded in the design:"); no question for the user',
    "step list": 'a list under "Next:" or "Plan:" of steps that the agent takes now without '
    "approval; no question for the user",
    "third party": 'a list under "Blocked on the reviewer:" or "Waiting for CI:" of waits on a '
    "subagent, CI or a reviewer; no question for the user",
    "findings": 'a list of findings, measurements or risks with "I recommend" inside a report '
    "that waits for nothing; no question for the user",
}
SCENARIOS = {
    "narration": (NARRATION, CONTROLS),
    "options": (OPTION_LISTS_NEGATIVE, None),
    **dict.fromkeys(LIST_BATCHES, (LISTS_NEGATIVE, None)),
}


def scenario(seed: int, number: int, batch: str) -> dict:
    rng = random.Random(f"{seed}:{number}")
    negative, controls = SCENARIOS.get(batch, (NEGATIVE, None))
    layouts = NARRATION_LAYOUTS if controls else LAYOUTS
    positive = POSITIVE
    if batch == "options":
        layouts, positive = OPTION_LAYOUTS, OPTION_LISTS
    if batch in LIST_BATCHES:
        layouts, positive = LIST_LAYOUTS, LIST_PATTERNS
    if rng.random() < NEGATIVE_SHARES[batch]:
        patterns = [negative[k] for k in rng.sample(sorted(negative), rng.randint(1, 3))]
        names = ["negative"]
    elif controls:
        layouts = LAYOUTS
        patterns = [controls[k] for k in rng.sample(sorted(controls), rng.randint(1, 2))]
        names = ["control"]
    else:
        wanted = rng.randint(1, 3)
        names = []
        while len(names) < wanted:
            names.append(rng.choices(list(positive), [w for w, _ in positive.values()])[0])
            names = list(dict.fromkeys(names))
        patterns = [positive[k][1] for k in names]
    return {
        "project": rng.choice(PROJECTS),
        "situation": rng.choice(SITUATIONS),
        "layout": rng.choices(list(layouts), list(layouts.values()))[0],
        "patterns": patterns,
        "pattern_names": names,
    }


def user_message(s: dict) -> str:
    patterns = "\n".join(f"- {p}" for p in s["patterns"])
    return (
        f"Project: {s['project']}\nAgent situation: {s['situation']}\nLayout: {s['layout']}\n"
        f"Question patterns:\n{patterns}"
    )


def generate(
    ask: Callable[[str, str, str], tuple[str, str]],
    template: Prompt,
    seed: int,
    number: int,
    count: int,
    batch: str,
) -> dict:
    s = scenario(seed, number, batch)
    content, finish = ask(
        template.text.replace("{count}", str(count)), user_message(s), f"synth{seed}-{number}"
    )
    try:
        found = json.loads(content)["messages"]
    except (ValueError, KeyError, TypeError):
        found = None
    ok = finish != "length" and isinstance(found, list) and all(isinstance(m, str) for m in found)
    return {
        "id": f"{seed}-{number}",
        "batch": batch,
        "session": f"synth:{seed}-{number}",
        "scenario": s,
        "prompt": template.digest,
        "status": "ok" if ok else "invalid",
        "messages": [m for m in found if m.strip()] if ok else [],
    }


def make_units(
    generations: list[dict], user: str
) -> tuple[list[Row], Counter, dict[str, tuple[str, int]]]:
    """The pipeline of corpus.py on the messages: normalise, units, redaction, letters, gate, and
    dedup against data/units.jsonl and inside the set."""
    ours = {r.text for r in load_units()[0]}
    kept: dict[str, Row] = {}
    origin: dict[str, tuple[str, int]] = {}
    counts: Counter = Counter()
    for generation in generations:
        for number, message in enumerate(generation["messages"]):
            counts["messages"] += 1
            for unit in units(*normalise(message)):
                text, lined, context = redacted(unit, user)
                counts["units"] += 1
                if not has_letters(text):
                    counts["no_letters"] += 1
                elif category(text) == "other":
                    counts["other"] += 1
                elif text in ours:
                    counts["duplicate_of_ours"] += 1
                elif text in kept:
                    counts["duplicate_inside"] += 1
                else:
                    kept[text] = Row(text, generation["session"], "train", lined, context)
                    origin[text] = (generation["id"], number)
                    counts[category(text)] += 1
    return list(kept.values()), counts, origin


def labelled_units(batches: list[str]) -> list[dict]:
    """The labelled units of the batches that training reads as the source "synth": gated,
    labelled with the current prompt, deduplicated like make_units."""
    prompt = load_prompt()
    generations = read_jsonl(GENERATIONS)
    batch_of = {g["id"]: g["batch"] for g in generations}
    rows, _, origin = make_units(generations, getpass.getuser())
    chosen = {r.text: r for r in rows if batch_of[origin[r.text][0]] in batches}
    latest = {
        r["text"]: r
        for r in read_jsonl(LABELS)
        if r["status"] == "labelled" and current(r, prompt) and r["text"] in chosen
    }
    return [r | {"lined": chosen[t].lined, "context": chosen[t].context} for t, r in latest.items()]


def report(examples: int, batch: str) -> None:
    generations = [g for g in read_jsonl(GENERATIONS) if g["batch"] == batch]
    rows, counts, origin = make_units(generations, getpass.getuser())
    prompt = load_prompt()
    labels = {r["text"]: r for r in read_jsonl(LABELS) if current(r, prompt)}
    done = [labels[r.text] for r in rows if r.text in labels]
    questions = [q for r in done for q in r["questions"]]
    print(
        json.dumps(
            {
                "calls": len(generations),
                "invalid_calls": sum(g["status"] != "ok" for g in generations),
                "pipeline": dict(counts),
                "units_per_message": counts["units"] / max(1, counts["messages"]),
                "gated_share_of_units": len(rows) / max(1, counts["units"]),
                "labelled": dict(Counter(r["status"] for r in done)),
                "units_with_questions": sum(bool(r["questions"]) for r in done),
                "empty_labels": sum(not r["questions"] for r in done),
                "questions": len(questions),
                "kinds": dict(Counter(q["kind"] for q in questions)),
                "options": sum(len(q["options"]) for q in questions),
                "corrections": sum(r["corrections"] for r in done),
            }
        )
    )
    shown = 0
    for generation in generations:
        for number, message in enumerate(generation["messages"]):
            found = [
                labels[r.text]
                for r in rows
                if origin[r.text] == (generation["id"], number) and r.text in labels
            ]
            if shown < examples and any(r["questions"] for r in found):
                shown += 1
                print(f"\n=== example {shown}: {generation['scenario']}\n{message}\n--- units")
                for r in found:
                    print(r["text"], "->", json.dumps(r["questions"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=0, help="generation calls to make")
    parser.add_argument("--count", type=int, default=5, help="messages per call")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", choices=BATCHES, default="base")
    parser.add_argument("--budget", type=float, default=5.0, help="dollars this run may spend")
    parser.add_argument("--peak", action="store_true", help="allow a call in DeepSeek peak hours")
    parser.add_argument("--report", type=int, nargs="?", const=5, help="print stats and N examples")
    args = parser.parse_args(argv)
    if args.report is not None:
        report(args.report, args.batch)
        return 0
    if args.calls and args.batch == "sonnet":
        print("synth: the batch sonnet is stored data; it has no generator", file=sys.stderr)
        return 1
    with open_session("synth", args.peak) as session:
        if session is None:
            return 1
        ledger = session.ledger
        start = ledger.total

        def ask(system: str, user: str, nonce: str) -> tuple[str, str]:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return complete(session, messages, nonce)

        template, prompt = load_prompt(PROMPT), load_prompt()
        done = {g["id"] for g in read_jsonl(GENERATIONS)}
        wanted = [n for n in range(args.calls) if f"{args.seed}-{n}" not in done]
        stop = []

        def write(number: int) -> None:
            if ledger.total - start > args.budget or stop:
                return
            try:
                append_jsonl(
                    GENERATIONS,
                    generate(ask, template, args.seed, number, args.count, args.batch),
                )
            except (BudgetExceeded, PeakHours) as error:
                stop.append(type(error).__name__)
            except APIError:
                stop.append("APIError")

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(write, wanted))
        generation_cost = ledger.total - start
        generations = read_jsonl(GENERATIONS)
        batch_of = {g["id"]: g["batch"] for g in generations}
        rows, counts, origin = make_units(generations, getpass.getuser())
        batch_rows = [r for r in rows if batch_of[origin[r.text][0]] == args.batch]
        labels = run(session, prompt, batch_rows, LABELS, WORKERS)
        print(
            f"synth: {len(wanted)} calls requested, stops {sorted(set(stop))}, generation "
            f"${generation_cost:.4f}, labels ${ledger.total - start - generation_cost:.4f}, "
            f"DeepSeek ledger ${ledger.total:.4f}, labelled {labels}, pipeline {dict(counts)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
