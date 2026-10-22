# pi-infer-questions

A [pi](https://github.com/earendil-works/pi) extension that finds the
questions in each reply of the agent and makes them answerable with the
mouse. A small local model reads the reply. No tool call, no change to
the agent's prompt and no network request is necessary.

In a completed reply, the extension adds clickable elements after each
question:

| The question | The elements |
|---|---|
| A choice ("Should I push now, or wait for CI?") | each option in place, then **Other** |
| A choice with one option | the option, then **Yes/Other** |
| A yes/no question or a proposal ("I recommend X. Say the word.") | **Yes/No/Other** |
| An open question ("What should the name be?") | **Answer** |

A click does not send anything. It adds the question as a quote and the
answer to the editor:

    > Should I push now, or wait for CI?

    wait for CI

You can answer several questions, edit the text, and then submit. A
second click on the same question replaces its answer. **Other** and
**Answer** add the quote with an empty answer.

## Requirements

- pi in fullscreen mode (`"tuiMode": "fullscreen"` in
  `~/.pi/agent/settings.json`). Clicks work only in fullscreen mode.
- macOS 15 or later on Apple silicon (the model runs on the Neural
  Engine), or Linux on arm64 (the model runs on the CPU through ONNX
  Runtime). Linux on x64 builds, but its models are not released yet:
  see [plans/linux.md](plans/linux.md).
- To build: Node 24 or later with its headers, Swift 6, cmake and curl.

On other platforms, or when the models do not load, the extension shows
one notice and pi renders replies as usual.

## Install

    make install DEST=~/.local/share/pi-infer-questions

`make` downloads the models for your platform from the GitHub release
(`MODELS_VERSION`, `RELEASE_URL`) and checks their sha256. Then add the
extension to `~/.pi/agent/settings.json`:

    "extensions": ["/Users/<you>/.local/share/pi-infer-questions/extension.ts"]

Restart pi. `install` replaces `questions.node` with a new file. Do not
overwrite it in place: on macOS, the next Node process that loads an
overwritten file is killed.

To work on the extension, load it from the checkout: `make` puts
`questions.node` and `models/` next to `extension.ts`, so the entry can
be `"<checkout>/extension.ts"`. After `make`, restart pi.

## How it works

The extension waits until a reply is complete. Then a native module
(Swift, one `.node` file) does these steps in the pi process:

1. It splits the reply's Markdown into units: paragraphs, and lists with
   the text that introduces them.
2. It skips units without `?` and without a decision word.
3. A question tagger (a 32M-parameter encoder) marks each question and
   its kind: yes/no, open or choice.
4. An option tagger marks the options in each choice question.

A reply takes a few milliseconds. See [docs/design.md](docs/design.md).

## Quality

One held-out evaluation of the shipped model on 2,939 units from real
coding-agent replies (PyTorch model; the Neural Engine and ONNX models
are within .02 on every metric):

| | |
|---|---|
| Questions found, with `?` (exact span) | 82% |
| Questions found, overlapping span | 80% |
| Proposals without `?` found | 52% |
| Question F1, exact span / overlapping span | .638 / .812 |
| Kind accuracy | .949 |
| Option F1, end to end / with the correct question | .577 / .850 |
| Units without a question that get one (false positives) | 4.1% |

It misses most often a proposal without `?`, and a list of decisions
without `?` (66% of their items found).

## Train your own model

The training pipeline is in [pipeline/](pipeline/). The training data is
not published: it comes from your own pi and Claude Code transcripts.
The labels come from DeepSeek (an API key and a few dollars).

    cd pipeline
    uv sync
    export DEEPSEEK_API_KEY=...
    uv run corpus                     # units from ~/.pi and ~/.claude transcripts
    uv run label-corpus --take question=all --take pool=all
    uv run synth --calls 100 --batch base     # optional synthetic messages
    uv run train --synth base         # seeds 0-2, selects one on dev
    uv run train --eval               # once, on the held-out split
    uv run convert-ane                # macOS: the Neural Engine pack
    uv run parity-ane
    uv run golden-ane
    uv run convert-onnx               # Linux: the ONNX pack
    uv run parity-onnx
    uv run golden-onnx

`uv run extract FILE` prints the questions of a text with the PyTorch
model. `uv run label FILE` prints DeepSeek's labels for a text.

### Privacy

- `corpus` reads every assistant message under `~/.pi/agent/sessions`
  and `~/.claude/projects` and keeps a copy in `data/messages.jsonl`.
  It replaces your user name, home paths, email
  addresses and secrets that it recognizes before anything goes to
  DeepSeek. It does not know the names of your private projects.
- Everything that the pipeline writes is in `data/`, which is not in
  git.
- The published models are trained on the author's transcripts after
  this redaction. The golden test files contain only the fixtures and
  synthetic messages.

## Tests

    make smoke    # the native module against the golden file, exactly
    make e2e      # pi in fullscreen on a private tmux server, fake provider
    cd pipeline && uv run pytest

`make e2e` needs `pi` and `tmux` on PATH. On Linux,
`linux/Containerfile` gives a build environment.

## License

Apache-2.0. The models are fine-tuned from
[jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m)
(MIT); see [NOTICE](NOTICE).
