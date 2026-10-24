# Design

This file gives the decisions and the rules that the code depends on.
[experiments.md](experiments.md) tells what was tried and rejected.

## One pipeline, three implementations of its rules

The rules from Markdown to questions exist in Python (`pipeline/`, the
reference) and in Swift (`native/`, the product). The golden files pin
the Swift output to the Python output exactly, on the same models:

- `golden-ane` writes `data/golden-pi.json` (macOS, Core ML).
- `golden-onnx` writes `data/golden-onnx-<arch>.json` (Linux, ONNX
  Runtime). int8 kernels differ between x64 and arm64, so each
  architecture has its own golden file.

All golden files use the same 209 cases: the fixtures and 200 synthetic
messages, selected by the text and its unit categories only (quotas:
50 with options, 70 with questions, 50 gated without `?`, 30 plain). No
case comes from a real transcript.

A change to a rule changes Python and Swift together, and then every
golden file.

## Units

A unit is the text that one model call reads (`units.py`, `Units.swift`).
Labelling, training, evaluation and inference use the same units.

1. Markdown normalization (`normalise.py`): only the lines of
   paragraphs and list items stay (not code blocks, tables, headings,
   block quotes, footnotes or front matter); a fully bold paragraph
   without `?` is dropped (a pseudo-heading), except when it introduces
   a list; inline markup, links and
   images become plain text. Each kept character keeps its offset in
   the original Markdown.
2. The text splits at blank rows.
3. A list joins the unit before it when that unit ends with `?` or `:`
   (before closing brackets and quotes), holds a list item, or starts
   with a heading or bold line. Joins chain. This keeps a question and
   its listed options in one unit, and a decision list with its lead-in.
   A heading line kept this way is context (`Unit.context`): a question
   span never starts in it.
4. `segments`: when a top-level list item's own text holds `?`, the
   text before the first item, each item with `?` (with its nested
   items) and each run of items without `?` become separate units.
   Top-level items are those with the marker kind (number, letter,
   bullet) of the first item. Without this rule, a list of ten
   questions was one unit, and the model lost questions in it. A list
   whose items have no `?` (a decision list) stays one unit with its
   lead-in: one unit per item found no more items.

The model input keeps the unit's lines: spaces and tabs collapse inside
a line, indentation goes, line breaks and blank lines stay. The gate and
the labeller read the collapsed one-line text, and gold spans map from
it to the lined text through the n-th non-space character.

## The gate

Only a unit with `?` or a decision word (`units.DECISION`: "should I",
"let me know", "recommend", "unless you", "propose", "waiting on your",
"once you've", "your go", `say "..."`, ...) goes to the model. The
others get no model call, in every phase. The gate decides cost and the
false-positive rate. It misses requests that have neither a `?` nor a
decision word. A three-sample label audit of 1,500 skipped units found
a question in 1.3% of them after removing the labeller's "Let's ..."
false positives, about 10% of all questions; most were next steps that
wait for the user. The cues that caught them are in the pattern. "Let's"
and "Let me" are not: they are almost always the agent's own steps.

## Labels

`label.py` sends each unit to DeepSeek (`deepseek-flash`, reasoning
effort low) with `unit_prompt.md`. The answer is JSON: the questions in
text order, each an exact substring with its kind (`choice`, `yes_no`,
`open`) and, for a choice, its options as exact substrings of the
question. A wrong answer gets one correction turn.

- Every real unit gets three samples, and a label needs two equal
  samples (`corpus.SAMPLES`); a unit without them is `disagree` and is
  not trained on. A synthetic unit gets one sample.
- The prompt's rules include: a proposal that waits for approval is a
  `yes_no` question, also without `?`; examples after a question are its
  options; alternatives joined by "or" are options, also as verbs or
  clauses; a context sentence before a question is not part of it; under
  a lead-in that announces decisions or things that wait for the user,
  each item that waits for the user is one question over the whole item,
  marker included (a recommendation or go-ahead is `yes_no`, named
  alternatives are `choice`, a request for a fact is `open`), and an
  item that reports status, a finding, a decision already made or the
  agent's next step is no question; an offer to do something for the
  user is a `yes_no` question over the offer sentence only; a list of
  findings, issues or notes is no question, only a request inside it
  is; a next step is a `yes_no` question only when it waits for a
  signal from the user; a question in prose is the sentence that asks
  or waits, with an earlier sentence only when the question cannot be
  understood without it.
- A label is valid only for the prompt digest it was made with. A
  prompt change relabels every unit, real and synthetic.
- Spend: `data/ledger.jsonl` reserves the worst case before each call
  and settles from the usage. The cap is $75 (`LIMIT`); a run starts no
  new unit above $72 (`STOP_SPEND`). Calls in DeepSeek peak hours need
  `--peak`.

`corpus` first copies every assistant reply of the live transcripts
into `data/messages.jsonl`, then builds units only from that file, so a
deleted transcript keeps its replies.

The split (train, dev, eval) is by session and frozen in
`data/units.jsonl` by the first `corpus` run. The unit texts of the
three splits are disjoint (`corpus.deduplicate`).

## Synthetic messages

`synth.py` asks DeepSeek for coding-agent replies from sampled scenarios
(`synth_prompt.md`) and labels their units with the same labeller. The
batches (`base`, `negatives`, `narration`, `options`, `lists`, and the
stored `sonnet` batch) are train-only and are never part of dev or eval.
`lists-check` is generated like `lists` and is never trained on: it
measures item recall on clean decision and question lists. A batch's
generation is fixed by its `--seed`; seeds are unique across batches.
Selection and every reported number use real units only.

## Models

Two fine-tunes of `jhu-clsp/ettin-encoder-32m` (pinned revision), token
taggers:

- The question tagger: 7 tags, `O` and `B`/`I` for each kind. The kind
  comes from the tags. It trains on gold spans snapped to whole
  sentences (`sentences.snap_gold`).
- The option tagger: 3 tags. It reads one choice question and runs only
  on choice spans.

`train` trains seeds 0–2 on real train units plus the chosen synthetic
batches and selects on real dev: the best mean of question F1 and option
F1 among seeds with an empty-unit false-positive rate below `MAX_FPR`
(.05). With no qualifying seed it exits with an error. `--eval` scores
the selected model once on the held-out split.

## Decoding

Questions are decoded per sentence (`sentences.decode`,
`Tagging.sentenceSpans`). A sentence (`sentences.segments`) ends at a
line end, or at `.`, `?` or `!` followed by a space; a dot inside a
number or a path and `e.g.`, `i.e.`, `etc.`, `vs.`, `cf.` do not end
it. A list item's marker is part of its first sentence.

- Average the softmax of each tag over the sentence's tokens. The
  sentence is `O` when the mean `O` is at least the largest mean `B`+`I`
  of a kind.
- A question sentence right after another one continues its question
  when its first token is not likelier a `B` than an `I` of its kind;
  else it starts a question.
- The kind of a question is the kind with the largest `B`+`I` mass,
  summed over its sentences.
- The sums run in token order in Python and Swift, so the two decode
  identically.
- Drop a question span without a letter (a lone list marker such as
  `2.`).

Options are decoded per token (`tagging.spans_of`):

- A span starts at a `B` tag; `I` tags continue it. An `I` without a
  `B` starts nothing. (A lenient decoder that joined runs of non-`O`
  tags scored worse on every dev metric.)
- Trim unbalanced brackets and quotes at the edges; cut trailing
  `, . ? ; :`; trim again.
- Balance backticks at the edges: extend the end to a near closing
  backtick, else the start over an opening one, else cut the edge one.
- Extend the end over the combining marks that follow it.
- Drop a span that is empty or holds only trailing characters.

Ranges in the result are Unicode code-point offsets into the original
Markdown. After the mapping, a range grows over every code span of the
original Markdown that it overlaps (`message_range`), because Markdown
normalization removes the backticks from the model input.

## Delivery on macOS: the Neural Engine

`convert-ane` (`ane.py`) writes `data/coreml-ane-pack`:

- One input length, 512 tokens. Several units share one call, packed
  greedily: `input_ids`, `position_ids` that restart at 0 for each unit,
  and a segment id per token; the model builds the block-diagonal mask
  and keeps the local attention window inside a unit.
- The layout of apple/ml-ane-transformers: (B, C, 1, S) tensors, 1x1
  convolutions, layer norm over channels.
- fp16 activations and fp16 weights in both models (62 MB each). int8
  weights changed a few question kinds and lost up to .057 option F1
  against PyTorch; int8 per block put convolutions on the CPU.
- Compute units `.cpuAndNeuralEngine`. There is no CPU or GPU fallback.
- `settings.json` holds the tag names per task, the pad id, the token
  limit, the decoder constants and the digests of the checkpoint, the
  packages and the source code. The native module refuses tag names it
  does not expect; `parity-ane` and `golden-ane` refuse a pack whose
  checkpoint digest differs from the checkpoint.

`parity-ane` has two gates:

1. Packing correctness: the same pack in fp32 must decode every dev
   input exactly like PyTorch on the CPU. This is a hard gate.
2. Quantization cost: the shipped pack against PyTorch; question F1,
   overlap F1 and option F1 may drop, and the false-positive rate may
   rise, by at most `MAX_DROP` (.02). Per-kind F1 and the count of
   inputs that decode differently are reported, not gated.

## Delivery on Linux: ONNX Runtime

`convert-onnx` (`onnx_pack.py`) exports both taggers to ONNX with
dynamic batch and sequence length (no packing; units are batched by
length) and int8 weights, into `data/onnx`. One int8 pack serves x64
and arm64; each architecture has its own release archive and golden
file. `parity-onnx` has the same two gates.

int8 results depend on the CPU's kernels. x64 is validated on AVX-512
VNNI (AMD EPYC Zen 5). Without VNNI, x64 adds pairs of 8-bit products
into a saturating 16-bit value; in qemu, which has no VNNI, option F1
dropped by .08. Such CPUs (AVX2 only) can lose option quality. fp32
models decode like PyTorch on every CPU, but are 4 times the size
(257 MB) and 2.4 times slower on x64.

The engine limits ONNX Runtime to `min(4, cores)` intra-op threads.
By default, ONNX Runtime sizes its pools by the cores of the host and
pins them: in a cpuset the pinning fails and logs to the terminal,
and under a cgroup quota the pools oversubscribe.

The Makefile downloads the official ONNX Runtime build
(pinned version and sha256), and the `.node` loads `libonnxruntime.so`
from its own directory.

The official Linux build sends telemetry by default. The engine calls
`DisableTelemetryEvents` after it creates the environment and sets the
log level to error, so that no warning reaches the TUI.

## The native module

One Swift module, built into `questions.node` with a small N-API shim
(`napi.c`). Only `CoreML.swift` and `Onnx.swift` differ by platform. The
Swift runtime is linked statically on Linux. Markdown parsing uses
cmark-gfm, built from a pinned source and linked statically on both
platforms.

It exports `start(models)`, which loads the models in the background,
and `questions(markdown)`, which runs the whole pipeline off the
JavaScript thread and returns JSON.

## Rendering in pi

Pi has no hook that renders an assistant message itself. The extension
uses two hooks that it has:

- `registerMarkdownTransformer` returns an empty string for a completed
  assistant message only if the extension holds an entry for exactly
  that text. Streaming, aborted and failed replies render as usual.
- On `message_end` with stop reason `stop`, the extension analyzes the
  text. If it has questions, it appends an entry (`appendEntry`) with
  the Markdown and the analysis. Entries do not enter the model's
  context, and a resumed session renders them without the model.
- `registerEntryRenderer` draws the entry through pi-tui's `Markdown`
  component, so it looks like a normal message. Sentinel characters mark
  the clickable ranges; the rendered lines give their cells, and a
  `MouseRegion` hit-tests against them. Question text is not styled;
  options and inline elements share one style (accent colour,
  underline, inverse on hover).
- A click edits the editor text: one block `> <question>\n\n<answer>`
  per question, found by its quote line. Pi exposes only whole-editor
  replacement, so the cursor goes to the end.
- Under a terminal multiplexer, pi enables only button-motion mouse
  tracking. The extension enables any-motion tracking (`?1003h`) while
  pi is in fullscreen, and again when pi enters it again, so that hover
  works.

## The question guideline

The model finds a question with `?` much more often than a proposal
without one. So the extension asks the agent to write decisions as
questions: in `before_agent_start` it adds five bullets to
`systemPromptOptions.promptGuidelines` (ask with `?` at the end of the
reply, one question per sentence or list item, name the alternatives of
a choice, ask for the go-ahead of a proposal, no questions before a tool
call). It adds to pi's prompt and does not replace it. The provider gets
the bullets in the rules section of the system prompt.

The flag `--no-question-guide` turns it off. The first run of a session
with the guideline appends one entry with `customType`
`pi-infer-questions-guide` and the bullets as `data.guidelines`. Replies
after that entry were written under the guideline, so a training round
can keep them apart: they are easier than replies of agents without it,
and an eval that mixes them overstates quality.

## Known limits

- Proposals and decision lists without `?` are the weakest case (55%
  of proposals and 59% of decision-list items found on eval). Most
  misses are whole lists that get no question; about half of the dev
  misses have arguable gold (findings lists, offers).
- About 3.5% of gated units without a question get one, mostly
  proposals, offers and the agent's own next steps ("Let's check
  ...").
- A run of `I` tags without a `B` is dropped by the strict decoder; the
  shipped seed loses one question this way in a test message where the
  other seeds find it.
- Two assistant messages with the same text: when one gets an entry,
  the transformer hides both.
- `/tree` and the HTML export show the raw text. Pi's normal mode has no
  mouse; there the extension renders but nothing is clickable.
