# Working in pi-infer-questions

[README.md](README.md) tells what this is and how to use it.
[docs/design.md](docs/design.md) gives the rules that the code depends on.
[docs/experiments.md](docs/experiments.md) tells what was tried and
rejected; read it before you propose an experiment. Open work is in
`plans/`, one file per plan; when a plan is done, its design moves into
`docs/` and the plan is deleted.

## Rules

- One pipeline. No old modes, flags, fallbacks or compatibility paths.
  When a rule changes, change Python (`pipeline/`) and Swift (`native/`)
  together and regenerate every golden file.
- Plain functions over data; no abstraction with one use; no dead code;
  one source of truth for each fact.
- No comments except a verified external constraint (a pi, Core ML,
  ONNX Runtime or OS quirk) or the reason for a workaround.
- Tests: only the CLI smoke tests in `pipeline/tests`, `make smoke` and
  `make e2e`. Do not add unit tests.
- Zero build warnings (`-warnings-as-errors`), ruff clean.
- `data/` is private and never in git: transcripts, labels, the ledger,
  models. Golden files and fixtures contain no transcript text.
- Paid calls (DeepSeek) go through the ledger in `data/ledger.jsonl`.
  Ask before a run whose cost is not agreed.
- Select on dev. Evaluate on eval once per shipped model.
- Communicate in ASD-STE100 English: with the user, in docs and in
  commit messages.

## Traps

- Do not overwrite `questions.node` in place. On macOS, the next Node
  process that loads it is killed (a stale code signature). `make`
  removes the old file first.
- Core ML packs and ONNX int8 decode a little differently from PyTorch;
  x64 and arm64 int8 differ from each other. Each backend and
  architecture has its own golden file. The x64 golden and parity come
  from a real x64 machine with AVX-512 VNNI, not from emulation.
- Model output changes the e2e expectations: the e2e uses fake replies
  whose questions the shipped model must find (for example, one fake
  reply must give exactly one option). After a model change, check them.
