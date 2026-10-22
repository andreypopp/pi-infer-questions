# Past experiments

The work started as a feature of a terminal app, and the pi extension
became the product later. This file records what was tried, the result,
and why the pipeline is as it is. The code of these experiments is not
in this repository.

Numbers are on the real dev split unless the text says eval. Two runs
are comparable only when they use the same units and the same labels;
the units and the labels changed several times, so numbers from
different sections are not comparable.

## Models and labels

- **Apple FoundationModels (the on-device LLM).** Tried early and
  replaced by DeepSeek labels and a small tagger. Not used again.
- **Hand labels.** Not used: all labels come from the DeepSeek
  labeller, so that a relabelling after a rule change is one command.
- **DeepSeek Flash at high reasoning effort.** On 152 units, 87% of the
  labels were identical to low effort. High effort found 2 more
  questions in 100 units and dropped 5 weak ones; on long units with
  many questions it hit the length limit. Low effort stays.
- **A CRF output layer and "two-ends" span tagging.** Gains were inside
  the seed spread; the plain BIO tagger stays.
- **Ettin 17M.** About twice as fast, but question F1 .628 against .657
  and a large seed spread on options. 32M stays.
- **Typed tags against a kind heuristic.** A rule that guessed the kind
  from the first word failed on list items ("2. **Label:** What ...").
  With the kind in the tags, kind accuracy rose from .87 to .97.
- **Ettin 68M.** One seed against three 32M seeds on the same dev:
  proposal recall .54 against about .45, false positives 3.2%. Not
  finished; 32M stays for now.
- **The whole message as labeller context, and DeepSeek V4 Pro as a
  judge.** On 300 dev units, the whole message raised units without a
  two-of-three vote from 2.3% to 7.0% (decision lists 5% to 25%) and
  dropped real list items. Pro equalled the Flash vote in 49% of the
  units it judged and gave the same label twice in 60%. Neither is
  used.
- **Three samples for every real unit.** One sample differed from the
  two-of-three vote in 10-14% of proposal, decision-list and `?`
  units. Train labels now also come from the vote. With three new rules
  (offers, findings lists, next steps), the new model against the old
  one on the same new eval labels: false positives 5.1% to 3.0%
  (decision lists 5.7% to 2.1%), overlap precision .794 to .833, kind
  accuracy .952 to .974; overlap recall .809 to .772, proposal recall
  .553 to .490, option F1 .582 to .562.
- **JEV-9B (a 9B decision model) through MLX.** A faithful port works
  (49 of 50 decisions equal to the reference in bf16), but it takes about
  150 ms and 9 GB for each call. Not used.

## Input

- **Lower-cased input.** Rejected; case stays.
- **One line per unit (all line breaks removed).** The model merged
  numbered questions on separate lines. Line breaks and blank lines now
  stay.
- **Bold markers in the input.** Option F1 on real data fell. The model
  input has no markup.
- **Several fixed input lengths.** One length of 512 tokens is simpler
  and loses little; longer units are cut.

## Units

- **Paragraphs only.** Questions and their listed options often cross a
  blank line. The list-join rule fixed most of these.
- **A whole list of questions as one unit.** With ten numbered questions
  in one unit, the model lost questions and options. The `segments` rule
  makes each list item with `?` its own unit; on the same messages, all
  questions were found afterwards.

## Decision lists

A decision list is a lead-in ("Open questions, with my
recommendations:") and items without `?` that wait for the user. The
model before this round found 38% of their items on dev.

- **One unit per item, with or without the lead-in copied in.** On
  labelled dev and eval lists, lead-in plus item found 42 of 60 gold
  items, the whole list 45, the item alone 14. It also broke every list
  whose items are the options of one question (0 of 51 kept). Not used.
- **A classifier for the list kind** (option list, question list,
  decision list) from the lead-in, to route a list to whole or per-item
  input: accuracy .857 against a majority baseline of .855. A cascade
  gained 2 or 3 of 54 items and added 6 or 7 false-positive lists of
  501. Not used.
- **What was done instead:** a prompt rule for decision lists, a heading
  or bold lead-in kept as context, a `lists` synthetic batch (1,566
  units), and a full relabelling of every unit with one prompt digest
  (dev labels now from three samples). On the same new dev gold, the
  new model against the old one: decision-list item recall .38 to .61
  at equal precision (.74), proposal recall .32 to .45, false positives
  2.9% to 2.6%; option F1 .55 to .53 and kind accuracy .976 to .965.
  On eval: decision-list recall .40 to .66, overall overlap F1 .759 to
  .812, false positives 3.8% to 4.1%. On the held-out `lists-check`
  batch, decision-list item recall went from .52 to .90.
- **The targets were not met** (decision-list recall .85, precision .80).
  Of 32 decision-list items missed on dev, 30 were in lists that got no
  question at all; misses do not depend on the item's position, so
  per-item units would not help. About half of these misses have
  arguable gold (findings lists, offers). Clean synthetic lists are
  learned well, so the limit is the real labels.

## Training data

- **Public data (`nlile/misc-merged-claude-code-traces-v1`).** A dry run
  found mostly tool narration and few questions to the user. Not used.
- **Synthetic messages.** The first batch raised option F1 on real dev
  from .490 to .547 (three seeds each) and also the false-positive rate
  (.010 to .018). Later batches targeted negatives, agent narration, and
  lettered option lists ("a) x b) y"; the first option was lost before).
  Their effects were small and partly inside the seed spread. One batch
  was generated with Claude Sonnet; the pipeline now generates with
  DeepSeek only.
- **Learning curve.** At 25%, 50% and 100% of the training data,
  question F1 was .590, .626 and .657; the model is not saturated.

## Decoding

- **A threshold on the begin tag.** Replaced by plain argmax.
- **A lenient decoder** (a span is any run of non-`O` tags, the kind is
  the most probable over the span). Worse on every dev metric (question
  F1 .607 to .505, false-positive rate .035 to .049).
- **An option fallback** (run the option model on each sentence of a
  multi-sentence question). Not needed with typed tags; removed.

## Delivery

- **Core ML on the GPU.** 1.8 ms per unit, about 200 MB process memory.
- **Core ML on the Neural Engine, direct conversion.** Slower than the
  GPU.
- **Core ML on the Neural Engine, rewritten and packed.** 0.75 ms per
  unit with 21 MB process memory (without the Neural Engine's own
  memory). Packing is exact in fp32. This is the shipped path.
- **Weight quantization.** It shrinks the pack and does not reduce
  latency. int4 embeddings had no measured loss on the lists-round
  model. On the next model, the option tagger lost .057 option F1 with
  int8 per-channel weights, with int4 or int8 embeddings; int8 per
  block (32) had no loss but put 62 convolutions on the CPU. The option
  tagger now ships in fp16 (63 MB), the question tagger in int8 with
  int8 embeddings (32 MB).
- **A Linux CPU backend.** ONNX Runtime through its C API, from the same
  Swift module. A TypeScript port of the pipeline and MLX on the CPU
  were considered and not chosen.

## Product

- **Inside the terminal app (Kido.app),** with Shift held to highlight
  questions in the terminal grid. Abandoned for the pi extension.
- **Pills below a message.** Replaced by elements in place, after each
  question's `?`.
- **A click that sends the answer.** Replaced by a click that fills the
  editor, so that several answers can go in one reply.
- **Approve / Reject / Other for yes/no questions and proposals.**
  Replaced by Yes / No / Other, which is more compact.
