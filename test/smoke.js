const { readFileSync } = require("node:fs");
const { isDeepStrictEqual } = require("node:util");
const [, , addon, models, goldenPath] = process.argv;

setTimeout(() => {
  console.error("smoke: timeout");
  process.exit(2);
}, 300_000).unref();

const native = require(require("node:path").resolve(addon));
const golden = JSON.parse(readFileSync(goldenPath, "utf8"));

const project = (a) => ({
  units: a.units,
  question_tokens: a.question_tokens,
  questions: a.questions.map((q) => ({
    unit: q.unit,
    span: q.span,
    range: q.range,
    kind: q.kind,
    options: q.options,
    option_spans: q.option_spans,
  })),
  option_tokens: a.option_tokens,
});
const options = (q) => q.options.map((o) => o.range ?? o);

native.start(require("node:path").resolve(models));

(async () => {
  let failed = 0;
  for (const c of golden.cases) {
    const got = JSON.parse(await native.questions(c.markdown, true));
    const want = { ...c, questions: c.questions.map((q) => ({ ...q, options: options(q) })) };
    const a = project(got);
    const b = project(want);
    const step = Object.keys(b).find((k) => !isDeepStrictEqual(a[k], b[k]));
    if (step) {
      failed++;
      console.error(`MISMATCH ${c.name}: first differing step: ${step}`);
      if (failed <= 3) {
        console.error(" native:", JSON.stringify(a[step]).slice(0, 400));
        console.error(" golden:", JSON.stringify(b[step]).slice(0, 400));
      }
    }
  }
  const questions = golden.cases.reduce((n, c) => n + c.questions.length, 0);
  console.log(`smoke: ${golden.cases.length - failed}/${golden.cases.length} cases equal, ${questions} questions`);
  process.exit(failed ? 1 : 0);
})();
