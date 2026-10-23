import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [extension] = process.argv.slice(2);
const here = import.meta.dirname;
const scratch = mkdtempSync("/tmp/pq-e2e-");
const socket = join(scratch, "socket");
const log = join(scratch, "log");
writeFileSync(log, "");
mkdirSync(join(scratch, "agent"));
mkdirSync(join(scratch, "work"));
writeFileSync(join(scratch, "agent", "settings.json"), JSON.stringify({ lastChangelogVersion: "1.2.0" }));

const tmux = (...args) => execFileSync("tmux", ["-S", socket, ...args], { encoding: "utf8" });
const pane = "e2e";
const screen = (escapes = false) => tmux("capture-pane", "-p", ...(escapes ? ["-e"] : []), "-t", pane);
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const userLines = () => readFileSync(log, "utf8").split("\n").filter((l) => l.startsWith("USER"));

async function until(what, check, ms = 20_000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const found = check();
    if (found) return found;
    await sleep(50);
  }
  throw new Error(`timeout: ${what}\n${screen()}`);
}

const pi = (...extra) =>
  [
    `cd ${scratch}/work`,
    `export PI_CODING_AGENT_DIR=${scratch}/agent PQ_LOG=${log} PI_OFFLINE=1`,
    `exec pi --no-extensions -e ${resolve(extension)} -e ${here}/fake.ts --provider fake --model m --session-dir ${scratch}/sessions --tui-mode fullscreen --no-skills --no-context-files ${extra.join(" ")}`,
  ].join(" && ");

const mouse = (code, x, y, up = false) => {
  const bytes = [...Buffer.from(`\x1b[<${code};${x + 1};${y + 1}${up ? "m" : "M"}`)].map((b) => b.toString(16));
  tmux("send-keys", "-t", pane, "-H", ...bytes);
};
const find = (text) => {
  const rows = screen().split("\n");
  const y = rows.findLastIndex((row) => row.includes(text));
  return y < 0 ? undefined : { x: rows[y].indexOf(text) + 1, y };
};
const row = (y) => screen(true).split("\n")[y];
const check = (what, ok) => {
  console.log(`${ok ? "PASS" : "FAIL"} ${what}`);
  if (!ok) throw new Error(what);
};
const occurrences = (text, part) => text.split(part).length - 1;

async function start(...extra) {
  tmux("new-session", "-d", "-s", pane, "-x", "100", "-y", "40", "-f", "/dev/null", `sh -c '${pi(...extra)}'`);
  await until("pi prompt", () => screen().includes("(auto)"));
}
const send = (text) => {
  tmux("send-keys", "-t", pane, "-l", text);
  tmux("send-keys", "-t", pane, "Enter");
};
const stop = () => tmux("kill-session", "-t", pane);

try {
  await start();
  send("ask");
  await until("inline answers", () => find("migration? Yes/No/Other"));
  const first = screen();
  check("the question text shows once", occurrences(first, "Should I continue with the migration?") === 1);
  check("the code block shows", first.includes("what is this?"));
  check("the question line keeps its height", first.includes("Which database should we use: PostgreSQL, SQLite or MySQL? Other\n"));
  check("any-motion tracking is on", tmux("display-message", "-p", "-t", pane, "#{mouse_all_flag}").trim() === "1");

  const at = find("PostgreSQL, SQLite or");
  const sqlite = { x: at.x + "PostgreSQL, ".length - 1, y: at.y };
  const before = row(sqlite.y);
  mouse(35, sqlite.x + 2, sqlite.y);
  await until("hover style", () => row(sqlite.y) !== before);
  check("hover changes the pill", true);
  mouse(35, 1, 0);
  await until("hover cleared", () => row(sqlite.y) === before);
  check("moving away clears the hover", true);

  const click = (at) => {
    mouse(0, at.x + 2, at.y);
    mouse(0, at.x + 2, at.y, true);
  };
  const editorText = () => {
    const rows = screen().split("\n");
    const bars = rows.flatMap((row, y) => (row.startsWith("───") ? [y] : []));
    return rows.slice(bars.at(-2) + 1, bars.at(-1)).map((row) => row.trim()).join("\n").replace(/^\n+|\n+$/g, "");
  };
  const quoteDb = "> Which database should we use: PostgreSQL, SQLite or MySQL?";
  const quoteMigration = "> Should I continue with the migration?";
  click(sqlite);
  await until("editor has the answer", () => editorText() === `${quoteDb}\n\nSQLite`);
  check("a click puts the quote and the answer in the editor", true);
  check("a click sends nothing", userLines().length === 1);
  const migration = find("migration? Yes/No/Other");
  const yes = { x: migration.x + "migration? ".length - 1, y: migration.y };
  click(yes);
  await until("two blocks", () => editorText() === `${quoteDb}\n\nSQLite\n\n${quoteMigration}\n\nYes`);
  check("a second question adds a block", true);
  // pi counts a second click within 500 ms in the same word ("Yes/No/Other") as a double click
  await new Promise((resolve) => setTimeout(resolve, 600));
  const again = find("migration? Yes/No/Other");
  click({ x: again.x + "migration? Yes/".length - 2, y: again.y });
  await until("replaced", () => editorText().endsWith(`${quoteMigration}\n\nNo`));
  check("another answer replaces the block's answer", editorText().split("> ").length === 3);
  tmux("send-keys", "-t", pane, "Enter");
  await until("sent", () => userLines().length === 2);
  check("Enter sends the composed reply", userLines()[1] === `USER ${JSON.stringify(`${quoteDb}\n\nSQLite\n\n${quoteMigration}\n\nNo`)}`);
  await until("reply", () => screen().includes("Got: > Which"));

  send("list");
  await until("list", () => find("Markdown or HTML?"));
  await sleep(500);
  const list = screen();
  check("the list keeps one row per item", list.split("\n").filter((l) => /^ \d\. The /.test(l)).length === 4);
  check("a choice question ends with Other", find("v2 or v3? Other") !== undefined);
  send("one");
  await until("options", () => find("acceptable"));
  click(find("acceptable"));
  await until("first answer", () => editorText().includes("risky?\n\nacceptable"));
  const plan = find("next week? Yes/No/Other");
  click({ x: plan.x + "next week? ".length - 1, y: plan.y });
  await until("stripped quotes", () => editorText() === "> Is the option to drop the cache acceptable, or is it too risky?\n\nacceptable\n\n> 2. Plan: Is there a plan to ship the docs next week?\n\nYes");
  check("quotes lose the markup", true);
  tmux("send-keys", "-t", pane, "Enter");
  await until("one sent", () => userLines().length === 5);
  send("open");
  await until("open question", () => find("called? Answer"));
  const open = find("called? Answer");
  click({ x: open.x + "called? ".length - 1, y: open.y });
  await until("quote with an empty answer", () => editorText() === "> What should the new service be called?");
  tmux("send-keys", "-t", pane, "-l", "Foo");
  await until("typing goes into the answer", () => editorText() === "> What should the new service be called?\n\nFoo");
  const second = find("called? Answer");
  click({ x: second.x + "called? ".length - 1, y: second.y });
  tmux("send-keys", "-t", pane, "-l", "d");
  await until("no second block", () => editorText() === "> What should the new service be called?\n\nFood");
  check("Answer adds the quote with an empty answer and keeps the cursor in it", true);
  tmux("send-keys", "-t", pane, "Enter");
  await until("open sent", () => userLines().length === 7);
  check("the typed answer is sent", userLines()[6] === `USER ${JSON.stringify("> What should the new service be called?\n\nFood")}`);
  send("abort");
  await until("aborted reply", () => screen().includes("Cut off"));
  check("an aborted reply stays visible", screen().includes("Should I continue with the migration?"));

  send("ask");
  await until("second question", () => screen().lastIndexOf("Trailing note.") > screen().lastIndexOf("Operation aborted"));
  const final = screen();
  stop();
  await sleep(300);
  await start("--continue");
  await until("resumed answers", () => find("migration? Yes/No/Other"));
  const resumed = screen();
  check("a resumed session shows the answers", resumed.includes("migration? Yes/No/Other"));
  const body = (text) => text.slice(text.lastIndexOf(" ask"), text.lastIndexOf("Trailing note."));
  check("the resumed screen has the same text", body(final) === body(resumed));
} finally {
  try {
    stop();
  } catch {}
  rmSync(scratch, { recursive: true, force: true });
}
