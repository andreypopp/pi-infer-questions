import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [cast] = process.argv.slice(2);
const root = resolve(import.meta.dirname, "..");
const scratch = realpathSync(mkdtempSync("/tmp/pq-demo-"));
const socket = join(scratch, "socket");
mkdirSync(join(scratch, "agent"));
mkdirSync(join(scratch, "shop"));
writeFileSync(join(scratch, "agent", "settings.json"), JSON.stringify({ lastChangelogVersion: "1.2.0", quietStartup: true }));

const tmux = (...args) => execFileSync("tmux", ["-S", socket, ...args], { encoding: "utf8" });
const pane = "demo";
const screen = () => tmux("capture-pane", "-p", "-t", pane);
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

async function until(what, check, ms = 20_000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const found = check();
    if (found) return found;
    await sleep(50);
  }
  throw new Error(`timeout: ${what}\n${screen()}`);
}
const find = (text) => {
  const rows = screen().split("\n");
  const y = rows.findLastIndex((row) => row.includes(text));
  return y < 0 ? undefined : { x: rows[y].indexOf(text) + 1, y };
};
const mouse = (code, x, y, up = false) => {
  const bytes = [...Buffer.from(`\x1b[<${code};${x + 1};${y + 1}${up ? "m" : "M"}`)].map((b) => b.toString(16));
  tmux("send-keys", "-t", pane, "-H", ...bytes);
};
const click = async (text, offset = 0) => {
  const at = await until(text, () => find(text));
  mouse(35, at.x + offset + 1, at.y);
  await sleep(700);
  mouse(0, at.x + offset + 1, at.y);
  mouse(0, at.x + offset + 1, at.y, true);
  mouse(35, 1, 0);
  await sleep(1200);
};
const type = async (text) => {
  for (const ch of text) {
    tmux("send-keys", "-t", pane, "-l", ch);
    await sleep(60);
  }
};

const pi = [
  `cd ${scratch}/shop`,
  `export HOME=${scratch} PI_CODING_AGENT_DIR=${scratch}/agent PI_OFFLINE=1`,
  `exec pi --no-extensions -e ${root}/extension.ts -e ${root}/scripts/demo.ts --provider demo --model agent --no-session --tui-mode fullscreen --no-skills --no-context-files`,
].join(" && ");

let recorder;
try {
  writeFileSync(join(scratch, "tmux.conf"), "set -g status off\nset -g extended-keys on\nset -g extended-keys-format csi-u\n");
  tmux("-f", join(scratch, "tmux.conf"), "new-session", "-d", "-s", pane, "-x", "90", "-y", "32", `sh -c '${pi}'`);
  await until("pi prompt", () => screen().includes("(auto)"));
  recorder = spawn(
    "asciinema",
    ["rec", "--headless", "--overwrite", "--window-size", "90x32", "-i", "1.5", "-c", `tmux -S ${socket} attach -t ${pane}`, resolve(cast)],
    { stdio: "ignore", env: { ...process.env, TMUX: undefined, TERM: "xterm-256color" } },
  );
  await sleep(1500);
  await type("Why does the nightly build fail?");
  await sleep(500);
  tmux("send-keys", "-t", pane, "Enter");
  await until("controls", () => find("pin? Answer"));
  await sleep(2500);
  await click("remove the cache");
  await click("that? Yes", "that? ".length);
  await click("pin? Answer", "pin? ".length);
  await type("24");
  await sleep(1500);
  tmux("send-keys", "-t", pane, "Enter");
  await until("second reply", () => find("Node 24."));
  await sleep(3000);
} finally {
  try {
    tmux("kill-session", "-t", pane);
  } catch {}
  if (recorder) await new Promise((done) => (recorder.exitCode === null ? recorder.on("exit", done) : done()));
  rmSync(scratch, { recursive: true, force: true });
}
