import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionUIContext, Theme } from "@earendil-works/pi-coding-agent";
import { getAgentDir, getMarkdownTheme, SettingsManager } from "@earendil-works/pi-coding-agent";
import {
  Container,
  Markdown,
  Spacer,
  visibleWidth,
  type Component,
  type TUI,
  type TuiMouseEvent,
} from "@earendil-works/pi-tui";

const TYPE = "pi-infer-questions";
const GUIDE_TYPE = `${TYPE}-guide`;
const GUIDE = [
  'When you need a decision or input from the user, ask it as a question that ends with "?", at the end of your reply.',
  "Put one question in one sentence or one list item.",
  'For a choice, name the alternatives in the question ("Should I A, B, or C?") or list them as items directly under the question.',
  'For a proposal, ask for the go-ahead ("Should I do X?"). Do not only write "I recommend X".',
  "Do not ask questions in text that comes before a tool call. Ask only when you stop.",
];
const BUTTON_MOTION = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h";
const ANY_MOTION = "\x1b[?1003h";
const OPEN = "\uE010";
const CLOSE = "\uE011";
const plain = (text: string) => text.replace(/[`*]|__|~~/g, "");
const TOKEN = /[\uE010\uE011]|\x1b\[[0-9;]*m|\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)|[^]/gu;
const SGR_RESET = /^\x1b\[(?:0|22|24|27|39)?m$/;

interface Analysis {
  range: [number, number];
  kind: "choice" | "yes_no" | "open";
  options: [number, number][];
}
interface Data {
  markdown: string;
  questions: Analysis[];
}

type Native = { start(models: string): void; questions(markdown: string): Promise<string> };
const loaded: { native: Native } | { error: string } = (() => {
  const platform = `${process.platform}-${process.arch}`;
  if (!["darwin-arm64", "linux-arm64", "linux-x64"].includes(platform)) return { error: `${platform} is not supported` };
  const name = `@andreypopp/pi-infer-questions-${platform}`;
  const require = createRequire(import.meta.url);
  let root: string;
  try {
    root = dirname(require.resolve(`${name}/package.json`));
  } catch {
    return { error: `the package ${name} is not installed; install with optional dependencies enabled` };
  }
  try {
    const addon = require(join(root, "questions.node"));
    addon.start(join(root, "models"));
    return { native: addon as Native };
  } catch (error) {
    return { error: `the module or the models in ${root} do not load: ${error instanceof Error ? error.message : error}` };
  }
})();

let hovered: string | undefined;
let moves = 0;
let lastHoverMove = 0;
let tui: TUI | undefined;
let editor: Pick<ExtensionUIContext, "getEditorText" | "setEditorText"> | undefined;
const patched = new WeakSet<TUI>();
const collapsed = new WeakSet<Component>();

const parentOf = (node: Component, child: Component): Container | undefined => {
  const children = (node as Container).children ?? [];
  if (children.includes(child)) return node as Container;
  for (const next of children) {
    const found = parentOf(next, child);
    if (found) return found;
  }
  return undefined;
};

// pi still draws a spacer line for the assistant message whose text the transformer hides, and the custom
// entry adds its own spacer, so the reply would move down one row when its entry replaces it.
const collapseHidden = (root: Component): boolean => {
  const entry = tui && parentOf(tui, root);
  const chat = entry && tui && parentOf(tui, entry);
  const hidden = chat?.children[chat.children.indexOf(entry) - 1];
  if (!hidden || collapsed.has(hidden)) return false;
  collapsed.add(hidden);
  const render = hidden.render.bind(hidden);
  hidden.render = (width) => {
    const lines = render(width);
    return lines.every((line) => visibleWidth(line) === 0) ? [] : lines;
  };
  return true;
};
let latest: Component | undefined;

const blocks = (cps: string[]): [number, number][] => {
  const found: [number, number][] = [];
  let begin = -1;
  let end = 0;
  let fenced = false;
  for (let start = 0; start <= cps.length; ) {
    const newline = cps.indexOf("\n", start);
    const stop = newline < 0 ? cps.length : newline;
    const line = cps.slice(start, stop).join("");
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    if (!line.trim() && !fenced) {
      if (begin >= 0) found.push([begin, end]);
      begin = -1;
    } else {
      if (begin < 0) begin = start;
      end = stop;
    }
    start = stop + 1;
  }
  if (begin >= 0) found.push([begin, end]);
  return found;
};

interface Reply {
  quote: string;
  answer: string;
  overwrite?: boolean;
}
interface Cell {
  row: number;
  from: number;
  to: number;
  id: string;
  reply?: Reply;
}

class Interactive implements Component {
  private cells: Cell[] = [];
  private last?: { lines: string[]; hover?: string; out: string[] };

  constructor(
    private readonly inner: Component,
    private readonly theme: Theme,
    private readonly id: string,
    private readonly replies: Reply[],
    private readonly send: (reply: Reply) => void,
  ) {}

  render(width: number): string[] {
    const lines = this.inner.render(width);
    const hover = hovered?.startsWith(`${this.id}:`) ? hovered : undefined;
    if (this.last?.lines === lines && this.last.hover === hover) return this.last.out;
    const [open, close] = this.theme.fg("accent", "\0").split("\0");
    this.cells = [];
    let option = -1;
    let seen = -1;
    let lit = false;
    const optionOn = () => `${open}${lit ? "\x1b[4;7m" : "\x1b[4m"}`;
    const optionOff = () => `${lit ? "\x1b[24;27m" : "\x1b[24m"}${close}`;
    const reapply = () => (option >= 0 ? optionOn() : "");
    const rendered = lines.map((line, row) => {
      let out = reapply();
      let column = 0;
      let cell: Cell | undefined;
      const begin = () => {
        const id = `${this.id}:${option}`;
        cell = { row, from: column, to: column, id, reply: this.replies[option] };
        this.cells.push(cell);
      };
      if (option >= 0) begin();
      for (const [token] of line.matchAll(TOKEN)) {
        if (token === OPEN) {
          option = ++seen;
          lit = `${this.id}:${option}` === hover;
          out += optionOn();
          begin();
        } else if (token === CLOSE) {
          out += optionOff();
          option = -1;
          cell = undefined;
        } else if (token.startsWith("\x1b")) {
          out += token + (SGR_RESET.test(token) ? reapply() : "");
        } else {
          out += token;
          column += visibleWidth(token);
          if (cell) cell.to = column;
        }
      }
      return out + (option >= 0 ? optionOff() : "");
    });
    this.last = { lines, hover, out: rendered };
    return rendered;
  }

  handleMouse(event: TuiMouseEvent) {
    const cell = this.cells.find((c) => c.row === event.y && event.x >= c.from && event.x < c.to);
    if (event.type === "click" && cell) {
      if (cell.reply) this.send(cell.reply);
      return { handled: true };
    }
    if (event.type !== "move" || !cell) return undefined;
    lastHoverMove = moves;
    if (hovered === cell.id) return { handled: true };
    hovered = cell.id;
    return { handled: true, render: true };
  }

  invalidate(): void {
    this.last = undefined;
    this.inner.invalidate();
  }
}

export default function (pi: ExtensionAPI) {
  if ("error" in loaded) {
    pi.on("session_start", (_event, ctx) => ctx.ui.notify(`${TYPE}: ${loaded.error}`, "error"));
    return;
  }
  const { native } = loaded;
  const texts = new Set<string>();

  pi.registerFlag("no-question-guide", {
    description: "Do not add the question-asking guideline to the system prompt",
    type: "boolean",
    default: false,
  });

  pi.on("before_agent_start", (event, ctx) => {
    if (pi.getFlag("no-question-guide")) return;
    event.systemPromptOptions.promptGuidelines.push(...GUIDE);
    if (!ctx.sessionManager.getEntries().some((entry) => entry.type === "custom" && entry.customType === GUIDE_TYPE)) {
      pi.appendEntry(GUIDE_TYPE, { guidelines: GUIDE });
    }
  });

  pi.registerMarkdownTransformer((markdown, context) =>
    context.messageType === "assistant" && !context.isStreaming && texts.has(markdown.trim()) ? "" : markdown,
  );

  pi.on("session_start", (_event, ctx) => {
    texts.clear();
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type === "custom" && entry.customType === TYPE) texts.add((entry.data as Data).markdown);
    }
    if (ctx.mode !== "tui") return;
    editor = ctx.ui;
    ctx.ui.setWidget(TYPE, (current) => {
      tui = current;
      const terminal = current.terminal as { write(data: string): void };
      if (!patched.has(current)) {
        patched.add(current);
        const write = terminal.write.bind(terminal);
        terminal.write = (data) => write(data.includes(BUTTON_MOTION) ? data + ANY_MOTION : data);
        const mouse = current as unknown as { handleMouseEvent?(raw: unknown): void };
        if (mouse.handleMouseEvent) {
          const handle = mouse.handleMouseEvent.bind(current);
          mouse.handleMouseEvent = (raw) => {
            const seen = ++moves;
            handle(raw);
            if (hovered !== undefined && lastHoverMove !== seen) {
              hovered = undefined;
              current.requestRender();
            }
          };
        }
      }
      if ((current as unknown as { altScreenActive?: boolean }).altScreenActive) terminal.write(ANY_MOTION);
      return { render: () => [], invalidate() {} };
    });
  });

  pi.on("message_end", (event) => {
    const message = event.message as { role: string; stopReason?: string; content: { type: string; text?: string }[] };
    if (message.role !== "assistant" || message.stopReason !== "stop") return;
    for (const block of message.content) {
      const markdown = block.type === "text" ? block.text?.trim() : undefined;
      if (!markdown) continue;
      native
        .questions(markdown)
        .then((json) => {
          const questions = JSON.parse(json) as Analysis[];
          if (questions.length === 0) return;
          texts.add(markdown);
          pi.appendEntry<Data>(TYPE, { markdown, questions });
          if (latest) collapseHidden(latest);
          tui?.invalidate();
          tui?.requestRender();
        })
        .catch(() => {});
    }
  });

  const outputPad = SettingsManager.create(process.cwd(), getAgentDir()).getOutputPad();

  pi.registerEntryRenderer<Data>(TYPE, (entry, _options, theme) => {
    if (!entry.data) return undefined;
    const { markdown, questions } = entry.data;
    const cps = Array.from(markdown);
    const markdownTheme = getMarkdownTheme();
    const root = new Container();
    latest = root;
    let placed = false;
    const render = root.render.bind(root);
    root.render = (width) => {
      if (!placed) {
        placed = true;
        if (collapseHidden(root)) tui?.requestRender();
      }
      return render(width);
    };
    const cut = (from: number, to: number) => cps.slice(from, to).join("");
    const quotes = questions.map(({ range: [start, end] }) =>
      cut(start, end)
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
        .split("\n")
        .map(plain)
        .map((line) => line.replace(/[ \t]+/g, " ").trim())
        .filter(Boolean)
        .join("\n"),
    );
    blocks(cps).forEach(([from, to], index) => {
      if (index) root.addChild(new Spacer(1));
      const marks: { at: number; text: string; reply?: Reply }[] = [];
      questions.forEach((question, number) => {
        const [start, end] = question.range;
        if (start >= to || end <= from) return;
        const quote = quotes[number];
        for (const [a, b] of question.options) {
          if (a < from || b > to) continue;
          marks.push({ at: a, text: OPEN, reply: { quote, answer: plain(cut(a, b)) } }, { at: b, text: CLOSE });
        }
        if (end > to) return;
        const labels =
          question.kind === "open"
            ? ["Answer"]
            : [...(question.kind === "yes_no" ? ["Yes", "No"] : question.options.length === 1 ? ["Yes"] : []), "Other"];
        labels.forEach((label, n) => {
          const answer = label === "Answer" || label === "Other" ? "" : label;
          marks.push({ at: end, text: `${n ? "/" : " "}${OPEN}${label}${CLOSE}`, reply: { quote, answer, overwrite: label === "Other" } });
        });
      });
      const ordered = [...marks].sort((a, b) => a.at - b.at);
      const text = cps.slice(from, to);
      for (const mark of [...ordered].reverse()) text.splice(mark.at - from, 0, mark.text);
      const replies = ordered.filter((mark) => mark.reply).map((mark) => mark.reply!);
      root.addChild(
        new Interactive(new Markdown(text.join(""), outputPad, 0, markdownTheme), theme, `${entry.id}:${index}`, replies, (reply) => {
          if (!editor) return;
          const quoted = reply.quote.split("\n").map((line) => `> ${line}`).join("\n");
          const current = editor.getEditorText();
          const head = `${quoted}\n\n`;
          const block = head + reply.answer;
          const found = current.indexOf(head);
          const at = found > 0 && current.slice(found - 2, found) !== "\n\n" ? -1 : found;
          if (at < 0) editor.setEditorText(current.trim() ? `${current}\n\n${block}` : block);
          else if (reply.answer || reply.overwrite) {
            const next = current.indexOf("\n\n> ", at + head.length);
            editor.setEditorText(current.slice(0, at) + block + (next < 0 ? "" : current.slice(next)));
          }
        }),
      );
    });
    return root;
  });
}
