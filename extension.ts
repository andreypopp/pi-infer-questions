import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
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

const here = dirname(fileURLToPath(import.meta.url));
type Native = { start(models: string): void; questions(markdown: string): Promise<string> };
const loaded: { native: Native } | { error: string } = (() => {
  const platform = `${process.platform}-${process.arch}`;
  if (!["darwin-arm64", "linux-arm64", "linux-x64"].includes(platform)) return { error: `${platform} is not supported` };
  try {
    const addon = createRequire(import.meta.url)(join(here, "questions.node"));
    addon.start(join(here, "models"));
    return { native: addon as Native };
  } catch (error) {
    return { error: `the module or the models in ${here} do not load: ${error instanceof Error ? error.message : error}` };
  }
})();

let hovered: string | undefined;
let moves = 0;
let lastHoverMove = 0;
let tui: TUI | undefined;
let editor: Pick<ExtensionUIContext, "getEditorText" | "setEditorText"> | undefined;
const wrapped = new WeakSet<TUI>();

const blocks = (cps: string[]): [number, number][] => {
  const found: [number, number][] = [];
  let begin = -1;
  let end = 0;
  let fenced = false;
  for (let start = 0; start <= cps.length; ) {
    const stop = cps.indexOf("\n", start) < 0 ? cps.length : cps.indexOf("\n", start);
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
  reset?: boolean;
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

  constructor(
    private readonly inner: Component,
    private readonly theme: Theme,
    private readonly id: string,
    private readonly replies: Reply[],
    private readonly send: (reply: Reply) => void,
  ) {}

  render(width: number): string[] {
    const [open, close] = this.theme.fg("accent", "\0").split("\0");
    this.cells = [];
    let option = -1;
    let seen = -1;
    let hover = false;
    const optionOn = () => `${open}${hover ? "\x1b[4;7m" : "\x1b[4m"}`;
    const optionOff = () => `${hover ? "\x1b[24;27m" : "\x1b[24m"}${close}`;
    const reapply = () => (option >= 0 ? optionOn() : "");
    return this.inner.render(width).map((line, row) => {
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
          hover = `${this.id}:${option}` === hovered;
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
  }

  handleMouse(event: TuiMouseEvent) {
    const cell = this.cells.find((c) => c.row === event.y && event.x >= c.from && event.x < c.to);
    if (event.type === "click" && cell) {
      if (cell.reply) this.send(cell.reply);
      return { handled: true };
    }
    if (event.type !== "move" && event.type !== "drag") return undefined;
    lastHoverMove = moves;
    if (hovered === cell?.id) return { handled: true };
    hovered = cell?.id;
    return { handled: true, render: true };
  }

  invalidate(): void {
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
      const terminal = current.terminal as { write(data: string): void; [TYPE]?: true };
      if (!terminal[TYPE]) {
        terminal[TYPE] = true;
        const write = terminal.write.bind(terminal);
        terminal.write = (data) => write(data.includes(BUTTON_MOTION) ? data + ANY_MOTION : data);
      }
      const mouse = current as unknown as { handleMouseEvent(raw: unknown): void };
      if (mouse.handleMouseEvent && !wrapped.has(current)) {
        wrapped.add(current);
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
      native!
        .questions(markdown)
        .then((json) => {
          const cps = Array.from(markdown);
          const questions = (JSON.parse(json) as Analysis[]).filter(
            (q) => q.options.length > 0 || cps.slice(q.range[0], q.range[1]).includes("?"),
          );
          if (questions.length === 0) return;
          texts.add(markdown);
          pi.appendEntry<Data>(TYPE, { markdown, questions });
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
    blocks(cps).forEach(([from, to], index) => {
      if (index) root.addChild(new Spacer(1));
      const marks: { at: number; text: string; reply?: Reply }[] = [];
      for (const question of questions) {
        const [start, end] = question.range;
        if (start >= to || end <= from) continue;
        const quote = cps
          .slice(start, end)
          .join("")
          .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
          .split("\n")
          .map(plain)
          .map((line) => line.replace(/[ \t]+/g, " ").trim())
          .filter(Boolean)
          .join("\n");
        for (const [a, b] of question.options) {
          if (a < from || b > to) continue;
          const answer = plain(cps.slice(a, b).join(""));
          marks.push({ at: a, text: OPEN, reply: { quote, answer } }, { at: b, text: CLOSE });
        }
        if (question.kind === "open" && end <= to) {
          marks.push({ at: end, text: ` ${OPEN}Answer${CLOSE}`, reply: { quote, answer: "" } });
        }
        if (end <= to && question.kind !== "open") {
          const answers = question.kind === "yes_no" ? ["Yes", "No"] : question.options.length === 1 ? ["Yes"] : [];
          const replies: Reply[] = [...answers.map((answer) => ({ quote, answer })), { quote, answer: "", reset: true }];
          replies.forEach((reply, n) => marks.push({ at: end, text: `${n ? "/" : " "}${OPEN}${reply.answer || "Other"}${CLOSE}`, reply }));
        }
      }
      const ordered = marks.map((mark, number) => ({ ...mark, number })).sort((a, b) => a.at - b.at || a.number - b.number);
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
          const at = found === 0 || (found > 0 && current.slice(found - 2, found) === "\n\n") ? found : -1;
          if (at < 0) editor.setEditorText(current.trim() ? `${current}\n\n${block}` : block);
          else if (reply.answer || reply.reset) {
            const next = current.indexOf("\n\n> ", at + head.length);
            editor.setEditorText(current.slice(0, at) + block + (next < 0 ? "" : current.slice(next)));
          }
        }),
      );
    });
    return root;
  });
}
