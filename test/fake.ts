import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { appendFileSync } from "node:fs";

const QUESTION =
  "I looked at the options.\n\nWhich database should we use: **PostgreSQL**, SQLite or MySQL?\n\n```sh\nwhat is this?\n```\n\nShould I continue with the migration?\n\nTrailing note.";
const LIST =
  "Four things need your decision:\n\n1. The cache: Should we keep the cache? Do you prefer to drop it or to shrink it?\n2. The API: Should we rename the API? Do you want v2 or v3?\n3. The tests: Should we add the tests now? Do you want unit or e2e tests?\n4. The docs: Should we update the docs? Do you prefer Markdown or HTML?";
const ONE =
  "Two points:\n\nIs dropping the cache acceptable, or is it too risky?\n2. **Plan:** Is there a plan to ship the docs next week?";
const zero = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

const reply = (model: any, text: string, stop: "stop" | "aborted") => {
  const stream = createAssistantMessageEventStream();
  const out: any = {
    role: "assistant",
    content: [{ type: "text", text }],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: { ...zero, totalTokens: 0, cost: { ...zero, total: 0 } },
    stopReason: stop,
    timestamp: Date.now(),
  };
  queueMicrotask(() => {
    stream.push({ type: "start", partial: out });
    stream.push(stop === "stop" ? { type: "done", reason: "stop", message: out } : { type: "error", reason: "aborted", error: out });
    stream.end();
  });
  return stream;
};

export default function (pi: ExtensionAPI) {
  pi.registerProvider("fake", {
    apiKey: "x",
    baseUrl: "http://localhost:1",
    api: "fake-api" as any,
    models: [{ id: "m", name: "Fake", reasoning: false, input: ["text"], cost: zero, contextWindow: 100000, maxTokens: 4000 }],
    streamSimple: (model: any, context: any) => {
      const last = context.messages[context.messages.length - 1];
      const text = typeof last.content === "string" ? last.content : last.content.map((c: any) => c.text ?? "").join("");
      appendFileSync(process.env.PQ_LOG!, `USER ${JSON.stringify(text)}\n`);
      if (text === "ask") return reply(model, QUESTION, "stop");
      if (text === "open") return reply(model, "Before I start.\n\nWhat should the new service be called?", "stop");
      if (text === "one") return reply(model, ONE, "stop");
      if (text === "list") return reply(model, LIST, "stop");
      if (text === "abort") return reply(model, QUESTION.replace("Trailing note.", "Cut off"), "aborted");
      return reply(model, `Got: ${text}`, "stop");
    },
  });
}
