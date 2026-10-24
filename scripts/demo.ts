import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

const REPLY =
  "The nightly build fails because the cache key ignores `package-lock.json`, so CI restores stale modules.\n\n" +
  "Should I fix the cache key in `ci.yml`, or remove the cache?\n\n" +
  "I can also add a check that fails when the lock file changes without a new key. Do you want that?\n\n" +
  "Which Node version should CI pin?";
const zero = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

export default function (pi: ExtensionAPI) {
  pi.registerProvider("demo", {
    apiKey: "x",
    baseUrl: "http://localhost:1",
    api: "demo-api" as any,
    models: [{ id: "agent", name: "Agent", reasoning: false, input: ["text"], cost: zero, contextWindow: 100000, maxTokens: 4000 }],
    streamSimple: (model: any, context: any) => {
      const turn = context.messages.filter((m: any) => m.role === "user").length;
      const text = turn === 1 ? REPLY : "Done: the cache key now includes the lock file, the new check is in `ci.yml`, and CI pins Node 24.";
      const stream = createAssistantMessageEventStream();
      const out: any = {
        role: "assistant",
        content: [{ type: "text", text: "" }],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: { ...zero, totalTokens: 0, cost: { ...zero, total: 0 } },
        stopReason: "stop",
        timestamp: Date.now(),
      };
      (async () => {
        stream.push({ type: "start", partial: out });
        stream.push({ type: "text_start", contentIndex: 0, partial: out });
        for (const word of text.match(/\S+\s*/g)!) {
          await new Promise((done) => setTimeout(done, 25));
          out.content[0].text += word;
          stream.push({ type: "text_delta", contentIndex: 0, delta: word, partial: out });
        }
        stream.push({ type: "text_end", contentIndex: 0, content: text, partial: out });
        stream.push({ type: "done", reason: "stop", message: out });
        stream.end();
      })();
      return stream;
    },
  });
}
