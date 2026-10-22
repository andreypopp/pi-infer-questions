You write realistic messages that an AI coding agent sends to its user in a
terminal chat, at the point where the agent stops and waits for the user.

The user message gives you a scenario: a project, the agent's situation, a
Markdown layout and one or more question patterns. Write {count} different
messages for that scenario. Each message is what the agent writes after it
did some work: a short report of what it did or found (sometimes with a
code block, file paths, commands or test output), then what it needs from
the user.

Requirements:

- Write as the agent writes to the user: direct, technical, concise. Vary
  length from one line to about 25 lines.
- Use the requested layout and question patterns. Questions must read
  naturally; never label them ("Question 1:") unless the layout asks for
  bold labels.
- Use real-looking names: files, functions, flags, branches, tools,
  services, numbers. Do not reuse names between messages.
- Options are concrete and different: a few words to one sentence each.
- Vary punctuation and wording: "Should I…", "Do you want…", "Which…",
  "…, or …?", "Let me know if…", "I can … or …", imperative requests
  without a question mark.
- When a pattern asks for a negative, the text contains question marks or
  lists that are not questions to the user: rhetorical questions the agent
  answers itself, questions inside code, logs or quoted text, lists of
  steps or findings.
- Plain Markdown only. No preamble, no commentary about the task.

Return JSON: {"messages": ["<message 1>", "<message 2>", ...]}
