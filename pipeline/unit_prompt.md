You find the questions that a coding agent asks the user, in one unit of one message of the agent.
A unit is one paragraph of the message as plain text on one line. A unit can also be a paragraph that ends with a question or a colon, followed by the list that belongs to it. A unit can also be a heading or a bold line that introduces a list; it is context and never part of a question. List markers ("-", "*", "1.", "2)", "a.") are kept. Return JSON only, in this shape:
{"questions": [{"question": "exact text", "kind": "choice|yes_no|open", "options": ["exact text", "exact text"]}]}
A question needs an answer from the user. List the questions in text order; they do not overlap. "question" is a copy of the question's text in the unit, character for character, from its first word to its last character, and it occurs once in the unit.
"kind" is "choice" (the user picks between options that the text names; "options" has them), "yes_no" (the user answers yes or no) or "open" (there is no finite answer, as in "What should the name be?"). Only "choice" has options.
"options" are copies of text inside the question, character for character: no list marker, no trailing full stop, inner punctuation kept, at least 3 characters, no duplicates, each one occurring once in the question. A list item gives one option.
If the unit has no question, return {"questions": []}.
Rules, each with an example:

Several questions in one unit are several questions.
Text: Should I push it now, or wait for CI? Also, do you want the docs updated too?
Answer: {"questions": [{"question": "Should I push it now, or wait for CI?", "kind": "choice", "options": ["push it now", "wait for CI"]}, {"question": "Also, do you want the docs updated too?", "kind": "yes_no", "options": []}]}

Two sentences that form one choice are one question.
Text: Should I keep the cache? Or drop it and rebuild?
Answer: {"questions": [{"question": "Should I keep the cache? Or drop it and rebuild?", "kind": "choice", "options": ["keep the cache", "drop it and rebuild"]}]}

A numbered list of questions is several questions, one for each item.
Text: I have two questions. 1. Should I update the docs? 2. What should the name be?
Answer: {"questions": [{"question": "1. Should I update the docs?", "kind": "yes_no", "options": []}, {"question": "2. What should the name be?", "kind": "open", "options": []}]}

A question followed by a list whose items answer it is one question, from the question to the last item. Each item is an option, also when it is a full sentence.
Text: Which should the CLI ship? 1. Only the 1,024 function, with chunking. 2. Both functions, and the CLI selects by text length. 3. Only the 2,048 function.
Answer: {"questions": [{"question": "Which should the CLI ship? 1. Only the 1,024 function, with chunking. 2. Both functions, and the CLI selects by text length. 3. Only the 2,048 function.", "kind": "choice", "options": ["Only the 1,024 function, with chunking", "Both functions, and the CLI selects by text length", "Only the 2,048 function"]}]}

A question can start with a clause.
Text: For the 2,048-token function, which should the CLI ship? - Only the 1,024 function - Both functions
Answer: {"questions": [{"question": "For the 2,048-token function, which should the CLI ship? - Only the 1,024 function - Both functions", "kind": "choice", "options": ["Only the 1,024 function", "Both functions"]}]}

A sentence before the question that only gives context is not part of the question.
Text: I checked the build. Which one do you prefer, the old cache or the new one?
Answer: {"questions": [{"question": "Which one do you prefer, the old cache or the new one?", "kind": "choice", "options": ["the old cache", "the new one"]}]}

A question can come after its options. It starts at the lead-in of the options.
Text: There are two ways: - update the docs - keep the change Which one do you want?
Answer: {"questions": [{"question": "There are two ways: - update the docs - keep the change Which one do you want?", "kind": "choice", "options": ["update the docs", "keep the change"]}]}

Inline options: the option is the noun phrase after "or" or the list word, without the verb that the options share.
Text: Should I update the docs or keep the change?
Answer: {"questions": [{"question": "Should I update the docs or keep the change?", "kind": "choice", "options": ["update the docs", "keep the change"]}]}

A yes/no question has no options. Words like "y/n" or "yes or no" are not options.
Text: Should I run the tests now? Reply y/n.
Answer: {"questions": [{"question": "Should I run the tests now?", "kind": "yes_no", "options": []}]}

A proposal that waits for the user's approval is a "yes_no" question, also without a question mark. It is the sentences that state the proposed action and ask for approval, in any wording: a recommendation, a suggestion, a plan, a next step that waits for a go-ahead, or an action that happens unless the user objects. A heading or label before them is not part of it. Text that does not wait for the user is not a question: a choice already made, an action already done or started, general advice, a report.
Text: The cache is stale. I recommend we drop it and rebuild it. Say the word and I'll do it.
Answer: {"questions": [{"question": "I recommend we drop it and rebuild it. Say the word and I'll do it.", "kind": "yes_no", "options": []}]}
Text: 1. Archive branch. My suggestion: commit the app changes to a separate branch. This needs git writes, so I wait for your go-ahead.
Answer: {"questions": [{"question": "My suggestion: commit the app changes to a separate branch. This needs git writes, so I wait for your go-ahead.", "kind": "yes_no", "options": []}]}
Text: Plan: split parser.ml into lexer.ml and parser.ml, then move the tests. Once you confirm, I'll start with the lexer.
Answer: {"questions": [{"question": "Plan: split parser.ml into lexer.ml and parser.ml, then move the tests. Once you confirm, I'll start with the lexer.", "kind": "yes_no", "options": []}]}
Text: Next I'll delete the three stale branches on origin, unless you object.
Answer: {"questions": [{"question": "Next I'll delete the three stale branches on origin, unless you object.", "kind": "yes_no", "options": []}]}
Text: I propose to pin the dependency to 2.3 until upstream fixes the leak. Happy to go another way if you prefer.
Answer: {"questions": [{"question": "I propose to pin the dependency to 2.3 until upstream fixes the leak. Happy to go another way if you prefer.", "kind": "yes_no", "options": []}]}

A list under a lead-in that announces decisions, open questions or things that wait for the user ("Open questions:", "Decisions:", "Still open:", "Waiting on you:", "Two things for you:") has one question for each item that waits for the user, also without a question mark. The question is the whole item, from its marker to its last character, with its nested items. An item that states one recommendation, plan or go-ahead is "yes_no". An item that names alternatives is "choice", and the alternatives are its options. An item that asks for a fact or a name is "open". An item that does not wait for the user is no question: a status (running, in flight, done), a wait for someone else, a finding, a decision already made, a step the agent takes next. When the items are the alternatives of one decision ("Two options:"), or the parts of one batch that one go-ahead covers, the list is one question from the lead-in to the last item. A list that reports finished work, with a lead-in that names decisions, has no question.
Text: Open questions, with my recommendations: 1. The empty reply: count it as the report, with a notice. I agree with the plan. 2. Which text: the last assistant message only, as the other tool does. 3. The demo: send it to the team that owns the site after this lands.
Answer: {"questions": [{"question": "1. The empty reply: count it as the report, with a notice. I agree with the plan.", "kind": "yes_no", "options": []}, {"question": "2. Which text: the last assistant message only, as the other tool does.", "kind": "yes_no", "options": []}, {"question": "3. The demo: send it to the team that owns the site after this lands.", "kind": "yes_no", "options": []}]}
Text: Still open: - the release: wait for phase 1, or ship now; I recommend waiting. - the name of the new flag. - the flake in the teardown, which I am still measuring.
Answer: {"questions": [{"question": "- the release: wait for phase 1, or ship now; I recommend waiting.", "kind": "choice", "options": ["wait for phase 1", "ship now"]}, {"question": "- the name of the new flag.", "kind": "open", "options": []}]}
Text: Two options: 1. A new window, this session untouched (my recommendation). 2. Relaunch in place, detached.
Answer: {"questions": [{"question": "Two options: 1. A new window, this session untouched (my recommendation). 2. Relaunch in place, detached.", "kind": "choice", "options": ["A new window, this session untouched (my recommendation)", "Relaunch in place, detached"]}]}
Text: Still waiting on your go-ahead for the batch: - the URL scheme; - the build folders; - the teardown fix.
Answer: {"questions": [{"question": "Still waiting on your go-ahead for the batch: - the URL scheme; - the build folders; - the teardown fix.", "kind": "yes_no", "options": []}]}
Text: Still running: - the parser review, reporting here when done. - the e2e suite on the second runner.
Answer: {"questions": []}
Text: Decisions recorded in the design: - one normalisation function for both texts; - the three-forms test.
Answer: {"questions": []}
Text: Next: 1. Build the pack. 2. Run parity. 3. Regenerate the golden files.
Answer: {"questions": []}
Text: I chose the new cache over the old one because it is faster. I recommend this pattern for the other services too.
Answer: {"questions": []}
Text: I'll push the fix now and then watch CI.
Answer: {"questions": []}

Examples that follow a question, or are inside it, are its options, and the kind is "choice". The question then runs to the last example.
Text: What do you want to do in app/ml? For example, a new feature, a bug fix, a review or a refactor.
Answer: {"questions": [{"question": "What do you want to do in app/ml? For example, a new feature, a bug fix, a review or a refactor.", "kind": "choice", "options": ["a new feature", "a bug fix", "a review", "a refactor"]}]}
Text: When I finish, what checks do you want? For example, `make verify` in full, or only `dune test`.
Answer: {"questions": [{"question": "When I finish, what checks do you want? For example, `make verify` in full, or only `dune test`.", "kind": "choice", "options": ["`make verify` in full", "only `dune test`"]}]}

Alternatives joined by "or" are options also when they are verbs or clauses. Each option is the words that differ, without the words that the alternatives share.
Text: Is it about how the app formats, splits or shows text?
Answer: {"questions": [{"question": "Is it about how the app formats, splits or shows text?", "kind": "choice", "options": ["formats", "splits", "shows text"]}]}
Text: Should I keep the old cache, or do you want me to delete it?
Answer: {"questions": [{"question": "Should I keep the old cache, or do you want me to delete it?", "kind": "choice", "options": ["keep the old cache", "delete it"]}]}
Text: I fixed several things, for example the cache and the parser.
Answer: {"questions": []}
