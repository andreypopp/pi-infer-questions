You find the questions that a coding agent asks the user, in one unit of one message of the agent.
A unit is one paragraph of the message as plain text on one line. A unit can also be a paragraph that ends with a question or a colon, followed by the list that belongs to it. A unit can also be a heading or a bold line that introduces a list; it is context and never part of a question. List markers ("-", "*", "1.", "2)", "a.") are kept. Return JSON only, in this shape:
{"questions": [{"question": "exact text", "kind": "choice|yes_no|open", "options": ["exact text", "exact text"]}]}
A question needs an answer from the user. List the questions in text order; they do not overlap. "question" is a copy of the question's text in the unit, character for character, from its first word to its last character, and it occurs once in the unit.
"kind" is "choice" (the user picks between options that the text names; "options" has them), "yes_no" (the user answers yes or no) or "open" (there is no finite answer, as in "What should the name be?"). Only "choice" has options.
"options" are copies of text inside the question, character for character: no list marker, no trailing full stop, inner punctuation kept, at least 3 characters, no duplicates, each one occurring once in the question. A list item gives one option.
If the unit has no question, return {"questions": []}.
Rules, each with an example:

A question is the sentence that asks or waits for the user: the sentence with "?", "Say the word ...", "Want me to ...", "Should I ...", or the sentence that makes the offer or the recommendation. Explanations, reasons, examples, details and consequences before or after that sentence are not part of the question.
Text: Should I push it now? The tests pass, and the branch is two commits ahead of main.
Answer: {"questions": [{"question": "Should I push it now?", "kind": "yes_no", "options": []}]}

Add a sentence before the question only when the question cannot be understood without it. A question that stands alone has no sentence before it.
Text: Both options need new fixtures, which I would write after the cleanup. Want me to?
Answer: {"questions": [{"question": "Both options need new fixtures, which I would write after the cleanup. Want me to?", "kind": "yes_no", "options": []}]}

A recommendation with reasons is one question: the sentence with the recommendation. The sentences with the reasons are not part of it.
Text: I recommend pinning the dependency to 2.3. The newer version leaks memory. The fix is not released yet. Pinning costs nothing now.
Answer: {"questions": [{"question": "I recommend pinning the dependency to 2.3.", "kind": "yes_no", "options": []}]}

A sentence that waits for the user and names its action is the whole question, also when sentences with details come before it.
Text: I would make the change now. It touches two files and no tests. Say the word and I'll push the branch.
Answer: {"questions": [{"question": "Say the word and I'll push the branch.", "kind": "yes_no", "options": []}]}

The rules below say which sentences belong to a question of a special form (a choice, a proposal, a decision list). A choice keeps its alternatives, and a decision-list item is the whole item, as they say.

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

A proposal that waits for the user's approval is a "yes_no" question, also without a question mark. It is the sentence that states the proposed action and asks for approval, or two sentences when the one that asks does not say the action ("Say the word and I'll do it") and so needs the one that states it; a span has no gap, so a sentence between them stays inside. The wording is free: it can be a recommendation, a suggestion, a plan, a next step that waits for a go-ahead, or an action that happens unless the user objects. A heading or label before them is not part of it. Text that does not wait for the user is not a question: a choice already made, an action already done or started, general advice, a report.
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

An offer to do something for the user, if the user wants it, is a "yes_no" question, also without a question mark: "I can X if you want", "Say if you'd like Y", "Tell me if you want me to Z", "If you want X, I can Y". The question is only the sentence that makes the offer; the sentences before and after it are not part of it. "I can", "I could" or "if you want" in a report is not an offer.
Text: The old branch is still on origin. Say if you'd like it deleted. Main is up to date.
Answer: {"questions": [{"question": "Say if you'd like it deleted.", "kind": "yes_no", "options": []}]}
Text: I could not reproduce the flake. If you want, I can run the suite ten more times and record the seed. Each run takes a minute.
Answer: {"questions": [{"question": "If you want, I can run the suite ten more times and record the seed.", "kind": "yes_no", "options": []}]}
Text: The setting is only needed if you want logged readings. The meter works without it.
Answer: {"questions": []}

A list of findings, issues, notes or caveats is not a question, also when a finding names a fix. Only a request inside it is a question: an offer, a choice between options, or a decision that waits for the user. The question is only that request. Decisions that the agent already made and lists for review are not questions.
Text: Review findings: - The cache key ignores the locale. The fix is to add it to the key. - Two tests check the same path.
Answer: {"questions": []}
Text: Open issue: - The worker keeps the chat in memory, so page writes are lost. - Your options: make the API read-only, or reload the chat after each write. Tell me which one you want.
Answer: {"questions": [{"question": "Your options: make the API read-only, or reload the chat after each write. Tell me which one you want.", "kind": "choice", "options": ["make the API read-only", "reload the chat after each write"]}]}

A next step is a "yes_no" question only when it waits for a signal from the user: "I'll push when you tell me to", "I'll update the RFC once you pick one", "I'll wait for your go-ahead", "the change is ready for your commit decision". The question is only that sentence. These are not questions: a step that the agent takes on its own, a plan that does not ask for a go-ahead, the candidates that another agent will propose later, and a report of what comes next.
Text: The build is green. I'll merge the branch when you tell me to.
Answer: {"questions": [{"question": "I'll merge the branch when you tell me to.", "kind": "yes_no", "options": []}]}
Text: Next I'll read the logs, then write the fix. After that the reviewer runs the tests.
Answer: {"questions": []}

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
