---
description: Write the plan for a task in the main session (methodology check, file-level plan, open questions), without editing files
argument-hint: <task file or description>
---

Plan this task yourself in the main session: $ARGUMENTS

Do not edit files. Use the `mechanic` subagent for broad searches instead of reading many files. Return:
1. Methodology check: does the task touch splits, thresholds, exclusions, channels, locked profiles, config
   hashes, fingerprints, contracts, the model, or the loss? Name any leakage risk.
2. Plan: files and exact changes, split into chunks for `implementer` (sonnet) and `mechanic` (haiku).
3. Tests and validators that prove each acceptance criterion.
4. Whether independent verification is required (see CLAUDE.md) and by which verifier.
5. Open methodology questions with options. If there are any, stop here and report them.
