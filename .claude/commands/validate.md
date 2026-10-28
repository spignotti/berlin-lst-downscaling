---
description: Run the full CI-equivalent checks via the test-runner subagent and summarize failures
argument-hint: "[optional nox session or pytest path]"
---

Use the `test-runner` subagent to run `uv run --locked nox` (or only `$ARGUMENTS` if given) and return its
PASS/FAIL summary. Do not fix anything in this step.
