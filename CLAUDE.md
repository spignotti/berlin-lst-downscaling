@AGENTS.md

# Claude Code notes

This file adds Claude-specific rules on top of `AGENTS.md`. Where both apply, the stricter rule wins.
Coding runs come from the project bot through the shared runner (`run-task.sh --tool claude`). The runner
pushes and opens the PR; you commit on the current branch only.

## Scientific rules (non-negotiable)

- **Contracts first.** Every data change is checked against its contract and validator:
  `src/berlin_lst_downscaling/data/{ard/contract.py,qa/contracts.py,features/contracts.py,training/contracts.py}`,
  `src/berlin_lst_downscaling/modeling/contracts.py`, `docs/data-sources-and-contracts.md`,
  `docs/pseudo-pair-tensor-contract.md`, `docs/features-v6.md`, and `scripts/validators/validate_*.py`.
  Never relax a contract, validator bound, or schema without writing down why (in the doc next to the
  contract and in the PR body), and only when the task asks for it.
- **No silent methodology changes.** Splits are fixed: 2017–2023 train, 2024 val, 2025 test, 2026 inference
  only. Do not change splits, thresholds, exclusions, channels, locked profiles (`stage*_locked`,
  `isolate_*`), Hydra configs, config hashes, or fingerprints unless the task says so explicitly. A
  methodological change is a new option; the previous behavior stays the default.
- **Stop and report.** If the task leaves a methodology question open (a threshold, an exclusion, a new
  channel, a split, a loss term, which data version to use), do not decide it. Stop, write the question and
  the options into the remaining-findings file or the final message, and leave the code unchanged there.
- **No leakage.** The 2025 test split never feeds checkpoint, model, hyperparameter, or threshold selection.
  Scalers fit on train only. Spatial holdouts keep a buffer.
- **Reproducible and fail-closed.** Seeds live in configs. Published data is never overwritten; a data
  change is a new version/prefix plus a new config hash. Missing inputs, hash mismatches, and contract
  violations raise; they are never skipped, clamped, or defaulted silently. Resampling is always explicit.
- **Evidence for every claim.** Each ticked acceptance criterion in a PR body names the test, validator, or
  command output that proves it (paste the relevant lines). "Should work" is not evidence.
- **No secrets.** Never read `.env*` files, print tokens, or write credentials to code, configs, or logs. Secrets live only in Infisical (thesis project, env `dev`: `/operator`, `/data`, `/vertex`) and are reached only through `with-gcp-operator.sh`, `sync-vm-env.sh`, or the Vertex entrypoint.
  Use the environment variables the code already expects.
- **Do not run pipelines.** No VM launchers, Vertex launches, GCS writes, or republishes from a coding run.
  Frozen docs (`docs/results/`, `docs/*-full-results.md`, `docs/stage1-*`, `docs/random-forest-*`) stay
  untouched.

## Standard workflow

The main session runs on Opus and is the orchestrator. It plans, delegates execution to cheap subagents,
checks their output itself, and calls an independent verifier only when the criteria below require it.

1. **Plan (Opus, main session).** Read the task and acceptance criteria, then write the plan yourself: methodology
   check, files and exact changes, tests and validators per criterion. There is no planner subagent. If the
   plan hits an open methodology question, stop and report it (see Scientific rules).
2. **Execute (cheap subagents).** Hand self-contained chunks of the plan to:
   - `implementer` (Sonnet): code and tests for a concrete plan chunk.
   - `mechanic` (Haiku): searches, renames, path or string replacements, other edits without design judgment.
   - `test-runner` (Haiku): nox/pytest and offline validators; returns only a PASS/FAIL summary.
   Do trivial one-line edits yourself when delegating would cost more than doing it.
3. **Check (Opus, main session).** Read the diff and the test-runner summary yourself. Compare against the plan and
   every acceptance criterion. Send fixes back to `implementer` or `mechanic`. At most 2 fix rounds; nits never
   block.
4. **Independent verification (only when required).** A verifier subagent runs in a fresh context, without your
   reasoning, and re-checks the diff, the tests, and the data/validator outputs itself.
   - **Required: `scientific-reviewer` (Opus)** whenever the change touches
     - the data pipeline (`src/berlin_lst_downscaling/data/`, `scripts/operators/`, `scripts/validators/`,
       `configs/` for selection, ARD, static, dynamic, features, training, QA, EDA),
     - a contract or schema (`*contract*.py`, `docs/data-sources-and-contracts.md`,
       `docs/pseudo-pair-tensor-contract.md`, `docs/features-v6.md`),
     - splits, holdouts, scalers, or anything that could leak 2025 test data,
     - the model, the loss, or training/ablation configs (`src/berlin_lst_downscaling/modeling/`,
       `configs/modeling/`, `configs/training/`),
     - thresholds, exclusions, config hashes, or fingerprints.
   - **Optional: `code-reviewer` (Sonnet)** for larger code-only changes (refactors, harness, CLI, IO) or when you
     are unsure. Not needed for docs, typos, CI config, or small fixes with a clear test.
   - Fix blocking findings from the verifier (at most 2 rounds), then re-run `test-runner`.
5. **Commit.** Conventional Commit on the current branch. Do not push, do not use `gh`, do not create or switch
   branches. The runner pushes and opens the PR; merges happen only after Silas' OK. The PR body lists each
   acceptance criterion with its evidence and states whether independent verification ran and its verdict.

## Efficiency rules

- Opus does the plan, the hard decisions, and the checks. Sonnet and Haiku do the volume work. Do not run
  long implementation or test loops in the main session.
- Keep context lean. Use Grep/Glob (or `mechanic`) to find the few files you need and read only the relevant
  ranges. `noxfile.py` (~1750 lines) and the large docs are read by section, not whole. Do not read generated
  data, lock files, or logs unless the task is about them.
- Never paste full test logs into the main session; `test-runner` summarizes.
- Give each subagent a self-contained brief (files, exact change, how to test); it does not see this session.

## Validation

- Full CI equivalent: `uv run --locked nox` (sessions `lint`, `typecheck`, `tests`, `validate-ablation-configs`,
  `smoke-training-contract-cpu`, `smoke-modeling`, `smoke-modeling-contract`).
- Fast loop: `uv run nox -s lint typecheck` and `uv run pytest tests/<file>.py -q`.
- Autoformat: `uv run nox -s fix`.
- Sessions prefixed `cloud-` touch GCS or the VM. Never run them.

## Custom commands

- `/plan-task <task>`: write the plan in the main session without editing files.
- `/validate [session or path]`: run the CI-equivalent checks through `test-runner`.
