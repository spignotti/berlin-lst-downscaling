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
- **No secrets.** Never read `.env*` files, print tokens, or write credentials to code, configs, or logs.
  Use the environment variables the code already expects.
- **Do not run pipelines.** No VM launchers, Vertex launches, GCS writes, or republishes from a coding run.
  Frozen docs (`docs/results/`, `docs/*-full-results.md`, `docs/stage1-*`, `docs/random-forest-*`) stay
  untouched.

## Efficiency rules

- The main session (Opus) orchestrates: it plans, decides the hard parts, and integrates. Delegate the rest
  to subagents in `.claude/agents/`:
  - `planner` (Opus): plan, methodology check, design decisions for non-trivial tasks.
  - `implementer` (Sonnet): code and tests from a concrete plan.
  - `test-runner` (Haiku): runs nox/pytest and validators, returns only a failure summary.
  - `code-reviewer` (Sonnet): every change.
  - `scientific-reviewer` (Opus): only when the diff touches the data pipeline, validators, contracts,
    configs, or `modeling/`. Skip it for docs, CI, and harness-only changes.
- Keep context lean. Use Grep/Glob to find the few files you need and read only the relevant ranges.
  `noxfile.py` (~1750 lines) and the large docs are read by section, not whole. Do not read generated data,
  lock files, or logs unless the task is about them.
- Never paste full test logs into the main session; the test-runner summarizes.
- Small, obvious changes (a typo, a one-line fix with a clear test) do not need the planner.
- At most 2 review/fix rounds. Nits never block.

## Validation

- Full CI equivalent: `uv run --locked nox` (sessions `lint`, `typecheck`, `tests`, `validate-ablation-configs`,
  `smoke-training-contract-cpu`, `smoke-modeling`, `smoke-modeling-contract`).
- Fast loop: `uv run nox -s lint typecheck` and `uv run pytest tests/<file>.py -q`.
- Autoformat: `uv run nox -s fix`.
- Sessions prefixed `cloud-` touch GCS or the VM. Never run them.

## Workflow

1. Read the task and its acceptance criteria. For non-trivial work, ask `planner` for a plan.
2. `implementer` writes code and tests; new computational logic (math, masks, splits, aggregation, hashing)
   gets offline unit tests.
3. `test-runner` runs the checks until green.
4. `code-reviewer` (and `scientific-reviewer` when it applies) review the diff; fix blocking findings.
5. Commit with a Conventional Commit message on the current branch. Do not push, do not use `gh`, do not
   create or switch branches. Merges happen only after Silas' OK.

Custom commands: `/plan-task` (planner on a task) and `/validate` (test-runner on the full suite).
