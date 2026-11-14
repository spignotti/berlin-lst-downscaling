@AGENTS.md

# Claude Code notes

This file adds Claude-specific scientific notes on top of `AGENTS.md`. It does not limit what an owner-started task may do.

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

## Validation

- Full CI equivalent: `uv run --locked nox` (sessions `lint`, `typecheck`, `tests`, `validate-ablation-configs`,
  `smoke-training-contract-cpu`, `smoke-modeling`, `smoke-modeling-contract`).
- Fast loop: `uv run nox -s lint typecheck` and `uv run pytest tests/<file>.py -q`.
- Autoformat: `uv run nox -s fix`.
- Sessions prefixed `cloud-` touch GCS or the VM. Run one only when the task asks for it.

## Custom commands

- `/plan-task <task>`: write the plan in the main session without editing files.
- `/validate [session or path]`: run the CI-equivalent checks through `test-runner`.
