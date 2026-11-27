# berlin-lst-downscaling

Cloud-native LST downscaling for Berlin (public thesis code). Landsat and Sentinel-2 come from Microsoft Planetary Computer STAC; ECOSTRESS comes from NASA CMR via earthaccess. Products live in GCS. `docs/` holds only technical docs; thesis notes, results summary and text live in the private workspace `~/code/masterarbeit-text`, result files in `~/vertex-runs`.

This file does not limit what an owner-started task may do.

## Compute and storage

- Canonical bucket: `gs://berlin-lst-training-data/` (GCP project `berlin-lst-training`, `europe-west3`).
- Local mount: rclone at `~/.mnt/berlin-lst/` (on the owner's Mac; gcsfuse is unavailable on this arm64 Mac).
- Heavy pipeline runs use the On-Demand VM `berlin-lst-vm`. Start, stop, SSH, and run launchers live only in `scripts/runners/vm/`. Application code consumes GCS and Hydra and does not know the VM.
- GCP and secret operations run from the owner's Mac or a Linux operator machine as the operator SA through `scripts/operators/with-gcp-operator.sh` (its key lives only in Infisical `/operator`). VM SSH uses OS Login; use `scripts/runners/vm/ssh-vm.sh`, not an unrelated local VM alias.
- Cloud Build uses the builder SA with the committed configs in `cloudbuild/`.
- Managed GPU training runs on Vertex AI (`docs/vertex-gpu-training.md`).
- Keep the VM stopped when it is not running a job. Deletion protection stays on; the boot disk is not auto-delete.
- The bucket holds immutable canonical products, retained QA evidence, and ephemeral run outputs. Do not delete or rewrite canonical products outside a planned task. Do not edit retained evidence. Remove ephemeral outputs in the task that created them.
- Logs live only at `<output_root>/logs/<pipeline>/`.
- VM launcher logs go to `gs://berlin-lst-training-data/logs/vm-runs/<YYYY-MM>/` through `scripts/runners/vm/_logging.sh` (see `docs/vm-operations.md`); never redirect them into `$HOME`.

## Runtime

- Pipeline telemetry goes through `log_event` in `data/io/run_logging.py`. `print()` is for validators, spikes, and human-oriented CLI summaries.
- W&B tracks experiments and records the Git commit per run.

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
- **Task first.** A change to method, a data product, or a training run needs an explicit task with acceptance criteria from the project owner (GitHub issue or agent task) stating current state, desired state, and what is out of scope. Hygiene (lint, harness, typos) needs no task.
- **Stop and report.** If the task leaves a methodology question open (a threshold, an exclusion, a new
  channel, a split, a loss term, which data version to use), do not decide it. Stop, write the question and
  the options into the remaining-findings file or the final message, and leave the code unchanged there.
- **No leakage.** The 2025 test split never feeds checkpoint, model, hyperparameter, or threshold selection.
  Scalers fit on train only. Spatial holdouts keep a buffer.
- **Reproducible and fail-closed.** Seeds live in configs. Published data is never overwritten; a data
  change is a new version/prefix plus a new config hash. Missing inputs, hash mismatches, and contract
  violations raise; they are never skipped, clamped, or defaulted silently. Resampling is always explicit.
- **Evidence for every claim.** A task with acceptance criteria: copy them verbatim into the PR body under `## Acceptance criteria` as a checklist. Tick an item only when it is implemented and tested, and name the test, validator, or command output that proves it (paste the relevant lines); list anything not done, and why. "Should work" is not evidence.
- **No secrets.** Never read `.env*` files, print tokens, or write credentials to code, configs, or logs. Secrets live only in Infisical (thesis project, env `dev`: `/operator`, `/data`, `/vertex`) and are reached only through `with-gcp-operator.sh`, `sync-vm-env.sh`, or the Vertex entrypoint.
  Use the environment variables the code already expects.

## Validation

- Full CI equivalent: `uv run --locked nox` (sessions `lint`, `typecheck`, `tests`, `validate-ablation-configs`,
  `smoke-training-contract-cpu`, `smoke-modeling`, `smoke-modeling-contract`). CI on `main` and pull requests runs it as job `validate`; job `secret-scan` runs gitleaks over the full history (locally: `uvx pre-commit install`, or `uv run nox -s secret-scan` with gitleaks on PATH).
- Fast loop: `uv run nox -s lint typecheck` and `uv run pytest tests/<file>.py -q`. Autoformat: `uv run nox -s fix`.
- Sessions prefixed `cloud-` touch GCS or the VM. Run one only when the task asks for it.
- New or changed computational logic (math, masks, splits, aggregation, hashing) needs unit tests. Tests are offline and small.
- Squash-merge into `main` after `validate` is green.

## Custom commands

- `/plan-task <task>`: write the plan in the main session without editing files.
- `/validate [session or path]`: run the CI-equivalent checks through `test-runner`.
