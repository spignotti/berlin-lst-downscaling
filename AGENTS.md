# berlin-lst-downscaling

Cloud-native LST downscaling for Berlin. Landsat and Sentinel-2 come from Microsoft Planetary Computer STAC; ECOSTRESS comes from NASA CMR via earthaccess. Products live in GCS.

## Compute and storage

- Canonical bucket: `gs://berlin-lst-training-data/` (GCP project `berlin-lst-training`, `europe-west3`).
- Local mount: rclone at `~/.mnt/berlin-lst/` (gcsfuse is unavailable on this arm64 Mac).
- Heavy pipeline runs use the On-Demand VM `berlin-lst-vm`. Start, stop, SSH, and run launchers live only in `.opencode/skills/google-access/scripts/`. Application code consumes GCS and Hydra and does not know the VM.
- Managed GPU training runs on Vertex AI (`docs/vertex-gpu-training.md`).
- Keep the VM stopped when it is not running a job. Deletion protection stays on; the boot disk is not auto-delete.
- The bucket holds immutable canonical products, retained QA evidence, and ephemeral run outputs. Do not delete or rewrite canonical products outside a planned task. Do not edit retained evidence. Remove ephemeral outputs in the task that created them.
- Logs live only at `<output_root>/logs/<pipeline>/`.

## Runtime

- Pipeline telemetry goes through `log_event` in `data/io/run_logging.py`. `print()` is for validators, spikes, and human-oriented CLI summaries.
- W&B tracks experiments and records the Git commit per run.

## Validation

- Default nox sessions are `lint`, `typecheck`, `tests`, `validate-ablation-configs`, `smoke-training-contract-cpu`, `smoke-modeling`, and `smoke-modeling-contract`. New or changed computational logic (math, masks, splits, aggregation, hashing) needs unit tests. Tests must be offline and small.
- CI on `main` and pull requests runs `uv run --locked nox` (job name `validate`).

## Git

- Work on a feature branch. Open a pull request into `main`. Squash-merge only after the `validate` check is green and the project owner (Silas) has approved the merge.
- A change to method, a data product, or a training run needs an explicit task with acceptance criteria from the project owner (a GitHub issue or an agent task). The task states the current state, the desired state, and what is out of scope. Agents must not decide new scientific choices (thresholds, exclusions, new channels, split changes) themselves; report them back instead. Hygiene (lint, harness, typos) needs no task. Notion holds the plan.
- PR titles and commits follow Conventional Commits (`feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `test: ...`, `chore: ...`).
- When a task comes with acceptance criteria, copy them verbatim into the PR description under `## Acceptance criteria` as a checklist. Tick an item only when it is implemented and tested, and give the file or test that covers it. List anything not done, and why.

## Coding agents

- Coding tasks run headless through the shared runner with Claude Code (`run-task.sh --tool claude`). Claude follows `CLAUDE.md` and the subagents in `.claude/agents/`.
- Review happens in the Claude review pipeline: the Opus main session checks every change, and an independent `scientific-reviewer` (Opus) verification is required for changes to the data pipeline, contracts, splits, the model, or the loss (see `CLAUDE.md`). There is no separate bot review on GitHub.
- Coding agents work on code only. They never publish data, start pipelines, republishes, or trainings, and they do not use the VM or Vertex launchers.
- Agents commit on their branch only. They do not push, open PRs, or use `gh`; the runner does that. Nothing is merged without the project owner's OK.
- Validation is `uv run --locked nox` (the default sessions).
- The bucket and evidence rules above still apply: do not delete or rewrite canonical products, do not edit retained evidence, and do not touch `docs/results/` or other frozen result write-ups.

## Planning

- Notion page: `28c35645-1f66-8057-b647-db5aebf191a5`
- GitHub: `spignotti/berlin-lst-downscaling`
