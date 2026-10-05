# berlin-lst-downscaling

Cloud-native LST downscaling for Berlin. Landsat and Sentinel-2 come from Microsoft Planetary Computer STAC; ECOSTRESS comes from NASA CMR via earthaccess. Products live in GCS.

## Compute and storage

- Canonical bucket: `gs://berlin-lst-training-data/` (GCP project `berlin-lst-training`, `europe-west3`).
- Local mount: rclone at `~/.mnt/berlin-lst/` (gcsfuse is unavailable on this x86_64 Mac).
- Heavy pipeline runs use the On-Demand VM `berlin-lst-vm`. Start, stop, SSH, and run launchers live only in `.opencode/skills/google-access/scripts/`. Application code consumes GCS and Hydra and does not know the VM.
- Managed GPU training runs on Vertex AI (`docs/vertex-gpu-training.md`).
- Keep the VM stopped when it is not running a job. Deletion protection stays on; the boot disk is not auto-delete.
- The bucket holds immutable canonical products, retained QA evidence, and ephemeral run outputs. Do not delete or rewrite canonical products outside a planned task. Do not edit retained evidence. Remove ephemeral outputs in the task that created them.
- Logs live only at `<output_root>/logs/<pipeline>/`.

## Runtime

- Pipeline telemetry goes through `log_event` in `data/io/run_logging.py`. `print()` is for validators, spikes, and human-oriented CLI summaries.
- W&B tracks experiments and records the Git commit per run.

## Validation

- Default nox sessions are `lint` and `typecheck`. There is no pytest session. Quality is real-data smoke and QA gates (`uv run nox -s smoke-*` and the stage validators). Do not add tests unless asked.
- CI on `main` and pull requests runs `uv run --locked nox`.

## Git

- Work on a feature branch. Open a pull request into `main`. Squash-merge after the `validate` check is green.
- Open a GitHub issue for a change to method, a data product, or a training run. The title says what changes. The body states the current state, the desired state, and what is out of scope. Hygiene (lint, harness, typos) needs no issue. Notion holds the plan.

## Planning

- Notion page: `28c35645-1f66-8057-b647-db5aebf191a5`
- GitHub: `spignotti/berlin-lst-downscaling`
