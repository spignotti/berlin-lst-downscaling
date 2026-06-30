# berlin-lst-downscaling

> Reproducible deep-learning pipeline that downscales Landsat land surface temperature from ~100 m to 10 m for Berlin via a fixed 2D U-Net and a 5-stage feature ablation.

100 m thermal data from Landsat TIRS is too coarse for block-level urban heat analysis. This project produces a 10 m LST time series plus a published model, using urban-context features (spectral indices, morphology, shadow/solar geometry, meteorology) evaluated through cumulative ablation.

## Setup

```bash
uv sync
```

## Pipeline

The ARD (Analysis-Ready Data) pipeline is run via a single orchestrator:

```bash
uv run python scripts/ard_run.py              # smoke: 1 source × year × 1 scene + visual QC
uv run python scripts/ard_run.py all          # full run: all sources × all years
uv run python scripts/ard_run.py plan         # show what would run, no execution
uv run python scripts/ard_run.py doctor       # GCP/GEE access check (5 checks, ~5s)
```

Sub-commands for individual stages: `export`, `process`, `validate`, `plan`, `doctor`. The orchestrator chains them automatically for the full pipeline. See `--help` for all options.

## Development

```bash
uv run nox -s fix        # lint and format
uv run nox -s test       # run tests
uv run nox               # full validation (lint + typecheck + test)
```

## GCP / GEE access

Mount the GCS bucket and verify access before any pipeline run:

```bash
mount-berlin                                        # rclone mount
uv run python scripts/ard_run.py doctor             # 5 access checks (mount, rclone, gcloud, Python GCS, Python GEE)
```

See `.opencode/skills/google-access/SKILL.md` for full reference (auth, troubleshooting, manual debugging).
