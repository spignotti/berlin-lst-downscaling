# Berlin LST Downscaling

Downscale Land Surface Temperature (LST) from Landsat / ECOSTRESS to
10 m Sentinel-2 resolution — Berlin case study. Built on **Microsoft
Planetary Computer** STAC for data access.

## Status

**Phase A complete** — Landsat C2-L2 + Sentinel-2 L2A ARD pipeline:

- Cloud masking (Landsat QA_PIXEL, S2 SCL + directional shadow projection)
- COG writer (deflate compression, 512×512 tiles, overviews [2,4,8,16])
- STAC item emission per scene
- PyArrow Parquet ledger with idempotency and resume support
- Hydra-driven config (smoke / full mode)
- YAML + JSONL logging

Phase B (ECOSTRESS) is planned.

## Quick start

```bash
uv sync
uv run nox                       # lint + typecheck

# Smoke test: one scene per source
uv run python scripts/run_ard.py --config-name smoke

# Full production run
uv run python scripts/run_ard.py --config-name full
```

## Pipeline modes

| Mode | What | When |
|------|------|------|
| `smoke` | Process 1 scene per source | Validation / CI |
| `full` | Process a manifest of scenes | Batch production |

Config lives in `configs/ard/`:
- `default.yaml` — shared settings (bbox, resolution, cloud parameters)
- `smoke.yaml` — one scene on 2024-06-29
- `full.yaml` — manifest-driven batch

## Stack

- **Python 3.12** — uv, ruff, pyright, nox
- **Data access** — pystac-client, odc-stac, Planetary Computer
- **Processing** — xarray, rioxarray, rasterio, numpy, scipy
- **Storage** — Cloud-Optimized GeoTIFF (COG), STAC Item JSON
- **State** — PyArrow Parquet ledger (BLAKE3 schema-hash for idempotency)
- **Config** — Hydra (OmegaConf)

## Output layout

```
data/ard/
├── ledger.parquet              # per-scene processing ledger
├── landsat-c2-l2/
│   └── 2024/
│       └── LC08_…_02_T1/
│           ├── LC08_…_02_T1.tif           # COG
│           └── LC08_…_02_T1.stac.json     # STAC item
├── sentinel-2-l2a/
│   └── 2024/
│       └── S2A_…/
│           ├── S2A_…tif
│           └── S2A_…stac.json
└── logs/
    └── {run_id}.jsonl
```

## GCP access

```bash
mount-berlin    # rclone mount of the project GCS bucket
```

See `.opencode/skills/google-access/SKILL.md` for full reference.

## Validation

```bash
uv run nox                # lint + typecheck
uv run nox -s fix         # lint and auto-fix
```
