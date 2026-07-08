# Berlin LST Downscaling

Downscale Land Surface Temperature (LST) from Landsat / ECOSTRESS to
10 m Sentinel-2 resolution — Berlin case study.

## Status

**Phase A complete** — Landsat C2-L2 + Sentinel-2 L2A ARD pipeline:

- Cloud masking (Landsat QA_PIXEL, S2 SCL + directional shadow projection)
- COG writer (deflate compression, 512×512 tiles, overviews [2,4,8,16])
- STAC item emission per scene
- PyArrow Parquet ledger with idempotency and resume support
- Hydra-driven config (smoke / full mode)
- YAML + JSONL logging

**Phase B complete** — ECOSTRESS L2T ARD pipeline:

- NASA Earthdata S3 → ephemeral stage → COG processing → GCS output
- Per-layer COGs (LST, cloud, water, QC) with QA flag band
- `aoi_overlap_px` metric: pixel count of COG∩AOI intersection (detects off-target swaths)
- Tenacity retry on download, GCS upload, and CMR metadata queries
- Self-contained smoke test: `nox -s smoke-ecostress` (local) or `nox -s smoke-ecostress-cloud` (GCS)

## Quick start

```bash
uv sync
uv run nox                           # lint + typecheck

# ── Manifest-driven ARD smoke test (Landsat + S2 + ECOSTRESS) ────────
uv run nox -s smoke-primary          # builds 3-row manifest, processes all sources

# ── Szenen-Selektion & Kopplung ──────────────────────────────────────
uv run nox -s smoke-selection-2024   # coupled manifest Mai–Sep 2024
uv run nox -s selection-scan         # metadata-only volume scan (Mai–Sep 2017–2025)

# ── Cloud pilot (requires ADC / Workload Identity) ───────────────────
uv run nox -s cloud-pilot            # smoke-primary targeting GCS
```

## Smoke tests

Each smoke test is self-contained: it stages raw inputs, runs the pipeline,
produces COG output and a visualisation, then cleans up the stage.  No manual steps.

| Session | What it validates |
|---------|-------------------|
| `smoke-primary` | Full ARD pipeline on all 3 sources (manifest-driven) |
| `smoke-selection-2024` | Scene coupling for Mai–Sep 2024 |
| `selection-scan` | Metadata-only volume scan (no pixel loads) |
| `cloud-pilot` | Same as smoke-primary but targeting GCS bucket |

## Full production run

```
Before triggering bulk mode:
  1. uv run nox -s lint typecheck        # code passes gates
  2. uv run nox -s smoke-primary         # local ARD smoke works for all 3 sources
  3. uv run nox -s smoke-selection-2024  # coupling produces reasonable results
  4. uv run nox -s cloud-pilot           # cloud ARD smoke works (requires ADC)
  5. THEN: build manifest → run bulk
```

### Build the manifest

```bash
# Full scan — metadata-only volume assessment (Mai–Sep 2017–2025)
uv run nox -s selection-scan

# Coupled export — write COGs for selected scenes
uv run python scripts/build_manifest.py \
    --config-dir configs/selection \
    --config-name full_2017_2025 \
    mode=couple
```

### Run bulk mode

```bash
uv run python scripts/run_ard.py --config-name full \
    manifest_uri=data/ard/manifest.parquet \
    output_root=gs://berlin-lst-data/ard/

# Single-source run:
uv run python scripts/run_ard.py --config-name full \
    sources=["ecostress"] \
    manifest_uri=data/ard/manifest.parquet \
    output_root=gs://berlin-lst-data/ard/
```

## Staging model

Raw L2T inputs are **ephemeral**: they exist only for the duration of a
processing run, then are deleted.  Final artefacts (COGs, STAC items) are
written to ``output_root`` and are never staged.

Supported URI schemes (``output_root``):

| Scheme | Mechanism |
|--------|-----------|
| ``local`` | ``pathlib.Path`` + ``shutil`` (POSIX) |
| ``gcs`` | ``google.cloud.storage`` (bucket → ``/vsigs/`` path for rasterio) |

ECOSTRESS staging path: NASA S3 → local tmp → GCS stage (``gs://bucket/_staging/``) → rasterio reads via ``/vsigs/`` → pipeline-internal cleanup.

## Storage layout

```
berlin-lst-data/                    # GCS bucket
├── _staging/                      # ephemeral raw inputs (auto-deleted)
│   └── ecostress/
│       └── {run_id}/
│           └── ECOv002_L2T_LSTE_…_33UUU_20180730T193555_0712_01/
│               ├── ECOv002_L2T_LSTE_…_LST.tif
│               ├── ECOv002_L2T_LSTE_…_cloud.tif
│               ├── ECOv002_L2T_LSTE_…_water.tif
│               └── ECOv002_L2T_LSTE_…_QC.tif
└── ard/                           # final COG output
    ├── ledger.parquet
    ├── landsat-c2-l2/
    ├── sentinel-2-l2a/
    └── ecostress/
        └── 2018/
            └── ECOv002_L2T_LSTE_…_33UUU_20180730T193555_0712_01/
                ├── lst.tif
                └── lst.stac.json

data/                              # local working directory
└── tmp/
    └── ecostress_stage/          # local smoke test stage (cleaned by nox)
```

## Stack

- **Python 3.12** — uv, ruff, pyright, nox
- **Data access** — pystac-client, odc-stac, Planetary Computer; earthaccess (ECOSTRESS)
- **Processing** — xarray, rioxarray, rasterio, numpy, scipy
- **Storage** — Cloud-Optimized GeoTIFF (COG), STAC Item JSON
- **State** — PyArrow Parquet ledger (BLAKE3 schema-hash for idempotency)
- **Config** — Hydra (OmegaConf)

## Validation

```bash
uv run nox                # lint + typecheck
uv run nox -s fix        # lint and auto-fix
```

## GCP access

```bash
mount-berlin    # rclone mount of the project GCS bucket
```

See `.opencode/skills/google-access/SKILL.md` for full reference.
