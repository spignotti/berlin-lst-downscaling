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

# ── Phase A (Landsat + Sentinel-2) ────────────────────────────────────
uv run nox -s smoke                  # smoke test (landsat + sentinel2, local disk)
uv run nox -s smoke-landsat         # landsat only
uv run nox -s smoke-sentinel2        # sentinel-2 only

# ── Phase B (ECOSTRESS) ───────────────────────────────────────────────
uv run nox -s smoke-ecostress       # local smoke test (stages → processes → cleans up)
uv run nox -s smoke-ecostress-cloud # GCS smoke test (requires ADC; opt-in)

# ── Phase C (Szenen-Selektion & Kopplung) ────────────────────────────
uv run nox -s smoke-selection        # coupled manifest smoke test (Juli 2024)
uv run nox -s selection-scan         # metadata-only volume scan (Mai–Sep 2017–2025)
```

## Smoke tests

Each smoke test is self-contained: it stages raw inputs, runs the pipeline,
produces COG output and a visualisation, then cleans up the stage.  No manual steps.

| Session | Source | Target | Notes |
|---------|--------|--------|-------|
| `smoke-landsat` | Landsat C2-L2 | Local disk | One scene on 2024-06-29 |
| `smoke-sentinel2` | Sentinel-2 L2A | Local disk | One scene on 2024-06-29 |
| `smoke-ecostress` | ECOSTRESS L2T | Local disk | Tile 33UUU, 2018-07-30 |
| `smoke-cloud` | Landsat + S2 | GCS | Requires rclone mount or ADC |
| `smoke-ecostress-cloud` | ECOSTRESS L2T | GCS | Requires ADC; **opt-in** (API costs) |
| `smoke-selection` | Landsat + S2 + ECOSTRESS | Manifest | Coupled scene selection, Juli 2024 |
| `selection-scan` | Landsat + S2 + ECOSTRESS | Scan report | Metadata-only volume scan |

## Full production run

```
Before triggering full-mode:
  1. uv run nox -s lint typecheck        # code passes gates
  2. uv run nox -s smoke-ecostress      # local smoke works
  3. uv run nox -s smoke-ecostress-cloud # cloud smoke works (requires ADC)
  4. Run a 10-granule pilot, spot-check one COG in QGIS
  5. THEN: build manifest → run full
```

### Build the manifest

```bash
# Smoke test — coupled manifest for Juli 2024
uv run nox -s smoke-selection

# Full scan — metadata-only volume assessment (Mai–Sep 2017–2025)
uv run nox -s selection-scan

# Coupled export — write COGs for selected scenes
uv run python scripts/build_manifest.py \
    --config-dir configs/selection \
    --config-name full_2017_2025 \
    mode=couple
```

### Run full mode

```bash
uv run python scripts/run_ard_ecostress.py \
    --config-name full \
    output_root=gs://berlin-lst-data/ard/ecostress
```

## Staging model

Raw L2T inputs are **ephemeral**: they exist only for the duration of a
processing run, then are deleted.  Final artefacts (COGs, STAC items) are
written to ``output_root`` and are never staged.

Supported URI schemes:

| Scheme | Mechanism |
|--------|-----------|
| ``local`` | ``pathlib.Path`` + ``shutil`` (POSIX) |
| ``gcs`` | ``google.cloud.storage`` (bucket → ``/vsigs/`` path for rasterio) |
| ``mounted`` | ``pathlib.Path`` + ``shutil`` via rclone FUSE mount |

Cloud staging path: NASA S3 → local tmp → GCS stage (``gs://bucket/_staging/``) → rasterio reads via ``/vsigs/`` → nox cleans GCS stage.

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
