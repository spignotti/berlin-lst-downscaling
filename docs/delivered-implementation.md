# Delivered implementation

Records the production pipelines, published roots, commands, and
observability for the preprocessing phase. Production paths live in
`gs://berlin-lst-data/`; local paths are scratch only.

## Pipeline graph

```text
Selection (build_manifest)
  └─ assets/manifests/v3/<bundle>/-r2/
       ├── ARD pipeline (run_ard)
       └── Dynamic pipeline (run_dynamic)
              └── DWD validation (run_dwd_validation)

Static sources (run_static_sources)
  └─ Static derived (run_static_derived)
       └── consumed by Dynamic (geometry)
```

Each output is anchored to a single immutable bundle prefix; nothing
reproduces or mutates historical artifacts.

## Published artefacts

| Pipeline | GCS root | Counts |
|----------|---------|--------:|
| Manifest bundle (canonical) | `gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/` | 509 manifest rows, 345 pairings |
| ARD ledger | `gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet` | 509 rows |
| Static sources | `gs://berlin-lst-data/static/sources/full/` | 8 ledger rows (4 original + 3 historical LoD + 1 existing 2024) |
| Static derived | `gs://berlin-lst-data/static/derived/full/_state/static/derived/ledger.parquet` | 18 ledger rows (6 original + 12 historical) |
| Dynamic full | `gs://berlin-lst-data/dynamic/full/` | 972 rows (`role=anchor`), 8-band ERA5 + carry-forward shadows |
| Dynamic inference | `gs://berlin-lst-data/dynamic/inference/2026/` | 63 rows (`role=inference`) |
| DWD validation r3 | `gs://berlin-lst-data/dwd_validation/r3/runs/dwd/9d5269f5/` | 345 anchors (historical, based on pre-rebuild ERA5) |

DWD r3 is historical — it validates the pre-rebuild scalar ERA5 products.
After the dynamic rebuild, a separate DWD rerun would be needed to
validate the new 8-band spatial ERA5 fields. DWD never feeds model training.

## Commands

```bash
# ARD (produces COG + flag + STAC + provenance + complete.json per scene)
uv run python scripts/run_ard.py --config-name full_all \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# ARD metadata repair (dry-run default, --apply to write sidecars)
uv run python scripts/finalize_ard.py \
    --ledger gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet

# ARD strict validation (exact manifest/ledger key-set, all four artifacts, STAC schemas)
uv run python scripts/validate_ard.py \
    --ledger gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet \
    --manifest gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# Dynamic full (manifest_uri required)
# Preferred: isolated runner (subprocess-per-scene, bounded memory)
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    --output-root gs://berlin-lst-data/dynamic/full \
    --config-name full --years 2017 2025 --dataset-role anchor --resume

# Dynamic inference
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    --output-root gs://berlin-lst-data/dynamic/inference/2026 \
    --config-name inference_2026 --years 2026 --dataset-role inference --resume

# Static sources
uv run python scripts/run_static_sources.py --config-name full

# Static derived
uv run python scripts/run_static_derived.py --config-name full

# Historical LoD morphometry vintages (2017, 2021, 2022)
# Additional Static A/B vintages — archive-first runner.
# (Requires terrain_height/2021, vegetation_height/2020, vegetation_dsm/2024
# already published in --upstream-source-root / --upstream-derived-root.)
uv run python scripts/run_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022

# Reconcile finalized artifacts into both Static ledgers
# (no archive download, no raster computation)
uv run python scripts/run_lod_vintages.py --reconcile-only \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --vintages 2017,2021,2022

# Build + upload a single archive from the local source dir, then free local copy
uv run python scripts/run_lod_vintages.py \
    --stage-local 2021 \
    --local-source-dir data/LoD2/LoD2_BE_1_33_2021 \
    --raw-root gs://berlin-lst-data

# Smoke against a small archive segment with separate smoke roots; read
# upstream terrain/vegetation products from the production roots.
uv run python scripts/run_lod_vintages.py \
    --smoke-archive data/LoD2/LoD2_2022.zip \
    --smoke-tile-count 100 \
    --smoke-bbox 13.586225,52.467717,13.616234,52.486040 \
    --source-root data/static/sources/lod_smoke \
    --derived-root data/static/derived/lod_smoke \
    --metadata-root data/static/geometry_vintages/smoke \
    --raw-root gs://berlin-lst-data \
    --upstream-source-root gs://berlin-lst-data/static/sources/full \
    --upstream-derived-root gs://berlin-lst-data/static/derived/full \
    --vintages 2022

# Validate published artifacts against production GCS roots (structural)
uv run python scripts/validate_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022

# Strict archive integrity verification (downloads ~4 GB sequentially)
uv run python scripts/validate_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022 \
    --verify-archives

# DWD validation
uv run python scripts/run_dwd_validation.py --config-name default \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    dynamic_full_root=gs://berlin-lst-data/dynamic/full \
    dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026 \
    output_root=gs://berlin-lst-data/dwd_validation/<run-id>
```

Run validation through the dedicated scripts — they reject malformed
input before opening ledgers:

```bash
uv run python scripts/validate_manifest.py \
    --manifest gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/full \
    --expected-role anchor --expected-scenes 324

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/inference/2026 \
    --expected-role inference --expected-scenes 21
```

## Memory-bounded Dynamic runner

`scripts/run_dynamic_isolated.py` runs the Dynamic pipeline with one
child process per scene, capping memory at roughly 1.2 GB per scene.
Use it on machines that cannot load the full pipeline worker; the
shared output root and ledger keep it idempotent.

```bash
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet \
    --output-root gs://berlin-lst-data/dynamic/full \
    --config-name full --years 2017 2025 --resume
```

## Smoke gates

The `noxfile.py` sessions are the documented smoke matrix:

- `smoke-primary` — bounded ARD with manifest-driven fixtures.
- `smoke-static-sources` / `smoke-static-derived` — bounded static runs.
- `smoke-dynamic` — single deterministic Landsat anchor through Dynamic.
- `smoke-dwd-validation` — bounded DWD run with a one-scene manifest.
- Cloud variants mirror local smokes and require ADC + CDS access.

Each smoke also runs the matching validator with assertion args.

## Logging contract

Every run emits two artifacts at `<output_root>/logs/<pipeline>/`:
- `<run_id>.jsonl` — structured event log
- `<run_id>.context.json` — redacted run context (Git commit, dirty state, pipeline, run_id, timestamp)

For GCS runs the JSONL is written to a local spool, uploaded atomically
on session exit, and the spool is deleted.  The context artifact is
written eagerly at session start.

```python
from berlin_lst_downscaling.data.io import RunLogSession, log_event

with RunLogSession(output_root, pipeline="ard", run_id=run_id):
    log_event(_logger, logging.INFO, "scene_done",
              scene_id=..., source=..., attempts=..., elapsed_s=...)
```

## Accessibility

- Manifest bundle: `gs://berlin-lst-data/manifests/v3/.../manifest.parquet`.
- AOI mask: `data/boundaries/aoi_10m.tif` (default), `aoi_100m.tif`.
- AOI polygon: `data/boundaries/berlin_landesgrenze.geojson`.

## Manual smoke command catalogue

- `.opencode/skills/google-access/` — rclone mount and ADC setup.
- `AGENTS.md` — repository conventions and current operational focus.
- `noxfile.py` — exact session definitions for every smoke listed above.
