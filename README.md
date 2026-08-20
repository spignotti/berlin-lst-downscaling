# Berlin LST Downscaling

Cloud-native land-surface-temperature downscaling pipeline for Berlin.
The preprocessing phase turns a published manifest of Landsat, Sentinel-2,
and ECOSTRESS scenes into a training-ready COG stack on Google Cloud
Storage.

## What ships in this phase

| Stage | Output | Count |
|-------|--------|-------|
| Selection (v3 manifest) | `gs://berlin-lst-data/manifests/v3/...` | 345 Landsat anchors, 509 scenes total |
| ARD | `gs://berlin-lst-data/ard/...` | Per-scene COG + flag COG + STAC + provenance |
| Static sources (A) | `gs://berlin-lst-data/static/sources/...` | 4 source types · 8 ledger rows (imperviousness 2016/2021, vegetation_height 2020, terrain_height 2021, lod2_morphology 2017/2021/2022/2024) |
| Static derived (B) | `gs://berlin-lst-data/static/derived/...` | building_dsm, vegetation_dsm, combined_dsm, horizon_building, horizon_vegetation, svf (per geometry vintage) |
| Dynamic scenes (C) | `gs://berlin-lst-data/dynamic/...` | 972 training + 63 inference = 1 035 products (era5_land + shadow_building + shadow_vegetation per scene) |
| Feature stacks | `gs://berlin-lst-data/features/v1/` | 324 per-anchor 24-band 10 m stacks + feature_valid masks (S2, indices, morphology, ERA5, shadows) |
| DWD validation | `gs://berlin-lst-data/dwd_validation/r3/` | Historical pre-rebuild check: 345 anchors, 1 508 matched pairs |

Details on the delivered products, their exact paths, and the phase-2
handoff: [`docs/phase-1-delivery.md`](docs/phase-1-delivery.md).

DWD head-line metrics (r3, historical — validated the pre-rebuild scalar
ERA5 products): bias −0.03 °C, MAE 0.77 °C, RMSE 0.98 °C. The DWD
validation subsystem is **retired**: the post-rebuild ERA5 fields cannot
be re-validated with it (a timezone join defect in the anchor-hour
comparison yields zero matches), so r3 stays as the historical record.
DWD was **validation-only** — it never fed into training or
normalisation.

## Architecture

```
v3 manifest (PC STAC + ECOSTRESS CMR)
   │
   ▼
ARD — Landsat/S2/ECOSTRESS Analysis-Ready Data (10–100 m, NaN-NoData)
   │
   ▼
Pipeline A — official archives → canonical 10 m COGs (sources)
   │
   ▼
Pipeline B — DSM + horizons + SVF (10 m derived geometry)
   │
   ▼
Pipeline C — ERA5-Land + shadows per Landsat anchor
```

Every product publishes four artifacts in its product directory: the
final COG, a STAC Item, `provenance.json`, and `complete.json`. The
completion marker is written last and is the only visibility gate; GCS
cannot publish multiple blobs atomically.

## Setup

```bash
uv sync
uv run nox -s lint typecheck
```

## Operations

Run order: selection → ARD (parallel to static) → dynamic. Production
paths below point at the canonical bundle
`manifests/v3/2017-2026-cutoff-20260717T235959Z-r2`; the VM runs the
heavy pipelines, GCS is the source of truth.

### Selection (manifest bundle)

```bash
# Build locally (couple mode; writes manifest.parquet + pairings.parquet
# + manifest_report.json under data/manifest_build/).
uv run python scripts/build_manifest.py --config-name full_2017_2026

# Publish to an immutable bundle prefix (already-published bundles refuse
# overwrite; use a -r2 style suffix for revisions).
uv run python scripts/publish_manifest.py \
    --local-root data/manifest_build/v3/2017-2026-cutoff-20260717T235959Z \
    --publish-root gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2

# Structural validation (schemas, hashes, report counts, FK consistency).
uv run python scripts/validate_manifest.py \
    --manifest gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet
```

### ARD

```bash
# output_root must point at GCS — the config default is the local scratch
# root `data/ard`.
uv run python scripts/run_ard.py --config-name full_all \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    output_root=gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z

# Strict validation (exact manifest/ledger key-set, all four artifacts, STAC schemas).
uv run python scripts/validate_ard.py \
    --ledger gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet \
    --manifest gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# S2-only republish (six-band contract) — reprocesses exactly the 158
# Sentinel-2 scenes whose schema version changed; Landsat/ECOSTRESS rows
# are untouched.
uv run python scripts/run_ard.py --config-name full_sentinel2_swir \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet
```

### Static sources (A)

```bash
uv run python scripts/run_static_sources.py --config-name full \
    source_root=gs://berlin-lst-data/static/sources/full
```

### Static derived (B)

```bash
# Consumes finalized Pipeline-A products from GCS.
uv run python scripts/run_static_derived.py --config-name full
```

### Historical LoD morphometry vintages

Additional Static A/B vintages (2017, 2021, 2022) — archive-first runner.
Requires `terrain_height/2021`, `vegetation_height/2020`,
`vegetation_dsm/2024` already published in the production roots.

```bash
# Full archive-backed backfill (VM, all three historical vintages).
uv run python scripts/run_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022

# Reconcile finalized artifacts into both Static ledgers (no archive
# download, no raster computation).
uv run python scripts/run_lod_vintages.py --reconcile-only \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --vintages 2017,2021,2022

# Derived-only repair — rebuilds only the requested products per vintage
# from existing finalised inputs. Currently supports combined_dsm,svf.
uv run python scripts/run_lod_vintages.py --derive-only \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022 \
    --derived-products combined_dsm,svf

# Structural validation (fast, no archive download).
uv run python scripts/validate_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022

# Strict archive integrity (downloads ~4 GB sequentially).
uv run python scripts/validate_lod_vintages.py \
    --source-root gs://berlin-lst-data/static/sources/full \
    --derived-root gs://berlin-lst-data/static/derived/full \
    --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \
    --raw-root gs://berlin-lst-data \
    --vintages 2017,2021,2022 \
    --verify-archives
```

### Dynamic scenes (C)

The isolated runner executes one child process per scene, capping memory
at roughly 1.2 GB per scene; the shared output root and ledger keep it
idempotent.

```bash
# Full retrospective (2017–2025 anchors).
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    --output-root gs://berlin-lst-data/dynamic/full \
    --config-name full --years 2017 2025 --dataset-role anchor --resume

# Inference (2026 anchors).
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    --output-root gs://berlin-lst-data/dynamic/inference/2026 \
    --config-name inference_2026 --years 2026 --dataset-role inference --resume
```

### Stage-1 raw-input QA gate

Validates every published raw input of the training universe (COG/grid
contracts, flag semantics, fixed physical ranges) and computes a
diagnostic joint-support statistic at 100 m: a target cell is fully
supported when the Landsat LST cell is valid and all present 10 m
subpixels of S2, static morphology, ERA5-Land, and shadows are valid.
The gate is **read-only and mask-free** by design: it reports support as
a metric and never persists a validity/selection mask. The final
training-eligibility mask is decided after feature computation and the
Stage-2 gate.

```bash
# Smoke: one deterministic pair on a bounded bbox, local ephemeral output
# (requires ADC). Output is removed after the run.
uv run nox -s smoke-qa-stage1

# Full: the published 2017-2025 universe (324 pairs), evidence to GCS.
uv run python scripts/run_qa_stage1_raw.py --config-name stage1_raw_full

# Independent report validation (schema, counts, fingerprints, no-mask invariant).
uv run python scripts/validate_qa_stage1_raw.py \
    --run-prefix gs://berlin-lst-data/qa/stage1_raw/<run-id>

# Full run on the VM (start → deploy → run → validate → stop).
scripts/run_qa_stage1_vm.sh main
```

Evidence (summary.json, scenes.parquet, scenes.csv, logs) lands under
`gs://berlin-lst-data/qa/stage1_raw/<run-id>/`. The gate exits non-zero
on any contract/range/input finding within the training universe; 2026
inference scenes are reported as exclusions, never failures.

### Scene feature stacks

Per paired Landsat anchor (2017-2025, 324 scenes): a 24-band 10 m
feature stack COG plus a co-registered `feature_valid` mask on the
canonical EPSG:25833 grid. Channels: six S2 ARD bands, four spectral
indices (NDVI, McFeeters NDWI, NDBI, six-band albedo proxy), four
morphology layers (building/vegetation/combined DSM + SVF, with the fixed
vegetation_height-2020 carry-forward), eight ERA5-Land bands, and two
shadow masks. A pixel is `feature_valid` only inside the exact Berlin
AOI with a clear S2 flag and all 24 channels finite and in-range;
invalid pixels are NaN in every channel. The mask is **not** a
training-eligibility mask — `training_eligible@100m` is a Stage-2
decision. The full channel contract and mask semantics live in
`docs/data-sources-and-contracts.md`.

```bash
# Smoke: one deterministic pair on a bounded bbox, local ephemeral output
# (requires ADC). Output is removed after the run.
uv run nox -s smoke-features

# Full: per-scene subprocess isolation (one child per scene; a full-grid
# scene peaks at ~7-8 GB RAM), canonical output to GCS.
uv run python scripts/run_features_isolated.py --config-name full

# Resume a previous full run (skips scenes already done + complete).
uv run python scripts/run_features_isolated.py --config-name full --resume

# Independent validation (ledger completeness against the expected
# 324 scenes, 24-band order, mask semantics, AOI, per-channel ranges,
# sidecars, coverage cross-check). Fails on any non-done/incomplete row.
uv run python scripts/validate_feature_stacks.py \
    --root gs://berlin-lst-data/features/v1 \
    --expected-scenes 324

# Full run on the VM (start → deploy → run → validate → stop).
scripts/run_features_vm.sh main
```

Products (COG, mask, STAC, provenance, `complete.json`) land under
`gs://berlin-lst-data/features/v1/<scene_id>/`; the ledger is at
`<root>/_state/features/ledger.parquet`; run reports at
`<root>/qa/features/<run-id>/report.json`.

### Validators

```bash
uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/full \
    --expected-role anchor --expected-scenes 324

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/inference/2026 \
    --expected-role inference --expected-scenes 21
```

## Smoke gates

Local smokes for each canonical pipeline (`noxfile.py`); they require ADC
+ CDS API access where noted. Only `smoke-primary` and `smoke-dynamic`
additionally run the standalone validators after the pipeline; the
remaining smokes run the pipeline and assert ledger state only.

```bash
uv run nox -s smoke-primary
uv run nox -s smoke-static-sources
uv run nox -s smoke-static-derived
uv run nox -s smoke-selection-couple  # builds the local smoke manifest below
uv run nox -s smoke-dynamic -- \
    data/manifest_build/v3/smoke/manifest.parquet
uv run nox -s smoke-qa-stage1         # Stage-1 QA gate, one real pair
uv run nox -s smoke-features          # Feature stacks, one real pair
```

Cloud variants (`cloud-smoke-*`) mirror the local smokes and assume ADC
+ CDS API access + the published v3 manifest.

## Logging

Pipeline telemetry goes through `log_event` (`data/io/run_logging.py`)
as JSONL at `<output_root>/logs/<pipeline>/<run_id>.jsonl`; GCS runs
publish the log after exit. `print()` is reserved for validators,
spikes, and human-oriented CLI summaries.

## Documentation

- `docs/phase-1-delivery.md` — delivered data products, metadata interface, phase-2 handoff.
- `docs/data-sources-and-contracts.md` — sources, canonical grid, manifest and ledger contracts.
- `docs/phase-2-preparation.md` — preparation state for the next QA phase.

## Stack

- Python 3.12, `uv` (lockfile committed)
- `pystac-client`, `odc-stac`, `rioxarray` for STAC + EO data
- `cdsapi`, `netcdf4` for ERA5-Land
- `pydantic-settings`, `hydra-core` for config
- `pyarrow` for ledger + Parquet IO
- `google-cloud-storage` for GCS access
- `nox`, `ruff`, `pyright` for validation
