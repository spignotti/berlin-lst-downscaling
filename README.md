# Berlin LST Downscaling

Cloud-native land-surface-temperature downscaling pipeline for Berlin.
The preprocessing phase turns a published manifest of Landsat, Sentinel-2,
and ECOSTRESS scenes into a training-ready COG stack on Google Cloud
Storage, plus an independent DWD sanity check on the ERA5-Land channel.

## What ships in this phase

| Stage | Output | Count |
|-------|--------|-------|
| Selection (v3 manifest) | `gs://berlin-lst-data/manifests/v3/...` | 345 Landsat anchors, 509 scenes total |
| ARD | `gs://berlin-lst-data/ard/...` | Per-scene COG + flag COG + STAC + provenance |
| Static sources (A) | `gs://berlin-lst-data/static/sources/...` | 4 sources × vintages (imperviousness 2016/2021, vegetation_height 2020, terrain_height 2021, lod2_morphology 2024) |
| Static derived (B) | `gs://berlin-lst-data/static/derived/...` | building_dsm, vegetation_dsm, combined_dsm, horizon_building, horizon_vegetation, svf |
| Dynamic scenes (C) | `gs://berlin-lst-data/dynamic/...` | 972 training + 63 inference = 1 035 products (era5_land + shadow_building + shadow_vegetation per scene) |
| DWD validation | `gs://berlin-lst-data/dwd_validation/...` | 345 anchors, 378 953 DWD observations, 1 508 matched pairs |

DWD head-line metrics: bias −0.03 °C, MAE 0.77 °C, RMSE 0.98 °C
(ERA5 `t2m_scene` minus DWD hourly 2 m temperature). DWD is
**validation-only** — it never feeds into training or normalisation.

## Architecture

```
v3 manifest (PC STAC + ECOSTRESS CMR + DWD)
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
   │
   ▼
DWD validation — independent sanity check on ERA5 t2m_scene
```

Every product publishes four artifacts in its product directory: the
final COG, a STAC Item, `provenance.json`, and `complete.json`. The
completion marker is written last and is the only visibility gate; GCS
cannot publish multiple blobs atomically.

## Quick start

```bash
uv sync
uv run nox -s lint typecheck

# Local smokes for each canonical pipeline
uv run nox -s smoke-primary
uv run nox -s smoke-static-sources
uv run nox -s smoke-static-derived
uv run nox -s smoke-dynamic -- \
    data/ard/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet
uv run nox -s smoke-dwd-validation -- \
    data/ard/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet \
    gs://berlin-lst-data/dynamic/full/<run_id>
```

Cloud smokes live next to the local smokes in `noxfile.py` and assume
ADC + CDS API access + the published v3 manifest.

## Documentation

- `docs/preprocessing-pipelines.md` — operations runbook for the four pipelines + DWD.
- `docs/run-logging.md` — JSONL log path layout and jq recipes.
- `docs/ard-manifest-schema.md` — v3 manifest schema, role contract, validation.
- `docs/lod2-vintage-qualification.md` — geometry vintage policy.
- `docs/data-availability.md`, `docs/additional-data-sources.md` — research records (status notes flag them as pre-shipment analyses).

## Stack

- Python 3.12, `uv` (lockfile committed)
- `pystac-client`, `odc-stac`, `rioxarray` for STAC + EO data
- `cdsapi`, `netcdf4` for ERA5-Land
- `wetterdienst` for DWD station access (validation only)
- `pydantic-settings`, `hydra-core` for config
- `pyarrow` for ledger + Parquet IO
- `google-cloud-storage` for GCS access
- `nox`, `ruff`, `pyright` for validation
