# Preprocessing Pipelines — Operations Runbook

Four pipelines build the training-ready data set on Google Cloud Storage.
Each pipeline writes a fixed product contract so downstream code can
discover inputs without scanning the bucket.

| Pipeline | Purpose | Entry script | Config root |
|----------|---------|--------------|-------------|
| **ARD** | Per-scene Landsat/Sentinel-2/ECOSTRESS Analysis-Ready Data | `scripts/run_ard.py` | `configs/ard/` |
| **Static sources (A)** | Official archives reprojected to canonical 10 m grid | `scripts/run_static_sources.py` | `configs/static_sources/` |
| **Static derived (B)** | DSM, horizons, SVF from Pipeline A outputs | `scripts/run_static_derived.py` | `configs/static_derived/` |
| **Dynamic scenes (C)** | ERA5-Land meteorology + per-scene shadows | `scripts/run_dynamic.py` | `configs/dynamic/` |
| **DWD validation** | Independent sanity check on ERA5 `t2m_scene` | `scripts/run_dwd_validation.py` | `configs/dwd_validation/` |

The selection phase that builds the v3 manifest bundle lives in
`data/selection/`. The DWD pipeline is **read-only** against the
published dynamic outputs and never feeds DWD data into training or
normalisation.

## Architecture

```text
v3 manifest bundle (manifest.parquet + pairings.parquet)
   │
   ▼
Pipeline ARD — Landsat/S2/ECOSTRESS COGs + flag COGs
   │
   ▼
Pipeline A — official archives → canonical 10 m COGs (sources)
   │
   ▼
Pipeline B — DSM + horizon + SVF from Pipeline A
   │
   ▼
Pipeline C — ERA5-Land channels + shadows per Landsat anchor
   │
   ▼
DWD validation — sanity check on ERA5 t2m_scene
```

Pipeline A is independent of Pipeline B. Pipelines A/B feed C; C feeds
DWD.

## Product Contract

Every product (ARD scenes, Pipeline A sources, Pipeline B geometry,
Pipeline C scenes) writes the same four artifacts under its product
directory:

| File | Written when | Purpose |
|------|-------------|---------|
| `<source>_<key>.tif` | first | Final COG on the canonical 10 m EPSG:25833 grid |
| `<source>_<key>.stac.json` | after COG validation | STAC Item with band metadata and links to provenance/COG |
| `provenance.json` | after COG validation | Source/archive metadata, config hash, QA statistics, retrieval date |
| `complete.json` | **last** | Publication marker — `reconcile()` only treats a product as final once this exists |

GCS cannot publish multiple blobs atomically, so the completion marker
is the only visibility gate.

### Canonical Grid

| Property | Value |
|----------|-------|
| CRS | `EPSG:25833` |
| Resolution | `10 m × 10 m` |
| Origin | `(369190, 5838410)` (upper-left corner) |
| Bounding box | `(369190, 5799570, 416180, 5838410)` |
| Shape | `4699 × 3884` pixels |

Local smokes use `smoke_grid(bbox_wgs84)` to produce a canonical-aligned
subset (e.g. 208×208 px for a 2×2 km extent).

### COG Profile

| Parameter | Value |
|-----------|-------|
| Dtype | `float32` |
| NoData | `NaN` |
| Blocksize | `512 × 512` |
| Overviews | `2, 4, 8, 16` |
| Compression | `deflate`, predictor `2` |
| BigTIFF | `IF_SAFER` |

Source-specific value ranges are enforced by `validate_secondary_cog`
via per-band `valid_range` on the `BandSpec` contract.

## Ledger Semantics

Every pipeline tracks item state in a Parquet ledger at
`<output_root>/_state/<pipeline>/ledger.parquet`:

1. **pending** — newly added, not yet processed
2. **exporting** — processing in progress (crash recovery marker)
3. **done** — output COG written, validated, and checksummed
4. **failed** — processing error, will be retried on next run
5. **skipped** — explicitly skipped, never processed

`reconcile()` decides what to process on each run:

- Done items with matching `config_hash` + confirmed output → skipped
- Exporting items (crashed) → retry with `interrupted` reason
- Failed items → retry with `retry` reason
- Config hash mismatch → reprocess with `config_changed` reason

For ARD, scene roles (`anchor`, `predictor`, `validation`, `inference`)
are carried alongside the ledger so downstream consumers can split data
without re-reading the manifest.

## Required Setup

### Local

- Python 3.12, `uv`, and project dev group (`uv sync`)
- `pip install -e .` once
- No GCS credentials needed for local smokes

### GCS access

- `gcloud auth application-default login`
- Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### Published Cloud Targets

| Output | URI |
|--------|-----|
| Manifest bundle | `gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z/` |
| Static sources (full) | `gs://berlin-lst-data/static/sources/full/...` |
| Static derived (full) | `gs://berlin-lst-data/static/derived/full/...` |
| Dynamic scenes (full) | `gs://berlin-lst-data/dynamic/full/...` |
| Dynamic 2026 inference | `gs://berlin-lst-data/dynamic/inference/2026/...` |
| DWD validation | `gs://berlin-lst-data/dwd_validation/...` |

## Smoke Tests

Each pipeline has a nox-managed smoke that exercises the full contract
locally where possible, and a separate cloud smoke when GCS access is
required.

### ARD

```bash
# Manifest-driven, all three sources. Runs twice for idempotency,
# asserts three ``done`` rows, then runs validate_ard.py.
uv run nox -s smoke-primary

# Cloud variant (requires ADC).
uv run nox -s cloud-pilot
```

### Static sources (A)

```bash
# Real-data, 2×2 km subset, all four sources. Runs twice for idempotency.
uv run nox -s smoke-static-sources

# Cloud smoke against GCS.
uv run nox -s cloud-static-sources
```

### Static derived (B)

```bash
# Consumes Pipeline A smoke output, produces DSM/horizons/SVF.
uv run nox -s smoke-static-derived

# Cloud variant.
uv run nox -s cloud-static-derived
```

### Dynamic (C)

```bash
# One fixed Landsat anchor, runs twice, asserts three done sources,
# then runs validate_dynamic.py.
uv run nox -s smoke-dynamic -- \
    data/ard/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet

# Cloud smoke.
uv run nox -s cloud-smoke-dynamic -- <manifest_uri>

# Isolated per-scene runner (one subprocess per scene; bounds memory).
uv run python scripts/run_dynamic_isolated.py \
    --manifest-uri <manifest_uri> \
    --output-root gs://berlin-lst-data/dynamic/full/<run_id> \
    --config-name full
```

### DWD validation

```bash
# Bounded: one fixed anchor + one-day DWD window. Runs against the
# published dynamic root and writes the standard four artifacts.
uv run nox -s smoke-dwd-validation -- \
    <manifest_uri> gs://berlin-lst-data/dynamic/full/<run_id>

# Full validation: all published scenes, historical + recent DWD.
uv run python scripts/run_dwd_validation.py --config-name default \
    manifest_uri=<manifest_uri> \
    dynamic_full_root=gs://berlin-lst-data/dynamic/full \
    dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026 \
    output_root=gs://berlin-lst-data/dwd_validation
```

## Validators

Standalone scripts that read the published ledger + artifacts without
running the pipeline:

| Validator | Purpose |
|-----------|---------|
| `scripts/validate_ard.py` | COG structural validation from the ARD ledger |
| `scripts/validate_dynamic.py` | Ledger status + per-source counts, role consistency |
| `scripts/validate_manifest.py` | Manifest schema + hash + upstream identity |

Each validator exits non-zero when findings need attention; safe to run
from CI.

## Source Adapters (Pipeline A)

### Imperviousness (Versiegelung)

- Vintages: 2016, 2021.
- Source: Umweltatlas Berlin — uncorrected raster (2.5 m, uint8 class codes).
- 16-code class lookup maps to sealing percent in `[0, 100]`.
- Reproject to canonical 10 m EPSG:25833 with `Resampling.average`.
- Output: `ard/static/sources/imperviousness/{vintage}/imperviousness_{vintage}.tif`.

### Vegetation Height

- Vintage: 2020 (Umweltatlas).
- 2-band COG: `vegetation_height_mean`, `vegetation_height_max`.
- Non-vegetated cells → 0; outside AOI → NaN.

### Terrain Height (DGM 1 m)

- Vintage: 2021 (ALS acquisition Feb–Mar 2021).
- Source: Geoportal Berlin — INSPIRE ATOM feed, 297 XYZ CSV tiles.
- Reproject from native 1 m to 10 m with `Resampling.average`.
- Scene-year mapping deferred to feature assembly.

### LoD2 Building Morphology

- Vintage: 2024 (ATOM feed 2026-03-26).
- Source: Geoportal Berlin — INSPIRE ATOM feed, ~925 CityGML tiles.
- 4-band COG: `building_height_mean`, `building_height_std`, `building_coverage_ratio`, `building_height_max`.
- Scene-year mapping deferred to feature assembly.

## Pipeline B Outputs

| Product | Description |
|---------|-------------|
| `building_dsm` | Surface model from LoD2 footprints, resampled to 10 m |
| `vegetation_dsm` | Surface model from vegetation-height canopy max |
| `combined_dsm` | Element-wise max of building + vegetation DSMs |
| `horizon_building` | 36-band horizon cube from building DSM |
| `horizon_vegetation` | 36-band horizon cube from vegetation DSM |
| `svf` | Sky view factor on the canonical 10 m grid |

Each product lives under
`ard/static/derived/<product>/<geometry_id>/` with the same four-artifact
contract.

## Pipeline C Outputs

Per Landsat anchor scene (`ard/dynamic/<source>/<scene_id>/`):

| Source | Bands |
|--------|-------|
| `era5_land` | `t2m_scene` (K), `ssrd_scene` (W/m²), `ssrd_antecedent_72h_mean` (W/m²) |
| `shadow_building` | sun-shadow mask from combined DSM |
| `shadow_vegetation` | sun-shadow mask from vegetation DSM |

ERA5-Land uses NetCDF retrieval (CDS `format=netcdf`); SSRD
accumulates within each UTC day and resets at 00:00 — conversion to
hourly W/m² follows the ECMWF differencing rule.

The dynamic geometry policy is **retrospective-static**: every 2017–2025
scene uses LoD2-2024, DGM-2021, and vegetation-height-2020.

## DWD Validation Outputs

```
<output_root>/
  _raw/dwd/<run_id>/
    station_inventory.parquet
    dwd_hourly_observations.parquet
  runs/dwd/<run_id>/
    anchor_comparison.parquet   # per-anchor join of ERA5 t2m vs DWD obs
    report.json                 # QA summary with bias/MAE/RMSE
    provenance.json
    complete.json
```

Anchor comparison uses historical precedence over recent (provisional)
DWD observations at the same station/timestamp.

## Logging

Every productive entrypoint uses `RunLogSession`
(`src/berlin_lst_downscaling/data/io/run_logging.py`). JSONL logs land
under `<output_root>/logs/<pipeline>/<run_id>.jsonl` and are uploaded
to GCS after the run exits. See `docs/run-logging.md` for the schema and
jq recipes.

## Adding a New Source (Pipeline A)

1. Acquire the raw archive, validate native metadata, compute the
   canonical output raster on the 10 m grid.
2. Return a `PreparedSecondaryProduct` with the canonical dataset,
   source metadata, and source-specific QA statistics.
3. Register the source in `data/secondary/source_pipeline.py`.
4. `product.finalize_secondary_product()` writes the four final artifacts
   and runs `validate_secondary_cog`.
5. Add a config entry and (optionally) a focused diagnostic nox session.
