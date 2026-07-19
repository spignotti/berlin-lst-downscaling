# Secondary-Data Pipeline — Operations Runbook

Shared foundation for acquiring, processing, and storing all non-satellite
data sources used in the LST downscaling ablation study.

## Architecture

Three pipelines share the product contract:

| Pipeline | Purpose | Entry point | Config prefix |
|----------|---------|-------------|---------------|
| **A — Source products** | Download official archives, reproject to canonical grid, publish validated COGs | `scripts/run_static_sources.py` | `configs/static_sources/` |
| **B — Derived geometry** | Combine source COGs into DSM, horizons, SVF | `scripts/run_static_derived.py` | `configs/static_derived/` |
| **C — Dynamic scene** | Per-scene ERA5-Land + shadow products from static geometry | `scripts/run_dynamic.py` | `configs/dynamic/` |

Pipeline A is strictly independent of Pipeline B; it does not
depend on any derived geometry products.

### Source products (Pipeline A)

```text
Official archives (Umweltatlas, Geoportal Berlin)
            │
            ▼
Pipeline A: download → reproject → COG + STAC + provenance + complete
            │
            ▼
    GCS: gs://berlin-lst-data/static/sources/{full|smoke}/ard/static/sources/
```

### Current source revisions

| Source | Vintage | Feed | License |
|--------|---------|------|---------|
| imperviousness | 2016, 2021 | Umweltatlas ATOM | dl-de/zero-2.0 |
| vegetation_height | 2020 | Umweltatlas ATOM | dl-de/zero-2.0 |
| terrain_height | 2021 | Geoportal Berlin DGM1 ATOM | dl-de/zero-2.0 |
| lod2_morphology | 2024 | Geoportal Berlin LoD2 ATOM | dl-de/zero-2.0 |

## Path Layout

All paths are relative to `source_root` (local path or `gs://bucket/prefix`).

| Path | Purpose |
|------|---------|
| `_raw/secondary/{source}/{period}/` | Raw downloaded archives — one per source and period/vintage |
| `ard/static/sources/{source}/{vintage}/` | Final source products (COG + STAC + provenance + completion marker) |
| `logs/static-sources/{run_id}.jsonl` | Structured JSONL run log (see [docs/run-logging.md](run-logging.md)) |
| `qa/static/sources/{run_id}/report.json` | Persisted per-run QA report |
| `_state/static/sources/ledger.parquet` | Persistent item-level processing ledger |

## Product Contract

Every secondary product (static or dynamic) produces exactly four
artifacts under its product directory:

| File | Written when | Purpose |
|------|-------------|---------|
| `{source}_{vintage}.tif` | first | Final COG on the canonical 10 m EPSG:25833 grid |
| `{source}_{vintage}.stac.json` | after COG + range/QA OK | STAC Item with canonical grid, raster band metadata, and links to provenance/COG |
| `provenance.json` | after COG OK | Source/archive metadata, config hash, QA statistics, retrieval date |
| `complete.json` | **last** | Publication marker — the product is considered final only after this file exists |

### Canonical Grid

All static products share one grid (see `common/grid.py`):

| Property | Value |
|----------|-------|
| CRS | `EPSG:25833` |
| Resolution | `10 m × 10 m` |
| Origin (upper-left corner) | `(369190, 5838410)` |
| Bounding box | `(369190, 5799570, 416180, 5838410)` |
| Shape | `4699 × 3884` pixels |

For local smoke tests, `smoke_grid(bbox_wgs84)` produces a canonical-aligned
subset grid (e.g. 208×208 px for a 2×2 km extent).

### COG Profile

| Parameter | Value |
|-----------|-------|
| Dtype | `float32` |
| NoData | `NaN` |
| Blocksize | `512 × 512` |
| Overviews | `2, 4, 8, 16` |
| Compression | `deflate`, predictor `2` |
| BigTIFF | `IF_SAFER` |

Source-specific value ranges are enforced by `validate_secondary_cog`.

### Publication Marker

The `complete.json` file is written **last**. Its absence means the
product is not considered final by `reconcile()`. This is the only
mechanism that guards against partial publication; GCS cannot atomically
publish multiple blobs, so `complete.json` is the visibility gate.

## Ledger Semantics

The `SecondaryLedger` tracks every item (`item_id + source + period`) through
its lifecycle:

1. **pending** — newly added, not yet processed
2. **exporting** — processing in progress (crash recovery marker)
3. **done** — output COG written, validated, and checksummed
4. **failed** — processing error, will be retried on next run
5. **skipped** — explicitly skipped, never processed

### Idempotency / Resume

`reconcile()` applies the same logic as the ARD pipeline:

- Done items with matching `config_hash` + confirmed output → skipped
- Exporting items (crashed) → retry with **interrupted** reason
- Failed items → retry with **retry** reason
- Config hash mismatch → reprocess with **config_changed** reason

## Required Setup

### GCS access

- Install ADC: `gcloud auth application-default login`
- Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### VM

- Instance: `berlin-lst-vm`, zone `europe-west3-a`, machine `n2-standard-2`
- OS: Debian 12, Spot preemptible
- Auth: Service account via ADC (no keys in repo)
- Bucket: `gs://berlin-lst-data` (co-located in `europe-west3`)

### Disk Budget

Each full source run **must** declare a peak scratch estimate before
execution.  The run will fail preflight if the estimate exceeds the
configured budget.

Current budget: **20 GB** (VM boot disk).
Override with `disk_budget_gb=50` in the Hydra command line.

## Smoke Tests

```bash
# Local real-data smoke — 2×2 km subset, all 4 sources.
# Downloads real archives, writes canonical-grid COGs, validates.
# Runs twice to confirm idempotency.
uv run nox -s smoke-static-sources

# Cloud smoke on VM — same subset, writes to GCS.
uv run nox -s cloud-static-sources

# Cloud full run on VM — all 297 DGM + ~925 LoD2 tiles, full AOI.
uv run python scripts/run_static_sources.py --config-name full \
    source_root=gs://berlin-lst-data/static/sources/full
```

### Legacy fixture sessions (no longer primary)

```bash
# Synthetic fixture smoke — validates contract lifecycle without downloads.
uv run nox -s smoke-secondary-all

# Individual-source diagnostics (local):
uv run nox -s smoke-secondary-imperviousness
uv run nox -s smoke-secondary-vegetation-height
```

## Source Adapters

### Imperviousness (Versiegelung)

**Vintages:** 2016, 2021 (both processed unconditionally)

**Source:** Umweltatlas Berlin — uncorrected raster (2.5 m, uint8 class codes)

**Processing:**
1. Download ZIP from official ATOM feed (preserved under `_raw/secondary/imperviousness/{vintage}/`).
2. Extract GeoTIFF from ZIP.
3. Convert uint8 class codes to float32 sealing percent using the verified 16-code lookup.
4. Reproject to canonical 10 m EPSG:25833 grid with `Resampling.average`.
5. Write COG to `ard/static/sources/imperviousness/{vintage}/imperviousness_{vintage}.tif`.

**Class codes verified (2026-07-14):**
| Code | Meaning | Output value |
|------|---------|-------------|
| 0 | Unsealed | 0 % |
| 5, 15, …, 95 | Sealing classes | class value (%) |
| 100 | Fully sealed (non-building) | 100 % |
| 101 | Building-shadow sealed | 100 % |
| 102 | Building footprint | 100 % |
| 103 | Rail ballast | 100 % |
| 110 | Shadow | 100 % |
| 255 | Nodata (2021 only) | NaN |

**Validation gates:**
- Structural: CRS, shape, origin, band count (reuses ARD `validate_cog`)
- Value range: all valid pixels ∈ [0, 100] with 0.01 tolerance
- Code set: hard fail on codes outside the verified 16-code scheme
- Idempotency: second run processes nothing

### Vegetation Height

**Vintages:** 2020

**Source:** Umweltatlas Berlin — GeoTIFF (1 m, float32)

**Processing:**
1. Download ZIP from official ATOM feed.
2. Reproject to canonical 10 m with `Resampling.average` (mean) and `Resampling.max` (max).
3. Normalize: non-vegetated cells → 0, outside AOI → NaN.
4. Write 2-band COG (`vegetation_height_mean`, `vegetation_height_max`).

**Validation:** range [0, 400] m.

### Terrain Height (DGM 1 m)

**Vintages:** 2021 (ALS acquisition Feb–Mar 2021)

**Source:** Geoportal Berlin — INSPIRE ATOM feed, 297 XYZ CSV tiles

**Processing:**
1. Parse ATOM feed to discover tiles intersecting the output grid.
2. Download each ZIP via `download_to_raw` (streaming SHA-256).
3. Read XYZ CSV (variable-size grid at 1 m, EPSG:25833, DHHN2016).
4. Reproject from native 1 m to output 10 m with `Resampling.average`.
5. Accumulate tile coverage onto the output grid.
6. Write COG to `ard/static/sources/terrain_height/{vintage}/`.

**Note:** The 2021 vintage is technically future for scenes 2017–2020.
Scene-year mapping is deferred to feature assembly; see `docs/lod2-vintage-qualification.md`.

### LoD2 Building Morphology

**Vintages:** 2024 (data revision 2024-04-22, ATOM feed 2026-03-26)

**Source:** Geoportal Berlin — INSPIRE ATOM feed, ~925 CityGML tiles

**Processing:**
1. Parse ATOM feed to discover tiles intersecting the output grid.
2. Download each ZIP and stream-parse CityGML XML (v1.0 and v2.0).
3. Extract `Building` elements: `measuredHeight` + `GroundSurface` polygons.
4. Rasterize footprints at 10 m: accumulate per-cell height sum, sum², count, area, max.
5. Compute four morphology bands: `building_height_mean`, `building_height_std`, `building_coverage_ratio`, `building_height_max`.
6. Write 4-band COG to `ard/static/sources/lod2_morphology/{vintage}/`.

**Temporal policy:** The 2024 feed is future for all scenes (2017–2025).
Scene-year mapping is deferred to feature assembly; see `docs/lod2-vintage-qualification.md`.

## Adding a New Source

Each source adapter produces one **prepared product** per vintage/scene:

1. Acquire raw archive, validate native metadata, compute canonical
   output raster.
2. Return a `PreparedSecondaryProduct` payload containing the canonical
   dataset, source metadata, and source-specific QA statistics.
3. Register the source in `source_pipeline.py`.
4. `product.finalize_secondary_product()` writes the four final
   artifacts (COG, STAC, provenance, completion marker), validates the
   COG, and returns the artifact URIs for the ledger.
5. Add a config entry and (optionally) a focused diagnostic nox session.
