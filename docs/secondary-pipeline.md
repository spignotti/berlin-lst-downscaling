# Secondary-Data Pipeline — Operations Runbook

Shared foundation for acquiring, processing, and storing all non-satellite
data sources used in the LST downscaling ablation study.

## Path Layout

All paths are relative to `output_root` (local path or `gs://bucket/prefix`).

| Path | Purpose |
|------|---------|
| `_raw/secondary/{source}/{period}/` | Raw downloaded archives — one per source and period/vintage |
| `_staging/secondary/{source}/{run_id}/` | Ephemeral processing scratch space |
| `ard/static/{category}/{source}/{vintage}/` | Final static products (COG + STAC + provenance + completion marker) |
| `ard/dynamic/meteorology/{scene_id}/` | Future: scene-keyed dynamic products (ERA5, shadows) |
| `qa/secondary/{run_id}/report.json` | Persisted per-run QA report |
| `ledger.parquet` | Persistent item-level processing ledger |

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

### STAC Item

Static products use a minimal STAC Item:

- `id`: `{source}-{vintage}` (future dynamic: `{source}-{scene_id}`)
- `datetime: null` + `start_datetime` / `end_datetime`: nominal vintage interval (not a fabricated acquisition timestamp)
- `geometry` / `bbox`: canonical-grid footprint in WGS84
- `proj:code`: `EPSG:25833`, `proj:shape`, `proj:transform` (Projection extension v2.0.0)
- `raster:bands` with COG dtype and nodata
- Assets: `data` → COG, `provenance` → provenance.json

### Publication Marker

The `complete.json` file is written **last**. Its absence means the
product is not considered final by `reconcile()`. This is the only
mechanism that guards against partial publication; GCS cannot atomically
publish multiple blobs, so `complete.json` is the visibility gate.

### Future Dynamic Identity

When dynamic sources are added (ERA5-Land, scene-level shadows), the
same product pattern applies under `ard/dynamic/...` keyed by
`source + scene_id` instead of `source + vintage`. The per-source
prepare handler produces a `PreparedSecondaryProduct` payload;
`product.finalize_secondary_product()` writes the four artifacts and
the completion marker.

## Modes

| Mode | Output Root | Purpose |
|------|-------------|---------|
| `fixture` | Local `data/smoke/secondary/{name}` | Validate pipeline lifecycle per registered source with small synthetic fixtures |
| `full` | Configurable | Production runs with real sources |

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

## Retry / Resume Behaviour

- HTTP downloads (via `download_to_raw`) use `tenacity` with
  3 attempts and exponential backoff (up to 10 s).
- Ledger upserts happen immediately — crash consistency is built in.
- An aborted run resumes by re-running the same command.  The second run
  processes only items that are missing, failed, or interrupted.

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
Use kachelweises/tilewise scratch cleanup when processing tile-based
sources (LoD2, DGM 1 m).

## Smoke Tests

```bash
# Local fixture smoke — exercises every registered source with
# small synthetic fixtures, no upstream downloads.
uv run nox -s smoke-secondary-all

# Cloud fixture smoke — fixture pattern on GCS.
uv run nox -s cloud-secondary-fixture

# Cloud full run (VM) — processes every real source against GCS.
uv run nox -s cloud-secondary-all

# Individual-source diagnostics (local, per source):
uv run nox -s smoke-secondary-imperviousness
uv run nox -s smoke-secondary-vegetation-height
uv run nox -s cloud-secondary-imperviousness
uv run nox -s cloud-secondary-vegetation-height
```

The local fixture smoke validates the framework (contract, STAC,
provenance, completion marker, idempotency, QA report) without
downloading any full upstream archive. Cloud acceptance runs validate
real provider inputs against GCS.

## Added Sources

### Imperviousness (Versiegelung)

**Vintages:** 2016, 2021 (both processed unconditionally)

**Source:** Umweltatlas Berlin — uncorrected raster (2.5 m, uint8 class codes)

**Processing:**
1. Download ZIP from official ATOM feed (preserved under `_raw/secondary/imperviousness/{vintage}/`).
2. Extract GeoTIFF from ZIP.
3. Convert uint8 class codes to float32 sealing percent using the verified 16-code lookup.
4. Reproject to canonical 10 m EPSG:25833 grid with `Resampling.average`.
5. Write COG to `ard/static/morphology/imperviousness/{vintage}/imperviousness_{vintage}.tif`.

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

**Runbook:**
```bash
# Local smoke (downloads both vintages)
uv run python scripts/run_secondary.py --config-name smoke_imperviousness

# With custom output root (local or GCS)
uv run python scripts/run_secondary.py --config-name imperviousness \
    output_root=gs://berlin-lst-data/secondary/full_20260714

# Override disk budget on VM
uv run python scripts/run_secondary.py --config-name imperviousness \
    output_root=gs://berlin-lst-data/secondary/full_20260714 \
    disk_budget_gb=50
```

**Validation gates:**
- Structural: CRS, shape, origin, band count (reuses ARD `validate_cog`)
- Value range: all valid pixels ∈ [0, 100] with 0.01 tolerance
- Code set: hard fail on codes outside the verified 16-code scheme
- Idempotency: second run processes nothing

### Terrain Height (DGM 1 m)

**Vintages:** 2021 (ALS acquisition Feb–Mar 2021)

**Source:** Geoportal Berlin — INSPIRE ATOM feed, 297 XYZ CSV tiles

**Processing:**
1. Parse ATOM feed to discover tiles intersecting the full AOI.
2. Download each ZIP via `download_to_raw` (streaming SHA-256).
3. Read XYZ CSV (2000×2000 points at 1 m, EPSG:25833, DHHN2016).
4. Reproject from native 1 m to canonical 10 m with `Resampling.average`.
5. Write COG to `ard/static/morphology/terrain_height/{vintage}/`.

**Note:** The 2021 vintage is technically future for scenes 2017–2020.
See `docs/lod2-vintage-qualification.md` for the temporal policy.

### LoD2 Building Morphology

**Vintages:** Pending qualification (current feed 2026-03-26 is future)

**Source:** Geoportal Berlin — INSPIRE ATOM feed, ~925 CityGML tiles

**Processing:**
1. Parse ATOM feed to discover tiles.
2. Download each ZIP and stream-parse CityGML XML.
3. Extract `Building` elements: `measuredHeight` + `GroundSurface` polygons.
4. Rasterize footprints at 10 m: accumulate per-cell height sum, sum², count, area.
5. Compute three morphology bands: `building_height_mean`, `building_height_std`, `building_coverage_ratio`.
6. Write 3-band COG to `ard/static/morphology/lod2_morphology/{vintage}/`.

**Temporal policy:** No future data for past scenes. Qualified vintages
must be documented in `docs/lod2-vintage-qualification.md`.

### Digital Surface Models (DSM)

**Keyed by:** input-vintage combination (terrain + LoD2 + VH vintages)

Three derived products per geometry vintage:

| Product | Formula | Description |
|---------|---------|-------------|
| `building_dsm` | terrain + LoD2 max height | Terrain with buildings added |
| `vegetation_dsm` | terrain + VH max height | Terrain with canopy added |
| `combined_dsm` | max(building_dsm, vegetation_dsm) | Full surface model |

**Source module:** `dsm.py` reads upstream COGs and combines them.

### Horizon Cubes

**Keyed by:** geometry vintage + component (building/vegetation)

36-band COG per component/vintage (0°–350°, 10° steps). Each band
encodes the maximum elevation angle visible from each cell along that
azimuth direction, stored as `uint16` centidegrees (×100) with
nodata=65535.

**Source module:** `horizon.py` — custom NumPy kernel, 200 m search radius.

**Purpose:** Pre-computed horizon enables fast per-scene shadow lookup
(Stage 3) without re-running the expensive ray-casting.

### Sky View Factor (SVF)

**Keyed by:** geometry vintage

Single-band float32 COG [0, 1] — fraction of sky hemisphere visible
from each cell. Computed via `xarray-spatial.sky_view_factor()` (Zakek
2011 algorithm, numba-backed, ~21s for full AOI).

**Source module:** `svf.py`

## Adding a New Source

Each source adapter produces one **prepared product** per vintage/scene:

1. Acquire raw archive, validate native metadata, compute canonical
   output raster.
2. Return a `PreparedSecondaryProduct` payload containing the canonical
   dataset, source metadata, and source-specific QA statistics.
3. Register the source in the pipeline runner (see `pipeline.py`).
4. `product.finalize_secondary_product()` writes the four final
   artifacts (COG, STAC, provenance, completion marker), validates the
   COG, and returns the artifact URIs for the ledger.
5. Add a config entry and (optionally) a focused diagnostic nox session.

The local fixture smoke auto-picks up every registered source via
`fixtures.registry()`; adding a new source should also add a small
synthetic fixture to keep `smoke-secondary-all` representative.
