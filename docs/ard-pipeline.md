# ARD Pipeline — Reprojection, QA, STAC, and GCS Orchestration

The ARD pipeline takes raw exports from Google Earth Engine (Landsat, Sentinel-2) and AppEEARS (ECOSTRESS), prepares Cloud-Optimized GeoTIFFs, runs quality checks, writes STAC sidecars, uploads outputs to GCS, and creates thumbnail contact sheets.

Landsat and Sentinel-2 are regridded to the canonical Berlin grid in **EPSG:25833**. ECOSTRESS is kept in its native projection because the validation product arrives as sparse ISS swath tiles and should be aligned during validation analysis, not during ARD ingestion.

---

## Source-to-Grid Mapping

| Source | Input CRS | Resolution | Processing |
|--------|-----------|------------|------------|
| Landsat 8/9 | EPSG:25833 (GEE export) | 100 m → 100 m | Select explicit WRS-2 path/row (Berlin: 193/23), regrid to canonical 100 m grid, align origin, intersect with AOI |
| Sentinel-2 | EPSG:25833 (GEE export) | 10 m → 10 m | Mosaic all AOI-intersecting tiles per datatake, regrid to canonical 10 m grid, align origin, intersect with AOI |
| ECOSTRESS | Native CRS (UTM zone, EPSG:32632 typical) | ~70 m → ~70 m | Passthrough, no reprojection; COG profile applied |

---

## Pipeline Flow

> **Orchestrator:** `scripts/ard_run.py` is the single entry point for the full pipeline. It chains the phases below via sub-commands (`smoke`, `all`, `plan`, `export`, `process`, `validate`, `doctor`). The individual scripts (`ard_export.py`, `ard_monitor.py`, `ard_process.py`, `ard_smoke_validation.py`) remain usable directly for dev/debugging.

### 1. Export phase

Driver: `scripts/ard_export.py` (called by `ard_run.py export` / `ard_run.py smoke` / `ard_run.py all`)

| Source | Export path | Output prefix | Config |
|--------|-------------|---------------|--------|
| Landsat 8/9 | GEE export via `gee_export.py` | `ard/dynamic/landsat/{year}/` | `configs/ard/gee_export.yaml`, `configs/ard/landsat.yaml` |
| Sentinel-2 | GEE export via `gee_export.py` | `ard/dynamic/sentinel2/{year}/` | `configs/ard/gee_export.yaml`, `configs/ard/sentinel2.yaml` |
| ECOSTRESS | AppEEARS area task via `ecostress_export.py` | `ard/validation/ecostress/{year}/` | `configs/ard/ecostress.yaml` |

GEE exports submit Earth Engine tasks for the configured year and source. ECOSTRESS export submits an AppEEARS area task, downloads the result bundle, converts GeoTIFF layers to COGs, and uploads them to GCS.

Task monitoring is split out into `scripts/ard_monitor.py` (called automatically by `ard_run.py` between export and process).

### 2. Processing phase

Driver: `scripts/ard_process.py` (called by `ard_run.py process` / `ard_run.py smoke` / `ard_run.py all`)  
Config: `configs/ard/ard_process.yaml`

For each selected source and year, the processor:

1. Lists input scenes from GCS with `list_scenes()`, sorted by file size descending.
2. Downloads each scene to `data/tmp/ard_process/`.
3. Reprojects or regrids the raster, or copies ECOSTRESS as a native-CRS COG.
4. Runs QA and writes `{scene_id}_qa.json`.
5. Writes `{scene_id}_stac.json`.
6. Generates a 512 px PNG thumbnail.
7. Uploads COG, QA, STAC, and thumbnail files to GCS.
8. Builds a year-level contact sheet after thumbnails are available.

Landsat and Sentinel-2 use `target_resolution` values from `ard.process.sources`. ECOSTRESS sets `target_resolution: null`, which triggers passthrough mode: source CRS, transform, and geometry are preserved while the COG profile is applied.

---

## GCS Output Structure

```text
ard/processed/{source}/{year}/
├── {scene_id}.tif                    # Processed COG
├── {scene_id}_qa.json                # QA report
├── {scene_id}_stac.json              # STAC metadata
├── thumbnails/{scene_id}.png         # Quicklook
├── _manifest.json                    # Completion tracking
├── _contact_sheet.png                # Year-level overview
```

`source` is one of `landsat`, `sentinel2`, or `ecostress`.

For ECOSTRESS:

| Stage | Prefix |
|-------|--------|
| Input | `ard/validation/ecostress/{year}/` |
| Output | `ard/processed/ecostress/{year}/` |

---

## Canonical Grid

Defined in `src/berlin_lst_downscaling/data/grid_spec.py` and `configs/ard/default.yaml`.

| Property | Value |
|----------|-------|
| CRS | EPSG:25833 (ETRS89 / UTM zone 33N) |
| Origin | `(368000, 5839000)` |
| AOI source | `data/boundaries/berlin_landesgrenze_2km_buffer.geojson` |
| 10 m grid | 4980 × 4150 pixels |
| 100 m grid | 498 × 415 pixels |

The 10 m and 100 m grids share the same origin. The 100 m grid is an exact 10 × 10 aggregate of the 10 m grid. AOI bounds are derived at runtime from the buffered Berlin boundary instead of being hardcoded in config.

---

## Processing Details

### Landsat 8/9

| Property | Value |
|----------|-------|
| Input source | GEE export |
| Input CRS | EPSG:25833 |
| Export scale | 100 m |
| Processing | Regrid to canonical 100 m origin, intersect with AOI bounds |
| Main bands | `ST_B10`, cloud mask, LST plausibility / QA flag |

The GEE export uses `crs="EPSG:25833"` and `scale=100`. Processing does not change the analytical resolution; it snaps the raster to the canonical grid origin and writes a COG with the configured profile.

### Sentinel-2

| Property | Value |
|----------|-------|
| Input source | GEE export |
| Input CRS | EPSG:25833 |
| Export scale | 10 m |
| Processing | Regrid to canonical 10 m origin, intersect with AOI bounds |
| 10 m bands | `B2`, `B3`, `B4`, `B8` |
| 20 m bands | `B5`, `B6`, `B7`, `B8A`, `B11`, `B12` |
| Classification / mask bands | `SCL`, cloud mask |

The GEE export uses `crs="EPSG:25833"` and `scale=10`. Continuous bands use the configured resampling method. Mask and classification bands use nearest-neighbour resampling when band descriptions identify them as `cloud_mask`, `lst_plausible`, or `scl`.

### ECOSTRESS

| Property | Value |
|----------|-------|
| Input source | AppEEARS area task |
| Product | `ECO_L2T_LSTE.002` |
| Input CRS | Native projection, EPSG:32632 typical for Berlin |
| Resolution | ~70 m |
| Processing | Passthrough, no reprojection |
| Layers | `LST`, `cloud`, `QC` as separate GeoTIFF/COG files from AppEEARS |

The processing stage preserves ECOSTRESS source CRS, transform, and geometry. It applies the shared COG profile:

| COG option | Value |
|------------|-------|
| Dtype | `float32` |
| Nodata | `NaN` |
| Compression | `ZSTD` |
| Tile size | 512 × 512 |
| Overviews | 2 levels |
| Overview resampling | `BILINEAR` |

Grid conformity QA is skipped for ECOSTRESS. `qa_passed` is true when at least one valid pixel exists. Single-band COGs report `cloud_fraction: -1.0` because cloud fraction is computed from the last band only when at least two bands are present.

---

## Quality Assurance

Implemented in `src/berlin_lst_downscaling/data/ard_qa.py`.

| Check | Landsat / Sentinel-2 | ECOSTRESS |
|-------|----------------------|-----------|
| CRS match | Must match EPSG:25833 | Skipped |
| Resolution match | Must match target resolution within ±1% | Skipped |
| Origin alignment | Checked against canonical origin with `<1e-6` tolerance | Skipped |
| AOI overlap | Checked against canonical AOI | Skipped |
| Radiometric stats | Per band: min, max, mean, std, nodata percentage | Same |
| Cloud fraction | Last band interpreted as cloud mask | `-1.0` for single-band COGs |
| Pass condition | CRS and resolution checks pass | Any valid pixel exists |

`detect_cohort_outliers()` is available for across-scene radiometric checks. It flags scenes with `|z| > 3` for min, max, mean, or standard deviation in any band/statistic pair. The per-scene processing path writes QA reports but does not fail scenes based on cohort outliers.

---

## STAC Metadata

Implemented in `src/berlin_lst_downscaling/data/stac_writer.py`.

Each processed COG gets a STAC 1.1.0 `Feature` JSON sidecar.

| Field | Content |
|-------|---------|
| `id` | Parsed scene ID |
| `bbox` | Raster bounds |
| `geometry` | Polygon from raster bounds, in the COG CRS used by the writer |
| `properties.datetime` | Parsed acquisition datetime when available |
| `properties.constellation` | Source constellation (`landsat`, `sentinel-2`, `ecostress`) |
| `properties.gsd` | Source ground sampling distance |
| `properties.cloud_fraction` | Value from QA report |
| `properties.sun_azimuth`, `properties.sun_elevation` | Computed from scene center and acquisition datetime when available |
| `properties.proj:transform` | First six affine transform values |
| `assets.cog` | Processed COG URI |
| `assets.qa-json` | QA report URI |

The writer also records `processing:level`, `processing:version`, optional `processing:collection_id`, and a config hash when the resolved Hydra config is provided.

---

## Thumbnails and Contact Sheets

Implemented in `src/berlin_lst_downscaling/data/quicklook.py`.

| Output | Path | Notes |
|--------|------|-------|
| Thumbnail | `thumbnails/{scene_id}.png` | 512 px PNG per scene |
| Contact sheet | `_contact_sheet.png` | Year-level grid of thumbnails, generated per source/year |

Contact sheet generation runs after scene processing for a source/year. It depends on thumbnails already being uploaded. Failures are logged as warnings and do not fail the processing run.

---

## Smoke Mode

Both export and processing scripts support `smoke=true`.

| Script | Behavior |
|--------|----------|
| `scripts/ard_export.py` | Limits GEE export to 1 scene per source; limits ECOSTRESS to 1 month and 1 file |
| `scripts/ard_process.py` | Processes 1 scene per selected source/year, then stops for inspection |

Smoke mode is intended for CI checks and manual validation before full runs.

---

## Resume Mode

`resume=true` is enabled by default in `configs/ard/ard_process.yaml`.

The processor reads `ard/processed/{source}/{year}/_manifest.json` from GCS and skips scene IDs already listed as completed. After each processed scene, it records success or failure. Failed scenes are stored with an error string and processing continues.

---

## Known Limitations

1. **ECOSTRESS sparse swath:** Single ECOSTRESS tiles can cover only a small fraction of Berlin because of the narrow ISS swath. Validation requires accumulating many tiles over the validation period.
2. **ECOSTRESS CRS:** Native CRS is typically EPSG:32632 for Berlin, not the canonical EPSG:25833 grid. Align ECOSTRESS during validation analysis.
3. **ECOSTRESS layer handling:** AppEEARS exports `LST`, `cloud`, and `QC` as separate files. The current GCS scene listing includes GeoTIFFs under the ECOSTRESS prefix; validation code should select the intended LST assets explicitly.
4. **AOI coverage QA:** `aoi_coverage_fraction` is now part of QA. Landsat/Sentinel-2 fail if coverage drops below the configured minimum; ECOSTRESS records coverage but does not gate on it.
5. **Cloud fraction for single-band COGs:** The QA code returns `-1.0` when a raster has fewer than two bands.
6. **Parallel processing:** `max_workers` defaults to 4. Sentinel-2 with 4 workers peaks at roughly 2 GB RAM. Increase only if the machine has enough memory.
7. **Contact sheets:** Contact sheets are generated per source/year after thumbnails are uploaded. Failures are warnings, not fatal errors.

---

## Debug Commands

```bash
# Dry run: show what would be processed
uv run python scripts/ard_process.py source=landsat year=2023

# Smoke test: process 1 ECOSTRESS scene
uv run python scripts/ard_process.py source=ecostress year=2023 smoke=true dry_run=false

# Process all sources for one year; omit source to select all sources
uv run python scripts/ard_process.py year=2023 dry_run=false

# Full configured run across all sources and configured years
uv run python scripts/ard_process.py dry_run=false

# Inspect raster output
rio info gs://berlin-lst-data/ard/processed/ecostress/2023/{scene_id}.tif

# Inspect QA JSON
gcloud storage cat gs://berlin-lst-data/ard/processed/ecostress/2023/{scene_id}_qa.json
```

---

## Related Files

| Path | Purpose |
|------|---------|
| `scripts/ard_export.py` | Export CLI for GEE and AppEEARS |
| `scripts/ard_process.py` | Processing CLI |
| `scripts/ard_monitor.py` | Task monitoring |
| `src/berlin_lst_downscaling/data/ard_processor.py` | Core processing and GCS orchestration |
| `src/berlin_lst_downscaling/data/ard_qa.py` | QA checks |
| `src/berlin_lst_downscaling/data/stac_writer.py` | STAC generation |
| `src/berlin_lst_downscaling/data/quicklook.py` | Thumbnail and contact sheet generation |
| `src/berlin_lst_downscaling/data/grid_spec.py` | Canonical grid specification |
| `src/berlin_lst_downscaling/data/gee_export.py` | GEE export logic |
| `src/berlin_lst_downscaling/data/ecostress_export.py` | ECOSTRESS/AppEEARS export logic |
| `configs/ard/` | Hydra configuration files |
