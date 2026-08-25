# Data sources and contracts

Single source-of-truth for the data the pipeline reads and the
contracts it writes. Each section lists the asset and the rules every
pipeline stage must respect.

## Sources

| Source | Adapter | Resolution | Role |
|--------|---------|-----------:|------|
| Landsat-8/9 (PC `landsat-c2-l2`) | `data/acquisition/landsat.py` | 100 m (target_low) | anchor (`landsat-c2-l2` + `role=anchor`) |
| Sentinel-2 L2A (PC `sentinel-2-l2a`) | `data/acquisition/sentinel2.py` | 10 m (SWIR 20 m native) | predictor (`role=predictor`) |
| ECOSTRESS L2T LSTE v002 (NASA CMR) | `data/acquisition/ecostress.py` | 70 m | validation (`role=validation`) |

**Spatial grid.** Canonical 10 m EPSG:25833 over Berlin bbox
`[13.08, 52.34, 13.76, 52.68]`. Helpers live in `common/grid.py`.

**Temporal policy.** May–September. Static-source vintage table — fixed
across all scenes (`geometry_temporal_mode: retrospective_static`):

| Product | Vintage | Notes |
|---------|--------:|-------|
| LoD2 CityGML morphometry | 2024 | `https://gdi.berlin.de/data/a_lod2/atom/` (CityGML v2.0 ZIP per 1 km tile) |
| DGM 1 m terrain height | 2021 | ALS acquisition Feb–Mar 2021 |
| Vegetation height (DOM − DGM) | 2020 | Berlin opacity/WMS, derived |
| Versiegelung (imperviousness) | 2016, 2021 | Hausumringe WMS, piece-wise constant per scene year |

Each scene year maps to a fixed source vintage
(`data/secondary/{lod2,dgm,vegetation_height,imperviousness}.py`).
Per the v3 manifest: 345 Landsat anchors, 509 manifest rows.

### Historical LoD morphometry vintages

The historical runner (`scripts/run_lod_vintages.py`) consumes three
CityGML archives supplied by the Senatsverwaltung Berlin — LoD1 (2017),
LoD2 (2021), and LoD2 (2022) — and publishes canonical-grid
`lod2_morphology/{2017,2021,2022}` products. The 2017 vintage applies
the LoD2-2021 stock filter against the LoD1-2017 footprints
(50% minimum footprint overlap) so only buildings present in 2017
survive, using the more detailed 2021 roof geometry. Vintages 2024 and
later remain served by the ATOM-feed product.

These are additional Static A/B vintages — once reconciled into the
Static ledgers, they are treated identically to the other source/derived
products. The carry-forward mapping wires per-scene geometry selection
for the Dynamic pipeline.

#### Archive contract

Raw archives live as one ZIP per vintage in GCS Standard storage.
The runner never expands these in GCS; it streams each ZIP into a
local temp directory once per vintage (context-managed; deleted on
exit) and iterates it as a regular ZipFile. No expanded XML ever
appears in the bucket.

```
gs://berlin-lst-data/lod_vintages/<vintage>/<archive_filename>
    ├─ LoD1_2017.zip         (1006 XML members, ~190 MB)
    ├─ LoD2_BE_1_33_2021.zip (928 XML members, ~2.5 GB)
    └─ LoD2_2022.zip         (928 XML members, ~1.7 GB)
```

Each archive carries an SHA-256 hash and the full member list, both
recorded in `<source_root>/ard/static/sources/lod_vintages/raw_manifest_<vintage>.json`.
The 2017 morphology provenance includes **both** input archive
manifests (LoD1-2017 + LoD2-2021).

Year → vintage carry-forward mapping is published at
`<metadata_root>/geometry_mapping.json` (default
`gs://berlin-lst-data/static/geometry_vintages/v1/`); the runner only
writes it after every requested vintage morphology AND every required
derived product has finalised, so the artefact can never point at
half-published state.

| Year | Vintage | geometry_id |
|------|--------:|-------------|
| 2017–2020 | 2017 | `dgm1-2021__lod2-2017__vh-2020` |
| 2021 | 2021 | `dgm1-2021__lod2-2021__vh-2020` |
| 2022–2023 | 2022 | `dgm1-2021__lod2-2022__vh-2020` |
| 2024–2026 | 2024 | `dgm1-2021__lod2-2024__vh-2020` (existing ATOM-feed product) |

The Dynamic pipeline resolves each scene's LoD vintage from
`geometry_mapping.json` per scene year and applies the matching
building-shadow horizon (`data/dynamic/geometry.py`).

#### Runner commands

The full LoD-vintage runner, reconcile, and derive-only commands live in
the README operations section (they are operational instructions, not
contracts).

## Manifest bundle (v3)

The canonical bundle is the only accepted manifest contract.
The reader fails fast on any other layout.

```
gs://berlin-lst-data/manifests/v3/<bundle-id>-r2/
    manifest.parquet       (schema_version 3)
    pairings.parquet       (schema_version 1)
    manifest_report.json   (publication gate)
```

Immutable history: `…-<bundle-id>/` (without `-r2`) is retained as a
historical record; never a canonical reference.

Read all three files together via
`data.selection.validate.load_bundle(manifest_uri)`. The helper validates
schemas, hashes, metadata, report counts, FK consistency, and the
count/fraction round-trip invariant in pairings.

**Manifest schema (v3).** Primary key `(scene_id, source)`. Required
fields: `scene_id`, `source`, `role` (`anchor|predictor|validation`),
`platform`, `year`, `acquisition_datetime` (UTC ts), `item_href`
(nullable for ECOSTRESS), `aoi_clear_frac` (≥ 0.05 for non-validation),
`cloud_cover`, `solar_azimuth`, `solar_elevation`. Landsat restricted to
`landsat-8`/`landsat-9`.

**Pairings schema (v1).** Primary key `landsat_scene_id`. Required
fields: `sentinel2_scene_id`, `dt_seconds`, `landsat_clear_px > 0`,
`joint_clear_px ∈ [0, landsat_clear_px]`, `joint_clear_frac ∈ [0, 1]`,
`score`. Invariant enforced: `np.float32(joint_clear_px / landsat_clear_px)`
must round-trip exactly through float32 to equal `joint_clear_frac`.

**Ledger roles.** ARD ledger carries scene+source rows. Per-pipeline
ledger carries `(item_id, source, period_or_vintage)` rows:
- ARD: rows are scene-centric, status tracks scene lifecycle.
- Static sources: per `(source, vintage)` item.
- Static derived: per `(product, geometry_id)` item.
- Dynamic: per scene, with `role ∈ {anchor, inference}`.

## Product contract

Every product is published as four co-located files:

```
<root>/
    <name>.tif             # Cloud-Optimised GeoTIFF
    <name>.stac.json       # STAC 1.0.0 Item
    provenance.json        # source/transform hash + upstream ids
    complete.json          # written last — publication gate
```

Per-pipeline root shape (exact layout from `data/ard/paths.py`,
`data/secondary/paths.py`, `data/dynamic/paths.py`):

- Static sources `<source_root>/ard/static/sources/<source>/<vintage>/<source>_<vintage>.tif`
- Static derived `<derived_root>/ard/static/derived/<product>/<geometry_id>/<product>_<geometry_id>.tif`
- ARD `<output_root>/<source>/<year>/<scene_id>/<scene_id>.tif`
  (+ `<scene_id>.flag.tif`, `<scene_id>.stac.json`, `provenance.json`, `complete.json`)
- Dynamic `<output_root>/ard/dynamic/<era5_land|shadow_building|shadow_vegetation>/<scene_id>/<source>_<scene_id>.tif`

### ARD flag band

Each ARD scene carries a separate single-band uint8 flag COG
(`<scene_id>.flag.tif`). Pixels are flagged, never silently deleted —
downstream consumers (joint validity masks, QA) treat any
non-zero flag as invalid. Bit layout (`data/ard/contract.py`).
Schema versions are per-source: Landsat v6, ECOSTRESS v7, Sentinel-2 v8
(six-band spectral contract, see below):

| Bit | Value | Flag | Sources |
|-----|------:|------|---------|
| 0 | 1 | fill | all (S2 SCL 0, Landsat QA fill/DN=0, ECOSTRESS cloud/water/QC fill) |
| 1 | 2 | cloudy | all |
| 2 | 4 | cloud shadow | S2 SCL 3 (+ directional projection), Landsat QA bit 4 |
| 3 | 8 | cirrus | S2 SCL 10, Landsat QA bit 2 |
| 4 | 16 | saturated | S2 SCL 1, ECOSTRESS QC degraded |
| 5 | 32 | snow/ice | S2 SCL 11 |

Sentinel-2 SCL class 11 (snow/ice) is flagged and excluded from clear
pixels — matching the selection layer, which already treats SCL 11 as
not clear. AOI metrics count it as `aoi_snow_ice_px` and include it in
`aoi_total_px`; `aoi_clear_px` counts only pixels with no flag bit set.

Land/Imperviousness products accept exact COG contracts via
`data.ard.contract.Contract` (`data.secondary.contract.Contract` was
consolidated into the ARD contract layer); each `BandSpec` carries
`valid_range` enforced by `validate_secondary_cog`. Dynamic products
embed scene `role` on the ledger row that publishes them.

### Sentinel-2 spectral group (6 bands)

Each `sentinel-2-l2a` data COG contains exactly these float32/NaN
bands in this order:

| Band | Native res. | Published res. | Notes |
|------|------------:|---------------:|-------|
| `B02` (blue), `B03` (green), `B04` (red), `B08` (NIR) | 10 m | 10 m | scaled reflectance [0,1] |
| `B11` (SWIR1), `B12` (SWIR2) | 20 m | 10 m | masked at 20 m, bilinearly resampled to 10 m |

SWIR handling: B11/B12 are masked at their native 20 m resolution
(any non-zero SCL flag → NaN) **before** the bilinear 20→10 m
resampling. NaN is declared as nodata, so invalid SWIR samples are
excluded from the interpolation kernel: NaN propagates through the
bilinear kernel from invalid 20 m samples (roughly the cell's 2×2 10 m
footprint, irregular at adjacent invalid cells), and valid neighbours
interpolate cleanly. Unlike the 10 m bands (NaN only on fill), SWIR
is NaN for every flagged class — this prevents cloudy/fill reflectance
from entering the interpolation. The 20 m mask uses the direct SCL
classes only; the directional cloud-shadow projection is a 10 m-only
enhancement. Consumers that need shadow-clean SWIR must consult the
flag band (`flag == 0`), which already excludes projected shadows.
STAC `spatial_resolution` reports 10 m for all bands; the native 20 m
origin of B11/B12 is carried in the BandSpec description and
provenance, not in the raster metadata.

The S2 schema version is 8; `full_sentinel2_swir` reprocesses exactly
the Sentinel-2 rows (schema change) and leaves Landsat/ECOSTRESS rows
at their published versions.

### ERA5-Land weather group (8 bands)

Each `era5_land` COG contains exactly these float32/NaN bands:

| Band | Unit | Derivation |
|------|------|------------|
| `t2m_scene` | K | Instantaneous 2m temperature at acquisition hour |
| `ssrd_scene` | W/m² | Hourly SSRD at acquisition hour (ECMWF differencing) |
| `ssrd_antecedent_72h_mean` | W/m² | 72-hour rolling mean of hourly SSRD |
| `vpd_scene` | kPa | Vapour pressure deficit (Tetens formula, t2m−d2m) |
| `wind_speed_10m_scene` | m/s | Magnitude of u10/v10 |
| `tp_0_24h` | mm | Total precipitation, 24h ending at acquisition |
| `tp_24_48h` | mm | Total precipitation, 24–48h before acquisition |
| `tp_48_72h` | mm | Total precipitation, 48–72h before acquisition |

All temporal quantities are derived on the native ERA5 0.1° grid **before**
bilinear reprojection to the canonical 10m grid.  The three precipitation
bins are non-overlapping 24-hour intervals.

Monthly NetCDF inputs are cached under
`<output_root>/_raw/dynamic/era5_land/YYYY-MM/` (`data/dynamic/era5.py`).
The cache is a required raw-input cache for the active Dynamic pipeline:
re-runs reuse cached months, and each cache file is validated before use.

### Shadow products

Each `shadow_building` / `shadow_vegetation` COG is uint8: 0=lit,
1=shadowed, 255=nodata.  Building shadows use the carry-forward LoD
vintage per scene year (via `geometry_mapping.json`).  Vegetation shadows
use the fixed VH-2020 horizon.

### Scene-before-patch split invariant

Scene IDs must be assigned to train/validation/test **before** creating
or sampling patches.  Every patch inherits its scene's split.  One scene
cannot occur in multiple splits.  Split policy and sampler enforcement
belong to a future task (WB2c-4).

### Stage-1 raw-input QA gate

The Stage-1 gate (`scripts/run_qa_stage1_raw.py`, core in
`data/qa/`) validates the published raw inputs of the training universe
before feature engineering. Contract:

- **Scope.** All 324 paired Landsat anchors with `role=anchor` dynamic
  products (2017-2025). The 21 paired 2026 anchors are `role=inference`
  scenes, outside the training universe, and are reported as explicit
  exclusions in the QA report — never dropped silently.
- **Checks.** COG/grid contracts (CRS EPSG:25833, canonical nested
  10/100 m grids, band count, dtype, origin) without full-band reads;
  ARD flag semantics (`flag == 0` is clear, `data/ard/aoi.py`); fixed
  physical ranges (Landsat LST 150-400 K, S2 reflectance 0-1, ERA5
  per-band ranges, shadows 0/1 with 255 nodata, static derived DSM/SVF
  ranges).
- **Joint support (diagnostic).** A 100 m target cell is counted as
  *all-100 supported* when the Landsat LST cell is valid and all 100
  corresponding 10 m subpixels of S2, static morphology (per-vintage
  DSM/SVF), ERA5-Land, and both shadows are valid. Edge-truncated cells
  (fewer than 100 present subpixels) can never be all-100. No
  resampling and no invented subpixels.
- **No mask policy.** The gate computes support as a statistic and
  persists only `summary.json`, `scenes.parquet`, `scenes.csv`, and
  logs under `qa/stage1_raw/<run-id>/`. It never writes a validity or
  selection mask. The final training-eligibility mask is a separate,
  later artifact decided after feature computation and the Stage-2 gate
  (user decision, planning session 2026-08-14).
- **Fail closed.** Any contract/range/input finding inside the training
  universe makes the run exit non-zero (hard finding blocks feature
  computation). Exclusions and 2026 scenes are reported, not failures.

### Scene feature stacks (28-band, 10 m)

Per paired Landsat anchor (2017-2025, 324 scenes), the features pipeline
(`scripts/run_features.py`, core in `data/features/`) publishes one
canonical-grid 10 m stack plus a co-registered validity mask:

```
<root>/<scene_id>/                         (root = gs://berlin-lst-data/features/v3)
    ├─ <scene_id>.tif                     # 28-band float32 COG
    ├─ <scene_id>.feature_valid.tif       # uint8 0/1 validity-mask COG
    ├─ <scene_id>.stac.json               # STAC Item (data + mask assets)
    ├─ provenance.json                    # inputs, policy, coverage
    └─ complete.json                      # publication marker (written last)
```

**Fixed 28-channel order** (the model input interface; reordering is a
breaking schema change → new URI version):

| # | Channel | Family | Unit | Range |
|---|---------|--------|------|-------|
| 1-6 | `B02`, `B03`, `B04`, `B08`, `B11`, `B12` | S2 ARD reflectance | 1 | [0, 1] |
| 7 | `ndvi` = (B08−B04)/(B08+B04) | index | 1 | [−1, 1] |
| 8 | `ndwi_mcfeeters` = (B03−B08)/(B03+B08) | index | 1 | [−1, 1] |
| 9 | `ndbi` = (B11−B08)/(B11+B08) | index | 1 | [−1, 1] |
| 10 | `s2_broadband_albedo` = 0.2266·B02+0.1236·B03+0.1573·B04+0.3417·B08+0.1170·B11+0.0338·B12 | index | 1 | [0, 1] |
| 11 | `building_height_mean` | morphology (LoD2) | m | [0, 200] |
| 12 | `building_height_std` | morphology (LoD2) | m | [0, 100] |
| 13 | `building_coverage_ratio` | morphology (LoD2) | 1 | [−0.01, 1.01] |
| 14 | `building_height_max` | morphology (LoD2) | m | [0, 200] |
| 15 | `vegetation_height_mean` | morphology (VH 2020) | m | [−0.01, 400.01] |
| 16 | `vegetation_height_max` | morphology (VH 2020) | m | [−0.01, 400.01] |
| 17 | `imperviousness` | morphology (Versiegelung) | % | [−0.01, 100.01] |
| 18 | `svf` | morphology (derived) | 1 | [−0.01, 1.01] |
| 19-26 | `t2m_scene`, `ssrd_scene`, `ssrd_antecedent_72h_mean`, `vpd_scene`, `wind_speed_10m_scene`, `tp_0_24h`, `tp_24_48h`, `tp_48_72h` | ERA5-Land | K / W/m² / kPa / m/s / mm | per-band |
| 27-28 | `shadow_building`, `shadow_vegetation` | shadow | 1 | {0, 1} |

- The albedo channel is a **six-band shortwave-reflectance proxy**
  (Bonafoni & Sekertekin 2020, IEEE GRSL 17(9):1618-1622; coefficients as
  published in the ALBEDO implementation, Zenodo 10.5281/zenodo.21111867),
  not a BRDF-corrected physical albedo. Applied to ARD reflectance
  already scaled to [0, 1]; the coefficients sum to 1.
- **Morphology channels are semantic predictors read directly from the
  static source products** (`lod2_morphology` bands 1-4 per the scene's
  LoD vintage via `geometry_mapping.json`, `vegetation_height/2020`
  bands 1-2, and `imperviousness` band 1 per scene year: 2017-2019 →
  vintage 2016, 2020-2025 → vintage 2021); only `svf` remains a derived
  product. The DSMs (`building_dsm`, `vegetation_dsm`, `combined_dsm`)
  are **not model inputs** — they remain internal auxiliaries for SVF,
  horizon, and shadow computation.
- The two vegetation-height bands use the **fixed vegetation_height/2020
  carry-forward**: the source COG of the 2024 geometry profile applies to
  every scene year. The policy is recorded in provenance as
  `vegetation_height_policy`.
- Shadows enter the stack as float 0/1; the uint8 nodata value 255 is
  cast to NaN.

**Validity mask.** `feature_valid == 1` iff the pixel is inside the exact
Berlin AOI (`data/boundaries/aoi_10m.tif`, reprojected onto the canonical
grid with nearest resampling — its own 10 m window is offset from the
canonical lattice), the S2 ARD flag band is `0` (clear), and all 28
channels are finite and within their declared ranges. Availability is
**per channel**: an unavailable channel is NaN in that band only — e.g.
the S2 bands and their derived indices under a non-clear flag, or a
genuine source-data gap in a morphology product — while the independently
known values of the other channels are preserved in the stack. Only
outside the AOI are all 28 channels NaN. The mask is the aggregate
"complete 28-channel vector" availability and is *not* a
training-eligibility mask: it excludes the Landsat target validity and
cannot authorize training selection — publication of
`training_eligible@100m` is a WB2c-4 (training-data preparation) decision.
Landsat LST (100 m) and ECOSTRESS (70 m) never enter the stack.

**LoD2 no-building semantics.** The four LoD2 channels
(`building_height_mean/std/max`, `building_coverage_ratio`) are `0` in
source-covered cells without a building and `NaN` in cells outside the
LoD2 source-tile coverage. Coverage is reconstructed at composition time
from the immutable raw archive manifests (2017/2021/2022) and the LoD
provenance tile receipts (2024) — see `data/features/lod_coverage.py`;
finite source values always win over the coverage classification. The
feature config hash includes the static-sources ledger fingerprint and
the per-vintage coverage-evidence fingerprints, so any change to these
inputs invalidates the published stacks.

**Sparse-support retention.** Structurally valid stacks are published even
when sparse or nearly empty. The generic ARD minimum-non-NaN check is
disabled for feature stacks (their validity is governed by the companion
mask, not by bulk non-NaN density); the mask/data pair is still validated
pixelwise (`mask == 1 ⇔ all 28 channels finite`, an all-zero mask is
valid). Scenes with low S2 clear-sky coverage produce few valid pixels and
are retained as diagnostic evidence, not excluded. Sparse support is
reported (non-gating) in the isolated-run summary as
`sparse_support_below_1pct` (fraction `feature_valid_px / inside_aoi_px
< 1%`) and does not by itself disqualify a stack — the training-eligibility
decision belongs to WB2c-4 (training-data preparation).

**Provenance and coverage.** Each `provenance.json` records the channel
order, config hash, AOI URI + fingerprint, the vegetation carry-forward
policy, and per-scene coverage: `total_px`, `inside_aoi_px`,
`outside_aoi_px`, `feature_valid_px`, `aoi_frac`, and
`feature_valid_frac_of_aoi`. The run report (`<root>/qa/features/<run-id>/report.json`)
aggregates these across scenes. The config hash covers the manifest,
geometry-mapping, and ledger fingerprints plus the AOI fingerprint, the
vegetation carry-forward geometry, and the full channel schema (names,
formulas, weights, ranges) — any change invalidates published stacks via
`config_changed` reconcile.

**Ledger and idempotency.** Rows live in
`<root>/_state/features/ledger.parquet` (keyed `feature_stack` +
`period_or_vintage = scene_id`) with the same `complete.json` visibility
gate as every other product. The full run uses per-scene subprocess
isolation (`scripts/run_features_isolated.py`) because one full-grid
scene peaks at ~7-8 GB of RAM; a fresh interpreter per scene bounds
memory and makes the run resume-safe.

## Reproducibility boundary

The pipeline is manifest-driven, idempotent, and deterministic in
layout: the immutable r2 bundle, SHA-256-pinned LoD archives, and the
ERA5 monthly cache (`_raw/dynamic/era5_land/`) let a fresh run
reproduce the published product set with the same COG/STAC/provenance
structure.

It is **not** byte-exact reproducible from an empty bucket:

- ARD/ECOSTRESS inputs are fetched live from PC STAC and NASA CMR at
  run time; published provenance records no input asset hashes.
- Ledger, provenance, STAC, and logs embed per-run `run_id` values and
  wall-clock timestamps.
- The published ARD ledger holds per-source schema versions: 345
  Landsat rows at v6, 6 ECOSTRESS rows at v7, 158 Sentinel-2 rows at
  v8. A fresh full run rewrites the older rows deterministically.

Validators and the pinned bundle/caches, not byte identity, are the
reproducibility basis.
