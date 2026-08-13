# Phase 1 delivery

Records what the preprocessing phase built and published: the pipeline
graph, the delivered data products on GCS, their metadata and stack
interface, and the handoff state for phase 2. Operational commands,
smoke gates, and logging live in the README; normative source and
artefact contracts live in `data-sources-and-contracts.md`.

## Purpose and scope

Phase 1 turns a manifest of Landsat, Sentinel-2, and ECOSTRESS scenes
into a training-ready COG stack on Google Cloud Storage, plus an
independent DWD sanity check on the ERA5-Land channel. All production
paths are in `gs://berlin-lst-data/`; the VM runs the pipelines, the
bucket is the source of truth.

## Pipeline graph

```text
Selection (build_manifest → publish_manifest)
  └─ manifests/v3/<bundle>-r2/
       ├── ARD (run_ard)                  → ard/full/<cutoff>/
       └── Dynamic (run_dynamic_isolated) → dynamic/full/, dynamic/inference/2026/
              └── DWD validation (run_dwd_validation)  → dwd_validation/r3/ (external check only)

Static sources (run_static_sources)   → static/sources/full/
  └─ Static derived (run_static_derived) → static/derived/full/
       └── consumed by Dynamic (geometry_mapping.json)
```

Every product publishes four co-located artefacts (COG, STAC Item,
`provenance.json`, `complete.json`); `complete.json` is written last
and is the only publication gate. ARD scenes additionally publish a
uint8 flag COG.

## Delivered data products

All counts are read from the published ledgers and the manifest report
at the time of this record.

| Pipeline | GCS root | Grain / layout | Counts | Role semantics |
|----------|----------|----------------|-------:|----------------|
| Manifest bundle (canonical) | `manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/` | one row per scene, three Parquet/JSON artefacts | 509 rows, 345 pairings | `anchor` (345 Landsat), `predictor` (158 S2), `validation` (6 ECOSTRESS) |
| ARD | `ard/full/2017-2026-cutoff-20260717T235959Z/<source>/<year>/<scene_id>/` | one scene dir with COG + flag + STAC + provenance + complete | 509 done | same as manifest |
| Static sources (A) | `static/sources/full/ard/static/sources/<source>/<revision>/` | one dir per (source, vintage) | 8 ledger rows | — |
| Static derived (B) | `static/derived/full/ard/static/derived/<product>/<geometry_id>/` | one dir per (product, geometry_id) | 18 ledger rows | — |
| Geometry mapping | `static/geometry_vintages/v1/geometry_mapping.json` | year → geometry_id | 10 year entries | — |
| Dynamic full | `dynamic/full/ard/dynamic/<source>/<scene_id>/` | per scene: era5_land + shadow_building + shadow_vegetation | 972 done = 324 scenes × 3 | `role=anchor` |
| Dynamic inference | `dynamic/inference/2026/ard/dynamic/<source>/<scene_id>/` | same layout | 63 done = 21 scenes × 3 | `role=inference` |
| DWD validation r3 | `dwd_validation/r3/runs/dwd/9d5269f5/` | per-anchor comparison + report | 345 anchors | external validation only |
| AOI | `boundaries/aoi_10m.tif`, `aoi_100m.tif`, `berlin_landesgrenze.geojson` | — | — | — |
| LoD raw archives | `lod_vintages/{2017,2021,2022}/` | one ZIP per vintage | 3 archives | — |

Retained QA evidence (`qa/repairs/s2-snow-ice-20260811T215612/`,
`qa/wb2c-2/`, `profiling/wb2c-1/`) documents one-off repairs and
profiling. It is not a training input.

## Metadata and stack interface for phase 2

- **Ledgers** enumerate every delivered product with its artefact URIs:
  - ARD: `ard/full/<cutoff>/ledger.parquet` — scene-centric, status tracks scene lifecycle.
  - Static sources: `static/sources/full/ledger.parquet` — per `(source, vintage)`.
  - Static derived: `static/derived/full/_state/static/derived/ledger.parquet` — per `(product, geometry_id)`.
  - Dynamic: `dynamic/<root>/_state/dynamic/ledger.parquet` — per scene with `role ∈ {anchor, inference}`.
- **Per-product artefacts**: COG (`<name>.tif`), STAC Item
  (`<name>.stac.json`), `provenance.json` (source/transform hashes +
  upstream ids), `complete.json` (publication gate). ARD scenes add
  `<scene_id>.flag.tif` (uint8 bitmask, `data/ard/contract.py`).
- **ERA5-Land group**: exactly 8 float32/NaN bands per `era5_land` COG
  (t2m_scene, ssrd_scene, ssrd_antecedent_72h_mean, vpd_scene,
  wind_speed_10m_scene, tp_0_24h, tp_24_48h, tp_48_72h).
- **Shadow products**: uint8 COGs, 0=lit, 1=shadowed, 255=nodata.
  Building shadows use the carry-forward LoD vintage per scene year
  (via `geometry_mapping.json`), vegetation shadows the fixed
  vegetation-height-2020 horizon.
- **Dynamic role policy**: `role=inference` products are excluded from
  training/validation/test; only `role=anchor` products are consumed as
  targets. Scene-before-patch split assignment is a phase-2 concern
  (`data-sources-and-contracts.md`).

## Reproducibility boundary and known limitations

The pipeline is manifest-driven, idempotent, and deterministic in
layout: the immutable r2 bundle, SHA-256-pinned LoD archives, and the
ERA5 monthly cache (`_raw/dynamic/era5_land/`) let a fresh run
reproduce the published product set with the same structure. It is
**not** byte-exact reproducible from an empty bucket — see
`data-sources-and-contracts.md` for the full boundary.

Known limits of the delivered state:

- The published ARD ledger is a deliberate mix of 434 schema-v6 + 75
  schema-v7 rows; current code writes v7, so a future full `run_ard`
  deterministically rewrites the v6 rows.
- DWD r3 is **historical**: it validated the pre-rebuild scalar ERA5
  products (run `dyn-20260721T092945-4a4de9`). It is not evidence for
  the current 8-band spatial ERA5 fields.

## QA status and handoff

- All canonical products validate: manifest 509/345, ARD 509/509,
  dynamic 972/972 + 63/63, static ledgers 8/18, LoD archives intact.
- DWD r3 metrics (historical): bias −0.03 °C, MAE 0.77 °C,
  RMSE 0.98 °C across 1 508 matched pairs (562 DWD-missing, 0
  ERA5-missing).
- **Open QA point (phase 2, not fixed here):** a post-rebuild DWD
  rerun against the current dynamic products is still required. The
  previous session's handoff reported a suspected UTC/timezone join
  mismatch in `data/dynamic/dwd_validation.py`; this could not be
  re-verified in the current bucket (only r3 exists) and must be
  reproduced and diagnosed before any DWD metric update.

## Documentation map

- `README.md` — pipeline operations, production and validation commands, smoke matrix.
- `data-sources-and-contracts.md` — sources, canonical grid, manifest/ledger/artefact contracts.
- This file — delivered products, metadata interface, handoff state.
