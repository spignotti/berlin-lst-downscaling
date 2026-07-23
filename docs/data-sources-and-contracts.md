# Data sources and contracts

Single source-of-truth for the data the pipeline reads and the
contracts it writes. Each section lists the asset and the rules every
pipeline stage must respect.

## Sources

| Source | Adapter | Resolution | Role |
|--------|---------|-----------:|------|
| Landsat-8/9 (PC `landsat-c2-l2`) | `data/acquisition/landsat.py` | 100 m (target_low) | anchor (`landsat-c2-l2` + `role=anchor`) |
| Sentinel-2 L2A (PC `sentinel-2-l2a`) | `data/acquisition/sentinel2.py` | 10 m | predictor (`role=predictor`) |
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
| Versiegelung (imperviousness) | 2021 | Hausumringe WMS, piece-wise constant per scene year |

Each scene year maps to a fixed source vintage
(`data/secondary/{lod2,dgm,vegetation_height,imperviousness}.py`).
Per the v3 manifest: 345 Landsat anchors, 509 manifest rows.

## Manifest bundle (v3)

The canonical bundle is the only accepted manifest contract.
The reader fails fast on any other layout.

```
gs://berlin-lst-data/manifests/v3/<bundle-id>/-r2/
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

Per-pipeline root shape:

- Static sources `<source_root>/<source>/<vintage>/<name>.tif`
- Static derived `<derived_root>/<product>/<geometry_id>/<name>.tif`
- ARD `<output_root>/<source>/<scene_id>/<name>.tif`
- Dynamic `<output_root>/<era5_land|shadow_building|shadow_vegetation>/<scene_id>/<name>.tif`

Land/Imperviousness products accept exact COG contracts via
`data.ard.contract.Contract` (`data.secondary.contract.Contract` was
consolidated into the ARD contract layer); each `BandSpec` carries
`valid_range` enforced by `validate_secondary_cog`. Dynamic products
embed scene `role` on the ledger row that publishes them.
