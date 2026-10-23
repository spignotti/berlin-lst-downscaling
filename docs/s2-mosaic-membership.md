# Selection v4: exact Sentinel-2 mosaic membership

`configs/selection/full_2017_2026_v4.yaml` enables
`sentinel2.mosaic.enabled: true` and the additive Landsat AOI-overlap gate.
It writes a manifest v4 bundle into its isolated output/checkpoint root.
Legacy configurations continue to write manifest v3.

## Shared acquisition API

Module: `berlin_lst_downscaling.data.acquisition.s2_mosaic_items`

```python
discover_s2_mosaic_items(
    seed_item, *, bbox, collection="sentinel-2-l2a", catalog=None
)  # list of STAC items, unique and sorted by item.id

s2_mosaic_group(item, *, bbox)
# {"solar_day": "YYYY-MM-DD", "datatake_id": "..."}

s2_item_membership(items)
# [{"item_id": "...", "item_href": "..."}, ...], sorted by item_id
```

Discovery consumes all search pages and includes every footprint intersecting
the configured AOI bbox in the exact seed's solar-day/datatake group. Solar day
uses the AOI-bbox center longitude. Separate datatakes remain separate even on
the same day. The ESA datatake processing-baseline suffix (`_Nxx.xx`) is omitted
from acquisition identity; member baselines may differ. The exact seed is
retained if absent from search. Missing identity/hrefs or a failed search raises;
there is no single-tile fallback.

## Manifest and ARD boundary

`MANIFEST_SCHEMA_V4` adds two nullable columns to the legacy schema. Every S2
predictor must populate both; other sources leave them null:

- `s2_items`: nonempty, sorted, unique list of `{item_id, item_href}` records.
- `s2_mosaic_group`: struct containing `solar_day` and `datatake_id`.

The first sorted member is the stable representative `scene_id`/`item_href`.
Pairings reference that representative; each group has one predictor row.
Selection scores precisely this membership, with first footprint-valid SCL in
sorted item-ID order. Finite, nonzero SCL is footprint-valid before cloud
classification. Cloud in an earlier donor is retained even if a later donor
is clear. SCL is loaded separately per item with nearest resampling. The S2
day is independent of the paired Landsat day.

For v4, ARD must resolve **every** `meta["s2_items"]` href with its exact expected
ID and process each item's own reflectance baseline before applying the same
donor order. It must not rediscover, substitute, add, or drop members. Missing
or unresolvable members fail the scene. Completion provenance belongs to ARD.

Legacy v3 remains structurally valid. A legacy ARD compatibility path can use
the shared discovery helper seeded from its exact item, but v3 records neither
the selection mosaic nor its membership. **A fresh v4 selection bundle is
required for the exact selection/ARD membership guarantee** and is recommended
for the immutable v10 ARD root.

The enabled mosaic policy and an explicitly configured
`landsat.anchor.min_aoi_overlap_frac` affect the selection policy fingerprint.
Absent/disabled additive keys preserve legacy hashes. Prefilter Landsat drops
and coupling drops both contribute to report totals and drop reasons; v4 also
records per-anchor reasons and available footprint metrics in `dropped_anchors`.

## Chosen versions

- Sentinel-2 ARD v10 (`configs/ard/sentinel2_mosaic_v10.yaml`, schema 10).
  Legacy schema 9 (`configs/ard/sentinel2_harmonized_v9.yaml`) keeps its old
  single-item behavior and published root.
- Selection/manifest v4 (`configs/selection/full_2017_2026_v4.yaml`,
  `MANIFEST_SCHEMA_V4`). Legacy selection configs keep writing manifest v3.
- The new S2 root is
  `gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10`;
  `publish_guard` additionally blocks schema-10 writes into the published v9
  root and leaves all schema-8 protections unchanged.

## Per-tile harmonization before mosaicking

Each recorded member is loaded alone, scaled with its own STAC processing
baseline (`reflectance_scale_from_item`, `(DN + BOA_ADD_OFFSET) / 10000`,
then `[0, 1]` clip), masked with its own SCL via the existing `mask_s2`
path (including native-20 m masked SWIR reprojection), and only then
composited. Overlaps use whole-donor selection: the first footprint-valid
donor in sorted item-ID order supplies all six bands plus the flag.
Footprint-valid means finite, nonzero SCL before cloud classification, so a
cloudy earlier donor is retained over a clear later donor.

## Coverage gate

Before any COG/STAC/ledger write, the union of member SCL footprints is
compared with the canonical 10 m Berlin AOI. Missing footprint (not cloud)
is measured as `aoi_missing_px / aoi_total_px`, logged as `s2_mosaic_prewrite`,
stored in provenance/coverage, and gated by configurable
`sentinel2.mosaic.max_missing_aoi_fraction` (default `0.005`). Excess missing
coverage, an empty/unreadable AOI, or an invalid threshold fails the scene
and marks the ledger row failed without writing products. A v4 bundle cannot
run under schema 9, schema 10 requires recorded v4 membership, and resume
pins the exact per-scene input contract (members, group, donor/policy rules,
coverage threshold, AOI URI and bytes) in `ledger.schema_hash`.

## Landsat AOI-overlap filter

Selection v4 adds `landsat.anchor.min_aoi_overlap_frac` (default `0.05`):
valid non-fill QA_PIXEL footprint ∩ AOI divided by total AOI area, measured
per exact scene before the unchanged clear-fraction gate. Cloudy pixels count
as footprint coverage; only fill/nodata/nonfinite QA pixels do not. Scenes
below the threshold are excluded with reason `insufficient_aoi_overlap`
(e.g. Landsat path/row 194023). Unavailable QA yields
`aoi_overlap_unavailable`, never a bbox fallback.
