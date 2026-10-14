# ECOSTRESS C2 collocation feasibility

Read-only inventory for issue #71, accessed 2026-10-14. It does not download
granules, infer a checkpoint, or add a pipeline stage. The scientific task
remains open:
[ECOSTRESS-C2: Validierungsdatenpfad und Kollokation prüfen](https://app.notion.com/p/3f8356451f66816b8b43f69f9db5386e).

Machine-readable counts: `docs/results/ecostress-feasibility/summary.json`.
Full tables:

`gs://berlin-lst-training-data/evaluation/ecostress-feasibility/v1/`

## Question and grain

The question is which published Landsat anchors have an `ECO_L2T_LSTE.002`
overpass close in time whose 100 km tiles meet Berlin.

One anchor row is one Landsat scene in the published v3 manifest
(`gs://berlin-lst-training-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet`,
345 anchors). One granule row is one CMR tile. One overpass is every granule
that shares `time_start`. Two Landsat path/row scenes a few seconds apart are
two anchors and one Landsat overpass.

## What was measured

CMR `granules.json` for `ECO_L2T_LSTE` version 002, bounding box
`[13.08, 52.34, 13.76, 52.68]`, temporal
`2017-05-01T00:00:00Z` to `2026-07-17T23:59:59Z`. No Earthdata login.
`CMR-Hits` was 8794 and 8794 rows were stored. Granule ids are unique.

Overlap is the planar area of the MGRS 100 km square, reprojected to
EPSG:25833, divided by `data/boundaries/berlin_landesgrenze.geojson`
(890.67 km²). An overpass unions the tiles that share `time_start`. This is
tile coverage. It is an upper bound on imaged area, because the square can
extend beyond the swath. It is not a count of valid LST pixels.

The Berlin boundary area above is the same GeoJSON the overlap uses.
`merge_asof` reproduced both the nearest-granule minute and the nearest
overlapping-overpass minute with zero disagreements. Tile corners of the
decoded squares stay within 0.062° of the CMR box.

229 CMR hits are antimeridian false positives (UTM zones 01 and 60). Their
boxes span about 360° of longitude, so a Berlin bbox search returns them.
Their tile intersection with Berlin is 0, and they are excluded from the
time match.

The first granule whose box does not wrap the antimeridian is
`2018-07-30T17:59:19Z`. 48 anchors are earlier than that, including all 28
anchors from 2017.

## Time windows

`±2 hours` is one sensitivity bin. It is not thermal simultaneity. Deltas
use granule `time_start` and the anchor `acquisition_datetime`, both UTC.
Among anchors that have any intersecting overpass inside 6 hours, the
closest delta has minimum 2.25 minutes and median 158 minutes (73 anchors).
272 anchors have no intersecting overpass inside 6 hours.

| Window | Any non-wrapping granule | Intersecting overpass | Overpass covering ≥ 10% |
|--------|-------------------------:|----------------------:|------------------------:|
| 15 min | 5 | 5 | 4 |
| 30 min | 9 | 9 | 7 |
| 60 min | 14 | 14 | 12 |
| 2 h | 27 | 27 | 17 |
| 3 h | 40 | 40 | 30 |
| 6 h | 77 | 73 | 59 |

The ≥10% column is the nearest overpass that covers at least 10% of Berlin,
which is the selection config's `overlap_min`. The intersecting column also
counts tile `33UVT`. That tile meets the boundary on 2169 granules at a
median fraction 0.000295, about 0.26 km². Fourteen of the 27 two-hour
intersections are that sliver.

On four of those fourteen, the next ECOSTRESS scene about one minute later
covers 55% or 100% of Berlin (`2021-06-20`, `2023-08-22`, `2025-06-15`).
Those four are inside the 17 and outside the 13 dates listed in
`within_2h_dates_overlap_at_least_0_10`.

The 17 anchors fall on 10 dates. Several dates contribute two path/row
scenes, so these are about 10 Landsat overpasses, not 17 independent days.
Every two-hour intersection is a CMR `DAY` granule between 10:00 and 13:00
Europe/Berlin. A gap of 15–120 minutes is still inside the diurnal cycle.
Cross-sensor temperature bias was not measured.

All 27 two-hour anchors have published Landsat `aoi_clear_frac` ≥ 0.05.
Nine of the 27 are ≥ 0.50. At most 7 anchors combine Landsat AOI clear
fraction ≥ 0.50 with an ECOSTRESS overpass inside 2 hours that covers ≥ 10%
of Berlin. ECOSTRESS cloud and QC can only remove pairs from that set.

## Code versus this inventory

`data/acquisition/ecostress.py` can download one named `ECO_L2T_LSTE.002`
granule. ARD masking and `scripts/validators/validate_ecostress_scenes.py`
exist for that product.

`configs/selection/_base.yaml` sets `window_hours: 2`. `pipeline.py` never
reads it. Selection resolves six allowlisted granules from 2018-08-25 and
writes `dt_hours` 0 and `overlap_frac` 1. Those six ids are in this CMR
extract, and the three tiles cover about 24%, 55%, and 76% of Berlin. The
nearest published anchor is about 190 hours earlier (`2018-08-17`). The
allowlist is not a collocation.

`data/selection/ecostress_subset.py` is absent. `couple.py` still leaves
the ECOSTRESS list empty for that step. The published manifest contains
those six granules and no other ECOSTRESS rows.

## What this inventory cannot support

ECOSTRESS cloud, QC, and the number of valid thermal pixels were not read.
Scene-level Landsat `cloud_cover` is not the Berlin clear fraction; the
manifest column `aoi_clear_frac` is. No temperature was compared. No
checkpoint was inferred.

## Later comparison, not built here

A later diagnostic can read the small set of overpasses inside 2 hours
whose tiles cover at least 10% of Berlin. The comparison grid is the
delivered ~70 m ECOSTRESS raster, reprojected to EPSG:25833 and area-weighted
onto the evaluation cells. Masks are the Berlin boundary, Landsat QA, and
ECOSTRESS cloud/QC. Report per-overpass MAE, RMSE, and bias, then stratify
by scene and by Stadtstruktur class only after those classes are approved.
Rank pairs by time delta and keep the sub-hour, high-coverage dates first.
If QC leaves only a handful of overpasses, the result is a case study.
That read needs its own approval. It is not a new training run and not a
required stage in the main pipeline.

## Reproduce

```bash
uv run python scripts/operators/inventory_ecostress_feasibility.py \
  --workdir /tmp/ecostress-feasibility \
  --output-root gs://berlin-lst-training-data/evaluation/ecostress-feasibility/v1
```
