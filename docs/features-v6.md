# features/v6 channel roster

The published features/v5 product stays the 28-channel contract.
`FEATURE_SCHEMA_VERSION` is 4 after the data-correctness bump and is
still that 28-channel roster. features/v6 is schema version 5. The
roster and its width come from `feature_schema(5)` in
`data/features/contracts.py`. Do not hard-code the width.

Order is spectral, index, morphology, ERA5, shadow.

| # | Channel | Family | Notes |
|---|---------|--------|-------|
| 1 | `B02` | spectral | unchanged |
| 2 | `B03` | spectral | unchanged |
| 3 | `B04` | spectral | unchanged |
| 4 | `B08` | spectral | unchanged |
| 5 | `B11` | spectral | unchanged |
| 6 | `B12` | spectral | unchanged |
| 7 | `ndvi` | index | unchanged |
| 8 | `ndwi_mcfeeters` | index | unchanged |
| 9 | `ndbi` | index | unchanged |
| 10 | `s2_broadband_albedo` | index | unchanged |
| 11 | `building_height_mean` | morphology | unchanged |
| 12 | `building_coverage_ratio` | morphology | unchanged |
| 13 | `vegetation_height_mean` | morphology | unchanged |
| 14 | `canopy_cover_fraction` | morphology | replaces `vegetation_height_max` |
| 15 | `imperviousness` | morphology | unchanged |
| 16 | `svf` | morphology | radius still configurable; default 30 m |
| 17 | `t2m_scene` | era5 | unchanged quantity |
| 18 | `ssrd_scene` | era5 | unchanged quantity |
| 19 | `wind_speed_10m_scene` | era5 | unchanged quantity |
| 20 | `tp_0_72h` | era5 | sum of the three precipitation windows; `log1p` then z-score |
| 21 | `shadow_building` | shadow | unchanged |
| 22 | `shadow_vegetation` | shadow | unchanged |

Removed from the v6 model input: `building_height_std`, `building_height_max`,
`vegetation_height_max`, `ssrd_antecedent_72h_mean`, `vpd_scene`, `tp_0_24h`,
`tp_24_48h`, `tp_48_72h`.

`building_height_std` and `building_height_max` stay on the LoD source
product because the building DSM still needs the max band.
`vegetation_height_max` stays on the vegetation source product because the
vegetation DSM reads it by description. `canopy_cover_fraction` is an
extra band on that product. The canopy threshold defaults to 2.5 m and is
provisional (`canopy_cover_threshold_m` in `configs/features/v6.yaml` and
`configs/static_sources/v6.yaml`).

ERA5 representation is a loader switch: `scene_value` (one value per
scene, broadcast; v6 default) or `bilinear` (the published interpolated
field). `scene_value` is the mean of the bilinear 10 m field inside the
exact Berlin AOI. The dynamic/v6 writer bilinear-reprojects and then
takes that mean. Composing from the published 8-band COG uses the same
mean, so the two paths agree. A scene-value COG is already constant, and
applying the mean again leaves the scalar unchanged. The published
dynamic product stays 8-band bilinear. `configs/dynamic/v6.yaml` writes
the new `dynamic/lod2-area-exact-v2` prefix.

SVF `svf_max_radius` stays 3 cells (30 m) in `configs/static_derived/v6.yaml`.
`scripts/validators/diagnose_svf_radius.py` compares 30/50/100/200 m on a
sample and does not write a product.

## Artifacts to regenerate

This change does not publish. `publish_guard` refuses Sentinel-2 writes
into the published schema-8 roots and feature writes into `features/v3`,
`features/v4`, and `features/v5`. Do not write `static/sources/imperviousness-v3`.

| Artifact | Why |
|----------|-----|
| `ard/full/2017-2026-s2-harmonized-v9` | schema-9 S2 ARD. The stack does not apply a second offset |
| `static/sources/lod2-area-exact-v2` | area-exact LoD2 morphology, plus vegetation height with a third `canopy_cover_fraction` band. Mean and max stay. Imperviousness objects stay the published v3 COGs; the new ledger points at them |
| `static/derived/lod2-area-exact-v2` | building DSM, combined DSM, building horizons, and SVF from the corrected heights. SVF radius stays 30 m unless the diagnostic changes it |
| `dynamic/lod2-area-exact-v2` | building shadows from the corrected horizons. ERA5 `scene_value` may be written here or applied at compose time to an 8-band COG; both use the AOI mean of the bilinear field |
| `features/v6` | 22-channel roster on the roots above |
| `training/v3` | eligibility, train-only scaler (`tp_0_72h` is `log1p` then z-score), patch index |
| EDA, random forest, ablation | new channels and a new training release. Frozen result write-ups stay as they are |

`building_height_std` and `building_height_max` stay on the LoD product
for the building DSM. Dropping them from the model input does not, by
itself, require a morphology republish. The area-exact LoD2 rebuild does.
