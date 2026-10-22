# features/v6 channel roster

features/v5 (schema version 3) stays the published 28-channel contract.
features/v6 is schema version 4. The roster and its width come from
`feature_schema(4)` in `data/features/contracts.py`. Do not hard-code the
width.

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
field). The published dynamic product stays 8-band bilinear unless
`configs/dynamic/v6.yaml` is used.

SVF `svf_max_radius` stays 3 cells (30 m) in `configs/static_derived/v6.yaml`.
`scripts/validators/diagnose_svf_radius.py` compares 30/50/100/200 m on a
sample and does not write a product.

## Artifacts to regenerate

This change does not publish. A later republish, after the data-correctness
fixes land, needs:

| Artifact | Why |
|----------|-----|
| `static/sources/v6` vegetation height | third band `canopy_cover_fraction`; mean and max stay |
| `dynamic/v6` ERA5 (optional) | four bands and `scene_value`. Compose can instead reduce the published 8-band COG |
| `static/derived` SVF | only if the radius changes after the diagnostic. The default 30 m does not |
| `features/v6` | new roster, canopy band, ERA5 reduction and representation |
| `training/v3` | eligibility, train-only scaler (`tp_0_72h` is `log1p` then z-score), patch index |
| EDA, random forest, ablation | new channels and a new training release. Frozen result write-ups stay as they are |

LoD morphology does not need a republish only to drop `building_height_std`
and `building_height_max` from the model input. Those bands remain inputs
to the building DSM.
