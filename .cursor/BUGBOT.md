# Bugbot review rules

Review-only rules for this repository. Flag methodological drift. A missing test or a style nit is not a scientific change.

## Holdout

- The 2025 test split is never used for checkpoint selection, model selection, hyperparameters, or thresholds. Validation is 2024. Training is 2017–2023. 2026 is inference only.
- The scaler is fit on the training split only. A spatial holdout keeps the holdout area out of the scaler, with a buffer against spatial leakage.

## Published data

- Published data is never overwritten. A data change is a new version or prefix plus a new config hash.
- Do not rewrite frozen result documents: `docs/results/`, `docs/baseline-full-results.md`, `docs/stage1-*`, `docs/ablation-full-results.md`, and `docs/random-forest-*`.

## Locked profiles

- `stage1_locked`, `stage2_locked` through `stage5_locked`, and `isolate_*` are not changed silently.
- A methodological change is a new option. The previous behavior stays the default.

## Rasters, masks, and temperature

- Resampling is always explicit. The `odc-stac` nearest default has corrupted the target before; do not rely on an implicit resampling.
- QA bitmasks are combined with OR. They are never averaged.
- Temperature is aggregated in Kelvin, not on normalized values.
- No silent clamps or clips.

## Reproducibility

- The seed lives in the config.
- Each run logs the git commit, the image, and the data version.

## Data products

- A pull request that touches a data product or an analysis names the data release root it used.

## Performance

- No full-raster loads inside Python loops. Use chunking or COG windows.
- Feature build is about 7–8 GB of RAM per scene. Flag changes that would raise that.

## Scientific choices

- Flag new scientific decisions in code (thresholds, exclusions, new channels, split changes) that do not come from an explicit task with acceptance criteria.

## Data-correctness invariants

- Sentinel-2 BOA reflectance is harmonized per scene from product metadata (`BOA_ADD_OFFSET`, `raster:bands`, or `s2:processing_baseline`). The acquisition date is not a fallback. The offset is applied to digital numbers before the published `[0, 1]` clip. Non-finite DN stays non-finite. Digital number 0 is a valid code and is not nodata; SCL class 0 is the nodata mask.
- A feature-step correction of already clipped ARD (`clip(DN / 10000, 0, 1)`) is not exact for DN above 10000. Features/v6 reads republished schema-9 ARD and does not apply a second offset.
- LoD2 heights are read from `BuildingPart` when that is where `measuredHeight` lives. A missing height is never stored as 0. The fallback is the unit's own Z extent, then a part may inherit its parent's direct `measuredHeight`, otherwise the unit is excluded and counted.
- Building coverage at 10 m is the polygon–cell intersection area divided by 100 m². A building that misses every cell center is still counted. Heights are area-weighted.
- `total_buildings` counts emitted buildings handed to the rasterizer. It is not left at zero.
- Diagnostics under `evaluation/diagnostics/<run-id>/` are read-only with respect to releases. The diagnostics script does not start the VM.
- `publish_guard` refuses Sentinel-2 writes into the published schema-8 roots and feature writes into `features/v3`, `features/v4`, and `features/v5`.

## Feature schema versions

- `FEATURE_CHANNELS` is the 28-channel roster. Do not drop, rename, or reorder those channels, and do not point `configs/features/v5.yaml` at the v6 roster or at `features/v6`.
- The code default `FEATURE_SCHEMA_VERSION` is 4 after the data-correctness bump. It is still that 28-channel roster.
- features/v6 channel roster is `FEATURE_SCHEMA_VERSION_V6` (5) from `feature_schema`. A literal channel count on a v6 path is a finding. The width is `schema.n_channels`.
- `building_height_std` and `building_height_max` stay on the LoD product for the building DSM. They are not v6 model inputs.
- `vegetation_height_max` stays on the vegetation product for the vegetation DSM. The v6 model input is `canopy_cover_fraction`.
- The canopy threshold default is 2.5 m and is provisional. Changing it is a config and config-hash change.
- `svf_max_radius` stays 3 cells (30 m) until a config change. The multi-radius diagnostic is read-only and must not write a product.
- ERA5 v6 representation defaults to `scene_value`. `bilinear` stays available. The published dynamic product stays the 8-band bilinear contract unless the v6 dynamic config is selected.
- `scene_value` is the mean of the bilinear 10 m field inside the exact Berlin AOI, in both the dynamic writer and the feature loader. A native-cell or bbox mean is a different quantity. The dynamic `scene_value` config hash includes the AOI file fingerprint. The domain label alone does not invalidate a changed mask.
- EDA tables follow the roster stored in the observation table. A v6 table includes `canopy_cover_fraction` and `tp_0_72h`.
- `tp_0_72h` uses `log1p` then z-score. The v5 scaler still uses the three precipitation windows. The published training-policy hash must not move when v6 is added.

## Acceptance criteria

- If the PR description has an `## Acceptance criteria` checklist, check each item against the diff. Report every item that the diff does not implement, or implements only partially, as a finding that names the item. Also flag a checked box with no matching change or test.
- If a PR changes method, a data product, or training code and has no `## Acceptance criteria` section, flag that.
