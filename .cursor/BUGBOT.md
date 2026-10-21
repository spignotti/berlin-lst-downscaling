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

## Acceptance criteria

- If the PR description has an `## Acceptance criteria` checklist, check each item against the diff. Report every item that the diff does not implement, or implements only partially, as a finding that names the item. Also flag a checked box with no matching change or test.
- If a PR changes method, a data product, or training code and has no `## Acceptance criteria` section, flag that.
