# Training-data EDA v1

Descriptive tables for the published Berlin LST training release. Issue
#77. The tables characterize coverage, the 100 m target, the 28 feature
channels, and Stadtstruktur contrasts. They do not choose a model, a
feature, or a class grouping.

This pass writes no figure, map, dashboard, or PDF. Later thesis figures
can be drawn from the saved tables without scanning the rasters again.

## Inputs

Read only:

- `gs://berlin-lst-training-data/training/v1` — manifest, eligible-cell
  index, per-scene provenance, scaler fingerprint
- Feature Release V3 — 28 channels and `feature_valid`
- the v3 anchor manifest — acquisition month and meteorological season
- `gs://berlin-lst-training-data/training/patch-index/v1` — patch
  acceptance, kept separate from cell counts
- `gs://berlin-lst-training-data/evaluation/stadtstruktur/v1` — share
  areas on cells whose centre is inside the Berlin boundary

Proposal B is the grouping in `class_review.py` (`PROPOSAL_B`). Official
06.07 `ststrnr` is the external reference. The map is not edited here.
Water and construction stay exclusion groups. A group is sparse when its
vintage area is under 1.0 km² or under 50 effective cells.

2026 is an inference reserve. It is counted in the inventory and has no
thermal or feature row. Scene years 2017–2020 use Stadtstruktur 2020,
2021–2023 use the same year, and 2024–2025 use 2024.

## Rules fixed before the scan

- A 100 m cell value of a feature is the mean of its 100 valid 10 m
  pixels, in the published physical unit. Precipitation stays in
  millimetres. A shadow value is the fraction of shadowed subpixels.
  `scaler.json` is train-only z-score, with `log1p` on the three
  precipitation channels, and is not the unit of these tables.
- LST is Kelvin. `lst_c = lst_k - 273.15`.
- Hot: LST at or above the scene's own q90. Cold: at or below the
  scene's own q10.
- Neighbors are the 4-neighborhood in the same scene. Both cells are
  eligible. A missing neighbor is left out of the pair.
- A channel is coarse-scale when the median within-scene standard
  deviation is under 1 percent of the between-scene standard deviation
  of scene means. That label does not drop the channel. ERA5 is a coarse
  field even where the flag is unset, because the source grid is much
  coarser than 10 m.
- `observation_area` weights each scene×cell by its class area.
  `cell_area` keeps one mean scene-anomaly per cell, so a repeatedly
  observed cell does not dominate a thermal contrast.
- 1 km blocks are 10 by 10 canonical cells. Their statistics use the
  eligible 100 m cells inside the block. They are not a 10 m truth.
- Cells without a structure polygon stay in the unstratified tables and
  drop out of the stratified ones only.

The cause of an invalid `feature_valid` subpixel is not stored and is
not inferred. The mask cross-tab reports target-valid versus 100/100
support, including edge cells with fewer than 100 present subpixels.

## Output

`gs://berlin-lst-training-data/evaluation/eda/v1/`

Tables are under `tables/`. One Parquet per scene with eligible cells is
under `observations/`. `summary.json` holds the input fingerprints, the
rules above, the population counts, and the data gaps. Logs are under
`logs/eda/`.

## Reproduce

```
uv run python scripts/operators/run_training_eda.py --output-dir /tmp/eda-v1
uv run python scripts/validators/validate_training_eda.py \
  --eda-dir /tmp/eda-v1 \
  --manifest gs://berlin-lst-training-data/training/v1/manifest.parquet
```

The full raster scan runs on `berlin-lst-vm` through
`.opencode/skills/google-access/scripts/run-training-eda-vm.sh`. That
launcher writes to `/workspace/eda-v1` on the boot disk. The validator's
`--self-check` locks the season map, the vintage map, the quantiles, the
neighbor rule, and the area weights without reading GCS.

## Findings

The numeric findings are filled from the published tables after the scan
and the validator pass. Until that directory exists, this section has no
results.
