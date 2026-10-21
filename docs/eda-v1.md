# Training-data EDA v1

Descriptive tables for the published Berlin LST training release. Issue
#77. The tables characterize coverage, the 100 m target, the 28 feature
channels, and Stadtstruktur contrasts. They do not choose a model, a
feature, or a class grouping.

This pass writes no figure, map, dashboard, or PDF. Later thesis figures
can be drawn from the saved tables without scanning the rasters again.

## Inputs

Read only:

- `gs://berlin-lst-training-data/training/v2` — manifest, eligible-cell
  index, per-scene provenance, scaler fingerprint
- Feature Release v5 — 28 channels and `feature_valid`
- the v3 anchor manifest — acquisition month and meteorological season
- `gs://berlin-lst-training-data/training/patch-index/v2` — patch
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

`gs://berlin-lst-training-data/evaluation/eda/v2/`

Tables are under `tables/`. One Parquet per scene with eligible cells is
under `observations/`. `summary.json` holds the input fingerprints, the
rules above, the population counts, and the data gaps. Logs are under
`logs/eda/`.

## Reproduce

```
uv run python scripts/operators/run_training_eda.py --output-dir /tmp/eda-v2
uv run python scripts/validators/validate_training_eda.py \
  --eda-dir /tmp/eda-v2 \
  --manifest gs://berlin-lst-training-data/training/v2/manifest.parquet
```

The full raster scan runs on `berlin-lst-vm` through
`.opencode/skills/google-access/scripts/run-training-eda-vm.sh`. That
launcher writes to `/workspace/eda-v2` on the boot disk. The validator's
`--self-check` locks the season map, the vintage map, the quantiles, the
neighbor rule, and the area weights without reading GCS.

## Findings

The counts in this section are the scan of
`gs://berlin-lst-training-data/evaluation/eda/v1/` over `training/v1`.
They stay until the `evaluation/eda/v2` scan replaces them. The
scan commit recorded in `summary.json` is
`d7074f0e073737cda6157d8f3c6d040cd3b3b9c0`. The copy under
`docs/results/eda-v1/summary.json` is that file. Counts below are exact.
Other figures are rounded from the published Parquet; the table keeps
the remaining digits. Nothing here drops a channel or refits proposal B.

### Inventory

The manifest holds 345 scenes. 294 are published and 51 are excluded.
The only exclusion reasons are `no_eligible_cells` (30 scenes) and
`inference_deferred` (21 scenes, all of 2026). Eligible scene×cell rows
are 5,520,165. Unique eligible cells are 69,875. The summary field
`n_training_scenes` is 240. That count is every scene with
`split == train`, including the 21 excluded train scenes. Published
train scenes are 219.

| Split | Scenes | Published | Excluded | Eligible rows | Unique cells |
|---|---:|---:|---:|---:|---:|
| train | 240 | 219 | 21 | 4,057,392 | 69,868 |
| validation | 41 | 37 | 4 | 696,942 | 69,679 |
| test | 43 | 38 | 5 | 765,831 | 69,669 |
| inference | 21 | 0 | 21 | 0 | 0 |

2026 is inventory only. It has no thermal row and no feature row.
Published scenes by year run from 25 in 2017 and 13 in 2021 up to 53 in
2023. There is no DJF scene. Summed across the split×year×season rows,
JJA has 200 scenes (173 published, 2,868,678 eligible rows), MAM has 69
(53 published, 1,370,799), and SON has 76 (68 published, 1,280,688).

### Coverage and the eligibility gap

Every year has 89,066 AOI cells. Across all years, 69,875 of them are
eligible at least once (fraction 0.784519), and 69,874 of those centres
lie inside the AOI lattice. One eligible cell lies outside that lattice.
The same outside count of 1 appears in every year row. 2021 is the
coverage hole: 49,240 unique eligible cells, fraction 0.552837. 2024
and 2025 were both described. They are the validation year and the test
year.

The raster scan covered 324 scenes: the 294 published scenes and the 30
`no_eligible_cells` scenes. Eligible cells match the published
provenance count, 5,520,165. Provenance records 20,169,061
target-valid scene×cells. On cells with all 100 subpixels present:

- target-valid and full 10 m support: 5,520,165 (the eligible set)
- target-valid and incomplete 10 m support: 14,541,533
- target-invalid and full 10 m support: 8,736,992

Cells with fewer than 100 present subpixels add 277,992 scene×cells.
The step from a target-valid interior cell to an eligible cell is
dominated by incomplete 10 m support, not by the edge class. The cause
of a failed `feature_valid` subpixel is not stored. Unique-cell columns
in `eligibility_gap_cells` are summed per scene, so one cell is counted
again in every scene where it falls in that class. The eligible class
sums to 1,456,899 scene-unique cells against 69,875 globally unique
eligible cells.

### Patches

Patch acceptance is not a cell count. 182 published scenes are in the
patch index and contribute 10,018 patches (train 6,873, validation
1,342, test 1,803). 112 published scenes have no valid patch: 93 train,
7 validation, and 12 test. 2026 contributes no patches. A scene can be
published for cell statistics and still be absent from training patches.

### LST

Observation weight uses each eligible scene×cell once. Equal-scene
weight uses one number per published scene, so a scene with one
eligible cell counts the same as a scene with 69,634. Published scenes
span that range. The median scene has 10,448 eligible cells.

Observation-weighted LST, 5,520,165 rows and 294 scenes: mean
304.173 K, standard deviation 6.685 K, median 304.380 K, q05 293.329 K,
q95 314.805 K, minimum 221.479 K, maximum 339.849 K. The mean of the
scene means is 301.941 K. Equal-scene weight: mean 302.300 K, standard
deviation 6.599 K, median 302.402 K, q05 291.695 K, q95 311.981 K,
minimum of scene medians 271.375 K, maximum of scene medians 318.517 K.

Observation-weighted means by split: train 304.290 K, validation
305.242 K, test 302.584 K. The test split is 2025 and is described
here, so this pass is not untouched holdout evidence. Observation-
weighted seasonal means, recomputed from the split×year×season rows,
are JJA 307.568 K, MAM 301.950 K, and SON 298.949 K.

Between-scene standard deviation of LST is 6.668 K. Mean within-scene
standard deviation is 3.212 K. The coarse-scale rule does not fire, and
the field is not globally flat. The same is true inside train,
validation, and test.

### A cold scene, and the hot tail

The minimum, 221.479 K, is one cell in
`LC09_L2SP_193024_20220827_02_T2` (train, JJA 2022, 5,621 eligible
cells). That scene's own mean is 269.543 K and its median is 271.375 K,
which is also the coldest scene median in the release. The same-day
neighbour `LC09_L2SP_193023_20220827_02_T2` has 1,421 eligible cells
and a mean of 271.127 K. Two scenes have a mean below 280 K, and both
are these rows. The JJA scene-mean distribution otherwise sits much
higher: 173 JJA scenes, mean of scene means 304.404 K, q05 of scene
means 294.655 K. The 221.479 K cell is 48.064 K below its scene mean,
and that scene mean is already far below the other August scenes. The
eligibility rule still accepted the cells. This pass does not assign a
sensor cause.

The maximum, 339.849 K, is in `LC09_L2SP_192024_20220804_02_T1`
(train, JJA 2022, 41,415 eligible cells, scene mean 317.436 K). It sits
at the top of a hot, well-populated scene. The largest scene anomaly is
22.450 K, in `LC09_L2SP_192024_20220516_02_T1`.

### Structure at 100 m

Hot and cold cells are the within-scene q90 and q10. That definition
yields 552,772 hot and 552,639 cold cells. Scene q10 has median
297.910 K. Scene q90 has median 306.016 K. The cold scene above is
what pulls the lowest scene q10 down to 244.687 K.

The 4-neighborhood, same scene, both cells eligible, has 10,452,446
pairs. The mean absolute difference is 0.845 K. The median of the
per-scene medians is 0.574 K. Across scenes that median sits between
0.350 K and 0.747 K at q05 and q95.

There are 88,450 scene×block rows at 1 km. Eligible-cell coverage has
mean 0.624 and median 0.75. Coverage is 1 on 24,210 blocks and below
0.5 on 33,263. Within-block standard deviation has mean 1.599 K, median
1.465 K, p95 3.334 K, and maximum 9.401 K. The within-block LST range
has mean 6.761 K, median 6.443 K, p95 13.754 K, and maximum 31.241 K.
On the 85,215 blocks that contain a neighbor pair, the mean absolute
neighbor difference has mean 0.838 K and median 0.776 K.

These block numbers describe the measured 100 m field inside a 1 km
block. They are the heterogeneity the 1000-to-100 m model can see.
They are not a measurement of 10 m structure.

### Features

Each feature value is the mean of the 100 valid 10 m pixels, in the
published unit. Precipitation stays in millimetres. Shadows are
fractions. Scaler z-scores are not these units.

The coarse-scale rule fires for three channels, all ERA5-Land
precipitation: `tp_0_24h`, `tp_24_48h`, and `tp_48_72h`. Their median
within-scene standard deviations are 0.57%, 0.62%, and 0.58% of the
between-scene standard deviation of scene means. No channel is
globally flat. The other five ERA5 channels stay above the 1% cutoff.
`t2m_scene` is the closest of them: median within-scene standard
deviation 0.133 K against a between-scene standard deviation of
4.615 K, ratio 0.029. ERA5 remains a coarse source grid where the flag
is false. The flag does not drop any channel.

`tp_0_24h` is skewed. The observation-weighted mean is 0.345 mm and the
median is 0.002 mm. The equal-scene mean is 0.993 mm. The same pattern
holds for the two antecedent precipitation channels.

At the 100 m cell mean, `building_height_mean` and
`building_height_max` have Pearson correlation 0.999804.
`building_height_std` has median 0 and q95 0. Its observation-weighted
mean is 0.000554 m. That is a description of the aggregated channel,
not a decision to remove it.

### Correlation with LST

Observation-level correlation mixes scenes, so a channel that changes
from scene to scene correlates with LST even when it barely varies
inside a scene. The largest observation-level Pearson values with LST
are `vpd_scene` 0.708, `t2m_scene` 0.693, `ssrd_scene` 0.471, and
`ssrd_antecedent_72h_mean` 0.461. All 5,520,165 rows are finite for
these pairs.

Within-scene correlations are the median across scenes and do not
weight a scene by its cell count. 17 scenes are skipped because they
have fewer than 30 finite pairs. On the other 277 scenes the largest
median Pearson values with LST are `ndbi` 0.675, `B04` 0.630, `B12`
0.608, `B02` 0.591, and `ndvi` −0.555. `vegetation_height_mean` is
−0.506. Within a scene, `t2m_scene` falls to 0.216 and `ssrd_scene` to
0.043. The three precipitation channels are near zero inside a scene.

The weather correlations are mostly between scenes. The surface-index
and vegetation-height correlations are mostly inside a scene. That
split is descriptive. It does not rank channels for a model.

### Stadtstruktur

Areas are the summed `area_m2` of `shares.parquet` on cells inside the
AOI. There is no majority class. No `typ` falls outside proposal B.
Water and construction stay exclusion groups. A group is sparse when
its vintage area is under 1.0 km² or its effective cell count is under
50. Three groups are sparse in every vintage: `sealed_plaza`
(comparison; 2024 area 0.474 km², 47.4 effective cells), `camping`
(boundary; 0.483 km², 48.3 effective cells), and `construction`
(exclusion; 0.529 km², 52.9 effective cells, sparse on area). Water is
an exclusion and is not sparse (2024 area 53.758 km²). The official
`ststrnr` null class has the same 2024 area. The 2024 proposal-B areas
sum to 790.819 km².

Thermal contrasts use `cell_area`: one mean scene-anomaly per cell,
weighted by that cell's class area, so a cell observed in more scenes
does not dominate. Vintage 2024 is the map for scene years 2024 and
2025, which are validation and test. Train uses vintages 2020–2023.

Cell-area mean scene anomaly, vintage 2024, for the groups that keep
their sign in every vintage:

| Group | Role | 2024 area km² | Anomaly K |
|---|---|---:|---:|
| water | exclusion | 53.758 | −6.011 |
| forest | comparison | 169.468 | −5.056 |
| park_cemetery | comparison | 45.710 | −1.286 |
| low_rise_residential | comparison | 151.545 | −0.214 |
| dense_closed_block | comparison | 34.195 | +2.371 |
| commercial_industrial | comparison | 58.724 | +3.330 |

Across the five vintages, water stays between −8.054 K and −5.410 K,
forest between −5.867 K and −4.760 K, closed blocks between +1.905 K
and +2.520 K, and commercial estates between +2.822 K and +3.569 K.
The inspected 2025 test rows keep those signs: water −5.822 K, forest
−4.520 K, closed blocks +2.145 K, commercial estates +3.091 K.
`agriculture` does not. Its train vintages change sign (+0.285 K,
−1.620 K, +0.928 K, −0.712 K), and validation 2024 is −0.610 K while
test 2025 is +0.491 K. Observation-area weighting, which lets repeated
scenes count again, moves the 2024 means by less than 0.5 K and does
not reorder water, forest, parks, closed blocks, or commercial estates.

`public_and_special` is heterogeneous by role (2024 anomaly +0.638 K).
The three sparse groups are too small to treat as thermal strata.

The same 2024 cell-area join, in physical units: forest has NDVI
0.501, imperviousness 5.8%, and mean building height 0.030 m. Closed
blocks have NDVI 0.220, imperviousness 78.6%, and height 1.807 m.
Commercial estates have NDVI 0.193, imperviousness 71.8%, and height
0.927 m, so the warmest comparison group is not the tallest. Water has
NDVI 0.108, building coverage 0.005, and imperviousness 86.5%. The
sealing value on the water group disagrees with the building layers.
This pass does not decide which product is wrong.

### What this pass does not claim

The 100 m tables do not validate a 10 m prediction. 2025 was read, so
the test year is not an untouched holdout. 2026 has no thermal or
feature evaluation. One eligible cell lies outside the AOI lattice.
The cause of an invalid 10 m subpixel is unknown. No feature is
removed, and proposal B is unchanged.
