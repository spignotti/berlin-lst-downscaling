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
`gs://berlin-lst-training-data/evaluation/eda/v2/` over `training/v2`
and `training/patch-index/v2`. Feature stacks are `features/v5`. The
scan commit recorded in `summary.json` is
`d454cf9e2dfe4486395d9db14a6d01ba4286c93c`. The copy under
`docs/results/eda-v1/summary.json` is that file. Counts below are exact.
Other figures are rounded from the published Parquet; the table keeps
the remaining digits. Nothing here drops a channel or refits proposal B.

### Inventory

The manifest holds 345 scenes. 283 are published and 62 are excluded.
Exclusion reasons are `no_eligible_cells` (39 scenes),
`inference_deferred` (21 scenes, all of 2026), and
`usgs_st_implausible_20220827` (2 scenes). Eligible scene×cell rows
are 4,728,053. Unique eligible cells are 69,870. The summary field
`n_training_scenes` is 240. That count is every scene with
`split == train`, including the 31 excluded train scenes. Published
train scenes are 209.

| Split | Scenes | Published | Excluded | Eligible rows | Unique cells |
|---|---:|---:|---:|---:|---:|
| train | 240 | 209 | 31 | 3,489,565 | 69,836 |
| validation | 41 | 36 | 5 | 608,091 | 69,450 |
| test | 43 | 38 | 5 | 630,397 | 69,671 |
| inference | 21 | 0 | 21 | 0 | 0 |

2026 is inventory only. It has no thermal row and no feature row.
Published scenes by year run from 25 in 2017 and 12 in 2021 up to 50 in
2023. There is no DJF scene. Summed across the split×year×season rows,
JJA has 200 scenes (164 published, 2,476,968 eligible rows), MAM has 69
(53 published, 1,256,072), and SON has 76 (66 published, 995,013).

### Coverage and the eligibility gap

Every year has 89,066 AOI cells. Across all years, 69,870 of them are
eligible at least once (fraction 0.784463), and 69,869 of those centres
lie inside the AOI lattice. One eligible cell lies outside that lattice.
The same outside count of 1 appears in every year row. 2021 is the
coverage hole: 46,015 unique eligible cells, fraction 0.516628. 2024
and 2025 were both described. They are the validation year and the test
year.

The raster scan covered 322 scenes: the 283 published scenes and the 39
`no_eligible_cells` scenes. The two implausible LST scenes are excluded
before the scan and have no mask. Eligible cells match the published
provenance count, 4,728,053. Provenance records 17,343,591
target-valid scene×cells. On cells with all 100 subpixels present:

- target-valid and full 10 m support: 4,728,053 (the eligible set)
- target-valid and incomplete 10 m support: 12,520,645
- target-invalid and full 10 m support: 9,499,389

Cells with fewer than 100 present subpixels add 276,276 scene×cells.
The step from a target-valid interior cell to an eligible cell is
dominated by incomplete 10 m support, not by the edge class. The cause
of a failed `feature_valid` subpixel is not stored. Unique-cell columns
in `eligibility_gap_cells` are summed per scene, so one cell is counted
again in every scene where it falls in that class. The eligible class
sums to 1,382,922 scene-unique cells against 69,870 globally unique
eligible cells.

### Patches

Patch acceptance is not a cell count. The published index keeps windows
with at least 244 eligible cells of 256. 166 published scenes are in
the patch index and contribute 6,619 patches (train 4,606, validation
895, test 1,118). 117 published scenes have no valid patch: 94 train,
7 validation, and 16 test. The 39 `no_eligible_cells` scenes are
separate from that 117. The two implausible scenes are skipped and are
not counted in either bucket. 2026 contributes no patches. A scene can
be published for cell statistics and still be absent from training
patches.

### LST

Observation weight uses each eligible scene×cell once. Equal-scene
weight uses one number per published scene, so a scene with one
eligible cell counts the same as a scene with 69,604. Published scenes
span that range. The median scene has 8,722 eligible cells.

Observation-weighted LST, 4,728,053 rows and 283 scenes: mean
304.641 K, standard deviation 6.503 K, median 304.790 K, q05 293.757 K,
q95 315.170 K, minimum 278.885 K, maximum 336.923 K. The mean of the
scene means is 302.658 K. Equal-scene weight: mean 302.969 K, standard
deviation 6.229 K, median 303.088 K, q05 292.218 K, q95 312.405 K,
minimum of scene medians 286.770 K, maximum of scene medians 318.872 K.

Observation-weighted means by split: train 304.795 K, validation
305.534 K, test 302.930 K. The test split is 2025 and is described
here, so this pass is not untouched holdout evidence. Observation-
weighted seasonal means, recomputed from the split×year×season rows,
are JJA 308.122 K, MAM 302.227 K, and SON 299.025 K.

Between-scene standard deviation of LST is 6.163 K. Mean within-scene
standard deviation is 2.975 K. The coarse-scale rule does not fire, and
the field is not globally flat. The same is true inside train,
validation, and test.

### The cold tail, and the hot tail

The two 27 August 2022 scenes named by `usgs_st_implausible_20220827`
are excluded from this release. They are not in the eligible set, and
no scene mean in the published rows is below 280 K.

The minimum, 278.885 K, is one cell in
`LC08_L2SP_192023_20230916_02_T1` (train, SON 2023, 932 eligible
cells, scene mean 297.145 K). The coldest scene mean is
`LC08_L2SP_194023_20190919_02_T1` (train, SON 2019, 1 eligible cell,
286.770 K), which is also the coldest scene median. The largest drop
below a scene mean is 20.116 K, in `LC08_L2SP_193023_20220803_02_T1`.
The JJA scene-mean distribution sits higher: 164 JJA scenes, mean of
scene means 305.429 K, q05 of scene means 296.217 K.

The maximum, 336.923 K, is in `LC09_L2SP_192024_20220804_02_T1`
(train, JJA 2022, 38,563 eligible cells, scene mean 317.970 K). It sits
at the top of a hot, well-populated scene. The largest scene anomaly is
19.511 K, in `LC08_L2SP_192023_20210606_02_T1`.

### Structure at 100 m

Hot and cold cells are the within-scene q90 and q10. That definition
yields 473,455 hot and 473,329 cold cells. Scene q10 has median
298.275 K. Scene q90 has median 306.462 K. The lowest scene q10 is
284.399 K.

The 4-neighborhood, same scene, both cells eligible, has 8,888,786
pairs. The mean absolute difference is 0.755 K. The median of the
per-scene medians is 0.530 K. Across scenes that median sits between
0.303 K and 0.687 K at q05 and q95.

There are 80,477 scene×block rows at 1 km. Eligible-cell coverage has
mean 0.588 and median 0.65. Coverage is 1 on 17,471 blocks and below
0.5 on 32,978. Within-block standard deviation has mean 1.478 K, median
1.335 K, p95 3.171 K, and maximum 8.741 K. The within-block LST range
has mean 6.128 K, median 5.794 K, p95 12.801 K, and maximum 30.769 K.
On the 77,434 blocks that contain a neighbor pair, the mean absolute
neighbor difference has mean 0.740 K and median 0.685 K.

These block numbers describe the measured 100 m field inside a 1 km
block. They are the heterogeneity the 1000-to-100 m model can see.
They are not a measurement of 10 m structure.

### Features

Each feature value is the mean of the 100 valid 10 m pixels, in the
published unit. Precipitation stays in millimetres. Shadows are
fractions. Scaler z-scores are not these units.

The coarse-scale rule fires for three channels, all ERA5-Land
precipitation: `tp_0_24h`, `tp_24_48h`, and `tp_48_72h`. Their median
within-scene standard deviations are 0.631%, 0.635%, and 0.683% of the
between-scene standard deviation of scene means. No channel is
globally flat. The other five ERA5 channels stay above the 1% cutoff.
`t2m_scene` is the closest of them: median within-scene standard
deviation 0.136 K against a between-scene standard deviation of
4.659 K, ratio 0.029. ERA5 remains a coarse source grid where the flag
is false. The flag does not drop any channel.

`tp_0_24h` is skewed. The observation-weighted mean is 0.303 mm and the
median is 0.002 mm. The equal-scene mean is 0.871 mm. The two
antecedent precipitation channels are skewed the same way:
observation-weighted means 0.721 mm and 0.762 mm, medians 0.069 mm and
0.068 mm.

At the 100 m cell mean, `building_height_mean` and
`building_height_max` have Pearson correlation 0.999816.
`building_height_std` has median 0 and q95 0. Its observation-weighted
mean is 0.000523 m. That is a description of the aggregated channel,
not a decision to remove it.

### Correlation with LST

Observation-level correlation mixes scenes, so a channel that changes
from scene to scene correlates with LST even when it barely varies
inside a scene. The largest observation-level Pearson values with LST
are `vpd_scene` 0.733, `t2m_scene` 0.730, `ssrd_antecedent_72h_mean`
0.475, and `ssrd_scene` 0.458. All 4,728,053 rows are finite for these
pairs, and no scene is skipped at observation level.

Within-scene correlations are the median across scenes and do not
weight a scene by its cell count. 17 scenes are skipped because they
have fewer than 30 finite pairs. On the other 266 scenes the largest
median Pearson values with LST are `ndbi` 0.705, `imperviousness`
0.686, `B04` 0.658, `B12` 0.624, `B02` 0.621, and `ndvi` −0.599.
`vegetation_height_mean` is −0.520. Within a scene, `t2m_scene` falls
to 0.213 and `ssrd_scene` to 0.040. The three precipitation channels
are near zero inside a scene.

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
`ststrnr` class id `na`, labeled Gewässer, has the same 2024 area. The
2024 proposal-B areas sum to 790.819 km².

Thermal contrasts use `cell_area`: one mean scene-anomaly per cell,
weighted by that cell's class area, so a cell observed in more scenes
does not dominate. Vintage 2024 is the map for scene years 2024 and
2025, which are validation and test. Train uses vintages 2020–2023.

Cell-area mean scene anomaly, vintage 2024, for the groups that keep
their sign in every vintage:

| Group | Role | 2024 area km² | Anomaly K |
|---|---|---:|---:|
| water | exclusion | 53.758 | −6.124 |
| forest | comparison | 169.468 | −5.101 |
| park_cemetery | comparison | 45.710 | −1.263 |
| low_rise_residential | comparison | 151.545 | −0.237 |
| dense_closed_block | comparison | 34.195 | +2.342 |
| commercial_industrial | comparison | 58.724 | +3.338 |

Across the five vintages, water stays between −7.825 K and −4.690 K,
forest between −5.813 K and −4.814 K, closed blocks between +1.922 K
and +2.423 K, and commercial estates between +2.707 K and +3.571 K.
The inspected 2025 test rows keep those signs: water −6.148 K, forest
−4.527 K, closed blocks +2.106 K, commercial estates +3.106 K.
`agriculture` does not. Its train vintages change sign (+0.121 K,
−1.743 K, +0.767 K, −0.800 K), and validation 2024 is −0.694 K while
test 2025 is +0.410 K. Observation-area weighting, which lets repeated
scenes count again, moves the 2024 means by at most 0.367 K and does
not reorder water, forest, parks, closed blocks, or commercial estates.

`public_and_special` is heterogeneous by role (2024 anomaly +0.624 K).
The three sparse groups are too small to treat as thermal strata.

The same 2024 cell-area join, in physical units: forest has NDVI
0.502, imperviousness 3.7%, and mean building height 0.030 m. Closed
blocks have NDVI 0.221, imperviousness 76.6%, and height 1.807 m.
Commercial estates have NDVI 0.193, imperviousness 68.9%, and height
0.926 m, so the warmest comparison group is not the tallest. Water has
NDVI 0.109, building coverage 0.005, imperviousness 6.4%, and mean
building height 0.076 m. The water group is lightly sealed. The
remaining percent is the non-103 class codes that sit inside water
polygons and keep their own percent. This pass does not decide a
product conflict.

### What this pass does not claim

The 100 m tables do not validate a 10 m prediction. 2025 was read, so
the test year is not an untouched holdout. 2026 has no thermal or
feature evaluation. One eligible cell lies outside the AOI lattice.
The cause of an invalid 10 m subpixel is unknown. No feature is
removed, and proposal B is unchanged. The two implausible LST scenes
are excluded from this description.
