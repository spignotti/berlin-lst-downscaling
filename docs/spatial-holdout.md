# Spatial holdout add-on test (all nine matrix arms)

pm #553. An addition to the v6 matrix, not a rerun. Test 2025 stays sealed: a
holdout run never scores it and the launcher refuses a holdout mode without
`--skip-test`.

Scope (owner decision 2026-11-16): all nine arms of the matrix (L1, L2, L3, L4,
L5, LOO-M, LOO-S, NO-NDVI-NDWI, NO-RED-NIR-IDX) times seeds 0, 1 and 2 run
under the existing assignment `v1`, 27 runs. L1 and L4 seed 0 ran first (image
`99a71cf`, stage 1); the other 25 runs follow the same rules, so the figures stay
comparable. The TESSERA arm (a sibling of L4, `docs/tessera-arm.md`) runs under the same
assignment `v1` with a separate mode and config; section "TESSERA arm" below.

Addition (owner decision 2026-11-16, pm #553): the single draw `v1` is skewed to the
inner city, so arms L1 and L4, seed 0, are also trained once per fold of a 5-fold
spatial block cross-validation over all anchors (assignments `cv5-fold0` to
`cv5-fold4`, section "Spatial cross-validation `cv5`"): ten more runs.

## Why

The matrix splits by time only (train 2017-2023, validation 2024, test 2025).
Of the 263 anchor positions of patch index v3, 250 carry both train and
validation patches, so the validation figures describe areas the model has
trained on. The add-on test asks:

- How much worse does the U-Net reconstruct areas it never saw in training?
- Does the distance L4 - L1 change on unseen areas?
- Do the other matrix comparisons (the ladder steps, the leave-one-out and
  input ablations against L4) keep their sign and size on unseen areas?

## Design

Fixed before any run; the block rule was approved by the owner on 2026-11-14.

- Unit: the 263 anchors of patch index v3 (1.6 km windows, no overlap). All
  263 stay the unit, also the 13 that lack train or validation patches.
- Blocks: fixed tessellation of the global anchor lattice into 3 x 3 anchors
  (4.8 km): `block = (row // 48, col // 48)`. The origin is the canonical grid
  origin, so there is no free offset. Only complete blocks (all nine anchors
  in the index) are eligible: 14 blocks.
- Draw: eligible blocks sorted by `(block_row, block_col)`, permuted with
  `numpy.random.default_rng(0)`; whole blocks until the held-out share of the
  263 anchors is at least 18 %. Above 22 % the draw fails.
- Position: a held-out anchor is `edge` when one of its eight neighbours is a
  training anchor (not held out, at least one train patch), else `interior`.
- Training: 2017-2023 without the held-out anchors. The scaler is refit on the
  remaining admitted train patches. Everything else is the arm's locked config
  (`v6_<arm>_locked`: 40 epochs, cosine, same seeds).
- Checkpoint: best validation MAE 2024 on the seen anchors only.
- Evaluation: validation 2024, all patches, by seen / unseen, and unseen by
  interior / edge.
- Comparison: the matrix run of the same arm and seed on the same anchors.

Quantities (MAE at 100 m in K, scene-equal, paired scene bootstrap as in the
matrix: B = 10 000, rng seed 0, 95 % percentile interval):

- `G` (transfer gap per arm) = holdout model - matrix model, unseen anchors,
  for every arm that has holdout and matrix exports.
- `K` (context under transfer) = L4 - L1, both holdout models, unseen anchors.
- Paired differences A - B of the matrix families
  (`notes/statistikplan-matrix.md`) on every stratum (unseen, unseen interior,
  unseen edge, seen), for the matrix models and for the holdout models:
  family A, the ladder: L2-L1, L3-L2, L4-L3, L5-L4; family B, against L4:
  LOO-M-L4, LOO-S-L4, L3-L4 (LOO-ERA5 is L3), NO-NDVI-NDWI-L4,
  NO-RED-NIR-IDX-L4; and K for the holdout models.

Runs: 27 (nine arms x seeds 0, 1, 2). Stage 1 was seed 0 of L1 and L4; its
stage-2 trigger (an interval of G or K excludes 0 or |point| >= 0.03 K) is
still printed from G of L1 and L4 and from K, but it no longer gates the
remaining runs.

## Assignment `v1`

`configs/modeling/spatial_holdout/v1.json`, written by
`scripts/operators/build_spatial_holdout.py`. The file holds the rule, the
anchor table (train and validation index rows per anchor) and the fingerprints
of the index it was drawn from. Loading re-derives the draw from the file and
refuses a file that disagrees with its own rule; a run refuses an assignment
drawn from another patch index.

| | held out | total | share |
| --- | --- | --- | --- |
| anchors | 54 | 263 | 20.5 % |
| train patches 2017 | 26 | 157 | 16.6 % |
| train patches 2018 | 166 | 1061 | 15.6 % |
| train patches 2019 | 79 | 410 | 19.3 % |
| train patches 2020 | 104 | 543 | 19.2 % |
| train patches 2021 | 67 | 241 | 27.8 % |
| train patches 2022 | 320 | 1421 | 22.5 % |
| train patches 2023 | 385 | 1690 | 22.8 % |
| train patches, all years | 1147 | 5523 | 20.8 % |
| validation patches 2024 | 199 | 1047 | 19.0 % |

Counts are index rows before admission. Six blocks
`(block_row, block_col)`: (2, 4), (3, 2), (3, 3), (3, 4), (3, 6), (4, 4).
Interior: 14 anchors (274 train, 47 validation patches). Edge: 40 anchors
(873 train, 152 validation patches).

```text
I interior, E edge, o kept, . no anchor; north up, one cell = 1.6 km
. . . . . . . o . . . . . . o o o . . . . . . . .
. . . . . o . o o . . o o o o o . . . . . . . . .
. . . . o . o o o o o o o o o o . . . . . . . . .
. . . . . . o o o o E E E o o o . . . . . . . . .
. . o . o o o o o o E I E o o o o o . . . . . . .
. . o o o o o . o o E I E o o o o o o . . . . . .
. o o o E E E E E E E I E o o o E E E . . . . . .
. o o o E I I I I I I I E o o . E I E o o . . . .
. . o o E E E E E E E I E o o o E E E o o o . . .
. . o o . o o o o o E I E o o o o o o o o . . . .
. o o . . o o o o o E I E o o o o o o o o . . . .
o o o . . o o o o o E E E o o o o o o o . . . . .
o o . . . o o o o o o o o o o o o o o o o . . . .
o o o o o o o o o o o o o o o o o o o o o o o . .
o o o o o o o o o o o o o o o o o o o o o . . . o
o o . . . . o . o o o o o . . o o o o o . o o . o
. . . . . . . . . . o o . . . . . o o o . o o . .
. . . . . . . . . . . . . . . . . . . . . o . . .
. . . . . . . . . . . . . . . . . . . . . . o . .
```

Stadtstruktur of the held-out and the kept anchors, as area shares of the
mapped area in percent (Umweltatlas 2024 vintage, 100 m shares of
`evaluation/stadtstruktur/v1`, Proposal B groups of `docs/stadtstruktur.md`;
street land is unmapped). Computed once for this document, not by a committed
script.

| Proposal B group | held out (54) | kept (209) | interior (14) | edge (40) |
| --- | --- | --- | --- | --- |
| vegetated_open | 25.1 | 35.6 | 19.2 | 27.1 |
| commercial_industrial | 19.4 | 9.0 | 27.2 | 16.7 |
| dense_closed_block | 13.1 | 4.4 | 17.4 | 11.7 |
| postwar_open_and_highrise | 11.4 | 9.1 | 10.0 | 11.8 |
| public_and_special | 11.1 | 7.0 | 11.6 | 10.9 |
| low_rise_residential | 9.0 | 22.8 | 3.4 | 10.9 |
| transport | 5.0 | 3.0 | 6.4 | 4.5 |
| water | 3.3 | 6.6 | 3.9 | 3.1 |
| interwar_block_and_row | 2.6 | 2.4 | 0.7 | 3.2 |
| construction | 0.1 | 0.1 | 0.1 | 0.1 |

```bash
# print the assignment and its statistics from the published index
uv run python scripts/operators/build_spatial_holdout.py
# verify the committed file against the published index
uv run python scripts/operators/build_spatial_holdout.py --check configs/modeling/spatial_holdout/v1.json
```

Both read the `gs://` patch index (operator credentials); `--patch-index-dir`
reads a local copy of `patch_index.parquet` and `complete.json`.

## Spatial cross-validation `cv5`

To answer "only one draw": a second assignment family partitions ALL 263 anchors into
five folds; fold `k` holds its anchors out, trains on the other four folds' anchors
and is scored on fold `k`. Arms L1 and L4, seed 0 only. A fold run is exactly a
holdout run with that fold's anchors held out (train 2017-2023 without them, scaler
refit on the remaining admitted train patches, job-local cache, checkpoint selected
on validation 2024 on the seen anchors, 40 epochs cosine, test 2025 never scored,
edge / interior by the same 8-neighbour rule with the fold as the held-out set).

Rule, fixed by the owner, no random draw:

- Blocks: the tessellation of `v1`, `block = (row // 48, col // 48)`, but every
  block that holds at least one anchor, complete or not: 45 blocks.
- Partition: blocks sorted by `(train patches descending, block_row, block_col)`;
  each block, in that order, goes to the fold that currently has the fewest train
  patches, ties to the lowest fold index. Train patches are the index rows per anchor
  stored in the files (the same patch index v3 as `v1`).

Files: `configs/modeling/spatial_holdout/cv5-fold0.json` to `cv5-fold4.json`, one
assignment file per fold in the existing schema (the fold's `holdout`, `blocks` and
positions, the whole anchor table, the index fingerprints, and the rule with
`partition: cv5` and `fold: k`; there is no seed or share bound). They are written by

```bash
# the anchor table and the fingerprints come from v1.json, which `--check` verified against
# the published index; nothing is read from the cloud
uv run python scripts/operators/build_spatial_holdout.py --partition cv5 \
  --anchors-from configs/modeling/spatial_holdout/v1.json \
  --output-dir configs/modeling/spatial_holdout          # or --check-dir to compare
```

Loading re-derives the whole partition from the file's own anchor table and refuses a
file that disagrees with its rule (a changed fold index, anchor, position or count). A
run refuses a fold drawn from another patch index, as for `v1`.

| fold | blocks | anchors | train patches | validation patches | interior / edge anchors |
| --- | --- | --- | --- | --- | --- |
| `cv5-fold0` | 9 | 49 | 1100 | 199 | 8 / 41 |
| `cv5-fold1` | 9 | 54 | 1107 | 201 | 10 / 44 |
| `cv5-fold2` | 9 | 54 | 1100 | 204 | 11 / 43 |
| `cv5-fold3` | 9 | 54 | 1111 | 213 | 10 / 44 |
| `cv5-fold4` | 9 | 52 | 1105 | 230 | 6 / 46 |
| all | 45 | 263 | 5523 | 1047 | 45 / 218 |

Index rows before admission. The train and validation counts of a fold are those of
its held-out anchors; the model trains on the other 5523 minus those rows. Interior
and edge anchors are labelled per fold with that fold as the held-out set; every
fold has both. Blocks per fold, `(block_row, block_col)`:

- fold 0: (1, 3), (1, 5), (2, 6), (3, 3), (3, 5), (5, 4), (5, 8), (6, 5), (7, 8)
- fold 1: (1, 4), (1, 6), (2, 5), (3, 1), (3, 4), (5, 1), (5, 6), (6, 2), (6, 6)
- fold 2: (2, 1), (2, 3), (2, 4), (4, 1), (4, 5), (4, 6), (5, 7), (6, 1), (6, 3)
- fold 3: (1, 2), (3, 2), (3, 6), (4, 0), (4, 2), (4, 4), (4, 7), (5, 3), (6, 8)
- fold 4: (2, 2), (3, 7), (4, 3), (5, 0), (5, 2), (5, 5), (6, 0), (6, 4), (6, 7)

```text
digit = fold that holds the anchor out, . no anchor; north up, one cell = 1.6 km
. . . . . . . 0 . . . . . . 0 0 1 . . . . . . . .
. . . . . 3 . 0 0 . . 1 1 0 0 0 . . . . . . . . .
. . . . 3 . 3 0 0 0 1 1 1 0 0 0 . . . . . . . . .
. . . . . . 4 2 2 2 2 2 2 1 1 1 . . . . . . . . .
. . 2 . 4 4 4 2 2 2 2 2 2 1 1 1 0 0 . . . . . . .
. . 2 2 4 4 4 . 2 2 2 2 2 1 1 1 0 0 0 . . . . . .
. 1 1 1 3 3 3 0 0 0 1 1 1 0 0 0 3 3 3 . . . . . .
. 1 1 1 3 3 3 0 0 0 1 1 1 0 0 . 3 3 3 4 4 . . . .
. . 1 1 3 3 3 0 0 0 1 1 1 0 0 0 3 3 3 4 4 4 . . .
. . 2 2 . 3 3 4 4 4 3 3 3 2 2 2 2 2 2 3 3 . . . .
. 2 2 . . 3 3 4 4 4 3 3 3 2 2 2 2 2 2 3 3 . . . .
3 2 2 . . 3 3 4 4 4 3 3 3 2 2 2 2 2 2 3 . . . . .
4 1 . . . 4 4 3 3 3 0 0 0 4 4 4 1 1 1 2 2 . . . .
4 1 1 1 4 4 4 3 3 3 0 0 0 4 4 4 1 1 1 2 2 2 0 . .
4 1 1 1 4 4 4 3 3 3 0 0 0 4 4 4 1 1 1 2 2 . . . 0
4 2 . . . . 1 . 2 2 4 4 4 . . 0 1 1 1 4 . 4 3 . 3
. . . . . . . . . . 4 4 . . . . . 1 1 4 . 4 3 . .
. . . . . . . . . . . . . . . . . . . . . 4 . . .
. . . . . . . . . . . . . . . . . . . . . . 0 . .
```

Selecting a fold: `spatial_holdout.assignment=cv5-fold<k>` on the L1 or L4 holdout
config, `--assignment cv5-fold<k>` on the launcher, `--assignments` on the batch
launcher. The guard (`assert_spatial_holdout`) and the launcher admit the folds for
the two arms L1 and L4 at seed 0 only; every other arm, the TESSERA arm, a prebuilt
cache, a run without `--skip-test` and any other assignment name are refused. The
assignment is part of the run label (`<prefix>-<arm>-cv5-fold<k>-seed0-ep40-cos-notest`),
so of the Vertex display name, the evidence prefix and the output root; with `v1` they
are unchanged. The job reads the fold from `VERTEX_SPATIAL_HOLDOUT_ASSIGNMENT`, which
only an image built from a commit that contains this code understands: an older image
would train `v1` under a fold label, so build the image from the merge commit first.

Run recipe (10 runs, no stage-1 skip; projected USD 24.00, launcher ceiling USD 31.67, per-job
caps USD 32.50 as the `--max-total-usd` limit; same environment as the batch launcher below):

```bash
S=scripts/runners/vertex/launch_v6_matrix.sh
F=cv5-fold0,cv5-fold1,cv5-fold2,cv5-fold3,cv5-fold4
$S holdout --dry-run --assignments $F --arms l1,l4 --seeds 0 --max-total-usd 32.50
$S holdout --preflight --assignments $F --arms l1,l4 --seeds 0
$S holdout --assignments $F --arms l1,l4 --seeds 0 --max-total-usd 32.50
```

Evaluation: `scripts/analyze_spatial_holdout.py --fold-export cv5-fold<k> <L1 export>
<L4 export>` once per fold next to the matrix exports of L1 and L4 (section
"Evaluation").

## Training option

`spatial_holdout: {enabled, assignment}` exists only in the nine configs
`configs/modeling/v6_<arm>_spatial_holdout.yaml` (`l1`, `l2`, `l3`, `l4`, `l5`,
`loo_morph`, `loo_shadow`, `no_ndvi_ndwi`, `no_red_nir_index`), one per matrix
arm. Each inherits its locked arm and differs from it in `spatial_holdout` and
`output_root` only (`SPATIAL_HOLDOUT_CONFIG_ARMS` in
`modeling/spatial_holdout.py`; a test composes all nine and compares them with
the locked configs). The guard admits the option on these nine configs only,
only with the assignment `v1` (on L1 and L4 also the folds `cv5-fold0` to `cv5-fold4`,
seed 0, see "Spatial cross-validation `cv5`"), and refuses a run that scores test 2025 or uses
a prebuilt cache. A run must pass
`evaluate_test=false` (the launcher's `--skip-test`); the guard refuses a
holdout run that would score the test split. (The tenth config, `tessera_arm_spatial_holdout`, is
the TESSERA arm's; it is not a matrix arm and has its own guard branch.) No
base or matrix config carries the key: absent means off, the resolved configs,
cache provenance and prebuilt-cache keys of the matrix runs are unchanged, and
`guard_modeling_config` rejects the option on any other config.

With the option on, `RealPatchDataModule`:

1. drops the train refs on held-out anchors before admission;
2. refits the scaler in one threaded pass over the admitted train patches
   (`holdout_scaler_fit` phase). Grain and transforms follow the published fit:
   finite 10 m pixels inside training-eligible 100 m cells, `log1p` before the
   moments of the precipitation channel, shadows unscaled, population std,
   8 decimals. The published scaler covers every eligible pixel of a train
   scene; the refit covers the pixels of the remaining train patches only;
3. builds the job-local cache with the refit (train rows, then all validation
   rows). A prebuilt cache is refused: it holds the published scaler (the TESSERA
   arm is the exception, see "TESSERA arm");
4. yields the validation patches on seen anchors for the validation loop, so
   `validation/mae_100m`, the checkpoint choice, `best_metric` and
   `validation_report` are seen-anchor figures;
5. after the checkpoint is frozen, scores the validation patches on held-out
   anchors once. `eval_validation_patches.parquet` holds both groups.

`full_summary.json`, and with it the evidence record
(`full.summary.spatial_holdout`), carries the assignment name and SHA-256, the
patch counts, the refit scaler and the report of the held-out patches. The
checkpoint is only usable together with that scaler.

## TESSERA arm

pm #524, pm #553 (owner decision 2026-11-16): the TESSERA arm also runs under `v1`, 3 seeds, next
to 3 seeds on the ordinary temporal split (mode `tessera`). It is a sibling of L4, not a ladder
stage: it keeps L4's 14 spectral, index and ERA5 channels and replaces the morphology and shadow
channels with 128 annual embedding channels projected to 8, so G_TESSERA and TESSERA minus L4 are
read next to the matrix, not inside its families A and B.

- Config `tessera_arm_spatial_holdout` inherits `tessera_arm` and adds `spatial_holdout`
  (assignment `v1`) and an output root. Launcher mode `tessera-spatial-holdout`: `--skip-test`,
  `--patch-cache prebuilt` and `--assignment v1` only (no fold, no job-local fallback); bounds are
  the TESSERA seed bounds (20400 s, USD 6.25, USD 1/h). The guard admits it with assignment `v1`
  and the prebuilt cache and refuses test scoring, a fold and a job-local cache.
- Training: the published prebuilt cache `patch-cache/v1/<key>` is read as it is (same key, same
  files); the held-out anchors only leave the train view. `docs/tessera-arm.md`, "Spatial holdout",
  gives the equivalence argument and the load-time checks. The order differs from the nine arms:
  the full train split is admitted with the published scaler, the cache is fetched and verified,
  then held-out rows are dropped, the scaler is refit from the cached rows and the kept feature
  channels are re-scaled when a sample is read. The mean check runs in the job as soon as the
  scaler is read, before admission and the download (the launcher preflight does not read the
  scaler, so it cannot run it). `train_removed` and `requested_per_split.train` count as for the
  nine arms (requested held-out train refs removed). The run summary
  (`spatial_holdout.feature_remap`) records the key, the patches checked and refit, the recorded
  filled pixels and the stored zeros.
- Evaluation: `scripts/analyze_spatial_holdout.py` accepts the arm `TESSERA` (an export labelled
  `tessera_arm`, `tessera_arm_spatial_holdout`, `tessera` or `TESSERA` is read as TESSERA; matrix
  exports are the temporal seeds, holdout exports the holdout seeds). The temporal seeds upload no
  per-patch export unless launched with `--export-validation` (the batch arm `TESSERA-TEMPORAL`
  does; plain `tessera` is unchanged without the flag). The arm drops patches with
  invalid embedding pixels, so every comparison that involves it is computed on the intersection of
  the patch sets of the compared runs (all seeds, both sides) and the output names the patches and
  scenes each side lost (`tessera.<quantity>.<stratum>.intersection`). The evaluation fails closed:
  G_TESSERA loses no patch on either side; in TESSERA minus L4 the TESSERA side loses none and L4
  only patches absent from the TESSERA export of the whole split (dropped at admission); anything
  else raises with the counts and example ids. With `--tessera-scope <data_scope.json of a TESSERA
  run>` L4's loss must also equal that run's recorded `tessera_invalid` validation patches;
  without it the output says so under "Skipped" and `scope_verified` is false. Quantities: `G_TESSERA`
  (holdout minus temporal TESSERA, unseen anchors), `TESSERA_minus_L4_holdout` (unseen anchors) and
  `TESSERA_minus_L4_matrix` (all validation patches, stratum `all`), each with the paired scene
  bootstrap and, with seeds 0, 1, 2 on both sides, the three criteria of the decision rule. All other
  comparisons keep the strict identical-patch-set check and their numbers.
- Batch: `launch_v6_matrix.sh holdout --arms TESSERA,TESSERA-TEMPORAL --seeds 0,1,2
  --max-total-usd 37.50` submits the three holdout and the three temporal seeds (arm `TESSERA` is
  mode `tessera-spatial-holdout`, `TESSERA-TEMPORAL` is mode `tessera`), with the batch's lock, at
  most 4 jobs in flight and the attach-only rule. `--arms all` stays the nine matrix arms.

Limits specific to the TESSERA arm: the annual embeddings include Sentinel imagery after the scene
date and from the validation year 2024, also on the held-out anchors, so the arm does not measure
causal transfer. The embeddings of the held-out anchors are not removed from the cache, and the
adapter is trained on the remaining anchors only. The arm replaces morphology and shadow channels
and drops patches with invalid embedding pixels; it is compared with L4 on the common patches.

## Launch

No job was started for this document. The image must be built from the merge
commit that contains this code (`docs/vertex-gpu-training.md`, "Build and push
the image"). Distinct run labels per arm are required: the label is the
evidence prefix.

One run, for example L5 seed 0 (run `--preflight` first, then without it):

```bash
scripts/operators/with-gcp-operator.sh -- \
  uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode v6-l5-spatial-holdout --seed 0 --max-epochs 40 --lr-schedule cosine --skip-test \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest> \
  --source-sha <merge-commit-sha> --run-label v6sh-<sha7>-l5 \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 1.00 --preflight
```

The modes are `<matrix mode>-spatial-holdout` (`v6-l1-spatial-holdout`, ...,
`v6-loo-morph-spatial-holdout`, `v6-no-red-nir-index-spatial-holdout`); the run
label becomes `v6sh-<sha7>-<arm>-seed0-ep40-cos-notest`. Each job uploads,
create-only, under
`gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/`:
`best.ckpt`, `epoch_metrics.json`, `eval_validation_patches.parquet`,
`evidence.json`.

### All arms: the batch launcher

`scripts/runners/vertex/launch_v6_matrix.sh holdout` submits the holdout runs of
the given arms and seeds (`--arms all` or a comma list, default all;
`--seeds`, default `0,1,2`; the two stage-1 runs, L1 and L4 seed 0, are left out
unless `--include-stage1` is given) seeds first, with at most `--max-parallel` jobs in
flight (default 4, maximum 4). It uses the environment of the matrix launcher
(`V6M_IMAGE_URI`, `V6M_SOURCE_SHA` = checkout HEAD, `V6M_LABEL_PREFIX`,
`V6M_RUN_DIR`; see its header) and the holdout bounds of the Python launcher:
`--skip-test`, job-local cache, 10800 s timeout, 600 s queue allowance, USD
3.25 per job at the USD 1/h cap.

```bash
export V6M_IMAGE_URI=europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest>
export V6M_SOURCE_SHA=<merge-commit-sha> V6M_LABEL_PREFIX=v6sh-<sha7> V6M_RUN_DIR=<dir outside the repo>
S=scripts/runners/vertex/launch_v6_matrix.sh
$S holdout --dry-run --max-total-usd 81.25     # the 25 runs that are left: plan, labels, summed caps; no cloud call
$S holdout --preflight --seeds 0               # launcher --preflight per run
$S holdout --max-total-usd 81.25               # the 25 runs that are left, 4 jobs at a time
# or in two calls: seed 0 (7 runs, USD 22.75), then seeds 1 and 2 (18 runs, USD 58.50)
$S holdout --max-total-usd 22.75 --seeds 0
$S holdout --max-total-usd 58.50 --seeds 1,2
```

By default the batch skips L1 and L4 seed 0 and prints a note: they ran in stage 1
(jobs `v6-l1-spatial-holdout-v6sh-99a71cf-l1-seed0-ep40-cos-notest` and the L4
equivalent, SUCCEEDED), under a label prefix a new image does not share, so the
displayName lookup could not find them. `--include-stage1` runs them again
(27 runs, USD 87.75).

- `--max-total-usd` is required. The launcher looks every run up by its Vertex
  displayName, then refuses (exit 2) before the first submission when the
  summed per-job caps of the runs it would start exceed it. Runs that already
  SUCCEEDED, and runs whose job is active, are not priced: the limit covers only
  jobs this invocation may submit. `--dry-run` makes no cloud call and prices every requested run.
- Active jobs found at the start (a resume after Ctrl-C) are watched first and hold
  their slots; nothing new starts until a slot is free, and a pair planned as
  attached never submits a new job in that invocation, even with
  `V6M_RESUBMIT_*` set (if its job ended meanwhile the run is reported failed).
- One batch per `V6M_RUN_DIR`: it holds the lock directory `holdout.lock` there and
  a second batch is refused (exit 2), naming the path. After a crash, remove the
  directory by hand once no batch runs. `--dry-run` and `--preflight` take no lock.
  The lock covers one run directory only: always resume with the same
  `V6M_RUN_DIR`. A second batch started with another directory while the first
  still runs can submit the same run twice.
- A cancel request that fails after the queue rule fired (`CANCEL_UNCONFIRMED`) is
  a failure, not a capacity miss: the old job may still start, so no new run
  starts and the job id is logged. A worker that dies without a result is also a
  failure.
- Idempotent and resumable: run the same command again. A pair whose only jobs
  are FAILED or CANCELLED refuses the whole batch (exit 1) unless the operator
  opts in with `V6M_RESUBMIT_FAILED=1` / `V6M_RESUBMIT_CANCELLED=1`.
- A run that gets no capacity (cancelled by the queue rule after 600 s, or
  refused at submission with `RESOURCE_EXHAUSTED` or a quota message) does not
  stop the batch; the others go on and the exit code is 3. Re-run it with
  `V6M_RESUBMIT_CANCELLED=1` to retry the cancelled ones. Any other failure
  starts no further run; jobs in flight are waited for, never cancelled.
- `--max-parallel` should not exceed the granted Vertex T4 training quota of
  `europe-west3`; its size is not written down in this repository (the quota
  was requested and approved, see "Region" in `docs/vertex-gpu-training.md`).
  Above the quota the queued jobs are cancelled after 600 s as capacity misses
  and need `V6M_RESUBMIT_CANCELLED=1`.

### Cost and time

| | per run | 25 further runs | all 27 |
| --- | --- | --- | --- |
| projected (0.6 h overhead + 40 x 0.045 h, the matrix planner figures) | 2.4 h, USD 2.40 | USD 60.00 | USD 64.80 |
| launcher ceiling (3 h timeout + 10 min queue) | USD 3.17 | USD 79.17 | USD 85.50 |
| per-job exposure cap checked by `--max-total-usd` | USD 3.25 | USD 81.25 | USD 87.75 |

USD 1.00/h is the user-authorized cap, not a list price. The per-run figure
rests on the measured matrix L4 job (2.38 h, job-local cache) and on the stage-1
holdout jobs (L1 1.66 h, L4 2.37 h). A holdout job trains on 21 % fewer patches
and adds the scaler fit pass. The total budget is the owner's to set; it enters
only through `--max-total-usd`.

Per-arm time bound: the written record has no measured v6 wall time for an arm
other than L4 (matrix) and L1 and L4 (stage 1). The channel counts run from 10
(L1) to 22 (L4, L5); L5 changes the loss only. For orientation, the v1 ladder on
the same job profile took 2 h 54 min at Stage 4 and 2 h 59 min at Stage 5
(`docs/ablation-full-results.md`, thermal-aware loss added): about 3 % more, which
would put L5 near 2.4 to 2.5 h. That is not a measurement of v6 L5. The 3 h
bound leaves 0.63 h over L4. Four concurrent jobs read the patch rasters from
the same bucket during data preparation; no measurement exists for that. Both
are watched on the first wave; the bound is not changed here.

Shared overhead: each job reads every admitted patch at the full 22-channel
width (admission, scaler fit, cache build) and only then selects its arm's
channels. That data preparation is the same work in every arm's job; the refit
scaler is identical. One full-width holdout cache, built once and read by all
arms, would save one data preparation per further run: an estimated 0.4 to
0.5 h of the 0.62 h fixed overhead. This is not built; it needs a new job mode
or an operator-side cache build with its own cost.

## Evaluation

`analysis/spatial_holdout_eval.py`, CLI `scripts/analyze_spatial_holdout.py`.
It reads local per-patch validation exports and the assignment, and is generic
over arms: the arm and seed of an export are read from the file. The anchor of
a patch is decoded from its `patch_id`. Patch sums are pooled per scene, the
scene MAE is averaged over the seeds of a model group, every scene counts
once, scenes are resampled. A scene without a patch on a stratum is not part
of that stratum's figure; every figure names its scenes and patches. G pairs
each holdout run with the matrix run of the same arm and seed, and both sides
must have scored the same patches.

Output per run of the CLI (`<prefix>.json`, `<prefix>.md`):

- MAE per model group (`matrix/<arm>`, `holdout/<arm>`) and stratum, and
  unseen minus seen per model.
- `G` per arm that has holdout and matrix exports, on the unseen strata.
- `K` (holdout L4 - L1) and `L4_minus_L1` of both model families, as before.
- `pairings`: the paired differences of the families A, B (and K) listed under
  "Design", for the matrix models and for the holdout models, on every stratum,
  with the same paired scene bootstrap (B = 10 000, rng seed 0, 95 %).
- With seeds 0, 1 and 2 on both sides, each pairing and stratum carries the
  three criteria of the matrix decision rule (`notes/statistikplan-matrix.md`):
  (a) the interval excludes 0, (b) the sign is the same in all seeds,
  (c) |difference| >= 2 x the larger seed SD of the two arms. The numbers come
  from `scene_bootstrap.compare_arms` run on the rows of the stratum: the
  interval, the seed SDs (ddof 1, pooled seed metric) and the sign count are
  the matrix's own, and no verdict label is printed. `scene_bootstrap.py` holds
  these ingredients (`SeedDiagnostics`); the combination into a rule is only
  written in the plan, so the three booleans are formed in the evaluation.
  With fewer seeds the criteria are marked unavailable.
- `notes`: an arm without exports, or a pairing whose two sides have other
  seeds, is skipped with a note and is not an error.
- The stage-2 check keeps its pre-registered inputs (G of L1 and L4, K).

```bash
P=gs://berlin-lst-training-data/qa/modeling/vertex-smoke
E=data/analysis/spatial-holdout/exports
mkdir -p $E
for arm in l1 l2 l3 l4 l5 loo-m loo-s no-ndvi-ndwi no-red-nir-idx; do for seed in 0 1 2; do
  label=v6m-070b33c-$arm-seed$seed-ep40-cos
  gsutil cp $P/v6m-070b33c-val-rescore-notest-r2/$label/eval_validation_patches.parquet \
    $E/$label.parquet
  label=v6sh-<sha7>-$arm-seed$seed-ep40-cos-notest
  gsutil cp $P/$label/eval_validation_patches.parquet $E/$label.parquet
done; done
# L1 and L4 seed 0 of stage 1 sit under the label prefix of stage 1, not v6sh-<sha7>;
# copy them to $E/v6sh-stage1-<arm>-seed0.parquet (any v6sh-*.parquet name works).

uv run python scripts/analyze_spatial_holdout.py \
  --matrix-export $E/v6m-*.parquet \
  --holdout-export $E/v6sh-*.parquet \
  --output-prefix data/analysis/spatial-holdout/all-arms
```

Run it on whatever has finished: missing arms and seeds are skipped with a note.
The two-arm stage-1 figures (L1 and L4, seed 0) are unchanged by the generic
evaluation; a regression test pins them to the output of the two-arm version
(`tests/test_spatial_holdout_eval.py`, `tests/fixtures/spatial_holdout_stage1_golden.json`).

### Cross-validation output

`scripts/analyze_spatial_holdout.py --fold-export cv5-foldK <files> ...` (once per fold;
the fold of an export is named, never inferred from the file name) with
`--matrix-export` of L1 and L4 (seed 0 is used) writes `<prefix>.json` and `<prefix>.md`
(`evaluate_cv` in `analysis/spatial_holdout_eval.py`, same bootstrap: B = 10 000,
rng seed 0, 95 %):

- per fold, on that fold's unseen anchors, what `evaluate` gives for one assignment: MAE
  seen / unseen / unseen interior / unseen edge per arm, `G` per arm against the matrix
  run of the same arm and seed, `K` = L4 - L1.
- the mean and range of `K` and of `G` per arm over the folds (unseen stratum); with
  a fold missing the spread names the folds it covers.
- out of fold: for every validation patch the error of the fold model that did NOT train
  on its anchor, so all 1047 patches are scored once. Scene-equal MAE of L1 and L4,
  `K_cv` = L4 - L1, and `G_cv` per arm against the matrix seed-0 model on the same
  patches. Every anchor with validation patches must be unseen in exactly one fold
  (checked); a fold or arm without export is listed and the out-of-fold summary is
  marked incomplete instead of being computed on a subset.

Fold runs need an image built from a commit that contains this code; an older image ignores
the fold and trains `v1` under a fold label. The evaluation therefore verifies the
assignment recorded by each run: the record next to each export (`<stem>.evidence.json`,
the run's evidence or `full_summary.json`) must name exactly the fold the export is filed
under, with the SHA-256 of the committed `cv5-fold<k>.json`; otherwise it refuses the export
and names the file. The run itself refuses a configured assignment that differs from the
fold named in its output root. (Not added for `v1` exports.)

Copy the exports of the ten fold runs into their own directory (not into the `$E` of the
`v1` recipe, whose `v6sh-*.parquet` glob would sweep them in), named by fold:

```bash
P=gs://berlin-lst-training-data/qa/modeling/vertex-smoke
C=data/analysis/spatial-holdout/exports-cv5
mkdir -p $C
for k in 0 1 2 3 4; do for arm in l1 l4; do
  label=v6sh-<sha7>-$arm-cv5-fold$k-seed0-ep40-cos-notest
  gsutil cp $P/$label/eval_validation_patches.parquet $C/$arm-fold$k.parquet
  gsutil cp $P/$label/evidence.json $C/$arm-fold$k.evidence.json   # the run's own record
done; done

uv run python scripts/analyze_spatial_holdout.py \
  --matrix-export $E/v6m-*-l1-seed0-*.parquet $E/v6m-*-l4-seed0-*.parquet \
  --fold-export cv5-fold0 $C/l1-fold0.parquet $C/l4-fold0.parquet \
  --fold-export cv5-fold1 $C/l1-fold1.parquet $C/l4-fold1.parquet \
  --fold-export cv5-fold2 $C/l1-fold2.parquet $C/l4-fold2.parquet \
  --fold-export cv5-fold3 $C/l1-fold3.parquet $C/l4-fold3.parquet \
  --fold-export cv5-fold4 $C/l1-fold4.parquet $C/l4-fold4.parquet \
  --output-prefix data/analysis/spatial-holdout/cv5
```


### Matrix models on the future strata

Run before any holdout job, on the six matrix exports (L1 and L4, seeds
0-2): `docs/results/spatial-holdout/matrix-v1.{json,md}`. MAE in K, 95 %
interval, n = scenes / patches.

| model | seen | unseen | unseen interior | unseen edge |
| --- | --- | --- | --- | --- |
| matrix L1 | 0.466 [0.424, 0.511] (27 / 848) | 0.443 [0.373, 0.512] (12 / 199) | 0.376 [0.325, 0.431] (8 / 47) | 0.449 [0.376, 0.520] (12 / 152) |
| matrix L4 | 0.446 [0.405, 0.489] (27 / 848) | 0.411 [0.352, 0.470] (12 / 199) | 0.334 [0.293, 0.378] (8 / 47) | 0.421 [0.358, 0.483] (12 / 152) |

The columns rest on different scenes, so their levels do not compare: 15 of
the 27 validation scenes have no patch on a held-out anchor (each of them
holds one to four patches in total). Within the 12 scenes that have both sides, unseen minus seen is
+0.026 K [-0.029, 0.079] for L1 and +0.032 K [-0.005, 0.077] for L4: the
held-out anchors are not reliably easier or harder for the matrix models.
L4 - L1 of the matrix models is -0.020 K [-0.041, 0.000] on seen anchors
(27 scenes) and -0.032 K [-0.053, -0.014] on the held-out anchors
(12 scenes).

## Limits

- `v1` is one holdout draw. `cv5` adds a five-fold partition of all anchors for L1 and
  L4 at seed 0; the other arms and seeds have no cross-validation. The block size is
  not derived from an autocorrelation range.
- The `cv5` folds are scattered blocks, so more anchors are edge and fewer are interior
  than in `v1` (6 to 11 interior anchors per fold against 14 in `v1`, which has one
  contiguous area). Every fold holds out about a fifth of the anchors, incomplete
  blocks included. The land-use mix of the folds was not measured. One seed per fold: the
  fold-to-fold spread mixes spatial and seed variation, and the out-of-fold interval
  carries scene variation only.
- No buffer beyond the window edge; the interior / edge split stands in.
- Five of the six blocks touch and form one cross-shaped area in the centre;
  one block lies apart. The 14 interior anchors exist because of that
  contact: a lone block has one interior anchor.
- The held-out anchors are more inner-city than the rest: about three times
  the share of dense closed blocks and twice the commercial and industrial
  share, less than half the low-rise residential share. G describes transfer
  to this mix, not to an average Berlin anchor.
- Only 12 of the 27 validation scenes have a patch on a held-out anchor, 8 of
  them an interior patch. G and K are scene-equal over those 12 scenes; four
  of them hold one or two unseen patches.
- One held-out anchor (row 144, col 96) has no train patch in the index, so
  the matrix models have not seen it either (one validation patch).
- The refit scaler differs from the published one in grain (patches instead
  of whole scenes) as well as in the removed anchors.
- L5 (thermal-aware loss) uses the published SSIM data range
  (`configs/modeling/ssim_data_range/v1.yaml`, L = 30.622 K) unchanged. It was
  derived on all 5523 train targets, the held-out anchors included, and is not
  refit under the holdout. The range is a fixed constant of the loss and the
  report, not a learned quantity, but L5's holdout run is therefore not
  strictly blind to the held-out anchors in that one constant. A test pins that
  the holdout config resolves the same range as the locked L5 config.
- The random forest is not retrained. No naive-baseline export on the same
  anchors is part of this evaluation yet.
- With one seed per arm (stage 1: L1 and L4 seed 0), G and K intervals carry
  scene variation, not seed variation; the decision-rule criteria need all
  three seeds.
