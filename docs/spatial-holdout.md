# Spatial holdout add-on test (L1, L4)

pm #553. An addition to the v6 matrix, not a rerun. Test 2025 stays sealed: a
holdout run never scores it and the launcher refuses a holdout mode without
`--skip-test`.

## Why

The matrix splits by time only (train 2017-2023, validation 2024, test 2025).
Of the 263 anchor positions of patch index v3, 250 carry both train and
validation patches, so the validation figures describe areas the model has
trained on. The add-on test asks:

- How much worse does the U-Net reconstruct areas it never saw in training?
- Does the distance L4 - L1 change on unseen areas?

## Design

Fixed before any run; the block rule was approved by the owner on 2026-11-16.

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
  remaining admitted train patches. Everything else is `v6_l1_locked` /
  `v6_l4_locked` (40 epochs, cosine, same seeds).
- Checkpoint: best validation MAE 2024 on the seen anchors only.
- Evaluation: validation 2024, all patches, by seen / unseen, and unseen by
  interior / edge.
- Comparison: the matrix run of the same arm and seed on the same anchors.

Quantities (MAE at 100 m in K, scene-equal, paired scene bootstrap as in the
matrix: B = 10 000, rng seed 0, 95 % percentile interval):

- `G` (transfer gap per arm) = holdout model - matrix model, unseen anchors.
- `K` (context under transfer) = L4 - L1, both holdout models, unseen anchors.

Stages: stage 1 is seed 0 for L1 and L4. Stage 2 (seeds 1 and 2) runs only if
an interval of G or K excludes 0 or |point| >= 0.03 K in stage 1; the
evaluation prints this check.

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

## Training option

`spatial_holdout: {enabled, assignment}` exists only in
`configs/modeling/v6_l1_spatial_holdout.yaml` and
`v6_l4_spatial_holdout.yaml`. Both inherit their locked arm and differ from it
in `spatial_holdout` and `output_root` only. A run must pass
`evaluate_test=false` (the launcher's `--skip-test`); the guard refuses a
holdout run that would score the test split. No
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
   rows). A prebuilt cache is refused: it holds the published scaler;
4. yields the validation patches on seen anchors for the validation loop, so
   `validation/mae_100m`, the checkpoint choice, `best_metric` and
   `validation_report` are seen-anchor figures;
5. after the checkpoint is frozen, scores the validation patches on held-out
   anchors once. `eval_validation_patches.parquet` holds both groups.

`full_summary.json`, and with it the evidence record
(`full.summary.spatial_holdout`), carries the assignment name and SHA-256, the
patch counts, the refit scaler and the report of the held-out patches. The
checkpoint is only usable together with that scaler.

## Launch (stage 1)

No job was started for this document. The image must be built from the merge
commit that contains this code (`docs/vertex-gpu-training.md`, "Build and push
the image"). Run each command with `--preflight` first, then without it.
Distinct run labels per arm are required: the label is the evidence prefix.

```bash
scripts/operators/with-gcp-operator.sh -- \
  uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode v6-l1-spatial-holdout --seed 0 --max-epochs 40 --lr-schedule cosine --skip-test \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest> \
  --source-sha <merge-commit-sha> --run-label v6sh-<sha7>-l1 \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 1.00 --preflight
```

```bash
scripts/operators/with-gcp-operator.sh -- \
  uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode v6-l4-spatial-holdout --seed 0 --max-epochs 40 --lr-schedule cosine --skip-test \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest> \
  --source-sha <merge-commit-sha> --run-label v6sh-<sha7>-l4 \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 1.00 --preflight
```

The run label becomes `v6sh-<sha7>-<arm>-seed0-ep40-cos-notest`. Each job
uploads, create-only, under
`gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/`:
`best.ckpt`, `epoch_metrics.json`, `eval_validation_patches.parquet`,
`evidence.json`.

### Cost

| | per run | stage 1 (2 runs) |
| --- | --- | --- |
| projected (0.6 h overhead + 40 x 0.045 h, the matrix planner figures) | 2.4 h, USD 2.40 | USD 4.80 |
| launcher ceiling (3 h timeout + 10 min queue) | USD 3.17 | USD 6.33 |

USD 1.00/h is the user-authorized cap, not a list price. The projection is
the measured matrix L4 job (2.38 h, job-local cache). Against it a holdout job
trains on 21 % fewer patches (about 30 s less per epoch) and adds the scaler
fit pass (no measurement; it reads 79 % of the train patches once with the
cache-build threads). Stage 2 would add four runs; the plan's total cap is
USD 15, and six runs at the per-job ceiling would exceed it, so check the
stage-1 cost before stage 2.

Shared overhead: each job reads every admitted patch at the full 22-channel
width (admission, scaler fit, cache build) and only then selects its arm's
channels. That data preparation is the same work in the L1 and the L4 job; the
refit scaler is identical. One full-width holdout cache, built once and read
by both arms, would save one data preparation per further run: an estimated
0.4 to 0.5 h of the 0.62 h fixed overhead, about USD 0.45 in stage 1. This is
not built; it needs a new job mode or an operator-side cache build with its
own cost.

## Evaluation

`analysis/spatial_holdout_eval.py`, CLI `scripts/analyze_spatial_holdout.py`.
It reads local per-patch validation exports and the assignment. The anchor of
a patch is decoded from its `patch_id`. Patch sums are pooled per scene, the
scene MAE is averaged over the seeds of a model group, every scene counts
once, scenes are resampled. A scene without a patch on a stratum is not part
of that stratum's figure; every figure names its scenes and patches. G pairs
each holdout run with the matrix run of the same arm and seed, and both sides
must have scored the same patches.

```bash
P=gs://berlin-lst-training-data/qa/modeling/vertex-smoke
for arm in l1 l4; do for seed in 0 1 2; do
  label=v6m-070b33c-$arm-seed$seed-ep40-cos
  gsutil cp $P/v6m-070b33c-val-rescore-notest-r2/$label/eval_validation_patches.parquet \
    data/analysis/spatial-holdout/exports/$label.parquet
done; done
for arm in l1 l4; do
  label=v6sh-<sha7>-$arm-seed0-ep40-cos-notest
  gsutil cp $P/$label/eval_validation_patches.parquet \
    data/analysis/spatial-holdout/exports/$label.parquet
done

uv run python scripts/analyze_spatial_holdout.py \
  --matrix-export data/analysis/spatial-holdout/exports/v6m-*.parquet \
  --holdout-export data/analysis/spatial-holdout/exports/v6sh-*.parquet \
  --output-prefix data/analysis/spatial-holdout/stage1
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

- One holdout draw, no spatial cross-validation. The block size is not
  derived from an autocorrelation range.
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
- The random forest is not retrained. No naive-baseline export on the same
  anchors is part of this evaluation yet.
- Stage 1 has one seed per arm: its G and K intervals carry scene variation,
  not seed variation.
