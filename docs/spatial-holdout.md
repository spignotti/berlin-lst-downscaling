# Spatial holdout add-on test (all nine matrix arms)

pm #553. An addition to the v6 matrix, not a rerun. Test 2025 stays sealed: a
holdout run never scores it and the launcher refuses a holdout mode without
`--skip-test`.

Scope (owner decision 2026-11-16): all nine arms of the matrix (L1, L2, L3, L4,
L5, LOO-M, LOO-S, NO-NDVI-NDWI, NO-RED-NIR-IDX) times seeds 0, 1 and 2 run
under the existing assignment `v1`, 27 runs. L1 and L4 seed 0 ran first (image
`99a71cf`, stage 1); the other 25 runs follow the same rules, so the figures stay
comparable. The TESSERA arm is not part of this test.

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

## Training option

`spatial_holdout: {enabled, assignment}` exists only in the nine configs
`configs/modeling/v6_<arm>_spatial_holdout.yaml` (`l1`, `l2`, `l3`, `l4`, `l5`,
`loo_morph`, `loo_shadow`, `no_ndvi_ndwi`, `no_red_nir_index`), one per matrix
arm. Each inherits its locked arm and differs from it in `spatial_holdout` and
`output_root` only (`SPATIAL_HOLDOUT_CONFIG_ARMS` in
`modeling/spatial_holdout.py`; a test composes all nine and compares them with
the locked configs). The guard admits the option on these nine configs only,
only with the assignment `v1`, and refuses a run that scores test 2025 or uses
a prebuilt cache. A run must pass
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
`--seeds`, default `0,1,2`) seeds first, with at most `--max-parallel` jobs in
flight (default 4, maximum 4). It uses the environment of the matrix launcher
(`V6M_IMAGE_URI`, `V6M_SOURCE_SHA` = checkout HEAD, `V6M_LABEL_PREFIX`,
`V6M_RUN_DIR`; see its header) and the holdout bounds of the Python launcher:
`--skip-test`, job-local cache, 10800 s timeout, 600 s queue allowance, USD
3.25 per job at the USD 1/h cap.

```bash
export V6M_IMAGE_URI=europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest>
export V6M_SOURCE_SHA=<merge-commit-sha> V6M_LABEL_PREFIX=v6sh-<sha7> V6M_RUN_DIR=<dir outside the repo>
S=scripts/runners/vertex/launch_v6_matrix.sh
$S holdout --dry-run --max-total-usd 81.25                    # plan, labels, summed caps; no cloud call
$S holdout --preflight                                        # launcher --preflight per run
$S holdout --max-total-usd 81.25                              # all 27 runs, 4 at a time
$S holdout --max-total-usd 6.50 --arms L5,LOO-M --seeds 0     # a subset
```

- `--max-total-usd` is required. The launcher looks every run up by its Vertex
  displayName, then refuses (exit 2) before the first submission when the
  summed per-job caps of the runs it would start exceed it. Runs that already
  SUCCEEDED, and runs whose job is active (the batch waits on it), are not
  priced. `--dry-run` makes no cloud call and prices every requested run.
- Idempotent and resumable: run the same command again. A pair whose only jobs
  are FAILED or CANCELLED refuses the whole batch (exit 1) unless the operator
  opts in with `V6M_RESUBMIT_FAILED=1` / `V6M_RESUBMIT_CANCELLED=1`.
- A run that gets no capacity (cancelled by the queue rule after 600 s, or
  refused at submission with `RESOURCE_EXHAUSTED` or a quota message) does not
  stop the batch; the others go on and the exit code is 3. Re-run it with
  `V6M_RESUBMIT_CANCELLED=1` to retry the cancelled ones. Any other failure
  starts no further run; jobs in flight are waited for, never cancelled.
- Stage-1 L1 and L4 seed 0 are skipped (their jobs SUCCEEDED) only when
  `V6M_LABEL_PREFIX` is the run-label prefix of stage 1; with another prefix
  use `--arms`/`--seeds` to leave them out.

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

uv run python scripts/analyze_spatial_holdout.py \
  --matrix-export $E/v6m-*.parquet \
  --holdout-export $E/v6sh-*.parquet \
  --output-prefix data/analysis/spatial-holdout/all-arms
```

Run it on whatever has finished: missing arms and seeds are skipped with a note.
The two-arm stage-1 figures (L1 and L4, seed 0) are unchanged by the generic
evaluation; a regression test pins them to the output of the two-arm version
(`tests/test_spatial_holdout_eval.py`, `tests/fixtures/spatial_holdout_stage1_golden.json`).

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
