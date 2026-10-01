# Stage-1 bounded learning probe — result (issue #45)

**Decision: NO-GO for the full Stage-1 temporal run as configured.** The probe
ran cleanly and produced a complete, finite six-epoch curve, but the model does
not approach the naive prior on the same cohort — it is about **193× worse** and
has barely moved off its initialization after six epochs. This is a
scientific-screen failure, not an infrastructure or evidence failure.

## What ran

| Field | Value |
|---|---|
| Config | `stage1_probe` (frozen Stage-1 method; bounded, scene-spread cohort) |
| Vertex job | `projects/996559849187/locations/europe-west3/customJobs/8186380564379467776` |
| Region / machine | `europe-west3` / `n1-standard-4` + 1× `NVIDIA_TESLA_T4` (on-demand) |
| Source SHA | `e079ad863965ebf01f05caf426c52e8a1aed3fb9` |
| Image | `europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:78430e03643a328775897db9a99b296d70621db73b5019c0b72ea68bff74777e` |
| Run label | `stage1-probe-20261001T115323Z-e9e8` |
| Terminal state | `JOB_STATE_SUCCEEDED` |
| Timing | created 11:54:45Z, started 12:06:47Z, ended 12:16:50Z (~22 min wall, ~10 min billed compute) |
| Evidence | `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/stage1-probe-20261001T115323Z-e9e8/evidence.json` |
| W&B run | `pretty-thunder-27` (project `berlin-lst-downscaling`) |
| Cost | ~$0.15 Vertex compute at the verified $0.90/h europe-west3 rate; Cloud Build 15m25s, within the default-pool daily free minutes |

Image built by Cloud Build from the probe tree (build `d6ac02c0-b3ce-4d9a-a982-a5844dabd40f`).

## Cohort (recorded in evidence)

| Split | Requested | Admitted | Skipped | Scenes | Years |
|---|---|---|---|---|---|
| train | 128 | 128 | 0 | 126 | 2017–2023 (7) |
| validation | 64 | 64 | 0 | 30 | 2024 |

Scene-spread selection (≤8 refs/scene), no test ref loaded, no exclusions. All
requested refs were admitted, so the minima (120/60, 16/8 scenes, 3 train
years) were met.

## Curve (cell-weighted masked MAE @ 100 m, valid cells 16,112/epoch)

| Epoch | train | validation |
|---|---|---|
| 1 | 303.0583 | 334.8163 |
| 2 | 301.9677 | 301.2557 |
| 3 | 300.8271 | 299.9706 |
| 4 | 299.4843 | 298.0887 |
| 5 | 297.9291 | 296.3768 |
| 6 | 296.0610 | **294.8369** (selected) |

Checkpoint reload recheck reproduced the selected score exactly
(`reload_recomputed = 294.83685302734375 = best_metric`); the best checkpoint
`best-05-294.8369.ckpt` is the epoch-6 model (Lightning filenames are 0-based).

## Predeclared screen

| Criterion | Result |
|---|---|
| All six train/val MAE finite | pass |
| Final train MAE ≥5% below epoch 1 | **fail** — 2.31% (303.0583 → 296.0610) |
| Best val MAE ≥5% below epoch 1 | pass — 11.94% |
| Final val MAE ≤1.25× best | pass — monotone; final = best |
| Best val MAE ≤10× same-cohort naive | **fail** — 294.8369 vs naive 1.5261 (≈193×) |

Same-cohort naive MAE 1.5261 K is recomputed over the probe's exact 64 admitted
validation patches from the committed baseline report; the full-validation
1.61010 K anchor is not comparable to this subset.

`uv run python scripts/validators/validate_stage1_probe.py --evidence <evidence.json>`
→ decidable, **NO-GO (exit 2)**.

## Interpretation

- This is a **model/optimization failure, not an evidence or infrastructure
  failure.** Every requested epoch completed with finite metrics, admission
  matched the requested cohort, and the checkpoint reload check passed exactly.
- The model has not learned the target scale: at epoch 1 the MAE (~303 K) is
  essentially the spread of LST from a near-zero prediction, and six epochs move
  it only ~2%. On the same cohort the naive prior-expand baseline is 1.5261 K.
  The configured schedule (LR 1e-3, batch 4, 128 training patches, 6 epochs) is
  far too slow to be a credible Stage-1 method — this is not a near-miss.
- **LR/batch assessment:** the frozen LR 1e-3 / batch 4 is not adequate for the
  full run at this scale. The failure is dominated by the distance to the target
  scale, not by noise; a longer schedule may move the curve, but the full run was
  not authorized and no retune was performed under this plan.

## Limitations

- The probe uses a bounded scene-spread subset (128/64), not the full split.
- The temporal holdout (2024 validation) recurs at locations also seen in train
  years, and the prior is derived from the same scene's Landsat observations —
  this is the intended pseudo-pair experiment, not spatial generalization or
  operational accuracy.
- Beating the naive baseline was **not** a probe requirement; the failure is on
  the predeclared "flat" and "far-from-naive" screens, not on the absence of a
  beating margin.

## Next step

Return to planning for the Stage-1 method — in particular the prediction/target
scale and optimization schedule — before any full temporal run. A probe re-run
is a new decision, not a retune of this one; this result stands as recorded.

## Reproduce

```bash
# Validate the retained evidence independently (needs ADC; reads the published index):
uv run python scripts/validators/validate_stage1_probe.py \
  --evidence <downloaded evidence.json>

# Guard self-check (no evidence, no submission):
uv run python scripts/validators/validate_stage1_probe.py --self-check
```
