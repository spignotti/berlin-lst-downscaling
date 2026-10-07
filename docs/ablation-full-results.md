# Tag-11 ablation full runs (issue #58)

Recorded outcomes of the cumulative ladder under the Stage-1 residual lock.
The anchor is `docs/stage1-full-results.md` (validation **0.50285 K**, one-shot
2025 test **0.53257 K**). Configs and the frozen lock are
`docs/ablation-stage-configs.md`. No width, learning-rate, or split retune
sits between these runs. Checkpoint selection stays masked MAE at 100 m.
Stage 5 changes the training loss only.

Single-family runs from issue #63 are in the last section. They are not rungs
of the ladder.

## 1. What is comparable

Every run uses the same residual prior, backbone, temporal split, patch
geometry, seed 0, 20 epochs, and cheap runtime as Stage 1. MAE is the
cell-weighted absolute error in Kelvin on valid 100 m cells after exact 10×10
pooling of the 10 m prediction. The 2025 test split is scored once after the
selected checkpoint reloads.

Scored cohort, from each evidence `data_scope`: train admitted 6872 (2017–2023),
validation admitted 1341 / 341 588 valid cells (2024), test 1803 / 459 564
valid cells. Those validation and test cell totals match the Stage-1 anchor.
Each ablation evidence also records one skipped train patch and one skipped
validation patch.

## 2. Ladder

| Stage | Inputs | Loss | Best val (K) | Epoch | Test 2025 (K) | Test − Stage 1 (K) |
|---|---|---|---:|---:|---:|---:|
| 1 | spectral + index, C=10 | masked L1 | 0.50285 | 19 | 0.53257 | — |
| 2 | + morphology, C=18 | masked L1 | 0.51167 | 19 | 0.51366 | −0.01891 |
| 3 | + shadows, C=20 | masked L1 | 0.50601 | 20 | 0.49991 | −0.03266 |
| 4 | + ERA5, C=28 | masked L1 | 0.49962 | 20 | 0.47535 | −0.05722 |
| 5 | same as Stage 4 | thermal-aware | 0.47624 | 19 | 0.45931 | −0.07326 |

Validation stays near 0.50 K through Stage 4. The test falls at each rung.
Stage 5 is the first rung that moves validation by more than a few thousandths,
and it is the only rung that changes the loss. Residual identity at init is
0.0 K on every rung. The selected checkpoint reload matches `best_metric`.
Pooled validation |correction| stays about 1.43–1.54 K.

## 3. Jobs

Stages 2–4 ran on image `modeling-vertex@sha256:95945d3c1e2ff7f590b140deea6007fe59d857f7760527d615808256968c3147`
(source `792168b7d8a8f5604740510780a2cd1da8ee31c2`). Stage 5 ran on
`modeling-vertex@sha256:b6e73e0fba9cf2506dfea193d1fb1a26725a11b5da06c8814b265422187fd282`
(source `3787fc931e404f75fd26d622c756b8b58a02befa`), which contains the
pad-free SSIM loss. An earlier Stage-5 job on the older image failed in the
first step and has no evidence. Machine is `n1-standard-4` + one on-demand T4
in `europe-west3`. Wall clock is about 3 h. Estimated compute at the admitted
$1.00/h rate is about $3 per job. Cloud Billing is the attributed source.

| Stage | Run label | Job | Wall (UTC) | W&B |
|---|---|---|---|---|
| 2 | `stage2-full-20261006T105815Z-ablation` | `8128860713083994112` | 11:04–13:58 | [s09hx959](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/s09hx959) |
| 3 | `stage3-full-20261006T135951Z-ablation` | `1003251308909559808` | 14:03–17:00 | [nfzywjnt](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/nfzywjnt) |
| 4 | `stage4-full-20261006T170106Z-ablation` | `5305877802908647424` | 17:05–19:59 | [fnqrmqnf](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/fnqrmqnf) |
| 5 | `stage5-full-20261006T223633Z-ablation` | `3522170875493220352` | 23:25–02:24 | [ivt0b55f](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/ivt0b55f) |

Evidence and `best.ckpt` for each label:

`gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/`

## 4. Reading

The large gap to the naive prior is already present at Stage 1 (test 0.533 K
versus naive 1.395 K). The ladder does not create that result. It asks what
the extra families add on top of it.

On the 2025 test, each cumulative rung is lower than the one before it.
ERA5, added between Stage 3 and Stage 4, is the largest feature step
(−0.02456 K). Morphology and shadows cannot be separated on this ladder,
because shadows are only added on top of morphology. Stage 5 shows that the
thermal-aware loss, with inputs held fixed, lowers both validation and test
relative to Stage 4. One seed. Validation through Stage 4 does not move with
the test, so the feature order should be read from the test column and not
from a validation ranking.

## 5. Single-family runs (issue #63)

Same lock and masked L1, on the Stage-5 image. These add one family to the
spectral block and nothing else.

| Run | Inputs | Best val (K) | Epoch | Test 2025 (K) | Test − Stage 1 (K) |
|---|---|---:|---:|---:|---:|
| `isolate-shadows-full-20261006T223633Z-ablation` | spectral + shadows, C=12 | 0.50417 | 19 | 0.52797 | −0.00460 |
| `isolate-era5-full-20261006T232533Z-retry` | spectral + ERA5, C=18 | 0.48504 | 19 | 0.47023 | −0.06234 |
| Stage 2 (from the ladder) | spectral + morphology, C=18 | 0.51167 | 19 | 0.51366 | −0.01891 |

Shadows barely move the test and leave validation on the Stage-1 value.
Morphology moves the test more than shadows. ERA5 alone moves it more than
either, and more than the whole cumulative feature ladder: its test (0.47023 K)
is slightly below the Stage-4 full stack (0.47535 K) on this one seed. The
first ERA5 attempt (`4037833032827142144`) was cancelled with no worker log and
no evidence. The row above is the retry.

W&B: shadows [w1piuu0f](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/w1piuu0f),
ERA5 [wi4uf6o3](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/wi4uf6o3).

## 6. Limitations

- One seed. The test gaps are small next to the Stage-1 gap versus naive, and
  next to Landsat's own uncertainty.
- Temporal holdout is not spatial generalization. The prior is a degradation of
  the same scene. Supervision is at 100 m (`docs/stage1-full-results.md` §5).
- The cumulative ladder confounds family effects. The two isolation runs answer
  only shadows-versus-morphology and ERA5-alone. They do not say whether
  morphology or shadows still help once ERA5 is present.
- Stage 5 is not a feature comparison. Its loss differs from Stages 1–4.
