# Pseudo-pair and tensor contract

Normative contract for the training data-to-model boundary (WB3). It
freezes how a training sample maps coarse 100 m LST supervision to the
10 m downscaling product, so the dataset, loss, and later patch work
share one definition.

Status: **implemented.** Patch geometry, the sampling index, the real patch
reader with the `lst_prior` construction (`modeling/patches.py`), the exact
pooling and masked metrics (`modeling/metrics.py`), the 10 m Lightning path
(`modeling/real_task.py`), and the naive prior-expand baseline
(`modeling/baseline.py`) are delivered under issues #18 and #19. The
modelling package keeps its synthetic scaffold alongside the real path (see
*Relation to the current modelling scaffold*).

Verification surface: `scripts/validators/validate_real_patches.py` and
`scripts/validators/validate_baseline.py` recompute the reader and the
baseline artifact independently, and `nox -s smoke-real-comparison` (opt-in,
requires ADC) runs both arms on one bounded real subset and asserts they
share a patch universe. Full training runs and the 2025 test comparison are
explicit invocations, not part of that gate.

## Scope

This document fixes the pseudo-pair semantics, the `lst_prior` input, the
tensor fields and their grids, and the masked-loss normalization.

Fixed here after the Stage-1 decision (issues #18/#19):

- the prior regime used for **every** compared split (train, validation,
  test), for both the model and the naive baseline,
- how invalid native 100 m cells inside a 1000 m `lst_prior` block are
  treated (masked mean over native-valid cells only),
- how invalid 10 m predictor pixels are handed to the model at its
  boundary (neutral zero fill after the train-only scaler).

Out of scope, left to later WB3 decisions:

- the sampler,
- the concrete loss family beyond masked aggregation,
- Zarr versus COG I/O.

## Fixed inputs referenced, not restated

These are already defined elsewhere and are inputs to this contract:

- **Feature Release V3** channel order and validity mask:
  `data-sources-and-contracts.md` § Scene feature stacks (28-band, 10 m).
- **`training_eligible@100m`**, temporal splits, cell identity, and the
  train-only scaler: `data-sources-and-contracts.md` § WB2c-4 training-data
  release; policy in `data/training/contracts.py`.
- **Canonical nested grid:** `common/grid.py`. The 100 m grid is
  `canon_grid_10m().zoom_out(10)`, so the two grids share an origin and
  every 100 m cell is a disjoint 10×10 block of 10 m pixels. The
  eligibility computation uses the same exact nested 10×10 aggregation
  (`data/training/eligibility.py`).

## Patch geometry (fixed)

The patch is 160×160 px at 10 m = 16×16 cells at 100 m (1.6 km), the
depth-4 U-Net edge requirement. Windows are complete 16×16 blocks
anchored on the global canonical 100 m lattice and are accepted when at
least 95% of their 256 cells are `training_eligible` (244 of 256). The
published `patch_index.parquet` enumerates them; its schema, anchor rule,
stride, and provenance are defined in `data-sources-and-contracts.md`
§ WB3 patch index. This contract consumes the index; it does not redefine
it.

## Variant B (main contract)

The model predicts at 10 m. Training compares a **10×10 nested block mean
of the 10 m prediction** against the **native 100 m Landsat LST**, on the
`training_eligible@100m` mask.

1. Nested 10×10 mean pooling of `prediction_10m` gives `prediction_100m`
   on the canonical 100 m grid. Pooling is exact over each disjoint 10×10
   block; no resampling and no partial-block handling beyond the mask.
2. The loss compares `prediction_100m` with `target_100m` only where
   `mask_100m` is true.

`mask_100m` is `training_eligible@100m`. A 100 m cell is eligible only
when the Landsat target cell is valid and all 100 of its 10 m
`feature_valid` subpixels are valid (strict 100/100 support). The rule is
defined in `data/training/contracts.py` and computed in
`data/training/eligibility.py`; it is not redefined here.

## Stage-1 objective and metrics (issues #18/#19)

Stage 1 uses **masked L1** and reports **masked MAE** as the selection
metric, computed identically for the model and the naive baseline.

- Training loss: sum of absolute errors over valid 100 m cells divided by
  the number of valid 100 m cells in the batch.
- Validation MAE (checkpoint selection): the same ratio accumulated over
  **all** validation batches (sum of numerators over sum of counts). It is
  never a mean of per-batch or per-patch values, so patches with unequal
  valid-cell counts do not carry equal weight.
- SSIM at 100 m is logged as a **secondary** diagnostic only. It never
  enters the loss and never gates checkpoint selection. It is evaluated on
  the pooled 100 m prediction with a fixed window and a fixed documented
  Kelvin data range; only windows whose contributing cells are all valid
  are counted, and the supported-window count is reported. When no window
  is supported the value is reported as unavailable, never as zero.
- Huber, thermal-aware, and other loss families stay out of Stage 1.

## Comparison universe

Both arms in Stage 1 read the **same** admitted patch set from the
published patch index and apply the same mask, the same nested 10×10
pooling, and the same reducer. A patch excluded by either arm is excluded
for both, with a recorded reason. Metrics are only comparable under this
identical universe.

## Loss normalization

Aggregate over **valid 100 m cells only**. Invalid cells are excluded from
both the numerator and the denominator, so they never dilute the mean.

- Never substitute zero for NaN.
- Never count an invalid cell in the denominator.
- Never treat the eligibility mask as all-valid.

The reduction family (mean, sum, weighted) beyond this masked-aggregation
rule is a later decision.

## `lst_prior`

`lst_prior` is an explicit model input, carried separately from the 28
frozen feature channels.

- **Training, validation, and test (the Stage-1 comparison):** native
  100 m LST is mean-pooled 10×10 to 1000 m, then block-expanded to the
  10 m grid. Each 10 m pixel takes the value of its containing 1000 m
  block. Both the model and the naive baseline use this regime, so their
  scores are directly comparable.
- **Inference:** native 100 m LST is block-expanded to the 10 m grid. Each
  10 m pixel takes the value of its containing 100 m cell.

Training and inference therefore use priors at different effective
resolutions (1000 m and 100 m). The 28-channel V3 order itself is
unchanged; adding a channel to the published stack would be a breaking
schema change. `lst_prior` enters the model as an additional single-band
input channel at the model boundary.

The stage-1 baseline is the 1000 m block-expanded prior itself. Expanding
the native 100 m target and pooling it back would reproduce the target
almost exactly and is **not** a baseline; that variant is rejected.

### Block geometry and alignment (fixed)

Prior blocks are formed on the **global canonical** EPSG:25833 100 m
lattice (`common/grid.py`), in groups of ten consecutive global rows and
columns, before any patch is cropped. Block identity therefore depends
only on global position: a given 1000 m block carries the same value in
every patch that touches it, and the value is identical on both sides of a
patch boundary. Whole contributing blocks are read even when they extend
past the 160×160 footprint; blocks are never aligned to patch corners and
the 16×16 patch is never pooled on its own.

### Native-invalid cells inside a block

A 1000 m block is built from the **native-valid** Landsat cells it
contains, using the same validity expression as
`training_eligible@100m` (ARD flag `0` and LST within `LST_RANGE_K`,
`data/qa/contracts.py`): the block value is the mean over its native-valid
cells only.

- Invalid native cells are never counted as zero and never as a value.
- A block with **no** native-valid cell has no prior. Any patch that
  requires such a block is excluded from the comparison, for both arms,
  with the reason recorded and counted. No value is invented for it.

This is a mean over native-valid observations, not over
`training_eligible` cells: the latter would make the degradation depend on
feature availability, which is exactly what the prior is meant to be
independent of. It does **not** impute any target value: invalid 100 m
target cells stay excluded from the loss and the metric as before.

### Model-boundary normalization and invalid predictor pixels

The prior enters the model as a fixed affine transform of the Kelvin
value, defined in `modeling/contracts.py` and applied by the reader; the
baseline consumes the same physical Kelvin prior. The transform is a fixed
constant, not a fitted statistic, so no split information leaks into it.

An admitted patch can still contain invalid 10 m **predictor** pixels
(individual feature channels are NaN where unavailable). At the model
boundary only:

- the published train-only scaler (`training/v1/scaler.json`) is applied
  first, exactly as fitted;
- invalid (non-finite) predictor pixels are then set to the neutral scaled
  value `0` so the model tensor is finite;
- the number of filled pixels is reported.

This touches neither the published feature stack, nor the eligibility
mask, nor the 100 m target. It is a model-input treatment, not target
imputation, and it does not change `mask_100m`.

## Tensor fields

Batch convention `(B, ...)`; `H100 = H10 / 10` and `W100 = W10 / 10`.

| Field | Grid | Shape | Role |
|-------|------|-------|------|
| `features` | 10 m | `(B, 28, H10, W10)` float32 | Fixed V3 channel order (first C of 28 for ablations) |
| `lst_prior` | 10 m | `(B, 1, H10, W10)` float32 | Explicit prior input, built per above |
| `target_100m` | 100 m | `(B, 1, H100, W100)` float32 | Native Landsat LST supervision |
| `mask_100m` | 100 m | `(B, 1, H100, W100)` bool | `training_eligible@100m` |
| `prediction_10m` | 10 m | `(B, 1, H10, W10)` float32 | Model output |

`prediction_100m` (the pooled prediction compared against `target_100m`)
is an intermediate derived from `prediction_10m`, not a separate field.

## Variant C (optional later baseline)

Variant C also feeds the native 100 m prior in training, instead of the
1000 m-pooled prior. It remains an **optional later baseline**, not the
main contract. It is recorded here only so the main contract is not
mistaken for the only option.

## Statement limits (Stage 1)

- The temporal split (2017-2023 train, 2024 validation, 2025 test) holds
  scenes apart by year. Accepted windows are anchored on a global lattice,
  so the same spatial anchor recurs across years: this is a temporal
  holdout at recurring locations, **not** evidence of independent spatial
  generalization. No spatial cross-validation is added in Stage 1.
- A pseudo-pair's coarse prior is derived from the same scene's Landsat
  observations. That is the intended degradation experiment (an explicit
  prior input), not a target-independent operational input, and results
  must be read as a comparison under that construction.
- The 2025 test split is never used for checkpoint selection.

## Relation to the current modelling scaffold

`modeling/contracts.py` keeps the synthetic `Batch`/`DatasetReader`/
`MaskedLoss` surface and `modeling/task.py`/`modeling/synthetic.py`
unchanged: they predict at 100 m and use plain MSE on an all-valid mask,
and they remain the CI lifecycle fixture. The real path is additive:
`RealBatch`/`RealSampleMeta` in `modeling/contracts.py`,
`modeling/patches.py` (reader, prior, scaler, fill), `modeling/metrics.py`
(pooling, masked L1/MAE, SSIM), `modeling/real_task.py` (10 m prediction
over features + prior), and `modeling/baseline.py` (naive prior-expand).
The synthetic configuration and its `validation/loss` smoke stay intact.

Alongside that MSE fixture, `ContractSyntheticDataModule`
(`modeling/synthetic.py`) emits contract-shaped tensors — 28x160x160
features, the normalized prior channel, a 16x16 target/mask, and a partially
valid mask (first row and column ineligible, invalid targets `NaN`) — and
feeds them through the same `RealLSTTask`, masked L1, and masked-MAE
checkpoint as the real path. `nox -s smoke-modeling-contract` exercises it
without reading GCS. It is a wiring gate only: a finite loss there proves the
tensor/loss/lifecycle path, never training quality.

The frozen geometry, temporal split, Stage-1 loss, and feature order are
declared in the `contract` block of `configs/modeling/_base.yaml` and asserted
against the contract constants by `modeling/run.py:contract_invariants`,
which fails closed on drift. They are declared for visibility, not exposed as
free parameters.
