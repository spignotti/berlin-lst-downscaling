# Pseudo-pair and tensor contract

Normative contract for the training data-to-model boundary (WB3). It
freezes how a training sample maps coarse 100 m LST supervision to the
10 m downscaling product, so the dataset, loss, and later patch work
share one definition.

Status: **normative, partly implemented.** Patch geometry and the sampling
index are delivered (see *Patch geometry* and the WB3 patch index in
`data-sources-and-contracts.md`); no dataset, loss, or training code in
the repository implements this contract yet, and the modelling package is
a synthetic scaffold (see *Relation to the current modelling scaffold*).

## Scope

This document fixes the pseudo-pair semantics, the `lst_prior` input, the
tensor fields and their grids, and the masked-loss normalization.

Out of scope, left to later WB3 decisions:

- the sampler,
- the concrete loss family beyond masked aggregation,
- Zarr versus COG I/O,
- handling of invalid native 100 m cells inside a 1000 m `lst_prior`
  block (masked mean versus NaN propagation).

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

- **Training:** native 100 m LST is mean-pooled 10×10 to 1000 m, then
  block-expanded to the 10 m grid. Each 10 m pixel takes the value of its
  containing 1000 m block.
- **Inference:** native 100 m LST is block-expanded to the 10 m grid. Each
  10 m pixel takes the value of its containing 100 m cell.

Training and inference therefore use priors at different effective
resolutions (1000 m and 100 m). The 28-channel V3 order itself is
unchanged; adding a channel to the published stack would be a breaking
schema change. `lst_prior` enters the model as an additional single-band
input channel at the model boundary.

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

## Relation to the current modelling scaffold

`modeling/contracts.py` (`Batch`, `DatasetReader`, `MaskedLoss`) and
`modeling/task.py` are a **synthetic scaffold**: they predict at 100 m and
use plain MSE on an all-valid mask. They do not implement this contract,
and this document does not change them. The real reader, the masked loss,
and the 10 m prediction path are WB3 work.
