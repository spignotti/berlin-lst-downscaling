# Stage-5 thermal-aware loss audit (issue #23)

Audit of the Stage-5 `thermal_aware` objective before the v6 ablation arm L5
runs. L5 uses the L4 inputs (22 channels, features/v6, training/v3), trains
with `16-mixed`, and selects checkpoints by masked MAE at 100 m on validation
2024. The 2025 test split informed nothing here. All numbers come from
offline synthetic experiments on CPU. No run, pipeline, or bucket was touched.

## Scope

The objective is (`modeling/metrics.py:151-188`):

```
loss = masked_L1_100m + 0.1 * (1 - mean SSIM over fully valid 7x7 windows)
                      + 0.1 * masked L1 of 4-neighbour finite differences
```

The weights are `THERMAL_AWARE_SSIM_WEIGHT = THERMAL_AWARE_GRAD_WEIGHT = 0.1`
(`metrics.py:71-72`). The SSIM uses a uniform 7×7 window without padding and
the fixed Kelvin data range `LST_RANGE_K = (150, 400)` (`metrics.py:28`,
`data/qa/contracts.py:34`). So `c1 = (0.01·250)² = 6.25 K²` and
`c2 = (0.03·250)² = 56.25 K²` (`metrics.py:237-238`).

The audit covers items a–f of the task. Two implementation bugs were found
and fixed (b and e). The SSIM term being close to inert is a methodology
question and is left unchanged in code.

| Item | Topic | Status |
|---|---|---|
| a | Term contributions and scales | **Methodology question**: the SSIM term is close to inert |
| b | SSIM support and masking | **Bug fixed**: NaN could leak from unsupported windows (eval path) |
| c | Scale of the loss (100 m vs 10 m) | OK, documented: nothing constrains 10 m structure |
| d | v6 L5 run-lock consistency | OK: L5 differs from L4 only by the loss |
| e | Numerical stability under `16-mixed` | **Bug fixed**: fp32 cancellation in the SSIM moments |
| f | Logging of term contributions | OK |

## Findings

### a. Term contributions and scales: methodology question

Setup: a throwaway script outside the repository. The table below is not
reproducible from the repository as is. Its central claim (weighted SSIM
< 1 % of L1 in value and gradient; gradient term 1–50 % of the L1 gradient)
is asserted on unmasked patches by
`tests/test_stage5_loss_audit.py::test_ssim_term_is_small_against_l1_on_realistic_patches`.
Batch 4 × 16×16 cells at 100 m (the real patch footprint,
`modeling/contracts.py:134-136`). The target is 300 K plus a Gaussian-smoothed
field (correlation ~2 cells) with local std σ ∈ {0.5, 1, 2, 3} K. Two samples
carry partial masks. Prediction cases:

- *prior*: the 8×8-cell block mean of the target, a stand-in for the coarse
  prior that a residual model starts from;
- *prior + 1 K*: the same with a 1 K bias;
- *target + N(0, 0.5)* and *target + N(0, 1.5)*: white-noise error bounds.

The table reports term values and the L2 norm of each **weighted** term's
gradient with respect to the pooled prediction.

| Case | σ (K) | L1 (K) | 1−SSIM | 0.1·(1−SSIM) | 0.1·grad-L1 (K) | ‖∇L1‖ | ‖∇ 0.1(1−SSIM)‖ | ‖∇ 0.1·grad‖ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior + 1 K | 0.5 | 0.983 | 2.9e-3 | 2.9e-4 | 0.016 | 3.32e-2 | 5.3e-5 | 2.6e-3 |
| prior + 1 K | 1.0 | 1.133 | 1.1e-2 | 1.1e-3 | 0.035 | 3.32e-2 | 1.0e-4 | 2.6e-3 |
| prior + 1 K | 2.0 | 1.551 | 3.8e-2 | 3.8e-3 | 0.062 | 3.32e-2 | 1.9e-4 | 2.6e-3 |
| prior + 1 K | 3.0 | 2.292 | 7.4e-2 | 7.4e-3 | 0.097 | 3.32e-2 | 2.5e-4 | 2.4e-3 |
| prior | 1.0 | 0.710 | 1.1e-2 | 1.1e-3 | 0.035 | 3.32e-2 | 1.0e-4 | 2.6e-3 |
| target + N(0, 0.5) | 1.0 | 0.418 | 4.8e-3 | 4.8e-4 | 0.059 | 3.32e-2 | 7.3e-5 | 4.7e-3 |
| target + N(0, 1.5) | 1.0 | 1.205 | 3.9e-2 | 3.9e-3 | 0.166 | 3.32e-2 | 2.0e-4 | 4.8e-3 |

Readout:

- **SSIM is close to inert.** The weighted SSIM term is 0.03–0.3 % of L1 in
  value. Its gradient norm is 0.2–0.8 % of the L1 gradient norm. The cause is
  the confirmed hypothesis: `c2 = 56.25 K²` dwarfs local 100 m variances of
  ~0.2–9 K². The contrast-structure factor `(2σxy + c2)/(σx² + σy² + c2)` then
  stays at ≈ 0.86–1 even for a structureless prediction. The luminance factor is
  ≈ 1 − ΔT²/(2·300²), so a 1 K bias moves it by ~1e-5. That is why *prior* and
  *prior + 1 K* have the same 1−SSIM.
- **Assumption.** These numbers assume local 100 m std of 0.5–3 K. The real
  local 7×7 variance of the train-split targets (2017–2023) was not measured
  here. The logged `thermal_*_weighted` shares of the first L5 runs are the
  check on real data.
- **The gradient term is active but secondary.** It is 2–15 % of L1 in value
  and about 7–15 % of the L1 gradient norm.
- In effect, L5 trains **masked L1 + 0.1 · gradient-L1**. Any L4→L5
  difference should be attributed to the gradient term, not to SSIM.
- Sensitivity (diagnostic only, not a proposal): with σ = 1 K and the *prior*
  case, 1−SSIM is 0.010 at data range 250 K (`c2 = 56.25`), 0.145 at 40 K
  (`c2 = 1.44`), and 0.437 at 10 K (`c2 = 0.09`).
- **No logged values are available as an in-repo check.**
  `docs/ablation-full-results.md` holds only the v1 Stage-5 MAE (validation
  0.4762 K vs Stage 4 0.4996 K, single seed, different image). The
  `train|validation/thermal_*_weighted` values exist only in W&B and are not
  reproduced in the repository. The single-seed v1 gain therefore cannot be
  split between terms from repository evidence.

### b. SSIM support and masking: bug fixed

What is correct (`metrics.py:273-320`):

- Invalid cells are zero-filled (`torch.where`). A window is *supported*
  only if all 49 cells are valid (`avg_pool2d(valid) == 1.0`, `metrics.py:309`).
  The filled values therefore never enter a supported window. This was
  verified by changing invalid predictions to −5000 K and invalid targets to
  NaN: the loss and the gradient are bit-identical, and the gradient at
  invalid cells is exactly 0
  (`test_thermal_objective_ignores_invalid_cell_values`).
- A patch with no supported window adds nothing to the sum or the count. A
  batch with no supported window drops the SSIM term, so the loss equals
  L1 + 0.1·grad (`metrics.py:171-176`,
  `test_thermal_objective_without_supported_window_is_l1_plus_gradient`). The
  mean is window-weighted across the batch, not patch-weighted, which matches
  the "support-weighted" wording in `docs/ablation-stage-configs.md`.
- Clamping: the training loss clamps the prediction to [150, 400] K
  (`metrics.py:166`). The validation diagnostic `validation/ssim_100m` does
  not clamp (`real_task.py:800-805`). The target is always clamped. This
  matches the docstrings. Inside training, the clamp zeroes the SSIM gradient
  only for predictions outside the physical range, where L1 still acts.

**Bug (fixed).** The per-sample sum was `(ssim_map * supported).sum()`. A
non-finite value in an *unsupported* window gives `0 · inf = NaN`, which made
the total NaN. The finiteness check only covers supported windows
(`metrics.py:316`), so the NaN passed silently. This happened on the
unclamped evaluation path, for example with a valid but huge prediction in a
cell that lies in no supported window. The fix sums
`torch.where(supported, ssim_map, 0)` (`metrics.py:317`). Regression test:
`test_ssim_ignores_non_finite_values_outside_supported_windows` (old code:
total NaN; new code: count 1, total 1.0). The training loss was not affected
because it clamps.

### c. Scale: OK, documented

All three terms act on `prediction_100m = pool_10m_to_100m(10 m output)`
(`real_task.py:768`). With `aggregation: arithmetic`, the pinned value in
every locked profile, this is the 10×10 block mean (`aggregation.py:124`).
The pooling is arithmetic, not radiometric (T4). Consequence: **nothing in
the loss constrains structure at 10 m.** Any zero-mean pattern inside a
100 m block leaves every term unchanged
(`test_thermal_objective_cannot_see_structure_inside_a_100m_block`: a ±5 K
checkerboard inside every block leaves the loss unchanged). The "structural"
and "gradient" terms are 100 m structure terms (7×7 cells ≈ 700 m, and
100 m neighbour differences). 10 m detail comes only from the inputs and the
network's inductive bias. No change was made.

### d. Run-lock consistency for v6 L5: OK

The v6 scaffold was read from PR #107 (`feat/v6-ablation-scaffold`, not
modified), which has since been merged into `main`. The references below are to `main`:

- `diff configs/modeling/v6_l4_locked.yaml configs/modeling/v6_l5_locked.yaml` differs only in the header
  comment, `ablation_arm` (L4/L5), `contract.loss` (`masked_l1` /
  `thermal_aware`), and `output_root` (`…-v6-l4/…` / `…-v6-l5/…`). The
  22 channels and their order, `cache_max_bytes`, and `feature_order:
  v6_registry` are identical. Everything else comes from the shared
  `_v6_ablation_base.yaml`: data roots, `training/v3`, batch 4, `16-mixed`,
  `aggregation: arithmetic`, U-Net depth 4 / width 32, lr 1e-3, wd 0,
  residual prior.
- `configs/ablation/v6_matrix.yaml:7-8`: L4 and L5 both use seeds 0, 1, 2 and
  `max_epochs: 20`.
- Enforcement: `_assert_ablation_lock_v6` (`modeling/guards.py:353`) pins
  every base field exactly (`_V6_LOCK_EXPECTED`, `guards.py:324`, including
  `trainer.precision` and `aggregation`). It checks channel names and order
  against the arm registry and the expected loss per arm (`channels.py:140-141`:
  L4 → `masked_l1`, L5 → `thermal_aware`). Seed and `max_epochs` are typed
  parameters. Checkpoint selection monitors `validation/mae_100m`. The same
  key is used by the optional early stopping, which defaults to off, for
  every arm (`run.py:716`).
- Run identity records git commit, image digest, data fingerprints,
  aggregation, seed, and early stopping. It does not record the loss name.
  The loss is visible through `resolved_config` and the config name.

Note: the bug fixes in this change touch `metrics.py`. They affect the L5
training loss (by ≲1e-3 in the SSIM map, see e) and the logged
`validation/ssim_100m` diagnostic of every arm. They do not affect masked L1
or MAE. To keep "only the loss differs" exact, L4 and L5 should run from the
same commit and image (see Recommendations).

### e. Numerical stability under `16-mixed`: bug fixed

- **dtype path in training.** The U-Net runs under autocast. The residual
  head output is cast back, and the prediction is
  `prior.float() + delta.float()` (`real_task.py:765`), so the 10 m
  prediction is fp32. Pooling is `reshape().mean()` on fp32 (autocast does
  not cast `mean`). L1 and the gradient term use elementwise fp32 ops only.
  SSIM runs inside `torch.autocast(enabled=False)` with explicit `.float()`
  inputs (`metrics.py:295`), which covers all of its `conv2d` calls.
  Validation disables autocast around the forward, the pooling, and both SSIM
  calls (`real_task.py:786`). Without the residual prior, a raw fp16 Kelvin
  output would have 0.25 K spacing at 300 K. All locked profiles require the
  residual prior, so this does not apply. The `|delta|` of ~1.5 K is
  resolved to ~1e-3 K in fp16.
- **Bug (fixed): fp32 cancellation in `E[x²] − μ²`.** At x ≈ 300 K,
  `E[x²] ≈ 9e4 K²`. The fp32 local variance had a mean error of 0.016 K² and
  a max error of 0.067 K², against local variances of 0.17–6.5 K²
  (3–10 % at σ ≤ 1 K). In the SSIM map this was up to ~1e-3 absolute error.
  That is the same order as `1 − SSIM` itself (3e-3–4e-2). The fix shifts
  predictions and targets by a detached per-sample reference (the mean of
  the clamped valid target cells) before forming the second moments. The luminance
  factor keeps the absolute means (`metrics.py:249-268, 297-308`). This is
  exact in real arithmetic, so constants, window, data range, clamp, and
  formula are unchanged. The reference uses the clamped target, so a huge
  finite target is still clamped as before
  (`test_ssim_reference_shift_keeps_target_clamp_for_huge_targets`). Regression test
  `test_ssim_fp32_matches_fp64_reference_at_kelvin_scale` (40 windows, fp32
  vs an independent fp64 two-pass reference): old max error 9.9e-4 (fails),
  new max error 1.2e-7 (passes).
- **Out-of-range predictions.** Training clamps to [150, 400] K before SSIM,
  so moments stay bounded. Evaluation does not clamp. Non-finite values in
  supported windows raise. In unsupported windows they are now ignored (b).
- **Non-finite raises mid-training.** `masked_abs_error_sums`
  (`metrics.py:59`), `masked_gradient_stats` (`metrics.py:118`), and
  `masked_ssim_stats` (`metrics.py:316`) raise `ValueError` on a non-finite
  value in a valid cell or window. This aborts the run. It does not let AMP
  skip the step. It matches the repository's fail-closed rule: no silent
  skipping or clamping. With the residual prior and clamped SSIM, a raise
  needs a non-finite network output. That is a real fault, not an fp16
  quirk. GradScaler still handles overflowing *gradients* by skipping steps.
  Status: intended.
- Minor, not a bug: `supported` relies on `avg_pool2d` of 49 ones being
  exactly 1.0. That holds for sum-then-divide kernels (verified on CPU). The
  v1 Stage-5 GPU run logged non-zero SSIM support.

### f. Logging: OK

For `thermal_aware`, train and validation each log per epoch
(`on_step=False, on_epoch=True`, `real_task.py:742-751`):
`thermal_l1_unweighted`, `thermal_l1_weighted`, `thermal_ssim_unweighted`,
`thermal_ssim_weighted`, `thermal_grad_unweighted`, and
`thermal_grad_weighted`. Validation also logs `ssim_coverage` and
`grad_coverage`, and train logs `ssim_supported`. The values are
support-weighted means (sum of numerators over sum of supports across the
epoch). The weighted metrics use the same `THERMAL_AWARE_*_WEIGHT` constants
as the loss (`real_task.py:672-679`). The epoch-metrics recorder reads these keys
(`tests/test_issue80_thermal_audit.py`), so term shares can be reported from
`epoch_metrics.json` as well as W&B. Caveat for the thesis: the epoch value
is a support-weighted mean, while the optimiser sees the per-batch ratio.
The two agree closely but are not identical.

## Numbers (summary)

| Quantity | Value |
|---|---|
| `c1`, `c2` | 6.25 K², 56.25 K² |
| Local 7×7 variance at 100 m (synthetic σ = 0.5–3 K) | 0.17–6.5 K² |
| Weighted SSIM term / L1 (value) | 0.03–0.3 % |
| ‖∇ weighted SSIM‖ / ‖∇ L1‖ | 0.2–0.8 % |
| Weighted gradient term / L1 (value) | 2–15 % |
| ‖∇ weighted grad‖ / ‖∇ L1‖ | 7–15 % |
| fp32 local-variance error, old formula (mean / max) | 0.016 / 0.067 K² |
| SSIM fp32 vs fp64 max error, old / new | 9.9e-4 / 1.2e-7 |

## Recommendations for the v6 L5 arm

1. **No code change is required before L5 beyond this branch.** Run L5 (and
   ideally all v6 arms) from a commit that contains these fixes. Keep L4 and
   L5 on the same commit and image, so the only difference between them is
   `contract.loss`. If L4 has already run on an older image, its MAE is
   unaffected. Only its `validation/ssim_100m` diagnostic can differ, by
   ≲1e-3.
2. **Report L5 honestly as "masked L1 + 0.1 · 100 m gradient-L1 (+ a
   near-zero SSIM term)".** Use the logged `thermal_*_weighted` shares to
   confirm this on real data in the thesis.
3. **Nothing to change in the v6 scaffold (PR #107).** Optional: record
   `contract.loss` in the v6 run identity payload, so the loss is visible
   next to seed and early stopping without opening `resolved_config`.

### Open questions for the author (not decided here)

- **Q1: SSIM scaling.** Keep the SSIM term as specified (near-inert, which
  must be stated in the thesis), or make it effective as a new option, with
  today's behaviour staying the default? Options:
  - *Data range from target statistics* (e.g. a train-split robust range or
    local std scale). Pro: standard SSIM semantics on the actual dynamic
    range. Con: a new constant derived from data, and it needs a train-only
    fit.
  - *Per-patch normalisation before SSIM* (z-score each patch with
    target statistics). Pro: scale-free and simple. Con: changes what
    "structure" means, and is unstable on flat patches.
  - *Re-weighting* (e.g. SSIM weight ~1–10). Pro: a single number. While `c2`
    dominates, 1−SSIM ≈ local variance of (prediction − target) / `c2`, i.e.
    a local high-pass L2 term on the error. Re-weighting (effective weight
    ~ w/56 K⁻²) would make that term matter. Con: it is then an L2 high-pass
    penalty rather than SSIM in the usual sense, and the weight is a new
    tuned number.
  - *Drop SSIM* and call the loss "L1 + gradient". Pro: honest and simplest.
    Con: departs from the specified Stage-5 loss and from v1 Stage 5.
- **Q2:** Is the window-weighted (rather than patch-weighted) SSIM mean the
  intended definition? It matches the current docs.

## Second loss variant

Budget context (figures given in the task, not measured here): ~2.5 USD and
~2.5 GPU-h per run, so an extra arm of
3 seeds costs ~7.5 USD and ~7.5 GPU-h. The 27-run matrix is ~68 USD of a
~200 USD budget, so money is not the constraint. Information is the
constraint.

- **L1 + SSIM only.** With today's constants, this is numerically L4: the
  weighted SSIM is < 0.3 % of the loss and < 1 % of the gradient. The
  expected result is a null indistinguishable from L4 seed noise. **Not
  worth it.**
- **L1 + gradient only.** This is numerically L5. The gradient carries
  essentially all of L5's non-L1 signal. It would duplicate L5. **Not worth
  it.**

**Recommendation: do not add a second loss arm in its current form.** Spend
the seeds on L5 itself. If the author wants to test SSIM, the only
informative extra arm is "L1 + SSIM with an effective scale" (Q1, e.g. a
target-derived data range). That needs a methodology decision first and is
a new option, not a change to L5.
