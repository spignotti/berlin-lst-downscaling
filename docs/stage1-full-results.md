# Stage-1 full temporal run — deferred pending efficiency gate (issue #53)

**Status: blocked.** The full-run invocation described below is superseded by
`docs/stage1-efficiency.md`. No full Stage-1 fit is authorized until the
bounded efficiency work is complete and a separate plan explicitly approves
the full run. The prior result thresholds are retained for later reference;
they do not authorize an invocation or test access during efficiency work.

The first **full temporal** Stage-1 fit under the recovered residual lock: every
published train and 2024-validation patch, checkpoint selection on validation
only, then a **one-shot** 2025 test score after selection is frozen. This is the
Stage-1 ablation anchor for Tag 11 (issues #54+). It is not a re-opened method
search.

The method is frozen by `configs/modeling/stage1_locked.yaml` and its guard
(`modeling/guards.py:assert_stage1_lock`): C=10 (first ten V3 channels), depth 4,
width 32, LR 1e-3, WD 0, seed 0, masked L1, temporal split, residual prior with a
zero-initialized head, 20 epochs. Batch size, AMP, workers, and patience remain
operational retunes; none is changed here.

Status: **deferred** (2026-10-02). The prior full-run pre-registration remains
historical context only; do not append a run outcome before the efficiency gate
and subsequent approval.

## 1. Representation and cohort

- `prediction_10m = reconstructed_Kelvin_prior_10m + U-Net_delta_10m`; the prior
  is reconstructed only from the normalized prior input and the fixed affine
  constants, never from the target (`modeling/run.py`, `:RealLSTTask.forward`).
- Fit splits are `["train", "validation"]`; the 2025 test split is admitted and
  scored **once**, after the best checkpoint and its validation reload succeed.
  Test never selects or retrains a checkpoint (frozen contract, `docs/pseudo-pair-tensor-contract.md`).
- Comparison universe: every published patch, identical admission, mask, nested
  10x10 pooling, and cell-weighted MAE as the naive anchor. A model arm is
  comparable only when its recorded admitted **and** skipped patch IDs equal the
  baseline universe per split, not merely the counts
  (`docs/baseline-full-results.md`, "Comparison universe").

## 2. Pre-registered decision table

Written before the first fit. Thresholds are screening choices, adapted from the
recovery-probe screen (`docs/stage1-debug-results.md` §3) to the **full** naive
anchors (validation **1.61010 K**, test **1.39523 K**), not measured full-run
results.

Required evidence for any anchor tier (all must hold):

1. 20 complete epochs; every train/validation MAE finite.
2. Final train MAE ≤ 0.95 × epoch-1 train MAE (≥ 5% learning).
3. Final validation MAE ≤ 1.25 × best validation MAE (no divergence).
4. Selected checkpoint reload reproduces `best_metric` within `1e-3`.
5. Residual identity at init ≤ `0.001` K; selected pooled validation correction
   ≥ 5% of the full-validation naive MAE (≥ **0.080505 K**), i.e. not a prior
   passthrough.
6. Exact same-universe reconciliation against the committed baseline per split
   (admitted + skipped patch IDs), with matching valid-cell totals.
7. A finite one-shot 2025 test score, and a checksum-verified selected
   checkpoint retained beside the evidence.

| Tier | Condition on the scored splits |
|---|---|
| **GO** | best validation ≤ **1.61010 K** **and** test ≤ **1.39523 K** |
| **Usable anchor** | not GO, but best validation ≤ **2.41515 K** and test ≤ **2.092845 K** (1.5× each split's own naive); deficits documented, not a performance win |
| **NO-GO** | a comparably completed run whose scores or learning evidence fail the above; flat, divergent, passthrough, or catastrophic/non-finite fit |

An incomplete run, absent test score, missing checkpoint, or corrupt evidence is
**technical incomplete: no anchor**, not a claimed model result. No outcome
authorizes a retune or a second full run in this issue. Beating naive is the
scientific target; staying near it without beating it is still a usable anchor if
documented honestly. Multi-10 K or init-scale scores are not.

## 3. Cost bound and artifacts

- One on-demand `n1-standard-4` + `NVIDIA_TESLA_T4` Vertex Custom Job in
  `europe-west3`, no retries, single worker pool. Server timeout **48 h**;
  projected compute ceiling **$50**, admitted only when the freshly verified
  regional rate is ≤ **$1.03/h** (`scripts/operators/launch_vertex_modeling.py`).
  The exposure check is an admission estimate, not a provider spending cap; the
  actual billed amount appears in Cloud Billing and is reported as *estimated*
  unless attributed.
- Retained under the approved QA prefix, all create-only: the selected
  checkpoint (`best.ckpt`) uploaded first, then `evidence.json` last, containing
  the epoch curve, summary, fit and test `data_scope`, checkpoint SHA-256/size,
  W&B reference, source SHA, and image digest.
- Full-run throughput is `[uncertain: the 6-epoch probe does not reliably predict
  streamed full-run runtime]`; the 48 h ceiling is roughly 2× a heuristic
  projection, not a measured guarantee.

## 4. Results

_No full Stage-1 run has been authorized or performed. Any future result requires
a separate approved plan after `docs/stage1-efficiency.md` is complete._

## 5. Limitations

- Temporal holdout at recurring locations: scenes are held apart by year, but
  accepted windows are anchored on a global lattice, so the same spatial anchor
  recurs across years. This is **not** evidence of independent spatial
  generalization.
- The coarse prior is derived from the same scene's Landsat observations; that is
  the intended degradation experiment, not a target-independent operational
  input. Results are a comparison under the pseudo-pair construction.
- Supervision is at 100 m; this does not establish 10 m detail accuracy.
- The 2025 test split is scored once and never used for checkpoint selection.
