# Stage-1 full temporal run (issue #53)

**Decision: GO.** The single Vertex full job succeeded, matched the committed
baseline universe, and beat both full-split naive anchors. Independent
validation (`validate_stage1_full.py`) returns **GO** (exit 0). Efficiency gate
closed (`docs/stage1-efficiency.md`); verification job cancelled
(`docs/archive/stage1-training-readiness.md`). Frozen runtime: job-local cache, batch 4,
workers 0, `pin_memory: false`, `16-mixed`.

The first **full temporal** Stage-1 fit under the recovered residual lock: every
published train and 2024-validation patch, checkpoint selection on validation
only, then a **one-shot** 2025 test score after selection is frozen. This is the
Stage-1 ablation anchor for Tag 11. It is not a re-opened method search.

The method is frozen by `configs/modeling/stage1_locked.yaml` and its guard
(`modeling/guards.py:assert_stage1_lock`): C=10 (first ten V3 channels), depth 4,
width 32, LR 1e-3, WD 0, seed 0, masked L1, temporal split, residual prior with a
zero-initialized head, 20 epochs. Batch size, AMP, workers, and patience remain
operational retunes; none is changed here.

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
- Measured wall clock for this job was ~2 h 45 min (start 10:38:54Z, end
  13:23:48Z). Estimated Vertex compute at the admitted $1.00/h rate is ~$2.75;
  Cloud Billing remains the attributed source.

## 4. Results

**Verdict: GO** — best validation **0.50285 K** ≤ naive **1.61010 K**, and
one-shot 2025 test **0.53257 K** ≤ naive **1.39523 K**. All seven required
evidence checks passed.

### What ran

| Field | Value |
|---|---|
| Config | `stage1_locked` (frozen Stage-1 method; full temporal train/val) |
| Vertex job | `projects/996559849187/locations/europe-west3/customJobs/8774705047146594304` |
| Region / machine | `europe-west3` / `n1-standard-4` + 1× `NVIDIA_TESLA_T4` (on-demand) |
| Source SHA | `8bd094b0758e661873bc338ef2ca6bfcad3f5883` |
| Image | `europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:8772b5c8a6349448e820404c75008c59f9df8b63aa1099088e7deffeee90d887` |
| Run label | `stage1-full-20261005T103017Z-anchor` |
| Terminal state | `JOB_STATE_SUCCEEDED` |
| Timing | created 10:32:06Z, started 10:38:54Z, ended 13:23:48Z (~2 h 45 min wall) |
| Evidence | `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/stage1-full-20261005T103017Z-anchor/evidence.json` |
| Checkpoint | `…/best.ckpt` SHA-256 `ae3a61620f6451e6066aa5b7a3d1b66f9bf81bb96d2998f9b31c4159e3e5f14b` (376 998 694 bytes) |
| W&B | [30e5g89u](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/30e5g89u) |
| Cost (estimate) | ~$2.75 Vertex compute at the admitted $1.00/h rate |

### Cohort (same universe as the committed baseline)

| Split | Requested | Admitted | Skipped | Valid cells | Years |
|---|---|---|---|---|---|
| train | 6872 | 6872 | 0 | (fit only) | 2017–2023 |
| validation | 1342 | 1341 | 1 | 341 588 | 2024 |
| test (one-shot) | 1803 | 1803 | 0 | 459 564 | 2025 |

Admitted and skipped patch IDs match `docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json` per split. Fit reads train/validation only; test is scored once after checkpoint freeze.

### Curve (cell-weighted masked MAE @ 100 m)

| Epoch | train MAE (K) | val MAE (K) |
|------:|--------------:|------------:|
| 1 | 1.0431 | 1.0098 |
| 2 | 0.9335 | 0.9760 |
| 3 | 0.8978 | 0.9194 |
| 4 | 0.8690 | 0.9215 |
| 5 | 0.8372 | 0.8522 |
| 6 | 0.7928 | 0.7971 |
| 7 | 0.7366 | 0.7386 |
| 8 | 0.6811 | 0.6923 |
| 9 | 0.6381 | 0.6280 |
| 10 | 0.6024 | 0.6175 |
| 11 | 0.5723 | 0.5992 |
| 12 | 0.5464 | 0.5868 |
| 13 | 0.5239 | 0.5806 |
| 14 | 0.5041 | 0.5682 |
| 15 | 0.4883 | 0.5486 |
| 16 | 0.4740 | 0.5264 |
| 17 | 0.4624 | 0.5303 |
| 18 | 0.4505 | 0.5234 |
| 19 | 0.4425 | **0.5028** (selected) |
| 20 | 0.4301 | 0.5143 |

Selected checkpoint: best validation **0.50285 K** at epoch **19**. Reload
recheck matched within tolerance. Residual identity at init **0.0 K**; pooled
validation \|correction\| **1.4846 K** (≫ 0.0805 K floor). One-shot 2025 test
MAE **0.53257 K** over 1803 patches / 459 564 valid cells.

### Same-universe naive compare

| Split | Model MAE (K) | Naive MAE (K) | Model / naive |
|---|---:|---:|---:|
| validation (best) | 0.50285 | 1.61010 | 0.31× |
| test (one-shot) | 0.53257 | 1.39523 | 0.38× |

### Predeclared screen

| Criterion | Result |
|---|---|
| 20 complete epochs; all train/val MAE finite | pass |
| Final train MAE ≤ 0.95 × epoch-1 (≥ 5% learning) | pass — 1.0431 → 0.4301 (~59% drop) |
| Final val MAE ≤ 1.25 × best (no divergence) | pass — 0.5143 / 0.5028 ≈ 1.023 |
| Reload reproduces `best_metric` within 1e-3 | pass |
| Residual identity ≤ 0.001 K; correction ≥ 0.0805 K | pass — 0.0 K / 1.4846 K |
| Same-universe IDs + valid-cell totals vs baseline | pass |
| Finite one-shot test + checksummed `best.ckpt` | pass |
| GO: best val ≤ 1.61010 K **and** test ≤ 1.39523 K | **pass** |

```text
uv run python scripts/validators/validate_stage1_full.py \
  --evidence <downloaded evidence.json>
```

→ **VERDICT: GO** (exit 0).

### Interpretation

- This is a **scientific GO and a usable Stage-1 ablation anchor.** The residual
  lock beats the committed full-split naive prior on both scored splits by a
  wide margin under the frozen method and the cheap runtime.
- Learning is clear and stable: train MAE falls ~59% over 20 epochs; validation
  improves nearly monotonically to epoch 19 with no late divergence.
- The run is the single authorized full temporal fit under issue #53. No retune
  and no second full run are authorized by this outcome.

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
