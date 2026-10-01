# Stage-1 learning recovery — diagnosis and pre-registration (issue #47)

Issue #47 recovery loop. The bounded Stage-1 probe (#45) was a **NO-GO** for the
full temporal run: six epochs ended at 294.84 K validation MAE against 1.5261 K
for the same-cohort naive prior-expand baseline (≈193× worse), barely moving off
its initialization (`docs/stage1-probe-results.md`). This document records (1)
the static and measured root-cause diagnosis, and (2) the **pre-registration** of
the recovery trial and its go/no-go screen, written before the first fit.

## 1. Static data-flow audit

**Observed facts** (post-`1f2645ef`; no fit):

| Element | Where | What it shows |
|---|---|---|
| Task output | `modeling/real_task.py:467-471` | `cat([features, lst_prior]) → UNet → prediction`; the head output **is** the 10 m LST map, with no prior bypass. |
| Head | `modeling/unet.py:133` | `head = Conv2d(widths[1], 1, 1)`, default init (bias ≈ 0); no custom init anywhere in `modeling/`. |
| Target | `modeling/contracts.py:183` | `target_100m` is native Landsat LST **in Kelvin**. |
| Prior input | `modeling/patches.py:534-536`, `contracts.py:144-152` | `lst_prior` enters the model **affinely normalized** to `[−1, 1]` (offset 275 K, scale 125 K). |
| Loss / metric | `modeling/metrics.py:47-96` | Exact 10×10 pooling, cell-weighted masked L1/MAE vs Kelvin target. |
| Naive arm | `modeling/baseline.py:211-227` | Consumes the **physical Kelvin** prior (`lst_prior_k`) through the same pooling/mask. |
| Active channels | `configs/modeling/stage1_locked.yaml:7-9`, `data/features/contracts.py` | C=10 = 6 S2 bands + 4 indices, all train-only z-scored; no channel carries a ~300 K offset. |
| Optimizer | `modeling/real_task.py:503-508` | Bare AdamW (LR 1e-3), no scheduler. |

**Grounded inference:** at initialization the head output is ≈ 0 K, while the
target is ≈ 300 K. The masked MAE of a near-zero prediction is therefore ≈ the
mean target — matching the recorded epoch-1 train MAE of 303.06 K. The network
must invent the entire Kelvin field; the prior is only a normalized input channel.

## 2. Bounded CPU zero-fit diagnostic (no fit, no GPU)

Read-only run against the published index. A deterministic scene-spread subset of
**4 train + 4 validation** admitted patches (`real_task.ProbeScope`, per-scene
cap 8). For each split: reconstructed the Kelvin prior from the normalized input
(`prior_k = lst_prior × 125 + 275`), pooled it with the shared metric path, and
compared the **untrained** absolute-head prediction to the target under the same
mask.

Reproduce (needs ADC; read-only; no submission):

```bash
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json \
uv run python - <<'PY'
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from berlin_lst_downscaling.modeling.contracts import PRIOR_AFFINE_OFFSET_K, PRIOR_AFFINE_SCALE_K
from berlin_lst_downscaling.modeling.metrics import masked_abs_error_sums, pool_10m_to_100m
from berlin_lst_downscaling.modeling.real_task import ProbeScope, RealLSTTask, RealPatchDataModule
from berlin_lst_downscaling.modeling.run import real_source_config

with initialize_config_dir(config_dir="configs/modeling", version_base=None):
    cfg = compose(config_name="stage1_locked")
dm = RealPatchDataModule(
    real_source_config(cfg), batch_size=4, mode="stream", num_workers=0,
    shuffle_train=False, n_active_channels=10, splits=("train", "validation"),
    probe_scope=ProbeScope(max_refs_per_split={"train": 4, "validation": 4}, max_refs_per_scene=8),
)
dm.setup()
task = RealLSTTask(n_active_channels=10, base_width=32, depth=4).eval()
for split, loader in (("train", dm.train_dataloader()), ("validation", dm.val_dataloader())):
    b = next(iter(loader))
    pk = b.lst_prior * PRIOR_AFFINE_SCALE_K + PRIOR_AFFINE_OFFSET_K
    naive = masked_abs_error_sums(pool_10m_to_100m(pk), b.target_100m, b.mask_100m)
    with torch.inference_mode():
        pred = pool_10m_to_100m(task(b))
        init = masked_abs_error_sums(pred, b.target_100m, b.mask_100m)
    print(split, "target_mean_K", float(b.target_100m[b.mask_100m.bool()].mean()),
          "prior_mean_K", float(pk.mean()), "naive_mae_K", float(naive[0]/naive[1]),
          "init_pred_mean_K", float(pred.mean()), "init_mae_K", float(init[0]/init[1]),
          "patch_ids", [m.patch_id for m in b.metadata])
PY
```

Result (recorded 2026-10-01, clean tree at `1f2645ef`):

| Split | Patches | Valid cells | Target mean (K) | Prior (K) mean | Naive pooled prior MAE (K) | Untrained pred mean (K) | Untrained pooled MAE (K) |
|---|---|---|---|---|---|---|---|
| train | 4 | 1014 | 305.379 (std 3.949) | 305.270 | **1.6501** | **−0.0754** | **305.454** |
| validation | 4 | 1007 | 304.155 (std 5.486) | 303.955 | **1.7725** | **0.0849** | **304.070** |

Patch IDs used — train: `LC08_L2SP_192023_20170830_02_T1:E394790N5833610`,
`LC08_L2SP_192023_20180513_02_T1:E394790N5833610`,
`LC08_L2SP_192023_20180529_02_T1:E394790N5833610`,
`LC08_L2SP_192023_20180716_02_T1:E394790N5833610`; validation:
`LC08_L2SP_192023_20240513_02_T1:E394790N5833610`,
`LC08_L2SP_192023_20240801_02_T1:E386790N5819210`,
`LC08_L2SP_192023_20240902_02_T1:E389990N5832010`,
`LC08_L2SP_192024_20240513_02_T1:E394790N5833610`. No exclusions.

**Interpretation:** the untrained absolute head predicts ≈ 0 K, so its masked MAE
≈ the mean target (304–305 K). The reconstructed Kelvin prior sits at the target
center and pools to a 1.65–1.77 K MAE. The scale gap is structural, not a slow
optimization schedule. Feature channels are ~O(1) after scaling (some large
per-pixel excursions come from zero-filled invalid predictor pixels), confirming
no active input channel carries the Kelvin offset.

**Decision:** the scale hypothesis is **confirmed**. Proceed to a probe-only
Kelvin-residual representation; no change to target, loss, pooling, mask, channel
order, or published data.

## 3. Pre-registration (written before the first fit)

### Representation under test (probe-only)

`prediction_10m = reconstructed_Kelvin_prior_10m + U-Net_delta_10m`, where the
prior is reconstructed only from the normalized prior input and the fixed affine
constants (`patches.py:534-536`, `contracts.py:151-152`) — never from the target.
The U-Net's final head is zero-initialized for the recovery mode only, so at init
the model reproduces the prior-expand arm exactly. Target, masked L1, pooling, and
the 100 m MAE remain unchanged. Non-probe profiles keep their current behavior.

### Cohort

The existing `stage1_probe` selection is reused unchanged: scene-spread
`train 128 / validation 64`, ≤ 8 refs/scene, train/validation only, 2025 test
never read. Because selection is deterministic over the same published index, the
admitted cohort should match #45 exactly; the same-cohort naive MAE is recomputed
from the retained patch IDs and the committed baseline report, and is expected to
be **1.5261 K**.

### Go/no-go screen (trial = 6 epochs)

A trial is **GO** only if all of the following hold:

1. Six complete epochs; every train/validation MAE finite; selected checkpoint
   reproduces on reload.
2. **Final train MAE ≤ 0.95 × epoch-1 train MAE** (≥ 5% improvement).
3. **Best val MAE ≤ min(1.5 × same-cohort naive, 8.0 K)** — on the #45 cohort the
   first bound is **≈ 2.289 K**.
4. **Final val MAE ≤ 1.25 × best val MAE** (no divergence).
5. **Selected checkpoint is not a passthrough:** mean absolute *pooled* correction
   on valid validation cells ≥ **5% of the same-cohort naive MAE**.
6. **Identity at init:** pooled zero-init residual output MAE vs the pooled prior
   arm on the same patch IDs ≤ **0.001 K** (fail-closed wiring check).

Beating the naive baseline is **not** required. Missing or non-finite evidence is
*inconclusive*, never a GO.

### Trials and budget

- **Trial 1** — `stage1_probe` (residual, LR 1e-3).
- **Trial 2** — `stage1_probe_lr3` (residual, **LR 3e-3**, all else equal), run
  **only** if trial 1 is finite and stable but its train curve is flat
  (criterion 2 not met while 3 and 4 are). One change family (LR), no other.
- At most **two** paid Vertex jobs, sequential, no automatic retry. Existing
  launcher bounds: `europe-west3`, T4, server timeout 10 800 s, client wait 600 s,
  ≤ **$3.00** projected compute exposure per job (≈ **$5.70** total at $0.90/h),
  create-only evidence under the existing QA prefix. No full temporal run.

## 4. Limitations

- The zero-fit diagnostic is a bounded 8-patch subset, not the #45 cohort; it
  establishes the scale gap, not cohort-level scores.
- The pseudo-pair construction derives the prior from Landsat observations of the
  same scene, so a residual GO justifies neither independent spatial
  generalization nor 10 m detail accuracy.
- `[uncertain: whether the residual representation learns the scale within six
  epochs — the re-probe decides.]`

## 5. Next step

Implement the guarded residual probe, run trial 1 (and trial 2 only if
triggered), and record GO/NO-GO in this document. The Stage-1 full lock
(`configs/modeling/stage1_locked.yaml`) is updated to the residual method **only
on GO**; on NO-GO it stays the old, unfrozen-as-approved method and the recovery
mode remains experimental.

## 6. Trial 1 result — GO

**Decision: GO for the residual representation.** The bounded residual probe
clears every pre-registered screen and, unlike #45, actually beats the
same-cohort naive prior. No second trial is triggered (trial 2 runs only when
trial 1 is finite and stable but flat; see §3).

| Field | Value |
|---|---|
| Config | `stage1_probe` (residual representation, LR 1e-3, scene-spread 128/64) |
| Vertex job | `projects/996559849187/locations/europe-west3/customJobs/2755039413770649600` |
| Region / machine | `europe-west3` / `n1-standard-4` + 1× `NVIDIA_TESLA_T4` (on-demand) |
| Source SHA | `0a2495675cba8da745d971364457ce37555cc6ad` |
| Image | `europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:b690a6e7b6773de2c008c62c83d3f094d35681e4d1f4ae79d5511bfdf299f50e` |
| Run label | `stage1-probe-20261001T144604Z-recovery` |
| Terminal state | `JOB_STATE_SUCCEEDED` |
| Evidence | `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/stage1-probe-20261001T144604Z-recovery/evidence.json` |
| Exposure | ~$2.85 projected / ~$0.15 actual compute at $0.90/h (one job) |

### Curve (cell-weighted masked MAE @ 100 m, valid cells 16,112/epoch)

| Epoch | train | validation |
|---|---|---|
| 1 | 1.4382 | 24.2345 |
| 2 | 1.3433 | 1.5293 |
| 3 | 1.2763 | 1.4735 |
| 4 | 1.2428 | 1.2955 |
| 5 | 1.1944 | **1.1929** (selected) |
| 6 | 1.1573 | 1.3350 |

The epoch-1 validation spike is the BatchNorm running-statistics warm-up (the
zero-init head makes epoch 1 the first pass that populates them); it settles from
epoch 2. Checkpoint reload reproduced the selected score exactly
(`reload_recomputed = 1.1929 = best_metric`).

### Screen

| Criterion | Result |
|---|---|
| All six train/val MAE finite; reload reproduced | pass |
| Final train MAE ≤ 0.95 × epoch 1 | pass — 1.1573 ≤ 1.3663 (19.5% below) |
| Best val MAE ≤ min(1.5 × naive, 8 K) | pass — 1.1929 ≤ 2.2892 |
| Final val MAE ≤ 1.25 × best val | pass — 1.3350 ≤ 1.4911 |
| Not a passthrough (pooled correction ≥ 5% naive) | pass — 1.2255 K ≥ 0.0763 K |
| Identity at init ≤ 0.001 K | pass — `residual_identity_mae_k = 0.0` |

Same-cohort naive validation MAE: **1.5261 K** (64 matched patches, identical to
the #45 cohort). Novel model **beats naive**: 1.1929 K vs 1.5261 K (**1.28×
better**). Against the #45 absolute-representation result (294.8369 K) this is a
**247× improvement**.

`uv run python scripts/validators/validate_stage1_recovery.py --evidence <evidence.json>`
→ **GO (exit 0)**. The historical `validate_stage1_probe.py` correctly returns
INCONCLUSIVE for `probe-residual` (no cross-screening).

### Interpretation

- The prediction is now a genuine learned correction over the physical prior
  (pooled mean |correction| 1.23 K), not a passthrough (identity 0.0 K at init).
- Train and validation curves track and improve monotonically to epoch 5; the
  small epoch-6 validation uptick is within the no-divergence cap.
- This is a **screening result on the pseudo-pair construction**, not an
  operational accuracy claim or a 10 m-detail result (see §4).

## 7. Frozen method after the GO

The residual representation is carried into the locked full method:
`stage1_residual_prior: true` and the zero-init head become part of
`configs/modeling/stage1_locked.yaml` and its guard (`assert_stage1_lock`), so the
full Stage-1 temporal run uses the representation proven here. The learning rate
stays at the locked **1e-3** (trial 2 was not needed). Batch size, AMP precision,
workers, and patience remain operational retunes; capacity and the spectral block
stay as frozen. No full temporal run is performed in this issue — the capability
and the frozen representation are delivered, the unbounded run stays a separate,
explicitly scheduled invocation.

