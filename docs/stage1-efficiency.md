# Stage-1 training efficiency — pre-registration

This protocol measures and reduces the cost of the existing Kelvin-residual
U-Net training path before any full Stage-1 run. It authorizes at most four
bounded Vertex efficiency jobs. **It does not authorize a full Stage-1 fit or
any test-pixel reads.**

Status: pre-registered on 2026-10-02, source `a6b42018ae382d93c5d51575067804f773369539`.

## Method invariants

The model and scientific method stay fixed: first ten V3 channels, U-Net depth
4 and width 32, residual prior, masked L1, LR `1e-3`, weight decay 0, seed 0,
temporal split. Evaluated learning runs use batch size 4, six epochs, and the
existing recovery-probe cohort (128 requested train / 64 requested validation).
No efficiency profile admits or reads the test split.

The existing reader remains authoritative for feature scaling, fill handling,
prior reconstruction, target, and mask. A job-local cache may change where these
already-computed tensors are read from, not their values, IDs, exclusions,
sampler order, or model inputs. Stage-1's 10-channel network does not imply that
reading ten channels from existing pixel-interleaved COGs transfers fewer remote
bytes; no such saving is assumed.

## Bounded cohorts

- Performance cohort: at most 512 requested train and 192 requested validation
  patches; require at least 384 and 128 admitted respectively. Select a fixed,
  scene-spread cohort from index metadata, covering at least three train years
  and including partially valid masks. Freeze requested IDs before admission;
  record exclusions without refill.
- Learning cohort: use `stage1_probe` unchanged for both J1 and J4: GPU
  accelerator, stream loader, seeded shuffle, seed 0, scene-spread 128/64
  selection, six epochs, existing cohort minima and residual recovery method.
  “CPU lifecycle reference” means only the current CPU reload/verification
  passes after GPU fitting, not CPU training or a different fit configuration.
- Performance scratch tasks never step an optimizer and never consume the
  learning fit's sampler or RNG state.
- Inspect metadata only for full-cohort size projections. Do not load full
  training, validation, or test pixel cohorts during this work.

## Cache protocol

Use the existing reader to construct job-local read-only NumPy memory-mapped
arrays for active float32 features, physical Kelvin prior, native target, and
boolean mask. Preserve metadata and the reader's full-stack fill count. Apply
the existing prior normalization at collation, not twice. Keep the existing
full-band source read: band-selective network savings are not established for
these COGs.

The readiness manifest is written only after every array has been flushed,
reopened, and checked. It binds the arrays to source and index fingerprints,
scaler digest, active-channel names/order, geometry, preprocessing version,
ordered patch IDs, admissions/exclusions, shapes, dtypes, and byte totals.
Missing readiness, source drift, a partial array, ID/shape mismatch, or capacity
failure is a hard failure; never silently fall back to streaming. Cache arrays
are disposable job-local scratch, never canonical data, retained artifacts, or
checkpoint state.

The planning-scale cache estimates are about 9 GiB for Stage-1 active inputs and
about 24 GiB for 28 channels over train/validation; the runtime must calculate
actual sizes from selected cohorts and enforce its disk budget. A small cache
that fits in memory does not establish full-cache performance under page-cache
pressure.

## Measurement protocol

Separate cache build, admission, loader wait, host-to-device transfer, forward,
backward, validation, checkpointing, reload, and verification. Report complete
wall time as well as CUDA-event device attribution. CUDA work is asynchronous;
do not use host hook time alone as GPU duration. Use a short profiler window and
warm-up, then at least three counterbalanced retained repeats; report median,
spread, and batch p95. Do not synchronize every training batch or sum overlapping
prefetch and device time. Label process-first-pass and warm-cache conditions;
do not claim caches were flushed or reads were remotely uncached.

Measure worker counts 0/2/4 and verify pinned memory on the actual custom batch
before including it. Keep default precision FP32. Measure 16-mixed on identical
inputs as a candidate; physical residual arithmetic, pooling, loss/metric
reductions, validation, and reload remain FP32. Batch 8 is forward/backward
compute-only and is not adopted for training by this protocol.

The measurement-tool search covered installed `torch.profiler`, CUDA events,
Lightning's timer, and NVIDIA Nsight Systems. The short profiler window is for
attribution; CUDA events plus whole-phase wall time are the throughput evidence.
Lightning's host-side timer alone does not establish completed GPU work. No
new profiling dependency is authorized.

## Job schedule and budget

At most four sequential submissions, each with a 2,700-second server timeout,
600-second client allowance, one on-demand `n1-standard-4` + T4, and no retry or
automatic substitution. Every failed or expired submission consumes its slot.
Before paid work, obtain an authoritative current Vertex Custom Training
europe-west3 aggregate rate for machine, memory, and T4; it must be no more than
$1.03/hour. A Compute Engine quote or historical rate is not a substitute.

The maximum projected compute exposure is approximately
`4 × (2700 + 600) / 3600 × $1.03 = $3.78`. The aggregate experiment ceiling is
$10 including builds, registry, storage and logging, leaving at most $6.22 for
those other expenses. This is an admission estimate, not a provider spending
cap. If a line item cannot be bounded, do not submit.

| Job | Configuration | Purpose |
|---|---|---|
| J1 | Current streaming, FP32, batch 4, workers 2 | Performance cohort measurements, then one fresh six-epoch recovery control on 128/64. |
| J2 | Job-local cache, FP32, batch 4 | Source/cache and loader/transfer comparisons; GPU-vs-CPU lifecycle equivalence. No optimizer steps. |
| J3 | J2 winner, FP32 vs 16-mixed; batch-8 diagnostic only | Measure precision and compute opportunity. No optimizer steps. Skip only if AMP cannot plausibly save 10% complete runtime. |
| J4 | Best evidenced settings, batch 4 | Repeat performance measurements and a fresh matched six-epoch 128/64 learning guard. |

Reserve J4 from the beginning. No fifth job. A timeout, failed cache, rate or
budget uncertainty, numerical mismatch, learning regression, or ambiguous
submission stops the sequence and consumes the slot. No shortened cohort or
omitted verification may be silently substituted.

## Acceptance and decision

Both J1 and J4 must pass the existing recovery screen: at least 5% train-MAE
improvement, existing same-cohort naive cap, no divergence, non-passthrough
correction, residual identity at most 0.001 K, and reload consistency. IDs,
exclusions, valid-cell totals, and epoch order must match. J4's best and final
validation MAE must each be no worse than J1 by more than `max(0.05 K, 5% of
the corresponding J1 score)`. This is a bounded screen, not statistical
equivalence or full-run accuracy.

The final report will give cache build and steady-state runtime separately,
then project the full 20-epoch lifecycle including startup, admission, cache
build, identity, epochs, checkpoint I/O, reload/verification, one-shot test
allowance, and evidence upload. Test time is extrapolated from validation
throughput; no test pixels are read. Report central and conservative-high
scenarios and per-phase costs without double-counting overlap.

Target at least 25% conservative projected runtime savings, or document that no
larger safe improvement was demonstrated. Stop tuning after two plausible
controlled knobs each yield under 10% incremental complete-runtime savings, or
when the four-job/$10 cap is reached. The proposed five-run allocation is at
most €175 of a €250 allowance, reserving €75; current available balance and FX
must be confirmed before this projection can be treated as affordable.

Possible verdicts: recommended measured configuration; no larger safe
improvement demonstrated; or no-go/inconclusive. **Every verdict keeps full
Stage-1 execution blocked. A separate approved plan is required before any full
fit.**

Validation commands are explicit QA steps, not new test-suite files:
`uv run nox`; `uv run nox -s smoke-modeling-contract`; and direct
`uv run python scripts/validators/validate_training_efficiency.py --self-check`
plus the corresponding cache self-check. The `smoke-real-comparison` session is
not run because it reads test-split patches.

## Results

_Pending. No efficiency jobs have been submitted._
