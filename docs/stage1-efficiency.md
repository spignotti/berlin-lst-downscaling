# Stage-1 training efficiency — pre-registration

This protocol measures and reduces the cost of the existing Kelvin-residual
U-Net training path before any full Stage-1 run. It authorizes at most four
bounded Vertex efficiency jobs. **It does not authorize a full Stage-1 fit or
any test-pixel reads.**

Status: complete — replacement J1–J4 executed and validated on 2026-10-03; pre-registered on 2026-10-02, source `a6b42018ae382d93c5d51575067804f773369539`.

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

- Performance cohort: request 384 train and 128 validation patches (the
  preregistered minima; caps remain 512 / 192) and require all requested counts
  admitted. Select a fixed scene-spread cohort from index metadata, covering at
  least three train years. Reserve one partial-mask candidate per split before
  filling the remaining scene-round-robin slots. Freeze requested IDs before
  admission; record exclusions without refill.
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
The project batch type implements Lightning's documented custom-batch
`pin_memory()` and `.to(device, non_blocking=...)` hooks; validate the CPU
transfer and verify all four tensors are actually pinned before the pinned GPU
candidate is considered.
For a learning fit, retain the GradScaler scale once per epoch; a scale decrease
marks at least one skipped optimizer update during that epoch, but is only a
lower bound on the number of skips. Do not poll the scaler every batch because
`get_scale()` synchronizes with the GPU and would distort the timing.

The measurement-tool search covered installed `torch.profiler`, CUDA events,
Lightning's timer, and NVIDIA Nsight Systems. The short profiler window is for
attribution; CUDA events plus whole-phase wall time are the throughput evidence.
Lightning's host-side timer alone does not establish completed GPU work. No
new profiling dependency is authorized.

## Job schedule and budget

### Preserved failed submission

The original J1 submission was `stage1-efficiency-j1-20261003T030056Z`, Vertex
resource `projects/996559849187/locations/europe-west3/customJobs/6995706228521304064`,
source SHA `514fae1cf27f73d08883b985f1cc3b5bcef31233`, and image digest
`sha256:6d2967c323410540b50365a32ec4a4e0bf4baff636c37d90bbaf0a641141c9f7`.
It failed before measurement setup because the entrypoint created
`logs/modeling/vertex-entrypoint.txt` inside its output root, then the measurer
rejected the non-empty root. The retained, incomplete evidence is at
`gs://berlin-lst-training-data/qa/modeling/vertex-smoke/stage1-efficiency-j1-20261003T030056Z/evidence.json`.
It has no measurements or checkpoint. Preserve that object and the original
local ledger; the failed submission consumes one of the five total submissions
allowed by the recovery plan.

### Replacement sequence

Run at most four sequential replacements, J1–J4, for at most five submissions
including the preserved failure. Each replacement uses one on-demand
`n1-standard-4` + T4, a 2,700-second server timeout, and at most 1,800 seconds
for provisioning. If the job has not entered `JOB_STATE_RUNNING` by that
deadline, cancel that exact active job once and confirm a terminal state. A
first observation after the deadline with `startTime` later than `createTime`
plus 1,800 seconds also misses the allowance: cancel if still active, or preserve
its actual terminal state and consume the slot if already terminal. An ambiguous
submit or cancel, failed job, failed validation, or failed learning/cache gate
consumes the slot and stops the sequence. Never resubmit a replacement slot or
automatically substitute hardware.

Use the user-authorized upper bound **$1.00/hour** for the supplied Frankfurt
Vertex Custom Training range. It is a planning estimate, not a verified active
Billing-account contract price or provider spending cap. The maximum projected
compute per replacement is `($2,700 + $1,800) / 3,600 × $1.00 = $1.25`.
Conservatively carry the failed J1 reservation of `$0.9167` and reserve `$2.00`
cumulatively for all non-compute costs, including the prior and replacement
image builds, registry storage, job storage, and logging. The resulting maximum
projection is `$0.9167 + 4 × $1.25 + $2.00 = $7.9167`, below the `$10` experiment
ceiling. The `$2.00` reserve supersedes the previous `$0.20` estimate; actual
charges and persistent image-retention cost remain unverified.

All replacements use the fixed recovery session
`stage1-efficiency-recovery-20261003`, its separate ignored ledger under
`data/runs/.stage1-efficiency-control/`, one committed source SHA, and one
digest-pinned image. The original failed ledger, evidence, and image remain
unchanged. No full Stage-1 run or test-pixel access is authorized.

### Operator-provided pre-launch estimate (2026-10-02; rate basis retained)

The operator supplied this Frankfurt Vertex Custom Trained Models itemization
in chat: N1 core `$0.190/h`, N1 RAM `$0.095/h`, T4 `$0.490/h`, and 100-GB disk
approximately `$0.006/h`, quoting an aggregate around `$0.78–$0.82/h`; the user
also authorized a planning range of `$0.80–$1.00/h`. Use the authorized upper
bound **`$1.00/h`**. A separate N1-standard-4 figure of `$0.21849885/h` was
also supplied, which does not equal the listed core-plus-RAM subtotal
(`$0.285/h`). The `$1.00/h` input conservatively exceeds both quoted worker
aggregates. `[uncertain: exact active Billing-account contract price and whether
an additional Vertex line item applies]`. This authorization applies only to
the four bounded efficiency jobs.

The original operator estimate for non-compute costs was less than `$0.20` for
one image build and the four jobs. It is superseded for the recovery sequence by
the larger `$2.00` cumulative reserve above, which includes the rebuild. These
are planning estimates, not actual billing or provider-enforced spend caps.

| Job | Configuration | Purpose |
|---|---|---|
| J1 | Current streaming, FP32, batch 4, workers 2 | Performance cohort measurements, then one fresh six-epoch recovery control on 128/64. |
| J2 | Job-local cache, FP32, batch 4 | Source/cache and loader/transfer comparisons; GPU-vs-CPU lifecycle equivalence. No optimizer steps. |
| J3 | J2 winner, FP32 vs 16-mixed; batch-8 diagnostic only | Measure precision and compute opportunity. No optimizer steps. If projected complete-runtime AMP gain is under 10%, J4 must use FP32. |
| J4 | Best evidenced settings, batch 4 | Repeat performance measurements and a fresh matched six-epoch 128/64 learning guard. |

Reserve J4 from the beginning. No sixth submission. A timeout, failed cache,
rate or budget uncertainty, numerical mismatch, learning regression, or
ambiguous submission stops the replacement sequence and consumes the slot. No
shortened cohort or omitted verification may be silently substituted.

After each job, download its create-only evidence and run the independent
efficiency validator. Supply J1 evidence to J2; J1 and J2 evidence to J3; and
J1/J2/J3 evidence to J4. The validator compares fixed IDs, exclusions, cache
provenance, timing structure, and the matched learning screen. J3 reuses the
loader configuration selected by J2 and projects AMP gain from J2's steady-loader
time and J3's batch timings; J3 does not remeasure a duplicate loader block.
Use `--mark-ledger` after validation, including a failing validation.

Two one-time, evidence-only revalidations are authorized for the exact retained
J1 and J3 jobs because the original validator mismatches were implementation
defects: J1's producer stores checkpoint metadata at the evidence top level, and
J3 consumes J2 loader timing without a duplicate `selected_cache_loader` block.
Use `--record-revalidation --mark-ledger` only with the original, hash-identical
evidence, after correcting the local validator. The ledger records the initial
failure and the revalidation verdict; it does not rerun or replace a Vertex job.
Any revalidation failure, any other failed/incomplete validation, or any failed
job blocks remaining slots. Vertex `SUCCEEDED` alone is not a quality or
cache-equivalence gate.

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
when the four-job/$10 cap is reached. Separately, the portfolio planning
assumption is at most €175 across five eventual substantive model runs (Stage 1
plus later phases), reserving €75 of a €250 allowance. That is not an assertion
that the allowance covers this efficiency budget as well; current balance and
FX must be confirmed before either projection is treated as affordable.

Possible verdicts: recommended measured configuration; no larger safe
improvement demonstrated; or no-go/inconclusive. **Every verdict keeps full
Stage-1 execution blocked. A separate approved plan is required before any full
fit.**

Validation commands are explicit QA steps, not new test-suite files:
`uv run nox`; `uv run nox -s smoke-modeling-contract`; and direct
`uv run python scripts/validators/validate_training_efficiency.py --self-check`
plus `uv run python scripts/validators/check_efficiency_output_root.py`,
`bash scripts/validators/check_vertex_efficiency_entrypoint.sh`,
`uv run --group operators python scripts/operators/launch_vertex_modeling.py --self-check`,
and the corresponding cache self-check. The `smoke-real-comparison` session is
not run because it reads test-split patches.

## Results

### Replacement jobs

The original J1 worker failed before measurement setup. The four replacement
jobs used source `ea10155e48ecf553db14bdcd635ae386972882c5` and image
`sha256:919012758eabdb8f6f34e9e98d2c2c951d908e3546fffa570f6fa8472f520601`.
All jobs succeeded, and their evidence passed independent validation. J1 and J3
each needed one local evidence-only revalidation. Their first failed verdicts
remain in the recovery ledger; neither job was rerun.

| Slot | Configuration | Provisioned seconds | Estimated compute at $1.00/h | Validation |
|---|---|---:|---:|---|
| J1 | Stream, FP32, batch 4, workers 2; six-epoch 128/64 learning control | 1116 | $0.31 | Pass after checkpoint-reference correction |
| J2 | Local cache, FP32, batch 4; worker 0 / pin memory false selected | 1268 | $0.35 | Pass; exact source/cache match and CPU/GPU lifecycle |
| J3 | Cache, worker 0; FP32/mixed batch-4 and FP32 batch-8 compute-only | 693 | $0.19 | Pass after removing a redundant J3 loader-timing requirement |
| J4 | Cache, mixed precision, batch 4, workers 0; matched six-epoch learning guard | 934 | $0.26 | Pass |

The ledger estimates about `$1.11` compute for the four replacements from
Vertex `startTime` to `endTime`. Including the original failed J1 reservation
of `$0.9167` and the cumulative `$2.00` non-compute reserve gives about `$4.03`
estimated experiment cost so far. This is not an invoice. The conservative
full-gate ceiling remains `$7.92` projected, below `$10`.

Vertex jobs:

| Slot | Resource ID | Evidence SHA-256 |
|---|---|---|
| J1 | `1504129432896405504` | `e1dc91ca05f6fe63d8585d7be07093b945ca344616af17b6efdc51a0d0920568` |
| J2 | `1400757747699417088` | `6b4dab8d1cae09b55c6d4bd825c99d3e66b4f2be1bdfa8dce7fd259418b8dfb4` |
| J3 | `2113452388730798080` | `17263a41474033cf7dc59990360a3e3b907dafe5776add8b99fbf65de43a201e` |
| J4 | `54744409069060096` | `9704b56759b11cce6e3665deb3bcd5ba7fed0b216db99f9293b41dbdc816698b` |

Each evidence object and checkpoint is retained under
`gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/`.
W&B recorded learning runs J1 `3749uejg` and J4 `exfo1fqp`. All four evidence
records report test access disabled and train/validation-only splits. No test
pixels were read.

### Loader and precision findings

On the fixed 384/128 performance cohort, J1's median steady streaming-loader
pass with two workers took `116.28 s` across train and validation. J2's selected
cached loader (zero workers, pin memory false) took `0.136 s` for the same pass.
The J2 cache build took `281.07 s` for `577,372,672` bytes, with another
`44.17 s` for cache admission. Counting both, the measured loader-time saving
repays the cache build and admission after about `2.8` repeated passes,
excluding shared source setup. Workers 2 and 4 were slower than worker 0 on the
cached loader.

J3's median batch-4 compute time was `0.1051 s` in FP32 and `0.0460 s` in
16-mixed. Using the validated J2 loader timing, the independent validator
projects **55.7% complete-runtime gain** for mixed precision, above the 10% J4
selection threshold. Batch 8 was compute-only and was not adopted.

J1's best validation MAE was `1.19290 K` at epoch 5; reload recomputed
`1.19290 K`. J4's best and final validation MAE were both `1.16232 K` at epoch
6; reload recomputed `1.16232 K`. J4 is about `2.6%` below J1's best score on
the matched cohort. Both runs improved train MAE by more than 5%, retained the
residual prior, had residual identity MAE `0.000 K`, and passed the same-cohort
naive cap and reload checks. Their mean absolute residual corrections were
`1.225 K` (J1) and `0.981 K` (J4), so neither is a prior passthrough.

### Full-run planning estimate

The baseline index has `10,018` patches: `6,873` train, `1,342` validation, and
`1,803` test (`docs/baseline-full-results.md:12–13, 32–34`). Full Stage-1
therefore requests `8,215` train/validation patches. Scaling the 192-patch,
six-epoch learning guard to 20 epochs and that requested universe gives a factor
of about `142.6`. At this scale, a linear cache-size estimate is about `9 GiB`;
using J2's measured cache-build rate gives about `1.31 h` to build it.

For a central cached/mixed scenario, J4's provisioned time minus its measured
performance phase is `322.7 s`. The 192-patch learning cache is `216.5 MB`;
estimating its build time from J2's measured build rate leaves about `217 s` for
the six-epoch learning and finalization. Scaling that portion gives about
`8.6 h`. Adding the full-cache build, scaled source/admission setup, a one-shot
test-score allowance extrapolated from J1 validation throughput, and a
10-minute launch-wait allowance gives roughly **`10.7 h`**, or about **`$10.7`
compute** at the authorized estimate.

For a conservative streaming/FP32 reference, J1's provisioned time minus its
performance phase is `471.5 s` for the same learning guard. Scaling it by the
same factor, then adding source/admission setup, the one-shot test-score
allowance, and launch wait gives roughly **`19.5 h`**, or about **`$19.5`
compute**. The cached/mixed projection is about 45% lower than this reference.
These are extrapolations from a small learning cohort, not measured full-run
times or provider quotes. Phase boundaries in the learning fit are incomplete,
so fixed checkpoint/finalization costs are included in the scaled fit tail.
The full 20-epoch run and test scoring remain blocked pending a separate plan.

The two experiment image builds each increased the Artifact Registry repository
size by about `6.13 GB` in the observed reads. At Google's published Artifact
Registry storage rate, retaining both images adds roughly `$1.1–$1.2/month`
before any account-level free allowance; actual billing units and credits were
not checked. The images and all create-only evidence remain retained for
reproducibility.

### Verdict

Recommended measured configuration for a future full Stage-1 plan: job-local
cache, batch 4, zero workers with pin memory false, and 16-mixed precision for
the learning fit. Batch 8 remains compute-only and is not adopted. The
conservative projection clears the 25% savings target, so this is not a
no-improvement outcome. Full Stage-1 execution and test scoring stay blocked
pending a separate approved plan.
