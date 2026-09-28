# Real patch read timing (issue #35)

Bounded measurement of the real patch reader's data phases on the GCP VM,
recorded to decide whether the duplicated target/mask COG window reads are
material at full-split scale. **Conclusion: they are not material; the reader
was left unchanged.**

## Question

`admission_reason` and `read_patch` both call `_read_target_and_mask` for an
admitted ref (`src/berlin_lst_downscaling/modeling/patches.py:623,642`), so the
native Landsat target COG and the eligibility-mask COG are opened twice per
admitted patch. The 12-patch smoke proved the streamed path is
read-invariant, not what those duplicate opens cost.

## Method

- Probe: `scripts/validators/measure_patch_reads.py`. It subclasses
  `RealPatchReader` to time `_read_target_and_mask` (the duplicate work),
  `_read_features`, and scene-prior builds, without instrumenting production
  code. It measures admission and one read pass separately, repeats the read
  pass, and wall-times a streamed `RealPatchDataset` `DataLoader` with
  `num_workers` in {0, 2}.
- Launcher: `.opencode/skills/google-access/scripts/run-patch-read-timing-vm.sh`,
  the shared fail-closed lifecycle (start, deploy pinned SHA, detached launch,
  poll, capture report, clean up, stop).
- Sample: deterministic and scene-spread. The probe takes up to
  `ceil(60 / 12) = 5` refs per scene in canonical index order until it reaches
  60 refs per split, so the sample spans several scenes instead of one scene's
  contiguous block.
- Read-only: the probe reads published sources and writes only local ephemeral
  output, which the launcher removes. Nothing was written to GCS.

## Environment

| Field | Value |
|---|---|
| Host | `berlin-lst-vm`, `n2-highmem-2` (2 vCPU / 16 GB), Debian 12 |
| Run | `patch-read-timing-20260928T192300Z-973E6A2B`, deployed SHA `a41fb25` |
| Config | `real_full` |
| Patch index | `gs://berlin-lst-training-data/training/patch-index/v1`, `patches_total=10018` (train 6873, validation 1342, test 1803) |
| Index provenance | `index_policy_hash=79add7e6a0736fb9`, `source_policy_hash=9f9287255803d759`, published 2026-09-23 |

## Sample

| Field | Value |
|---|---|
| Requested refs | 180 (60 per split) |
| Admitted refs | 180 (no exclusions) |
| Distinct scenes | 46 |

## Results

Admission and read phases, seconds for the 180-ref sample:

| Phase | Total | Target + mask | Features | Scene priors |
|---|---|---|---|---|
| Admission | 16.47 | 5.50 (180 calls) | - | 10.93 (46 builds) |
| Read, first pass | 98.24 | 9.74 | 88.44 | 0.00 |
| Read, warm (passes 2-3) | 89.50 | 10.42 | 79.03 | 0.00 |

The read-pass target/mask time is the duplicate cost: admission already paid
the same two windows for every admitted ref.

| Duplicate target/mask | Cold | Warm |
|---|---|---|
| Seconds per ref | 0.0541 | 0.0579 |
| Share of admission + read | 8.49% | 9.83% |
| Share of the read pass | 9.92% | 11.64% |

Streamed `DataLoader` wall time over the same 180 admitted refs, median of 3
passes:

| `num_workers` | Median pass | Passes |
|---|---|---|
| 0 | 60.30 s | 64.55, 60.30, 59.20 |
| 2 | 38.75 s | 41.20, 38.66, 38.75 |

Two workers are 1.56x faster than in-process loading on this 2-vCPU host.

Extrapolating the warm per-ref duplicate cost to the full index:

| Scope | Refs | Duplicate cost |
|---|---|---|
| All splits | 10018 | ~9.7 min |
| Train + validation | 8215 | ~7.9 min |

## Decision

The plan's rule requires **both** conditions.

1. Duplicate opens at least 20% of admission + read: measured 9.83% warm /
   8.49% cold. **Not met.**
2. Extrapolated saving at least 10% of the data phase or 30 min per epoch:
   measured 9.83% and ~7.9 min for train + validation. **Not met.**

Outcome: record the measurement and close the issue without an optimization.
Feature reads dominate the read pass (88% warm), so the duplicate target/mask
opens are not the lever they were suspected to be.

## Limitations

- "Cold" here means the first in-process pass. OS page caches were not
  flushed, so these are not uncached remote reads and the cold/warm gap is a
  lower bound.
- The sample is 1.8% of the index. The extrapolation applies sample unit costs
  to full ref counts; the full-index admitted share is unknown, so both
  full-scale figures are approximations.
- Admission per-ref cost amortizes scene-prior builds over 46 sampled scenes.
  The full index has a different scene count, so the admission share is
  approximate.
- Loader wall time includes collation and worker IPC, not only COG reads, and
  the worker arm runs one concatenated dataset rather than per-split modules.
- With two warm passes the reported warm figure is the upper of the two
  sorted values, not an interpolated median, so it leans slightly toward
  overstating the duplicate cost.
- The measurement excludes model fitting by design.

## Reproduce

```bash
.opencode/skills/google-access/scripts/run-patch-read-timing-vm.sh <branch>
```
