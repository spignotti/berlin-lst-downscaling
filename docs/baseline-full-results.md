# Full naive baseline, validation and test (issue #39)

The Stage-1 comparison anchor: the naive prior-expand baseline scored over the
complete published validation and test patch splits. Every number below is read
from the unchanged run artifact committed beside this note
(`docs/results/baseline-full-20260929T084243Z-29A5F946/`).

## Result

| Split | Requested | Evaluated | Excluded | Valid cells | Masked MAE @ 100 m (K) | SSIM (secondary) |
|---|---|---|---|---|---|---|
| validation | 1342 | 1341 | 1 | 341,588 | 1.61010 | 0.94266 |
| test | 1803 | 1803 | 0 | 459,564 | 1.39523 | 0.95352 |

- Primary metric: cell-weighted masked MAE at 100 m over the valid
  `training_eligible@100m` cells.
- SSIM is a diagnostic only, never a selection gate: 7x7 uniform window,
  fully valid windows only, 130,175 supported windows on validation and
  176,349 on test.
- One validation patch is excluded, not scored. It is accounted for below.

## Provenance

- Run `baseline-full-20260929T084243Z-29A5F946`, method `naive_prior_expand`,
  written `2026-09-29T09:07:01Z`.
- Runner Git revision `9d959601daf670327ba06b85ab2b5ffa9376a913` (dirty: no),
  deployed to `berlin-lst-vm` at that exact SHA.
- Artifact `baseline_report.json`, SHA-256
  `5a0daca809055195028c6688da7f337b4448723038c07bc4b30c24c9353d62d6`.
- Config `baseline_full` with `splits=[validation,test]` and
  `max_patches_per_split=null`, so both published splits were scored in full.
- Published patch index `gs://berlin-lst-training-data/training/patch-index/v1`:
  `patches_total=10018`, `index_policy_hash=79add7e6a0736fb9`,
  `source_policy_hash=9f9287255803d759`, published `2026-09-23T12:32:45Z`.
- Sources: `training/patch-index/v1`, `training/v1`, `features/v3`,
  `ard/full/2017-2026-cutoff-20260717T235959Z`.
- Host: CPU-only `berlin-lst-vm`. No GPU and no Vertex AI were used.

## Independent validation

`scripts/validators/validate_baseline.py` re-derived the claims from the
published sources with its own code and recomputed 50 patches (25 per split).
It passed on the retained artifact with exit code 0:

- **Selection and accounting.** Test is the full 1803 requested rows with no
  gap. Validation is the full 1342 requested rows with exactly one gap, equal to
  the one recorded exclusion. Provenance: the report fingerprints equal the
  index completion marker, and `patches_total` equals the index row count.
- **Numerics.** Recomputed prior, native target, eligibility window and masked
  error sums matched the artifact on every recomputed patch, as did the SSIM
  support counts. The SSIM **values** come from the shared metrics path and are
  not recomputed here, so only their support is independently checked.
- **Exclusion proof.** The single exclusion is
  `LC08_L2SP_192023_20240513_02_T1:E394790N5812810` with reason
  `no_valid_prior_block`. One required 1000 m prior block, global block
  `(27, 25)`, holds zero native-valid Landsat cells, while the patch's
  eligibility window holds 244 cells, matching its index row. Each required
  block is read directly from the Landsat COG and flag, which is a different
  method from the reader's whole-scene block grid. The contract excludes such a
  patch from both arms with a recorded reason
  (`docs/pseudo-pair-tensor-contract.md`, "Native-invalid cells inside a block").

## Comparison universe

The anchor covers the full published validation (1342) and test (1803) splits
under the shared reader, eligibility mask, nested 10x10 pooling and
cell-weighted reducer. A model arm is comparable only when its recorded
`data_scope.json` admitted and `skipped_refs` patch IDs equal this universe per
split. A matching count alone is not sufficient.

## Limitations

- The temporal split (2024 validation, 2025 test) holds scenes apart by year,
  but accepted windows are anchored on a global lattice, so the same spatial
  location recurs across years. This is a temporal holdout at recurring
  locations, not evidence of independent spatial generalization.
- The coarse prior is derived from the same scene's Landsat observations. That
  is the intended degradation experiment, not a target-independent operational
  input.
- One validation patch is excluded rather than scored; it is accounted for, not
  dropped silently.
- The 2025 test split is never used for checkpoint selection.
- Per-patch numeric recomputation was bounded to 25 patches per split. The
  selection audit, exclusion proof, provenance check and aggregate accounting
  were exhaustive over the full requested splits.

## Reproduce

```bash
# The baseline run is expensive and is not repeated here. It removes the VM's
# ephemeral output on success, so the run id and hash above belong to this
# delivered artifact only; a new run needs its own run id and hash.
.opencode/skills/google-access/scripts/run-baseline-vm.sh <branch>

# Re-read the committed artifact against the published sources (needs working
# ADC). The VM-side output of run baseline-full-20260929T084243Z-29A5F946 was
# removed after re-validation, so the VM launcher path no longer applies to it:
#   .opencode/skills/google-access/scripts/run-baseline-validation-vm.sh <branch> <run-id> <report-sha256>
uv run python scripts/validators/validate_baseline.py \
  --report docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json \
  --max-patches 25
```
