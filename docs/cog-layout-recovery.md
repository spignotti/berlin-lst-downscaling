# COG Layout Recovery Runbook

## Incident Summary

**Date:** 2026-08-03  
**Duration:** ~2.5 hours (14:19–16:35 UTC)  
**Affected assets:** 2,079 canonical COGs  
**Root cause:** COG repair process without fail-closed validation

### Timeline

| Time (UTC) | Event |
|------------|-------|
| 14:19:03 | First overwrite — 464 uploads without audit |
| 14:42:42 | Second phase — 638 uploads with audit |
| 16:03:16 | Third phase — 305 uploads, interrupted |
| 16:35:43 | Last overwrite |
| 17:40:16 | VM stopped |

### Impact

- **1,243** assets are strict-valid (zero errors, zero warnings)
- **164** assets have missing-overview warning only (flag COGs)
- **672** assets have hard IFD/offset layout errors
- **1,407** assets were overwritten; only **638** have durable audit
- **769** assets lack post-upload verification evidence

## Recovery Rules

**Absolute rule — no workarounds:**

- Warnings are failures. No suppression, `--force`, fallback engine, skipped cohort, inferred original generation, best-effort continuation, or retry after a generation conflict.
- Every mutation requires:
  - Generation guard
  - Recoverable original
  - Post-write verification
  - Immutable event record

## Prerequisites

1. **GCS access:** Service account with `storage.objectAdmin` on both main and recovery buckets
2. **Recovery bucket:** `gs://berlin-lst-data-recovery` in same region (`europe-west3`)
3. **Cogger binary:** `/tmp/cog-repair/cogger` (SHA-256: `79f6e988...`)
4. **VM:** `berlin-lst-vm` started and accessible

## Recovery Procedure

### Phase 1: Preflight

Verify bucket policies, IAM, and inventory key set.

```bash
# Start VM
.opencode/skills/google-access/scripts/start-vm.sh

# SSH to VM
.opencode/skills/google-access/scripts/ssh-vm.sh

# Run preflight
cd /workspace/app
uv run python scripts/repair_cog_layout.py preflight \
    --config configs/cog_repair/recovery.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03
```

**Expected output:**
- Config hash recorded
- Main bucket: versioning disabled, 7-day soft delete
- Recovery bucket: accessible
- Inventory count: 2,079
- Soft delete deadline: remaining hours > 0

**STOP conditions:**
- Count mismatch
- Soft delete deadline passed
- Recovery bucket not accessible

### Phase 2: Classify

Build inventory + evidence matrix.

```bash
uv run python scripts/repair_cog_layout.py classify \
    --config configs/cog_repair/recovery.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03 \
    --workers 4
```

**Expected output:**
- Layout classes: strict_clean=1243, missing_overview=164, hard_layout=672
- Evidence classes: audited_match=638, unaudited_overwrite=769, untouched=672
- Total: 2,079

**STOP conditions:**
- Unexpected layout class
- Count mismatch
- Classification error

### Phase 3: Backup

Preserve all current live objects to recovery bucket.

```bash
uv run python scripts/repair_cog_layout.py backup \
    --config configs/cog_repair/recovery.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03
```

**Expected output:**
- All 2,079 objects backed up
- CRC32C verified
- Backup root: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/backups/current/`

**STOP conditions:**
- Any backup failure
- CRC mismatch

### Phase 4: Stage Candidates

Generate repair candidates to recovery bucket.

```bash
uv run python scripts/repair_cog_layout.py stage \
    --config configs/cog_repair/recovery.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03 \
    --cogger-bin /tmp/cog-repair/cogger
```

**Expected output:**
- Cogger: 672 candidates for hard layout errors
- GDAL COG: 164 candidates for missing-overview flags
- All candidates pass strict validation
- All candidates pass semantic comparison

**STOP conditions:**
- Candidate validation failure
- Semantic mismatch
- Cogger/GDAL error

### Phase 5: Promote

Guarded canonical promotion with rollback.

```bash
uv run python scripts/repair_cog_layout.py promote \
    --config configs/cog_repair/recovery.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03
```

**Expected output:**
- Each candidate promoted with generation guard
- Each promoted object verified
- Rollback on failure
- Immutable events recorded

**STOP conditions:**
- Generation conflict
- Verification failure
- Rollback failure

### Phase 6: Verify Recovery

Independent verification of all canonical assets.

```bash
uv run python scripts/repair_cog_layout.py verify-recovery \
    --config configs/cog_repair/recovery.yaml \
    --workers 4
```

**Expected output:**
- All 2,079 assets valid
- Zero errors, zero warnings
- Verification PASSED

**STOP conditions:**
- Any invalid asset

## Validation Gates

After recovery, run all validators:

```bash
# ARD validation
uv run python scripts/validate_ard.py \
    --ledger gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet \
    --manifest gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# Dynamic validation
uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/full \
    --expected-role anchor \
    --expected-scenes 324

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/inference/2026 \
    --expected-role inference \
    --expected-scenes 21

# Profiling validation
uv run python scripts/validate_profiling.py \
    --output-root gs://berlin-lst-data/profiling/wb2c-1 \
    --require-clean --expected-assets 2079

# Full validation gate
uv run nox
```

## Rollback Procedure

If recovery fails, restore from backup:

```bash
# Load backup inventory
uv run python -c "
from berlin_lst_downscaling.data.ard.cog_repair import load_table
table = load_table('gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/snapshots/inventory.parquet')
print(f'Inventory: {table.num_rows} URIs')
"

# Restore from backup
# (Implementation in cog_recovery_gcs.py)
```

## Evidence Retention

Recovery artifacts retained until approved deletion date:

- Backups: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/backups/`
- Candidates: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/candidates/`
- Events: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/events/`
- Reports: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03/reports/`

## Stop Authority

The approved plan, including its Build Stop Conditions, is an execution contract. Before each task or package, read its Scope, Validation, Risk checkpoints, and Stop Conditions.

**STOP immediately when:**
- A plan Stop Condition is observed
- Implementation requires a new dependency, external operation, configuration/infra/schema/public-contract change not named in the plan
- Repository or runtime evidence contradicts a plan assumption
- Validation cannot pass through a one- or two-line self-inflicted correction in the same file and current task

## Contact

For questions or issues during recovery, consult the plan documentation or halt and ask for clarification.
