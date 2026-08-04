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
3. **Cogger binary:** verified against official `v0.1.1` Linux AMD64 ZIP hash
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
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation
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

### Phase 2: Rebaseline

Build immutable baseline: inventory, Soft Delete catalog, and run manifest.

```bash
uv run python scripts/repair_cog_layout.py rebaseline \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \
    --run-id <run-id>
```

**Expected output:**
- Layout classes: strict_clean=1243, missing_overview=164, hard_layout=672
- Evidence classes: audited_match=638, unaudited_overwrite=769, untouched=672
- Soft Delete catalog: 1,407 canonical generations
- Deadline runway: 3× runtime + 24h margin
- Run manifest persisted

**STOP conditions:**
- Layout count mismatch
- Soft Delete count mismatch
- Insufficient deadline runway

### Phase 3: Canary

Prove GCS transaction semantics on isolated scratch paths.

```bash
uv run python scripts/smoke_cog_recovery.py \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-canary
```

**Expected output:**
- All tests pass (basic copy, soft delete restore, rewrite with metadata, event persistence)
- Canary report persisted

**STOP conditions:**
- Any test failure

### Phase 4: Capture Originals

Preserve all 1,407 pre-incident generations before hard-delete deadline.

```bash
uv run python scripts/repair_cog_layout.py capture-originals \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \
    --run-id <run-id> \
    --execute
```

**Expected output:**
- 1,407 originals captured
- All canonical objects restored to baseline
- Immutable events for each capture

**STOP conditions:**
- Any capture failure
- Deadline exhaustion
- Metadata mismatch

### Phase 5: Stage Candidates

Generate 164 GDAL and 672 Cogger candidates.

```bash
uv run python scripts/repair_cog_layout.py stage \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \
    --run-id <run-id> \
    --cogger-bin /path/to/cogger \
    --execute
```

**Expected output:**
- 164 GDAL COG candidates (missing-overview flags)
- 672 Cogger candidates (hard-layout errors)
- All candidates strict-clean and semantically equivalent

**STOP conditions:**
- Candidate validation failure
- Semantic mismatch
- Cogger/GDAL error

### Phase 6: Validate Candidates

Independent verification of all candidates.

```bash
uv run python scripts/validate_cog_recovery.py \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \
    --run-id <run-id> \
    --workers 4
```

**Expected output:**
- All 836 candidates valid
- Validation report persisted

**STOP conditions:**
- Any invalid candidate

### Phase 7: Promote

Guarded promotion with bounded batches and rollback.

```bash
uv run python scripts/repair_cog_layout.py promote \
    --config configs/cog_repair/remediation.yaml \
    --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \
    --run-id <run-id> \
    --execute
```

**Expected output:**
- Each candidate promoted with generation guard
- Each promoted object independently verified
- Rollback on failure (stops run)
- Immutable events recorded

**STOP conditions:**
- Generation conflict
- Verification failure
- Rollback failure

### Phase 8: Verify Recovery

Independent verification of all canonical assets.

```bash
uv run python scripts/repair_cog_layout.py verify-recovery \
    --config configs/cog_repair/remediation.yaml
```

**Expected output:**
- All 2,079 assets strict-clean
- Zero errors, zero warnings

**STOP conditions:**
- Any invalid asset

## Validation Gates

After recovery, run all validators:

```bash
# Full validation gate
uv run nox
```

## Evidence Retention

Recovery artifacts retained until approved deletion date:

- Backups: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/backups/`
- Originals: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/originals/`
- Candidates: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/candidates/`
- Events: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/events/`
- Reports: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/reports/`

## Architecture

```text
Fresh baseline + Soft Delete catalog
             │
             ▼
 per-object original capture saga
             │
     ┌───────┴────────┐
     ▼                ▼
164 GDAL flags   672 Cogger layouts
     └───────┬────────┘
             ▼
 independent candidate validation
             ▼
 guarded promotion + rollback
             ▼
 fresh 2,079-object final validation
```

## Key Files

| File | Purpose |
|------|---------|
| `src/.../cog_recovery_state.py` | Event models, reducer, classification |
| `src/.../cog_recovery_gcs.py` | GCS operations, descriptors, events |
| `src/.../cog_recovery.py` | Saga orchestrator |
| `scripts/repair_cog_layout.py` | CLI |
| `scripts/smoke_cog_recovery.py` | GCS transaction canary |
| `scripts/validate_cog_recovery.py` | Independent candidate validator |
| `configs/cog_repair/remediation.yaml` | Immutable run configuration |
