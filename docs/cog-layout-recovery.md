# COG Layout Recovery Runbook

## Incident Summary

**Date:** 2026-08-03
**Duration:** ~2.5 hours (14:19–16:35 UTC)
**Affected assets:** 2,079 canonical COGs
**Root cause:** COG repair process without fail-closed validation

### Impact

- **1,243** assets are strict-valid (zero errors, zero warnings)
- **164** assets have missing-overview warning only (flag COGs)
- **672** assets have hard IFD/offset layout errors
- **1,407** assets were overwritten; **672** untouched hard-layout failures

## Recovery Strategy

```text
1,243 strict-clean  → untouched (already correct)
  164 flag COGs     → GDAL COG from captured originals
  672 hard-layout   → Cogger from legacy payload backups
```

Source sets:
- **164 GDAL candidates:** from captured pre-incident originals
- **672 Cogger candidates:** from verified legacy payload backups
- **Legacy candidates (164):** explicitly forbidden, never adopted

## Prerequisites

1. VM running, code at exact committed SHA
2. Cogger v0.1.1 installed at `/workspace/tools/cogger-v0.1.1/cogger` with ZIP SHA `79f6e988…`
3. Recovery bucket `gs://berlin-lst-data-recovery` accessible
4. Legacy recovery root: `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03`

## Execution

### Step 0: Deploy and verify

```bash
# On local machine — deploy exact commit to VM
S=.opencode/skills/google-access/scripts
COMMIT=$(git rev-parse HEAD)
"$S/start-vm.sh"
"$S/ssh-vm.sh" -- "cd /workspace/app && git fetch origin && git checkout $COMMIT && uv sync --frozen && uv run nox"

# Verify Cogger
"$S/ssh-vm.sh" -- "sha256sum /workspace/tools/cogger-v0.1.1/cogger && /workspace/tools/cogger-v0.1.1/cogger -h"
```

### Step 1: Preflight

```bash
S=.opencode/skills/google-access/scripts
CONFIG=configs/cog_repair/remediation.yaml
ROOT=gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation
RUN_ID="cog-recovery-$(date -u +%Y%m%dT%H%M%SZ)"

"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py preflight \
    --config $CONFIG --recovery-root $ROOT"
```

**Gate:** config hash, inventory 2,079, deadline > 0.

### Step 2: Canary

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/smoke_cog_recovery.py \
    --config $CONFIG --recovery-root ${ROOT}-canary"
```

**Gate:** all 4 test suites pass.

### Step 3: Rebaseline

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py rebaseline \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID"
```

**Gate:** layouts 1243/164/672, Soft Delete 1407, deadline runway OK.

### Step 4: Capture originals

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py capture-originals \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID --execute"
```

This step for each of 1,407 overwritten assets:
1. Restores pre-incident original to canonical path
2. Copies restored original to `originals/` in recovery root
3. Rewrites canonical from legacy backup with frozen metadata
4. Verifies and records events

**Gate:** 1,407 captured, zero failed.

### Step 5: Stage candidates

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py stage \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID \
    --cogger-bin /workspace/tools/cogger-v0.1.1/cogger --execute"
```

**Gate:** 836 staged (164 GDAL + 672 Cogger), zero failed.

### Step 6: Validate candidates

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/validate_cog_recovery.py \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID --workers 4"
```

**Gate:** 836 valid, zero invalid.

### Step 7: Promote

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py promote \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID --execute"
```

Batches: 1 GDAL canary, 1 Cogger canary, 10, 50, then 100s until all 836 promoted.
Each batch: intent → guarded rewrite → independent verify → terminal event.
On failure: rollback from legacy backup → stop entire run.

**Gate:** 836 promoted, zero rolled back.

### Step 8: Final verification

```bash
"$S/ssh-vm.sh" -- "cd /workspace/app && uv run python scripts/repair_cog_layout.py verify-recovery \
    --config $CONFIG --recovery-root $ROOT --run-id $RUN_ID --workers 4"
```

Scans all 2,079 canonical assets. Writes `complete.json` on success.

**Gate:** 2,079 strict-clean, `complete.json` exists.

### Step 9: Stop VM

```bash
"$S/stop-vm.sh"
```

## Evidence

All artifacts in `gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation/<run-id>/`:
- `manifests/` — run manifest
- `snapshots/` — inventory
- `originals/` — captured pre-incident generations
- `candidates/` — GDAL and Cogger repair candidates
- `events/` — immutable per-object event chains
- `reports/` — validator reports
- `complete.json` — final completion marker
