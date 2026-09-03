#!/usr/bin/env bash
# Run the scene feature-stack pipeline (full, 324 anchors) on the
# On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): release preflight → start VM →
# deploy pinned branch → launch full run → poll the run marker → validate
# the published feature stacks (independent validator + V2→V3 comparison)
# → stop VM. Products (28-band COGs, masks, sidecars, ledger) live in GCS
# under gs://berlin-lst-data/features/v3/, not on the VM disk.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts in this directory. The VM is stopped
# on completion, failure, or discovery failure — except on connection
# loss, where it is intentionally left running until an operator confirms
# the remote job state.
#
# Usage:
#   run-features-vm.sh [branch]
#   run-features-vm.sh main

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Feature stacks"
MARKER_CONFIG="full"
FEATURES_ROOT="gs://berlin-lst-data/features/v3"
BASELINE_ROOT="gs://berlin-lst-data/features/v2"
EXTRA_MARKER_JSON="  \"features_root\": \"$FEATURES_ROOT\",
"
REMOTE_CMD="uv run python scripts/runners/run_features_isolated.py --config-name full"

vm_init_run "features"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── release preflight (hard go/no-go gate) ───────────────────────────
echo "Running release preflight..."
vm_preflight_pushed

if ! uv run python scripts/operators/preflight_feature_release.py --config-name full; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll

# The isolated runner prints the summary path; extract the run id.
vm_discover_run_id() {
  RUN_ID=$(ssh_cmd "
    grep -o '$FEATURES_ROOT/logs/features/isolated_summary_[0-9T]*' '$REMOTE_LOG' 2>/dev/null \
      | tail -1 | sed 's#.*isolated_summary_##; s#\.json##'
  " 2>/dev/null || echo "")
  if [[ -z "$RUN_ID" ]]; then
    echo "WARNING: could not discover run id from remote log (validation still covers the full root)."
  else
    echo "  Run ID:      $RUN_ID"
  fi
}

vm_finish

# ── validate published feature stacks (local, read-only) ─────────────
VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $FEATURES_ROOT ..."
  uv run python scripts/validators/validate_feature_stacks.py \
    --root "$FEATURES_ROOT" \
    --aoi data/boundaries/aoi_10m.tif \
    --expected-scenes 324 \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "Comparing $BASELINE_ROOT -> $FEATURES_ROOT ..."
  uv run python scripts/operators/compare_feature_releases.py \
    --baseline-root "$BASELINE_ROOT" \
    --candidate-root "$FEATURES_ROOT" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: feature stacks published and validated."
  echo "  Root:   $FEATURES_ROOT"
  echo "  Run ID: ${RUN_ID:-unknown}"
else
  echo "FAIL: feature-stack run or validation failed."
  echo "  Root:       $FEATURES_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi