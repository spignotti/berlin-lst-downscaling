#!/usr/bin/env bash
# Run the WB2c-4 training-data release (full, 324 assessable scenes) on
# the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch full run → poll the run marker → validate the published
# release locally → stop VM. Evidence (release artifacts) lives in GCS,
# not on the VM disk. The VM is stopped on completion, failure, or
# discovery failure — except on connection loss, where it is
# intentionally left running until an operator confirms the remote job
# state.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts in this directory.
#
# Usage:
#   run-training-data-vm.sh [branch]
#   run-training-data-vm.sh main

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Training data release"
MARKER_CONFIG="training_full"
RELEASE_ROOT="gs://berlin-lst-data/training/v1"
REMOTE_CMD="uv run python scripts/runners/run_training_data.py --config-name full"

vm_init_run "training-data"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

# ── validate published release (local, read-only) ────────────────────
VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $RELEASE_ROOT ..."
  uv run python scripts/validators/validate_training_data.py \
    --release-root "$RELEASE_ROOT" \
    --expected-scenes 345 \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: training-data release completed and validated."
  echo "  Release: $RELEASE_ROOT"
else
  echo "FAIL: training-data release failed or validation found findings."
  echo "  Release:    $RELEASE_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi