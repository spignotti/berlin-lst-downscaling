#!/usr/bin/env bash
# Run the Stage-2 feature-stack QA gate (full, 324 stacks) on the
# On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch full QA → poll the run marker → validate the published report
# bundle locally → stop VM. Evidence (QA report) lives in GCS, not on the
# VM disk. The VM is stopped on completion, failure, or discovery failure
# — except on connection loss, where it is intentionally left running
# until an operator confirms the remote job state.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts in this directory.
#
# Usage:
#   run-qa-stage2-vm.sh [branch]
#   run-qa-stage2-vm.sh main

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Stage-2 feature QA"
MARKER_CONFIG="stage2_features_full"
QA_OUTPUT_ROOT="gs://berlin-lst-data/qa/stage2_features"
RUN_PREFIX_GREP="$QA_OUTPUT_ROOT/[0-9a-f]\\{8\\}"
REMOTE_CMD="uv run python scripts/runners/run_qa_stage2_features.py --config-name stage2_features_full"

vm_init_run "qa-stage2"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

# ── validate published report bundle (local, read-only) ──────────────
VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" && -n "${RUN_PREFIX:-}" ]]; then
  echo "Validating $RUN_PREFIX ..."
  uv run python scripts/validators/validate_qa_stage2_features.py \
    --run-prefix "$RUN_PREFIX" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: Stage-2 feature QA completed and validated."
  echo "  Evidence: $RUN_PREFIX"
else
  echo "FAIL: Stage-2 feature QA validation failed or the gate found findings."
  echo "  Evidence:    ${RUN_PREFIX:-unknown}"
  echo "  Remote log:  $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi