#!/usr/bin/env bash
# Run the Stage-1 raw-input QA gate (full, 324 pairs) on the On-Demand
# berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch full QA → poll the run marker → validate the published report
# bundle locally → stop VM. Products (QA evidence) live in GCS, not on
# the VM disk.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts in this directory.
#
# Usage:
#   run-qa-stage1-vm.sh [branch]
#   run-qa-stage1-vm.sh main

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Stage-1 QA"
MARKER_CONFIG="stage1_raw_full"
QA_OUTPUT_ROOT="gs://berlin-lst-data/qa/stage1_raw"
RUN_PREFIX_GREP="$QA_OUTPUT_ROOT/[0-9a-f]\\{8\\}"
REMOTE_CMD="uv run python scripts/runners/run_qa_stage1_raw.py --config-name stage1_raw_full"

vm_init_run "qa-stage1"
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
  uv run python scripts/validators/validate_qa_stage1_raw.py \
    --run-prefix "$RUN_PREFIX" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: Stage-1 raw QA completed and validated."
  echo "  Evidence: $RUN_PREFIX"
else
  echo "FAIL: Stage-1 QA validation failed or the gate found findings."
  echo "  Evidence:    ${RUN_PREFIX:-unknown}"
  echo "  Remote log:  $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi