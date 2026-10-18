#!/usr/bin/env bash
# Read-only EDA of the published training release on berlin-lst-vm.
#
# The scan reads Landsat LST and the 28 V3 channels for every eligible
# cell. Outputs land in GCS. The VM is stopped on completion or on a
# pre-launch failure. A lost connection leaves the VM running.
#
# Usage:
#   run-training-eda-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-analysis/eda-v1}"
PIPELINE_LABEL="Training EDA"
MARKER_CONFIG="eda_v1"
RELEASE_ROOT="gs://berlin-lst-training-data/evaluation/eda/v1"
REMOTE_CMD="uv run python scripts/operators/run_training_eda.py --output-dir /workspace/eda-v1 && uv run python scripts/validators/validate_training_eda.py --eda-dir /workspace/eda-v1 --manifest gs://berlin-lst-training-data/training/v1/manifest.parquet && uv run python scripts/operators/run_training_eda.py --output-dir /workspace/eda-v1 --upload-only --upload-uri ${RELEASE_ROOT}"
POLL_MAX_SECONDS=64800

vm_init_run "training-eda"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish
vm_stop

if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: training EDA completed, validated, and uploaded."
  echo "  Release: $RELEASE_ROOT"
else
  echo "FAIL: training EDA failed."
  echo "  Release:    $RELEASE_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
