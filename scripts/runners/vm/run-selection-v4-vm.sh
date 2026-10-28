#!/usr/bin/env bash
# Step 1 of docs/republish-v6.md: selection v4 manifest bundle on berlin-lst-vm.
#
# Builds the additive v4 selection policy (exact Sentinel-2 mosaic
# membership + Landsat AOI-overlap gate) into the new manifests/v4 GCS
# release root. Follows the run-training-eda-vm.sh pattern: release
# preflight → start VM → deploy pinned branch → launch → poll → validate
# → stop. The VM is stopped on completion or failure — except on
# connection loss, where it is left running for operator inspection.
#
# Usage:
#   run-selection-v4-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Selection v4 manifest"
MARKER_CONFIG="selection_v4"
CONFIG_NAME="full_2017_2026_v4"
CONFIG_DIR="configs/selection"
OUTPUT_ROOT="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z"
MANIFEST_URI="${OUTPUT_ROOT}/manifest.parquet"
REMOTE_CMD="uv run python scripts/runners/build_manifest.py --config-dir ${CONFIG_DIR} --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=64800

vm_init_run "selection-v4"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config:  $CONFIG_DIR/$CONFIG_NAME"
echo "  Output:  $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step selection_v4 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish

VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $MANIFEST_URI ..."
  uv run python scripts/validators/validate_manifest.py \
    --manifest "$MANIFEST_URI" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: selection v4 manifest published and validated."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: selection v4 run or validation failed."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
