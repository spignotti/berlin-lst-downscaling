#!/usr/bin/env bash
# Step 8 of docs/republish-v6.md: patch index v3 on berlin-lst-vm.
#
# Enumerates the 160x160 px windows over the training/v3 eligibility
# masks into training/patch-index/v3. Follows the
# run-training-eda-vm.sh pattern with an explicit config name, an
# explicit upstream input root, and a new output root. training/v1 and
# patch-index v1/v2 stay immutable.
#
# Usage:
#   run-patch-index-v3-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Patch index v3"
MARKER_CONFIG="patch_index_v3"
CONFIG_NAME="patch_index_v3"
CONFIG_DIR="configs/training"
SOURCE_ROOT="gs://berlin-lst-training-data/training/v3"
OUTPUT_ROOT="gs://berlin-lst-training-data/training/patch-index/v3"
REMOTE_CMD="uv run python scripts/runners/run_patch_index.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=21600

vm_init_run "patch-index-v3"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config: $CONFIG_DIR/$CONFIG_NAME"
echo "  Source: $SOURCE_ROOT"
echo "  Output: $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step patch_index_v3 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
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
  echo "Validating $OUTPUT_ROOT ..."
  uv run python scripts/validators/validate_patch_index.py \
    --index-root "$OUTPUT_ROOT" \
    --source-root "$SOURCE_ROOT" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: patch index v3 completed and validated."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: patch index v3 failed or validation found findings."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
