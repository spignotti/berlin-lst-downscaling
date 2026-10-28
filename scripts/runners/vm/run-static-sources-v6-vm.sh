#!/usr/bin/env bash
# Step 3 of docs/republish-v6.md: static sources v6 on berlin-lst-vm.
#
# Rebuilds the area-exact LoD2 morphology (BuildingPart heights) plus the
# vegetation-height product with the third canopy_cover_fraction band into
# the new static/sources/lod2-area-exact-v2 prefix. Follows the
# run-training-eda-vm.sh pattern with an explicit config name and a new
# output root. Imperviousness stays on the published v3 COGs; the new
# ledger points at them.
#
# Usage:
#   run-static-sources-v6-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Static sources v6"
MARKER_CONFIG="static_sources_v6"
CONFIG_NAME="v6"
CONFIG_DIR="configs/static_sources"
OUTPUT_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
REMOTE_CMD="uv run python scripts/runners/run_static_sources.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=172800

vm_init_run "static-sources-v6"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config: $CONFIG_DIR/$CONFIG_NAME"
echo "  Output: $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step static_sources_v6 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
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
LEDGER_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Checking static-sources ledger (fail-closed)..."
  uv run python scripts/operators/check_static_sources_ledger.py \
    --source-root "$OUTPUT_ROOT" \
    && LEDGER_OK=0 || LEDGER_OK=1
fi

if [[ "$LEDGER_OK" -eq 0 ]]; then
  echo "Validating $OUTPUT_ROOT (2024 coverage only; historical lod2_morphology vintages live in older releases) ..."
  uv run python scripts/validators/validate_lod_coverage.py \
    --static-sources-root "$OUTPUT_ROOT" \
    --vintages 2024 \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$LEDGER_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: static sources v6 published and validated."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: static sources v6 run or validation failed."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
