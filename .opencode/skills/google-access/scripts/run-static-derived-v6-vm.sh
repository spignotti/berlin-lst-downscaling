#!/usr/bin/env bash
# Step 4 of docs/republish-v6.md: static derived v6 on berlin-lst-vm.
#
# Builds building/vegetation/combined DSMs, horizons, and SVF (radius 3
# cells / 30 m, part of the derived config hash) from the v6 static
# sources into the new static/derived/lod2-area-exact-v2 prefix. Follows
# the run-training-eda-vm.sh pattern with an explicit config name,
# explicit upstream input root, and a new output root. There is no
# dedicated derived validator; the publisher-side readback plus the
# step-6 features inventory gate cover the products.
#
# Usage:
#   run-static-derived-v6-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Static derived v6"
MARKER_CONFIG="static_derived_v6"
CONFIG_NAME="v6"
CONFIG_DIR="configs/static_derived"
SOURCE_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
OUTPUT_ROOT="gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
REMOTE_CMD="uv run python scripts/runners/run_static_derived.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=172800

vm_init_run "static-derived-v6"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config: $CONFIG_DIR/$CONFIG_NAME"
echo "  Source: $SOURCE_ROOT"
echo "  Output: $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step static_derived_v6 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish
vm_stop

if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: static derived v6 run completed (publisher readback only)."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: static derived v6 run failed."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
