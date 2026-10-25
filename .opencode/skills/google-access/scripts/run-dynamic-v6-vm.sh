#!/usr/bin/env bash
# Step 5 of docs/republish-v6.md: dynamic v6 on berlin-lst-vm.
#
# Builds building shadows from the v6 horizons and the ERA5 v6
# scene_value representation (AOI mean of the bilinear 10 m field) from
# the v4 manifest into the new dynamic/lod2-area-exact-v2 prefix. Follows
# the run-training-eda-vm.sh pattern with an explicit config name,
# explicit upstream input roots, and a new output root.
#
# Usage:
#   run-dynamic-v6-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Dynamic v6"
MARKER_CONFIG="dynamic_v6"
CONFIG_NAME="v6"
CONFIG_DIR="configs/dynamic"
MANIFEST_URI="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
SOURCE_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
DERIVED_ROOT="gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
OUTPUT_ROOT="gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2"
REMOTE_CMD="uv run python scripts/runners/run_dynamic.py --config-name ${CONFIG_NAME} manifest_uri=${MANIFEST_URI}"
POLL_MAX_SECONDS=64800

vm_init_run "dynamic-v6"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config:   $CONFIG_DIR/$CONFIG_NAME"
echo "  Manifest: $MANIFEST_URI"
echo "  Sources:  $SOURCE_ROOT"
echo "  Derived:  $DERIVED_ROOT"
echo "  Output:   $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step dynamic_v6 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
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
  uv run python scripts/validators/validate_dynamic.py \
    --output-root "$OUTPUT_ROOT" \
    --expected-role anchor \
    --expected-scenes-from-manifest "$MANIFEST_URI" \
    --check-bands --era5-channel-set v6 \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: dynamic v6 published and validated."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: dynamic v6 run or validation failed."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
