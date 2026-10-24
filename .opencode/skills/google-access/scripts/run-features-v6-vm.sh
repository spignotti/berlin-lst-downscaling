#!/usr/bin/env bash
# Step 6 of docs/republish-v6.md: feature stacks v6 on berlin-lst-vm.
#
# Composes the 22-channel v6 stacks (canopy_cover_fraction, slim ERA5 with
# scene_value, no second S2 offset) from ARD v10 + the v6 static/dynamic
# roots into features/v6. Follows the run-training-eda-vm.sh pattern with
# an explicit config name, explicit upstream input roots, and a new output
# root. The isolated runner is used, as for the published features run.
#
# Usage:
#   run-features-v6-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Feature stacks v6"
MARKER_CONFIG="v6"
CONFIG_NAME="v6"
CONFIG_DIR="configs/features"
MANIFEST_URI="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
ARD_ROOT="gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
STATIC_SOURCES_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
STATIC_DERIVED_ROOT="gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
DYNAMIC_ROOT="gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2"
FEATURES_ROOT="gs://berlin-lst-training-data/features/v6"
EXTRA_MARKER_JSON="  \"features_root\": \"$FEATURES_ROOT\",
"
REMOTE_CMD="uv run python scripts/runners/run_features_isolated.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=172800

vm_init_run "features-v6"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config:   $CONFIG_DIR/$CONFIG_NAME"
echo "  Manifest: $MANIFEST_URI"
echo "  ARD:      $ARD_ROOT"
echo "  Statics:  $STATIC_SOURCES_ROOT $STATIC_DERIVED_ROOT"
echo "  Dynamic:  $DYNAMIC_ROOT"
echo "  Output:   $FEATURES_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step features_v6 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
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
# ARD v10 ledger for the S2 mosaic AOI coverage-gate exclusions: pairings
# whose S2 partner failed the gate drop out of the expected features count
# (conservative rule, no replacement partner). Derived from ARD_ROOT (same
# root as configs/features/v6.yaml ard_root); never a hard-coded run id.
ARD_LEDGER="${ARD_ROOT}/ledger.parquet"
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $FEATURES_ROOT ..."
  uv run python scripts/validators/validate_feature_stacks.py \
    --root "$FEATURES_ROOT" \
    --aoi data/boundaries/aoi_10m.tif \
    --expected-scenes-from-manifest "$MANIFEST_URI" \
    --ard-ledger "$ARD_LEDGER" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: feature stacks v6 published and validated."
  echo "  Root: $FEATURES_ROOT"
else
  echo "FAIL: feature-stack v6 run or validation failed."
  echo "  Root:       $FEATURES_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
