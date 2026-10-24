#!/usr/bin/env bash
# Step 7 of docs/republish-v6.md: training-data v3 on berlin-lst-vm.
#
# Builds the eligibility masks, train-only scaler (tp_0_72h log1p then
# z-score), and release artifacts from features/v6 into training/v3.
# training/v1 and training/v2 stay immutable. Follows the
# run-training-eda-vm.sh pattern with an explicit config name, explicit
# upstream input roots, and a new output root. The expected count is
# derived from the upstream v4 bundle (the AOI-overlap gate changes the
# assessable count versus the v2 universe), never hard-coded.
#
# Usage:
#   run-training-v3-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Training data v3"
MARKER_CONFIG="training_v3"
CONFIG_NAME="v3"
CONFIG_DIR="configs/training"
MANIFEST_URI="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
FEATURES_ROOT="gs://berlin-lst-training-data/features/v6"
ARD_ROOT="gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
RELEASE_ROOT="gs://berlin-lst-training-data/training/v3"
REMOTE_CMD="uv run python scripts/runners/run_training_data.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=64800

vm_init_run "training-v3"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config:   $CONFIG_DIR/$CONFIG_NAME"
echo "  Manifest: $MANIFEST_URI"
echo "  Features: $FEATURES_ROOT"
echo "  ARD:      $ARD_ROOT"
echo "  Release:  $RELEASE_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step training_v3 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish

# ARD v10 ledger for the S2 mosaic AOI coverage-gate exclusions: pairings
# whose S2 partner failed the gate are accepted as documented exclusions
# (reason 'ard sentinel2 not done', no eligibility mask). Derived from
# ARD_ROOT (same root as configs/training/v3.yaml ard_root); the release
# manifest row count is unchanged (one row per pairing).
ARD_LEDGER="${ARD_ROOT}/ledger.parquet"
VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $RELEASE_ROOT ..."
  uv run python scripts/validators/validate_training_data.py \
    --release-root "$RELEASE_ROOT" \
    --expected-scenes-from-manifest "$MANIFEST_URI" \
    --ard-ledger "$ARD_LEDGER" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: training-data v3 completed and validated."
  echo "  Release: $RELEASE_ROOT"
else
  echo "FAIL: training-data v3 failed or validation found findings."
  echo "  Release:    $RELEASE_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
