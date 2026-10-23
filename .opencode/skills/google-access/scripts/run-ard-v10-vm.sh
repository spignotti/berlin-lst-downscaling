#!/usr/bin/env bash
# Step 2 of docs/republish-v6.md: Sentinel-2 ARD v10 mosaics on berlin-lst-vm.
#
# Processes the recorded-member S2 mosaics from the v4 selection bundle
# into the new ARD v10 root. Landsat/ECOSTRESS ARD is reused, not
# reprocessed: both contracts are unchanged (landsat-c2-l2 schema 7 with
# area-mean lwir11 + native_bit_or QA, ecostress schema 7 per
# src/berlin_lst_downscaling/data/ard/contract.py). The remote command
# first seeds the v10 ledger idempotently with the published
# Landsat/ECOSTRESS rows (scripts/operators/seed_ard_v10_reuse.py; COGs
# stay referenced in place), then runs the S2-only mosaic config, which
# keeps seeded rows and processes only S2 (reconcile iterates manifest
# S2 rows). Follows the run-training-eda-vm.sh pattern with an explicit
# config name, explicit upstream input root, and a new output root.
#
# Usage:
#   run-ard-v10-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Sentinel-2 ARD v10"
MARKER_CONFIG="ard_v10"
CONFIG_NAME="sentinel2_mosaic_v10"
CONFIG_DIR="configs/ard"
MANIFEST_URI="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
REUSE_ROOT="gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019"
OUTPUT_ROOT="gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
REMOTE_CMD="uv run python scripts/operators/seed_ard_v10_reuse.py --source-ledger ${REUSE_ROOT}/ledger.parquet --manifest-uri ${MANIFEST_URI} --dest-root ${OUTPUT_ROOT} && uv run python scripts/runners/run_ard.py --config-name ${CONFIG_NAME} manifest_uri=${MANIFEST_URI}"
POLL_MAX_SECONDS=172800

vm_init_run "ard-v10"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config:   $CONFIG_DIR/$CONFIG_NAME"
echo "  Manifest: $MANIFEST_URI"
echo "  Reuse:    $REUSE_ROOT (Landsat/ECOSTRESS ledger seed, contracts unchanged)"
echo "  Output:   $OUTPUT_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step ard_v10 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
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
  uv run python scripts/validators/validate_ard.py \
    --ledger "${OUTPUT_ROOT}/ledger.parquet" \
    --manifest "$MANIFEST_URI" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: Sentinel-2 ARD v10 published and validated."
  echo "  Release: $OUTPUT_ROOT"
else
  echo "FAIL: ARD v10 run or validation failed."
  echo "  Release:    $OUTPUT_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
