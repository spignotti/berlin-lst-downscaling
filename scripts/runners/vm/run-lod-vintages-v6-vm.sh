#!/usr/bin/env bash
# Step 4b of docs/republish-v6.md: historical LoD vintages v6 on berlin-lst-vm.
#
# Backfills the 2017/2021/2022 area-exact LoD2 morphology (archive-first
# via lod_vintages.py: LoD1-2017 stock filter, BuildingPart heights) plus
# the per-vintage derived bundles (building/combined DSMs,
# horizon_building, SVF at 100 m) into the v6 static roots. Runs after
# step 4 (needs terrain_height/2021 + vegetation_height/2020 in the v6
# source root and the vintage-agnostic vegetation_dsm/2024 in the v6
# derived root) and before step 5. Follows the run-training-eda-vm.sh
# pattern with an explicit config name and existing output roots.
#
# Usage:
#   run-lod-vintages-v6-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Historical LoD vintages v6"
MARKER_CONFIG="lod_vintages_v6"
CONFIG_NAME="v6"
CONFIG_DIR="configs/lod_vintages"
SOURCE_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
DERIVED_ROOT="gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
REMOTE_CMD="uv run python scripts/runners/run_lod_vintages_v6.py --config-name ${CONFIG_NAME}"
POLL_MAX_SECONDS=172800

vm_init_run "lod-vintages-v6"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Config: $CONFIG_DIR/$CONFIG_NAME"
echo "  Sources: $SOURCE_ROOT"
echo "  Derived: $DERIVED_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step lod_vintages_v6 --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish

SOURCES_OK=1
DERIVED_OK=1
VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Checking static-sources ledger (fail-closed)..."
  uv run python scripts/operators/check_static_sources_ledger.py \
    --source-root "$SOURCE_ROOT" \
    && SOURCES_OK=0 || SOURCES_OK=1
  echo "Checking static-derived ledger (fail-closed)..."
  uv run python scripts/operators/check_static_derived_ledger.py \
    --derived-root "$DERIVED_ROOT" \
    && DERIVED_OK=0 || DERIVED_OK=1
fi

if [[ "$SOURCES_OK" -eq 0 && "$DERIVED_OK" -eq 0 ]]; then
  echo "Validating $SOURCE_ROOT (all four LoD vintages) ..."
  uv run python scripts/validators/validate_lod_coverage.py \
    --static-sources-root "$SOURCE_ROOT" \
    --vintages 2017,2021,2022,2024 \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$SOURCES_OK" -eq 0 && "$DERIVED_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: historical LoD vintages v6 published and validated."
  echo "  Sources: $SOURCE_ROOT"
  echo "  Derived: $DERIVED_ROOT"
else
  echo "FAIL: historical LoD vintages v6 run or validation failed."
  echo "  Sources:    $SOURCE_ROOT"
  echo "  Derived:    $DERIVED_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
