#!/usr/bin/env bash
# Read-only feature-correctness diagnostics on berlin-lst-vm.
#
# Follows the training-EDA launcher: identity check via start-vm.sh,
# pinned commit deploy, detached launch, poll, then stop. Outputs go
# only to gs://berlin-lst-training-data/evaluation/diagnostics/<run-id>/.
# The operator script itself never starts the VM; this launcher does.
#
# Usage:
#   run-feature-diagnostics-vm.sh [branch] [lod2-tile ...]
#
# With no tile paths the LoD2 parser scope is omitted. Tile paths are
# paths on the VM and must not contain single quotes.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="main"
TILES=()
if [[ $# -ge 1 && "$1" != *.gml && "$1" != *.xml && "$1" != *.zip && "$1" != /* ]]; then
  BRANCH="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  TILES=("$@")
fi

PIPELINE_LABEL="Feature diagnostics"
MARKER_CONFIG="feature_diagnostics"
POLL_MAX_SECONDS=64800

vm_init_run "feature-diagnostics"
UPLOAD_URI="gs://berlin-lst-training-data/evaluation/diagnostics/${WRAP_RUN_ID}/"

SCOPES="--scope s2-baseline --scope eligibility-gaps"
LOD2_ARGS=""
if [[ ${#TILES[@]} -gt 0 ]]; then
  SCOPES="${SCOPES} --scope lod2-parser"
  for tile in "${TILES[@]}"; do
    LOD2_ARGS="${LOD2_ARGS} --lod2-tile ${tile}"
  done
fi

REMOTE_CMD="uv run python scripts/operators/diagnose_feature_correctness.py ${SCOPES} --yes --run-id ${WRAP_RUN_ID} --output-dir /workspace/feature-diagnostics --upload-uri ${UPLOAD_URI} --training-manifest gs://berlin-lst-training-data/training/v2/manifest.parquet --selection-manifest gs://berlin-lst-training-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet --features-root gs://berlin-lst-training-data/features/v5 --ard-root gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019${LOD2_ARGS}"

echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Upload: $UPLOAD_URI"

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish
vm_stop

if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: feature diagnostics completed and uploaded."
  echo "  Release: $UPLOAD_URI"
else
  echo "FAIL: feature diagnostics failed."
  echo "  Release:    $UPLOAD_URI"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
