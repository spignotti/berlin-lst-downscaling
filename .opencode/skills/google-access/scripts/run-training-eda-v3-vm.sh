#!/usr/bin/env bash
# Step 9 of docs/republish-v6.md: read-only EDA v3 on berlin-lst-vm.
#
# Scans training/v3 with the AOI-intersected denominators from #89
# (eligibility_rates.parquet weighted by the 100 m AOI fractions).
# Follows the run-training-eda-vm.sh pattern with an explicit arg set,
# explicit upstream input roots, and a new output root. The scan writes
# nothing outside evaluation/eda/v3. The VM is stopped on completion or
# on a pre-launch failure. A lost connection leaves the VM running.
#
# Usage:
#   run-training-eda-v3-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-analysis/eda-v1}"
PIPELINE_LABEL="Training EDA v3"
MARKER_CONFIG="eda_v3"
TRAINING_ROOT="gs://berlin-lst-training-data/training/v3"
V3_MANIFEST="gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
PATCH_INDEX_ROOT="gs://berlin-lst-training-data/training/patch-index/v3"
RELEASE_ROOT="gs://berlin-lst-training-data/evaluation/eda/v3"
REMOTE_CMD="uv run python scripts/operators/run_training_eda.py --output-dir /workspace/eda-v3 --training-root ${TRAINING_ROOT} --v3-manifest ${V3_MANIFEST} --patch-index-root ${PATCH_INDEX_ROOT} && uv run python scripts/validators/validate_training_eda.py --eda-dir /workspace/eda-v3 --manifest ${TRAINING_ROOT}/manifest.parquet && uv run python scripts/operators/run_training_eda.py --output-dir /workspace/eda-v3 --training-root ${TRAINING_ROOT} --v3-manifest ${V3_MANIFEST} --patch-index-root ${PATCH_INDEX_ROOT} --upload-only --upload-uri ${RELEASE_ROOT}"
POLL_MAX_SECONDS=64800

vm_init_run "training-eda-v3"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Training: $TRAINING_ROOT"
echo "  Manifest: $V3_MANIFEST"
echo "  Patches:  $PATCH_INDEX_ROOT"
echo "  Release:  $RELEASE_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py --step eda_v3 --config-dir configs/eda --config-name v3; then
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
  echo "SUCCESS: training EDA v3 completed, validated, and uploaded."
  echo "  Release: $RELEASE_ROOT"
else
  echo "FAIL: training EDA v3 failed."
  echo "  Release:    $RELEASE_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
