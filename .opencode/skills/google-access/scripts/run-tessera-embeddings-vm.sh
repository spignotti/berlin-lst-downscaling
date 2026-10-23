#!/usr/bin/env bash
# Canonical TESSERA annual embedding publish on berlin-lst-vm (NOT executed in CI).
#
# Reads TESSERA v1.1-dclimate for the Berlin bbox for one year (2017-2025),
# reprojects int8+scale onto the canonical EPSG:25833 10 m grid with nearest
# resampling, and publishes under:
#   gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate/<year>/
# then runs the independent validator. The VM is stopped on completion or on
# a pre-launch failure. A lost connection leaves the VM running.
#
# Code only in this PR: this launcher is not executed here.
#
# Usage:
#   run-tessera-embeddings-vm.sh [branch] [year]
#   run-tessera-embeddings-vm.sh feat/tessera-arm 2024

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-feat/tessera-arm}"
YEAR="${2:-2024}"
PIPELINE_LABEL="TESSERA embeddings"
MARKER_CONFIG="tessera_embeddings_v1"
RELEASE_ROOT="gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate"
# The acquisition operator runs in an ISOLATED uv script env (PEP 723 block
# at the top of run_tessera_embeddings.py: geotessera / icechunk / zarr>=3),
# separate from the main training env (torch 2.2.2 / zarr<3) that the
# validator below uses. PYTHONPATH=src exposes the repo package to the
# isolated script env. The remote CWD is the repo checkout (APP_DIR).
REMOTE_CMD="PYTHONPATH=src uv run --script scripts/operators/run_tessera_embeddings.py --year ${YEAR} --output-root ${RELEASE_ROOT} && uv run python scripts/validators/validate_tessera_embeddings.py --root ${RELEASE_ROOT} --year ${YEAR}"
POLL_MAX_SECONDS=64800

vm_init_run "tessera-embeddings"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Year: $YEAR | Run: $WRAP_RUN_ID"

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish
vm_stop

if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: TESSERA embeddings year ${YEAR} published and validated."
  echo "  Release: $RELEASE_ROOT/$YEAR/"
else
  echo "FAIL: TESSERA embeddings year ${YEAR} failed."
  echo "  Release:    $RELEASE_ROOT/$YEAR/"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
