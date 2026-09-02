#!/usr/bin/env bash
# Run a Dynamic pipeline config on the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy code → launch
# pipeline → poll → validate → stop VM. The boot disk is retained after
# stop; pipeline products live in GCS.
#
# Every remote command uses ssh-vm.sh (strict host-key verification).
# Each run writes an immutable marker on the VM so that a later
# status-dynamic-vm.sh can report running/completed/failed/connection-lost
# without relaunching the process. The VM is stopped on completion,
# failure, or discovery failure — except on connection loss, where it is
# intentionally left running until an operator confirms the remote job
# state.
#
# Usage:
#   run-dynamic-vm.sh full [branch]
#   run-dynamic-vm.sh inference_2026 [branch]
#
# Examples:
#   run-dynamic-vm.sh full main
#   run-dynamic-vm.sh inference_2026 fix/dynamic-resumable-on-demand

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

CONFIG="${1:-}"
BRANCH="${2:-main}"

if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 <full|inference_2026> [branch]" >&2
  exit 1
fi

if [[ "$CONFIG" != "full" && "$CONFIG" != "inference_2026" ]]; then
  echo "ERROR: config must be 'full' or 'inference_2026', got: $CONFIG" >&2
  exit 1
fi

PIPELINE_LABEL="Dynamic pipeline"
MARKER_CONFIG="$CONFIG"
MANIFEST_URI="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
REMOTE_CMD="uv run python scripts/runners/run_dynamic.py --config-name $CONFIG manifest_uri=$MANIFEST_URI"

vm_init_run "$CONFIG"
echo "Config: $CONFIG | Branch: $BRANCH | Run ID: $WRAP_RUN_ID"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

PIPELINE_ERRORS=$(ssh_cmd "
  grep -c 'error\\|ERROR\\|Traceback\\|SystemExit' '$REMOTE_LOG' 2>/dev/null || echo 0
" 2>/dev/null || echo "unknown")
echo "  Error lines: $PIPELINE_ERRORS"

# ── validate (local, read-only) ──────────────────────────────────────
if [[ "$CONFIG" == "full" ]]; then
  EXPECTED_ROLE="anchor"
  EXPECTED_SCENES="324"
  OUTPUT_ROOT="gs://berlin-lst-data/dynamic/full"
else
  EXPECTED_ROLE="inference"
  EXPECTED_SCENES="21"
  OUTPUT_ROOT="gs://berlin-lst-data/dynamic/inference/2026"
fi

VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $OUTPUT_ROOT ..."
  uv run python scripts/validators/validate_dynamic.py \
    --output-root "$OUTPUT_ROOT" \
    --expected-role "$EXPECTED_ROLE" \
    --expected-scenes "$EXPECTED_SCENES" \
    --check-bands \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: $CONFIG completed and validated."
  echo "  Run ID: $WRAP_RUN_ID"
else
  echo "FAIL: $CONFIG validation failed or pipeline had errors."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Check VM-side files with: status-dynamic-vm.sh --run-id $WRAP_RUN_ID"
  exit 1
fi