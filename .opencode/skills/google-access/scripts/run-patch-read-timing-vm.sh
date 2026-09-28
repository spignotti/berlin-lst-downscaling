#!/usr/bin/env bash
# Measure the real patch reader's data phases on the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch scripts/validators/measure_patch_reads.py → poll the run marker
# → verify the bounded evidence → echo the report → clean this run's ephemeral
# output → stop VM.
#
# The probe publishes nothing to GCS: it reads the published sources and writes
# only local ephemeral output under data/smoke/patch-read-timing, which this
# launcher removes after the report is echoed locally. The evidence is
# therefore the bounded remote log plus the report echoed here.
#
# Run it on the VM, never on a workstation: the loader arm uses the Linux
# process model and multiple worker processes. This is a bounded measurement,
# not a training run.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and the
# fail-closed lifecycle scripts in this directory.
#
# Usage:
#   run-patch-read-timing-vm.sh [branch] [extra probe args...]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
shift || true
EXTRA_ARGS="${*:-}"

PIPELINE_LABEL="Patch read timing"
MARKER_CONFIG="patch_read_timing"
POLL_MAX_SECONDS=3600
REPORT_REL="data/smoke/patch-read-timing/report.json"
REPORT_PATH="$APP_DIR/$REPORT_REL"
REMOTE_CMD="uv run python scripts/validators/measure_patch_reads.py --output $REPORT_REL $EXTRA_ARGS"

vm_init_run "patch-read-timing"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

# ── capture the bounded evidence while the VM is still up ────────────
# Success needs the terminal exit code AND both in-session markers, with no
# traceback in the log. The report is echoed before the VM stops so it is
# reachable after the ephemeral root is removed.
EVIDENCE_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]] \
  && ssh_cmd "
    grep -q 'OK: patch read timing measured' '$REMOTE_LOG' && \
    grep -q 'REPORT_JSON:' '$REMOTE_LOG' && \
    ! grep -q 'Traceback (most recent call last)' '$REMOTE_LOG'
  "; then
  EVIDENCE_OK=0
  echo "===== PATCH READ TIMING REPORT ====="
  ssh_cmd "cat '$REPORT_PATH'"
  echo "===== END PATCH READ TIMING REPORT ====="
  ssh_cmd "rm -rf $APP_DIR/data/smoke/patch-read-timing" || true
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$EVIDENCE_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: patch read timing measured; report above."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG (bounded evidence; nothing published to GCS)"
else
  echo "FAIL: patch read timing did not complete cleanly (exit $PIPELINE_EXIT)."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Inspect VM-side files with: $VM_SCRIPTS/status-vm.sh"
  exit 1
fi
