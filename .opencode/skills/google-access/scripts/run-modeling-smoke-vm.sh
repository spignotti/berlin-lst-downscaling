#!/usr/bin/env bash
# Run the bounded WB3 modeling real smoke on the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch `nox -s smoke-real-comparison` → poll the run marker → verify the
# bounded evidence → clean this run's ephemeral smoke output → stop VM.
#
# The smoke publishes nothing to GCS: it writes only local ephemeral output
# under data/smoke/, which the session itself removes in `finally`. The
# evidence is therefore the bounded remote log (Nox session output plus the
# streaming validator's summary), not a GCS artifact — there is no separate
# local validation step.
#
# Run it on the VM, never on a workstation: the loader-worker arm uses the
# Linux process model, and this is a bounded check (four patches per split,
# one epoch), not a training run. The poll is bounded by POLL_MAX_SECONDS; on
# expiry the run is left for operator inspection rather than claimed or
# killed.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and the
# fail-closed lifecycle scripts in this directory.
#
# Usage:
#   run-modeling-smoke-vm.sh [branch]
#   run-modeling-smoke-vm.sh feat/30-stream-real-patch-dataset

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Modeling real smoke"
MARKER_CONFIG="modeling_real_smoke"
POLL_MAX_SECONDS=2700
REMOTE_CMD="uv run nox -s smoke-real-comparison"

# Ephemeral local smoke roots the session writes on the VM.
SMOKE_ROOTS="$APP_DIR/data/smoke/modeling-real $APP_DIR/data/smoke/baseline-real"

vm_init_run "modeling-smoke"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

# ── verify the bounded evidence while the VM is still up ─────────────
# Success needs the terminal exit code AND both in-session markers, with no
# traceback in the log. Checked before the VM stops so the log is reachable.
EVIDENCE_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]] \
  && ssh_cmd "
    grep -q 'OK: streaming path is read-invariant' '$REMOTE_LOG' && \
    grep -q 'Session smoke-real-comparison was successful' '$REMOTE_LOG' && \
    ! grep -q 'Traceback (most recent call last)' '$REMOTE_LOG'
  "; then
  EVIDENCE_OK=0
  # The session's own finally removes these; this is a bounded backstop so a
  # green run never leaves ephemeral output on the VM disk.
  ssh_cmd "rm -rf $SMOKE_ROOTS" || true
fi

# ── stop VM + report ─────────────────────────────────────────────────
vm_stop

if [[ "$EVIDENCE_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: modeling real smoke completed and passed its in-session checks."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG (bounded evidence; nothing published to GCS)"
else
  echo "FAIL: modeling real smoke did not complete cleanly (exit $PIPELINE_EXIT)."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Inspect VM-side files with: $VM_SCRIPTS/status-vm.sh"
  exit 1
fi
