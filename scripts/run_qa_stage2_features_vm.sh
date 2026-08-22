#!/usr/bin/env bash
# Run the Stage-2 feature-stack QA gate (full, 324 stacks) on the
# On-Demand berlin-lst-vm.
#
# Lifecycle: start VM → deploy committed branch → launch full QA →
# poll the run marker → validate the published report bundle locally →
# stop VM. Evidence (QA report) lives in GCS, not on the VM disk. The VM
# is stopped on completion, failure, or discovery failure — except on
# connection loss, where it is intentionally left running until an
# operator confirms the remote job state (see the exit-2 path).
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts from .opencode/skills/google-access/.
#
# Usage:
#   scripts/run_qa_stage2_features_vm.sh [branch]
#   scripts/run_qa_stage2_features_vm.sh main

set -euo pipefail

VM_SCRIPTS="$(cd "$(dirname "$0")/../.opencode/skills/google-access/scripts" && pwd)"
source "$VM_SCRIPTS/vm-identity.sh"

APP_DIR="/workspace/app"
QA_OUTPUT_ROOT="gs://berlin-lst-data/qa/stage2_features"

CONNECTION_RETRIES=5
CONNECTION_RETRY_WAIT=30

# ── fail-safe VM cleanup ─────────────────────────────────────────────
# After a successful start, any pre-launch setup/deploy failure must stop
# the VM (AGENTS.md: VM stays stopped when not actively running). Once the
# QA run is launched, every ambiguous exit leaves the VM RUNNING — a
# possibly-still-running job must never be killed mid-write. The explicit
# connection-loss path below also disarms stopping via leave_running.
vm_started=0
leave_running=0
pipeline_launched=0

cleanup() {
  local rc=$?
  # leave_running is authoritative: any path that sets it (connection loss
  # before or after launch) must never trigger an automatic VM stop.
  if [[ "$leave_running" -eq 1 ]]; then
    exit "$rc"
  fi
  if [[ "$pipeline_launched" -eq 1 ]]; then
    echo "WARNING: abnormal exit after QA launch — VM left RUNNING."
    echo "  Inspect the marker/log under $APP_DIR/logs/runs/$WRAP_RUN_ID/"
    echo "  (status-dynamic-vm.sh or direct ssh), then stop manually:"
    echo "  $VM_SCRIPTS/stop-vm.sh"
    exit "$rc"
  fi
  if [[ "$vm_started" -eq 1 ]]; then
    echo "Stopping VM (cleanup on pre-launch failure)..."
    "$VM_SCRIPTS/stop-vm.sh" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

# ── args ─────────────────────────────────────────────────────────────

BRANCH="${1:-main}"

WRAP_RUN_ID="qa-stage2-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$APP_DIR/logs/runs/$WRAP_RUN_ID"
MARKER="$LOG_DIR/marker.json"
STATUS_FILE="$LOG_DIR/exit_status"
REMOTE_LOG="$LOG_DIR/nohup.log"
REMOTE_PID_FILE="$LOG_DIR/pid"

echo "Stage-2 feature QA | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── SSH helper ───────────────────────────────────────────────────────

ssh_cmd() {
  "$VM_SCRIPTS/ssh-vm.sh" -- "$@"
}

# sshd can briefly refuse connections right after boot even after an initial
# successful probe (observed 2026-08-22). Pre-launch calls therefore retry.
ssh_cmd_retry() {
  local attempt
  for attempt in 1 2 3; do
    if ssh_cmd "$@"; then
      return 0
    fi
    echo "  [$(date +%H:%M:%S)] ssh attempt $attempt/3 failed. Retrying in 15s..."
    sleep 15
  done
  return 1
}

# ── start VM + deploy ────────────────────────────────────────────────

echo "Starting VM..."
"$VM_SCRIPTS/start-vm.sh"
vm_started=1
trap cleanup EXIT

echo "Pushing branch $BRANCH to origin..."
git push origin "$BRANCH" --quiet

echo "Deploying code on VM..."
# A never-deployed feature branch needs an explicit fetch refspec (the
# default fetch may only track main); create/reset the local branch from
# that ref so the new code actually runs.
ssh_cmd_retry "
  cd $APP_DIR && \
  git fetch origin $BRANCH:refs/remotes/origin/$BRANCH && \
  git checkout -B $BRANCH refs/remotes/origin/$BRANCH && \
  git reset --hard refs/remotes/origin/$BRANCH && \
  uv sync --frozen --quiet
"

# ── marker + launch ──────────────────────────────────────────────────

echo "Creating run marker: $WRAP_RUN_ID"
ssh_cmd_retry "
  mkdir -p '$LOG_DIR' && \
  cat > '$MARKER' <<MARKER_JSON
{
  \"run_id\": \"$WRAP_RUN_ID\",
  \"config\": \"stage2_features_full\",
  \"branch\": \"$BRANCH\",
  \"started\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"pid\": 0,
  \"log\": \"$REMOTE_LOG\",
  \"status_file\": \"$STATUS_FILE\"
}
MARKER_JSON
"

echo "Launching Stage-2 feature QA on VM..."
# The detached wrapper writes the pipeline's real exit code to STATUS_FILE
# from inside the process (wait on a sibling subshell would return 127).
#
# stdin of the detached process comes from /dev/null, yet launch sessions
# through sshd can STILL hang after the remote command has finished
# (observed 2026-08-21/22). Bound the wait locally: if the session has not
# returned within LAUNCH_WAIT_SECONDS, kill ONLY the local ssh client —
# the remote job is nohup-detached and unaffected — then verify the launch
# via the pid file before marking the VM as occupied.
LAUNCH_WAIT_SECONDS=60
ssh_cmd "
  cd $APP_DIR && \
  nohup sh -c 'uv run python scripts/run_qa_stage2_features.py --config-name stage2_features_full; rc=\$?; echo \"\$rc\" > $STATUS_FILE' \
    > $REMOTE_LOG 2>&1 < /dev/null &
  PID=\$! && echo \$PID > $REMOTE_PID_FILE
" &
LAUNCH_SSH_PID=$!

for _ in $(seq 1 "$LAUNCH_WAIT_SECONDS"); do
  kill -0 "$LAUNCH_SSH_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$LAUNCH_SSH_PID" 2>/dev/null; then
  echo "  [$(date +%H:%M:%S)] Launch session did not return within ${LAUNCH_WAIT_SECONDS}s — killing local ssh client only."
  kill "$LAUNCH_SSH_PID" 2>/dev/null || true
  wait "$LAUNCH_SSH_PID" 2>/dev/null || true
fi

sleep 5
REMOTE_PID=""
REACHABLE=0
# A failed ssh here is indistinguishable from a connection blip while the
# detached job runs — only a successful readback may prove "not launched".
if VERIFY=$(ssh_cmd "cat '$REMOTE_PID_FILE' 2>/dev/null || true" 2>/dev/null); then
  REMOTE_PID="$VERIFY"
  REACHABLE=1
fi

if [[ "$REACHABLE" -eq 0 ]]; then
  # Cannot verify because contact was lost — the QA run may be running.
  # Never stop a possibly-running job.
  echo "WARNING: launch verification lost contact — leaving VM RUNNING for inspection."
  leave_running=1
  exit 2
fi

if [[ "$REMOTE_PID" =~ ^[0-9]+$ ]] && ssh_cmd "kill -0 $REMOTE_PID" 2>/dev/null; then
  pipeline_launched=1
  echo "Remote PID: $REMOTE_PID (launch verified)"
else
  # Reachable but no live process behind the pid file: nothing was
  # launched or it died instantly — safe pre-launch failure.
  echo "ERROR: could not confirm the remote QA run is running."
  exit 1
fi

if [[ "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
  ssh_cmd "
    sed -i 's/\"pid\": 0/\"pid\": $REMOTE_PID/' '$MARKER'
  " 2>/dev/null || true
fi

# ── poll for completion ──────────────────────────────────────────────

echo "Polling for Stage-2 QA completion ($WRAP_RUN_ID)..."
POLL_FAILURES=0

while true; do
  sleep 60

  TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""

  if [[ -n "$TERMINAL" ]]; then
    if [[ "$TERMINAL" == "0" ]]; then
      echo "  [$(date +%H:%M:%S)] Stage-2 QA completed successfully."
    else
      echo "  [$(date +%H:%M:%S)] Stage-2 QA exited with code $TERMINAL."
    fi
    break
  fi

  IS_RUNNING=$(ssh_cmd "
    if kill -0 $REMOTE_PID 2>/dev/null; then echo running; else echo stopped; fi
  " 2>/dev/null) || {
    POLL_FAILURES=$((POLL_FAILURES + 1))
    if [[ $POLL_FAILURES -ge $CONNECTION_RETRIES ]]; then
      echo "  [$(date +%H:%M:%S)] Lost contact after $POLL_FAILURES attempts."
      echo ""
      echo "CONNECTION LOST — remote process may still be running."
      echo "  Run ID:     $WRAP_RUN_ID"
      echo "  Remote PID: $REMOTE_PID"
      echo "  Marker:     $MARKER"
      echo "The VM will NOT be stopped automatically."
      leave_running=1
      exit 2
    fi
    echo "  [$(date +%H:%M:%S)] Connection lost (attempt $POLL_FAILURES/$CONNECTION_RETRIES). Retrying in ${CONNECTION_RETRY_WAIT}s..."
    sleep "$CONNECTION_RETRY_WAIT"
    continue
  }

  POLL_FAILURES=0

  if [[ "$IS_RUNNING" == "stopped" ]]; then
    sleep 5
    TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""
    if [[ -n "$TERMINAL" ]]; then
      break
    fi
    # The process is gone without writing its exit status — ambiguous
    # (crash, interrupted write, or delayed publication). Never stop a
    # possibly-active job; leave the VM for operator inspection.
    echo "No exit status written — leaving VM RUNNING for manual inspection."
    leave_running=1
    exit 2
  fi

  LAST_LOG=$(ssh_cmd "
    tail -1 '$REMOTE_LOG' 2>/dev/null || echo 'waiting...'
  " 2>/dev/null || echo "polling...")
  echo "  [$(date +%H:%M:%S)] $LAST_LOG"
done

# ── exit code + run prefix discovery ─────────────────────────────────

PIPELINE_EXIT=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null || echo "unknown")
echo ""
echo "Stage-2 QA finished."
echo "  Exit code:   $PIPELINE_EXIT"
echo "  Run ID:      $WRAP_RUN_ID"

# The pipeline prints the evidence URIs; extract the run prefix.
RUN_PREFIX=$(ssh_cmd "
  grep -o '$QA_OUTPUT_ROOT/[0-9a-f]\\{8\\}' '$REMOTE_LOG' 2>/dev/null \
    | tail -1
" 2>/dev/null || echo "")

if [[ -z "$RUN_PREFIX" ]]; then
  echo "ERROR: could not discover QA run prefix from remote log."
  PIPELINE_EXIT="discovery-failed"
else
  echo "  Evidence:    $RUN_PREFIX"
fi

# ── validate published report bundle (local, read-only) ──────────────

VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" && -n "$RUN_PREFIX" ]]; then
  echo "Validating $RUN_PREFIX ..."
  uv run python scripts/validate_qa_stage2_features.py \
    --run-prefix "$RUN_PREFIX" \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM ──────────────────────────────────────────────────────────

echo "Stopping VM..."
vm_started=0
pipeline_launched=0
"$VM_SCRIPTS/stop-vm.sh"

# ── report ───────────────────────────────────────────────────────────

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: Stage-2 feature QA completed and validated."
  echo "  Evidence: $RUN_PREFIX"
else
  echo "FAIL: Stage-2 feature QA validation failed or the gate found findings."
  echo "  Evidence:    ${RUN_PREFIX:-unknown}"
  echo "  Remote log:  $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi