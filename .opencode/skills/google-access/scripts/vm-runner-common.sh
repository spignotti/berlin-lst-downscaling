#!/usr/bin/env bash
# Shared fail-closed VM pipeline runner for the berlin-lst-vm.
#
# Sourced by the pipeline-specific run-<pipeline>-vm.sh launchers in this
# directory. Provides the lifecycle that is identical for every pipeline:
#
#   preflight → start → deploy pinned commit → detached launch → poll → stop
#
# with the fail-safe rules:
#   - any pre-launch failure stops the VM (AGENTS.md: VM stays stopped
#     when not actively running);
#   - once launched, no ambiguous exit or connection loss ever stops the
#     VM automatically — a possibly-still-running job must never be
#     killed mid-write; those paths exit 2 and leave the VM RUNNING for
#     operator inspection.
#
# The caller must run `set -euo pipefail` BEFORE sourcing this file.
#
# Launcher contract (set before calling the vm_* functions):
#   PIPELINE_LABEL   human label, e.g. "Feature stacks"
#   MARKER_CONFIG    value of the marker "config" field
#   REMOTE_CMD       remote command to run detached (repo-relative paths
#                    resolve from APP_DIR on the VM)
#   BRANCH           branch to deploy (validated; the VM runs exactly the
#                    clean, pushed local HEAD)
#   Optional:
#     EXTRA_MARKER_JSON  extra marker fields, one JSON line ending with a
#                        comma, inserted before the "started" field
#     RUN_PREFIX_GREP    grep pattern (BRE) to discover the evidence/run
#                        prefix from the remote log; sets RUN_PREFIX
#     vm_discover_run_id()  optional function; called by vm_finish to
#                           discover a pipeline-specific run id

# ── pinned VM access ─────────────────────────────────────────────────

VM_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$VM_SCRIPTS/vm-identity.sh"

APP_DIR="/workspace/app"
CONNECTION_RETRIES=5
CONNECTION_RETRY_WAIT=30
LAUNCH_WAIT_SECONDS=60

# ── fail-safe VM cleanup ─────────────────────────────────────────────
# Any pre-launch failure must stop the VM. Once the pipeline is launched,
# every ambiguous exit leaves the VM RUNNING — the connection-loss and
# no-exit-status paths set leave_running before exiting.
vm_started=0
leave_running=0
pipeline_launched=0

vm_cleanup() {
  local rc=$?
  # leave_running is authoritative: any path that sets it (connection
  # loss before or after launch) must never trigger an automatic stop.
  if [[ "$leave_running" -eq 1 ]]; then
    exit "$rc"
  fi
  if [[ "$pipeline_launched" -eq 1 ]]; then
    echo "WARNING: abnormal exit after pipeline launch — VM left RUNNING."
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

# ── SSH helpers ──────────────────────────────────────────────────────

ssh_cmd() {
  "$VM_SCRIPTS/ssh-vm.sh" -- "$@"
}

# sshd can briefly refuse connections right after boot even after an
# initial successful probe (observed 2026-08-22). Retry pre-launch calls.
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

# ── run identity ─────────────────────────────────────────────────────

vm_init_run() {
  local prefix="$1"
  # Two-part branch validation: the strict allowlist excludes every shell
  # metacharacter and quote (so the branch is safe to interpolate into
  # remote shell commands and marker JSON), and git check-ref-format
  # rejects malformed refs (e.g. 'foo//bar', 'foo.lock') that would fail
  # later during push/fetch/checkout.
  if [[ ! "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || ! git check-ref-format --branch "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: invalid branch name: $BRANCH"
    exit 1
  fi
  WRAP_RUN_ID="${prefix}-$(date -u +%Y%m%dT%H%M%SZ)"
  LOG_DIR="$APP_DIR/logs/runs/$WRAP_RUN_ID"
  MARKER="$LOG_DIR/marker.json"
  STATUS_FILE="$LOG_DIR/exit_status"
  REMOTE_LOG="$LOG_DIR/nohup.log"
  REMOTE_PID_FILE="$LOG_DIR/pid"
}

# ── preflight (hard go/no-go gate for guarded releases) ─────────────

vm_preflight_pushed() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree not clean — refusing to publish."
    exit 1
  fi
  LOCAL_SHA=$(git rev-parse HEAD)
  ORIGIN_SHA=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "missing")
  if [[ "$LOCAL_SHA" != "$ORIGIN_SHA" ]]; then
    echo "ERROR: HEAD $LOCAL_SHA != origin/$BRANCH $ORIGIN_SHA (not pushed)."
    exit 1
  fi
  echo "  Branch: $BRANCH | SHA: $LOCAL_SHA (clean, pushed)"
}

# ── start VM + wait for SSH ─────────────────────────────────────────

vm_start_and_wait_ssh() {
  echo "Starting VM..."
  "$VM_SCRIPTS/start-vm.sh"
  vm_started=1
  trap vm_cleanup EXIT

  # sshd may still be booting after the instance reaches RUNNING — wait
  # for the first connection instead of failing the whole run on a race.
  echo "Waiting for SSH readiness..."
  local attempt
  for attempt in $(seq 1 10); do
    if ssh_cmd "echo ready" >/dev/null 2>&1; then
      echo "  SSH ready."
      return 0
    fi
    echo "  [$(date +%H:%M:%S)] SSH not ready (attempt $attempt/10). Retrying in 15s..."
    sleep 15
  done
  echo "ERROR: SSH never became ready. Stopping VM."
  vm_started=0
  "$VM_SCRIPTS/stop-vm.sh"
  exit 1
}

# ── deploy pinned commit ─────────────────────────────────────────────

vm_push_deploy() {
  # Exact-commit pinning: refuse to deploy unless the local HEAD is the
  # clean, pushed state, and verify the VM runs exactly that SHA.
  vm_preflight_pushed

  echo "Pushing branch $BRANCH to origin..."
  git push origin "$BRANCH" --quiet

  echo "Deploying code on VM..."
  # A never-deployed feature branch needs an explicit fetch refspec (the
  # default fetch may only track main); create/reset the local branch
  # from that ref so the new code actually runs.
  ssh_cmd_retry "
    cd $APP_DIR && \
    git fetch origin $BRANCH:refs/remotes/origin/$BRANCH && \
    git checkout -B $BRANCH refs/remotes/origin/$BRANCH && \
    git reset --hard refs/remotes/origin/$BRANCH && \
    uv sync --frozen --quiet
  "
  DEPLOYED_SHA=$(ssh_cmd_retry "cd $APP_DIR && git rev-parse HEAD")
  if [[ "$DEPLOYED_SHA" != "$LOCAL_SHA" ]]; then
    echo "ERROR: deployed SHA $DEPLOYED_SHA != local SHA $LOCAL_SHA — refusing to launch."
    exit 1
  fi
  echo "  Deployed SHA verified: $DEPLOYED_SHA"
}

# ── marker + detached launch ─────────────────────────────────────────

vm_write_marker() {
  local config_name="$1"
  echo "Creating run marker: $WRAP_RUN_ID"
  ssh_cmd_retry "
    mkdir -p '$LOG_DIR' && \
    cat > '$MARKER' <<MARKER_JSON
{
  \"run_id\": \"$WRAP_RUN_ID\",
  \"config\": \"$config_name\",
  \"branch\": \"$BRANCH\",
  \"sha\": \"$DEPLOYED_SHA\",
  ${EXTRA_MARKER_JSON:-}\"started\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"pid\": 0,
  \"log\": \"$REMOTE_LOG\",
  \"status_file\": \"$STATUS_FILE\"
}
MARKER_JSON
  "
}

vm_launch_detached() {
  echo "Launching $PIPELINE_LABEL on VM..."
  # The detached wrapper writes the pipeline's real exit code to
  # STATUS_FILE from inside the process (wait on a sibling subshell
  # would return 127).
  #
  # stdin of the detached process comes from /dev/null, yet launch
  # sessions through sshd can STILL hang after the remote command has
  # finished (observed 2026-08-21/22). Bound the wait locally: if the
  # session has not returned within LAUNCH_WAIT_SECONDS, kill ONLY the
  # local ssh client — the remote job is nohup-detached and unaffected —
  # then verify the launch via the pid file before marking the VM as
  # occupied.
  ssh_cmd "
    cd $APP_DIR && \
    nohup sh -c '$REMOTE_CMD; rc=\$?; echo \"\$rc\" > $STATUS_FILE' \
      > $REMOTE_LOG 2>&1 < /dev/null &
    PID=\$! && echo \$PID > $REMOTE_PID_FILE
  " &
  local launch_ssh_pid=$!

  local _i
  for _i in $(seq 1 "$LAUNCH_WAIT_SECONDS"); do
    kill -0 "$launch_ssh_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$launch_ssh_pid" 2>/dev/null; then
    echo "  [$(date +%H:%M:%S)] Launch session did not return within ${LAUNCH_WAIT_SECONDS}s — killing local ssh client only."
    kill "$launch_ssh_pid" 2>/dev/null || true
    wait "$launch_ssh_pid" 2>/dev/null || true
  fi

  sleep 5
  local verify="" remote_pid="" reachable=0
  # A failed ssh here is indistinguishable from a connection blip while
  # the detached job runs — only a successful readback may prove "not
  # launched".
  if verify=$(ssh_cmd "cat '$REMOTE_PID_FILE' 2>/dev/null || true" 2>/dev/null); then
    remote_pid="$verify"
    reachable=1
  fi

  if [[ "$reachable" -eq 0 ]]; then
    echo "WARNING: launch verification lost contact — leaving VM RUNNING for inspection."
    leave_running=1
    exit 2
  fi

  if [[ "$remote_pid" =~ ^[0-9]+$ ]] && ssh_cmd "kill -0 $remote_pid" 2>/dev/null; then
    pipeline_launched=1
    REMOTE_PID="$remote_pid"
    echo "Remote PID: $REMOTE_PID (launch verified)"
  else
    # Reachable but no live process behind the pid file: nothing was
    # launched or it died instantly — safe pre-launch failure.
    echo "ERROR: could not confirm the remote pipeline is running."
    exit 1
  fi

  ssh_cmd "
    sed -i 's/\"pid\": 0/\"pid\": $REMOTE_PID/' '$MARKER'
  " 2>/dev/null || true
}

# ── poll for completion ──────────────────────────────────────────────

vm_poll() {
  echo "Polling for $PIPELINE_LABEL completion ($WRAP_RUN_ID)..."
  local poll_failures=0 terminal="" is_running=""
  while true; do
    sleep 60

    terminal=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || terminal=""

    if [[ -n "$terminal" ]]; then
      if [[ "$terminal" == "0" ]]; then
        echo "  [$(date +%H:%M:%S)] $PIPELINE_LABEL completed successfully."
      else
        echo "  [$(date +%H:%M:%S)] $PIPELINE_LABEL exited with code $terminal."
      fi
      break
    fi

    is_running=$(ssh_cmd "
      if kill -0 $REMOTE_PID 2>/dev/null; then echo running; else echo stopped; fi
    " 2>/dev/null) || {
      poll_failures=$((poll_failures + 1))
      if [[ $poll_failures -ge $CONNECTION_RETRIES ]]; then
        echo "  [$(date +%H:%M:%S)] Lost contact after $poll_failures attempts."
        echo ""
        echo "CONNECTION LOST — remote process may still be running."
        echo "  Run ID:     $WRAP_RUN_ID"
        echo "  Remote PID: $REMOTE_PID"
        echo "  Marker:     $MARKER"
        echo "The VM will NOT be stopped automatically."
        leave_running=1
        exit 2
      fi
      echo "  [$(date +%H:%M:%S)] Connection lost (attempt $poll_failures/$CONNECTION_RETRIES). Retrying in ${CONNECTION_RETRY_WAIT}s..."
      sleep "$CONNECTION_RETRY_WAIT"
      continue
    }

    poll_failures=0

    if [[ "$is_running" == "stopped" ]]; then
      sleep 5
      terminal=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || terminal=""
      if [[ -n "$terminal" ]]; then
        break
      fi
      # The process is gone without writing its exit status — ambiguous
      # (crash, interrupted write, or delayed publication). Never stop a
      # possibly-active job; leave the VM for operator inspection.
      echo "No exit status written — leaving VM RUNNING for manual inspection."
      leave_running=1
      exit 2
    fi

    echo "  [$(date +%H:%M:%S)] $(ssh_cmd "
      tail -1 '$REMOTE_LOG' 2>/dev/null || echo 'waiting...'
    " 2>/dev/null || echo "polling...")"
  done
}

# ── exit code + run id discovery ─────────────────────────────────────

vm_finish() {
  PIPELINE_EXIT=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null || echo "unknown")
  echo ""
  echo "$PIPELINE_LABEL finished."
  echo "  Exit code:   $PIPELINE_EXIT"
  echo "  Run ID:      $WRAP_RUN_ID"

  if [[ -n "${RUN_PREFIX_GREP:-}" ]]; then
    RUN_PREFIX=$(ssh_cmd "
      grep -o '$RUN_PREFIX_GREP' '$REMOTE_LOG' 2>/dev/null \
        | tail -1
    " 2>/dev/null || echo "")
    if [[ -z "$RUN_PREFIX" ]]; then
      echo "ERROR: could not discover run prefix from remote log."
      PIPELINE_EXIT="discovery-failed"
    else
      echo "  Evidence:    $RUN_PREFIX"
    fi
  fi

  if declare -F vm_discover_run_id >/dev/null 2>&1; then
    vm_discover_run_id
  fi
}

# ── stop VM (normal completion) ──────────────────────────────────────

vm_stop() {
  echo "Stopping VM..."
  vm_started=0
  pipeline_launched=0
  "$VM_SCRIPTS/stop-vm.sh"
}