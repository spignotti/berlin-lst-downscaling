#!/usr/bin/env bash
# Read-only probe for a remote Dynamic pipeline run.
#
# Reports whether the remote process is running, completed, failed,
# or if the connection was lost.  Uses the guarded ssh-vm.sh and
# the run marker written by run-dynamic-vm.sh.
#
# NEVER starts, stops, or relaunches a remote process.
#
# Usage: scripts/status-dynamic-vm.sh --run-id <run-id>

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/vm-identity.sh"

APP_DIR="/workspace/app"
RUN_ID=""

# ── parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)  RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    *)         echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 --run-id <run-id>" >&2
  exit 1
fi

RUN_DIR="$APP_DIR/logs/runs/$RUN_ID"
MARKER="$RUN_DIR/marker.json"

# ── probe ────────────────────────────────────────────────────────────────────

ssh_cmd() {
  "$SCRIPTS_DIR/ssh-vm.sh" -- "$@"
}

echo "=== Run status: $RUN_ID ==="

# Check marker exists
MARKER_CONTENTS=$(ssh_cmd "cat '$MARKER'" 2>/dev/null) || {
  echo "Status: CONNECTION_LOST"
  echo "  Cannot reach VM or marker not found."
  echo "  The remote process may still be running."
  echo "  Do NOT launch a new run until this is resolved."
  exit 2
}

echo "Marker found: $MARKER"
echo ""

# Parse marker
CONFIG=$(echo "$MARKER_CONTENTS" | rg '"config"\s*:\s*"([^"]+)"' -o -r '$1' 2>/dev/null || echo "(unknown)")
REMOTE_PID=$(echo "$MARKER_CONTENTS" | rg '"pid"\s*:\s*(\d+)' -o -r '$1' 2>/dev/null || echo "0")
START_TIME=$(echo "$MARKER_CONTENTS" | rg '"started"\s*:\s*"([^"]+)"' -o -r '$1' 2>/dev/null || echo "(unknown)")

echo "  Config:    $CONFIG"
echo "  PID:       $REMOTE_PID"
echo "  Started:   $START_TIME"

# Check terminal status
STATUS_FILE="$RUN_DIR/exit_status"
TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""

if [[ -n "$TERMINAL" ]]; then
  echo ""
  if [[ "$TERMINAL" == "0" ]]; then
    echo "Status: COMPLETED"
    echo "  Exit code: 0"
  else
    echo "Status: FAILED"
    echo "  Exit code: $TERMINAL"
  fi
  LOG_FILE=$(echo "$MARKER_CONTENTS" | rg '"log"\s*:\s*"([^"]+)"' -o -r '$1' 2>/dev/null || echo "")
  [[ -n "$LOG_FILE" ]] && echo "  Log: $LOG_FILE"
  exit 0
fi

# No terminal status — process is either still running or connection was lost
# Verify the PID is still alive
PID_ALIVE=$(ssh_cmd "
  if kill -0 $REMOTE_PID 2>/dev/null; then
    echo alive
  else
    echo dead
  fi
" 2>/dev/null) || {
  echo ""
  echo "Status: CONNECTION_LOST"
  echo "  Cannot reach VM to verify PID $REMOTE_PID."
  echo "  The remote process may still be running."
  echo "  Do NOT launch a new run until this is resolved."
  exit 2
}

if [[ "$PID_ALIVE" == "alive" ]]; then
  echo ""
  echo "Status: RUNNING"
  LOG_FILE=$(echo "$MARKER_CONTENTS" | rg '"log"\s*:\s*"([^"]+)"' -o -r '$1' 2>/dev/null || echo "")
  if [[ -n "$LOG_FILE" ]]; then
    LAST_LINE=$(ssh_cmd "tail -1 '$LOG_FILE'" 2>/dev/null || echo "(unavailable)")
    echo "  Last log: $LAST_LINE"
  fi
  exit 0
else
  # PID is dead but no terminal status — ambiguous
  echo ""
  echo "Status: AMBIGUOUS"
  echo "  PID $REMOTE_PID is no longer running but no exit status was written."
  echo "  The process may have been killed, crashed, or the VM rebooted."
  echo "  Check the VM-side log manually before launching a new run."
  exit 3
fi