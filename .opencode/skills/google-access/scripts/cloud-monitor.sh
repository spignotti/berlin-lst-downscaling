#!/usr/bin/env bash
# Read-only Herdr pane monitor for cloud-ops dashboard.
#
# Modes: vm | bucket | mount | run --run-id <id>
#
# Default periodic refresh (30 s, configurable via REFRESH_INTERVAL).
# Use ONCE=1 or ONCE=true for single-shot output.
#
# NEVER starts, stops, or mutates any cloud resource.  NEVER runs
# lifecycle scripts.  NEVER writes to GCS.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=""
RUN_ID=""
REFRESH_INTERVAL="${REFRESH_INTERVAL:-30}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"

# ── parse arguments ──────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    vm|bucket|mount) MODE="$1"; shift ;;
    run)             MODE="$1"; shift ;;
    --run-id)        RUN_ID="$2"; shift 2 ;;
    *)               shift ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <vm|bucket|mount|run --run-id <id>>" >&2
  exit 1
fi

if [[ "$MODE" == "run" && -z "$RUN_ID" ]]; then
  echo "Usage: $0 run --run-id <id>" >&2
  exit 1
fi

# ── rendering (failed probes never abort the refresh loop) ───────────

_clear_screen() { printf '\033[2J\033[H'; }

_separator() { printf '\n%s\n\n' "$(printf '=%.0s' {1..60})"; }

# ── vm monitor ───────────────────────────────────────────────────────

monitor_vm() {
  _separator
  echo "VM STATUS"
  _separator
  if [[ -f "$SCRIPTS_DIR/status-vm.sh" ]]; then
    bash "$SCRIPTS_DIR/status-vm.sh" 2>&1 || true
  else
    echo "status-vm.sh not found at $SCRIPTS_DIR/status-vm.sh"
  fi
}

# ── bucket monitor ───────────────────────────────────────────────────

monitor_bucket() {
  _separator
  echo "BUCKET CONTENTS"
  _separator
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage ls "gs://berlin-lst-data/" --project=masterarbeit-berlin-lst-v2 2>&1 || true
  else
    echo "gcloud not available — cannot list bucket"
  fi
}

# ── mount monitor ────────────────────────────────────────────────────

monitor_mount() {
  _separator
  echo "RCLONE MOUNT STATUS"
  _separator
  if pgrep -f "rclone mount" >/dev/null 2>&1; then
    echo "Mount: RUNNING"
  else
    echo "Mount: NOT RUNNING"
  fi
  if [[ -d "$HOME/.mnt/berlin-lst" ]]; then
    echo ""
    echo "Contents of ~/.mnt/berlin-lst:"
    ls "$HOME/.mnt/berlin-lst/" 2>&1 || echo "(unable to list)"
  fi
}

# ── run monitor ──────────────────────────────────────────────────────

monitor_run() {
  _separator
  echo "RUN STATUS: $RUN_ID"
  _separator
  if [[ -f "$SCRIPTS_DIR/status-dynamic-vm.sh" ]]; then
    bash "$SCRIPTS_DIR/status-dynamic-vm.sh" --run-id "$RUN_ID" 2>&1 || true
  else
    echo "status-dynamic-vm.sh not found"
  fi
  _separator
  echo "LAST 10 LOG LINES"
  _separator
  REMOTE_LOG="/workspace/app/logs/runs/$RUN_ID/nohup.log"
  if [[ -f "$SCRIPTS_DIR/ssh-vm.sh" ]]; then
    bash "$SCRIPTS_DIR/ssh-vm.sh" -- tail -10 "$REMOTE_LOG" 2>&1 || true
  else
    echo "ssh-vm.sh not available"
  fi
}

# ── refresh loop ─────────────────────────────────────────────────────

render() {
  case "$MODE" in
    vm)     monitor_vm ;;
    bucket) monitor_bucket ;;
    mount)  monitor_mount ;;
    run)    monitor_run ;;
  esac
}

trap 'clear; exit 0' INT TERM

while true; do
  _clear_screen
  render
  [[ "${ONCE:-}" == "1" || "${ONCE:-}" == "true" ]] && exit 0
  sleep "$REFRESH_INTERVAL"
done
