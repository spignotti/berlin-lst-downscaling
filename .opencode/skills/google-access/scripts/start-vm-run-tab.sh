#!/usr/bin/env bash
# Start an approved VM runner in a dedicated, monitored Herdr tab.
#
# Creates a temporary tab, runs the launcher, waits for the run ID,
# renames the tab, and sets up read-only monitoring panes.
#
# Usage:
#   start-vm-run-tab.sh <launcher> [args...]
#
# Examples:
#   start-vm-run-tab.sh run-dynamic-vm.sh full main
#   start-vm-run-tab.sh run-features-vm.sh main
#
# Requires HERDR_ENV=1 (must run inside Herdr).
# NEVER starts, stops, or mutates any cloud resource.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID_WAIT_SECONDS=120

# ── preflight ────────────────────────────────────────────────────────

if [[ "${HERDR_ENV:-}" != "1" ]]; then
  echo "ERROR: HERDR_ENV=1 required — must run inside Herdr." >&2
  exit 1
fi

LAUNCHER="${1:-}"
if [[ -z "$LAUNCHER" ]]; then
  echo "Usage: $0 <launcher> [args...]" >&2
  echo "Approved launchers: run-dynamic-vm.sh run-features-vm.sh" >&2
  echo "  run-training-data-vm.sh run-qa-stage1-vm.sh run-qa-stage2-vm.sh" >&2
  exit 1
fi
shift

# Resolve launcher path (allow bare names by resolving against SCRIPTS_DIR)
if [[ "$LAUNCHER" != /* ]]; then
  LAUNCHER="$SCRIPTS_DIR/$LAUNCHER"
fi

# Allowlist: only the existing fail-closed launchers
BASENAME="$(basename "$LAUNCHER")"
case "$BASENAME" in
  run-dynamic-vm.sh|run-features-vm.sh|run-training-data-vm.sh|run-qa-stage1-vm.sh|run-qa-stage2-vm.sh)
    ;;
  *)
    echo "ERROR: not an approved launcher: $BASENAME" >&2
    echo "Approved: run-dynamic-vm.sh, run-features-vm.sh, run-training-data-vm.sh," >&2
    echo "  run-qa-stage1-vm.sh, run-qa-stage2-vm.sh" >&2
    exit 1
    ;;
esac

if [[ ! -f "$LAUNCHER" ]]; then
  echo "ERROR: launcher not found: $LAUNCHER" >&2
  exit 1
fi

WORKSPACE_ID="${HERDR_WORKSPACE_ID:?}"

# ── create temporary run tab ─────────────────────────────────────────

TAB_JSON=$(herdr tab create --label "run-$(date -u +%H%M%S)" --no-focus --workspace "$WORKSPACE_ID")
ROOT_PANE=$(printf '%s\n' "$TAB_JSON" | jq -r '.result.root_pane.pane_id')

if [[ -z "$ROOT_PANE" || "$ROOT_PANE" == "null" ]]; then
  echo "ERROR: could not determine root pane ID from tab create" >&2
  exit 1
fi

echo "Run tab created: root pane $ROOT_PANE"

# ── launch the runner ────────────────────────────────────────────────

echo "Starting $BASENAME $* ..."
herdr pane run "$ROOT_PANE" "$LAUNCHER" "$@"

# ── wait for run ID or completion ────────────────────────────────────

echo "Waiting for run ID (up to ${RUN_ID_WAIT_SECONDS}s)..."
herdr pane wait-output "$ROOT_PANE" \
  --regex "(Run ID:|Run:)" \
  --timeout "$((RUN_ID_WAIT_SECONDS * 1000))" \
  --source recent-unwrapped --lines 120 || true

# ── parse run ID from output ─────────────────────────────────────────

OUTPUT=$(herdr pane read "$ROOT_PANE" --source recent-unwrapped --lines 120 2>/dev/null || echo "")

RUN_ID=""
if RUN_ID=$(printf '%s\n' "$OUTPUT" | rg 'Run ID:\s*(\S+)' -o -r '$1' 2>/dev/null | tail -1); then
  : # got it
elif RUN_ID=$(printf '%s\n' "$OUTPUT" | rg 'Run:\s*(\S+)' -o -r '$1' 2>/dev/null | tail -1); then
  : # got it
fi

if [[ -n "$RUN_ID" ]]; then
  herdr tab rename "$(printf '%s\n' "$TAB_JSON" | jq -r '.result.tab.tab_id')" "run-$RUN_ID" --workspace "$WORKSPACE_ID"
  echo "Tab renamed: run-$RUN_ID"
else
  echo "WARNING: no run ID found within ${RUN_ID_WAIT_SECONDS}s."
  echo "  Monitor pane still active. Rename manually once the run ID appears."
  RUN_ID="unknown"
fi

# ── create read-only monitoring panes ────────────────────────────────

echo "Setting up monitoring panes..."

# VM status pane (right of launcher)
SPLIT_JSON=$(herdr pane split --pane "$ROOT_PANE" --direction right --no-focus --workspace "$WORKSPACE_ID")
VM_PANE=$(printf '%s\n' "$SPLIT_JSON" | jq -r '.result.pane.pane_id')
herdr pane run "$VM_PANE" "bash $SCRIPTS_DIR/cloud-monitor.sh vm --once"

# Bucket contents pane (below launcher)
SPLIT_JSON=$(herdr pane split --pane "$ROOT_PANE" --direction down --no-focus --workspace "$WORKSPACE_ID")
BUCKET_PANE=$(printf '%s\n' "$SPLIT_JSON" | jq -r '.result.pane.pane_id')
herdr pane run "$BUCKET_PANE" "bash $SCRIPTS_DIR/cloud-monitor.sh bucket --once"

# Run status pane (right of bucket)
SPLIT_JSON=$(herdr pane split --pane "$BUCKET_PANE" --direction right --no-focus --workspace "$WORKSPACE_ID")
RUN_STATUS_PANE=$(printf '%s\n' "$SPLIT_JSON" | jq -r '.result.pane.pane_id')
if [[ "$RUN_ID" != "unknown" ]]; then
  herdr pane run "$RUN_STATUS_PANE" "bash $SCRIPTS_DIR/cloud-monitor.sh run --run-id $RUN_ID --once"
else
  herdr pane run "$RUN_STATUS_PANE" "echo 'Run status: waiting for run ID...'"
fi

echo "Done. Monitoring panes created."
