#!/usr/bin/env bash
# Run a Dynamic pipeline config on the On-Demand berlin-lst-vm.
#
# Lifecycle: start VM → deploy code → launch pipeline → poll → validate → stop VM.
# The boot disk is retained after stop; pipeline products live in GCS.
#
# Every remote command uses ssh-vm.sh (strict host-key verification).
# Each run writes an immutable marker on the VM so that a later
# status-dynamic-vm.sh can report running/completed/failed/connection-lost
# without relaunching the process.
#
# Usage:
#   scripts/run-dynamic-vm.sh full [branch]
#   scripts/run-dynamic-vm.sh inference_2026 [branch]
#
# Examples:
#   scripts/run-dynamic-vm.sh full main
#   scripts/run-dynamic-vm.sh inference_2026 fix/dynamic-resumable-on-demand

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/vm-identity.sh"

APP_DIR="/workspace/app"
MANIFEST_URI="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"

CONNECTION_RETRIES=5
CONNECTION_RETRY_WAIT=30

# ── args ─────────────────────────────────────────────────────────────────────

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

RUN_ID="${CONFIG}-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$APP_DIR/logs/runs/$RUN_ID"
MARKER="$LOG_DIR/marker.json"
STATUS_FILE="$LOG_DIR/exit_status"
REMOTE_LOG="$LOG_DIR/nohup.log"
REMOTE_PID_FILE="$LOG_DIR/pid"

echo "Config: $CONFIG | Branch: $BRANCH | Run ID: $RUN_ID"

# ── SSH helper ───────────────────────────────────────────────────────────────

ssh_cmd() {
  "$SCRIPTS_DIR/ssh-vm.sh" -- "$@"
}

# ── start VM ─────────────────────────────────────────────────────────────────

echo "Starting VM..."
"$SCRIPTS_DIR/start-vm.sh"

# ── push branch so VM can fetch ─────────────────────────────────────────────

echo "Pushing branch $BRANCH to origin..."
git push origin "$BRANCH" --quiet

# ── deploy code on VM ───────────────────────────────────────────────────────

echo "Deploying code on VM..."
ssh_cmd "
  cd $APP_DIR && \
  git fetch origin && \
  git checkout $BRANCH && \
  uv sync --frozen --quiet
"

# ── prepare run directory + marker ──────────────────────────────────────────

echo "Creating run marker: $RUN_ID"
ssh_cmd "
  mkdir -p '$LOG_DIR' && \
  cat > '$MARKER' <<MARKER_JSON
{
  \"run_id\": \"$RUN_ID\",
  \"config\": \"$CONFIG\",
  \"branch\": \"$BRANCH\",
  \"started\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"pid\": 0,
  \"log\": \"$REMOTE_LOG\",
  \"status_file\": \"$STATUS_FILE\"
}
MARKER_JSON
"

# ── launch pipeline ─────────────────────────────────────────────────────────

echo "Launching pipeline on VM..."
ssh_cmd "
  cd $APP_DIR && \
  nohup uv run python scripts/run_dynamic.py \
    --config-name $CONFIG \
    manifest_uri=$MANIFEST_URI \
    > '$REMOTE_LOG' 2>&1 &
  echo \$! > '$REMOTE_PID_FILE'
"

# Read PID and update marker
REMOTE_PID=$(ssh_cmd "cat '$REMOTE_PID_FILE'" 2>/dev/null || echo "unknown")
echo "Remote PID: $REMOTE_PID"

# Update marker with actual PID
ssh_cmd "
  sed -i 's/\"pid\": 0/\"pid\": $REMOTE_PID/' '$MARKER'
" 2>/dev/null || true

# ── write terminal status on exit (detached wrapper) ─────────────────────────

echo "Registering terminal status writer..."
ssh_cmd "
  ( while kill -0 $REMOTE_PID 2>/dev/null; do sleep 5; done; \
    echo \$? > '$STATUS_FILE' ) &
"

# ── poll for completion ─────────────────────────────────────────────────────

echo "Polling for pipeline completion (run $RUN_ID)..."
POLL_FAILURES=0

while true; do
  sleep 60

  # Check terminal status file first — avoids PID-reuse ambiguity
  TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""

  if [[ -n "$TERMINAL" ]]; then
    if [[ "$TERMINAL" == "0" ]]; then
      echo "  [$(date +%H:%M:%S)] Pipeline completed successfully."
    else
      echo "  [$(date +%H:%M:%S)] Pipeline exited with code $TERMINAL."
    fi
    break
  fi

  # No terminal status yet — check if PID is still alive
  IS_RUNNING=$(ssh_cmd "
    if kill -0 $REMOTE_PID 2>/dev/null; then echo running; else echo stopped; fi
  " 2>/dev/null) || {
    # Connection failure — do not give up immediately
    POLL_FAILURES=$((POLL_FAILURES + 1))
    if [[ $POLL_FAILURES -ge $CONNECTION_RETRIES ]]; then
      echo "  [$(date +%H:%M:%S)] Lost contact after $POLL_FAILURES attempts."
      echo ""
      echo "CONNECTION LOST — remote process may still be running."
      echo "  Run ID:     $RUN_ID"
      echo "  Remote PID: $REMOTE_PID"
      echo "  Marker:     $MARKER"
      echo ""
      echo "To check later: scripts/status-dynamic-vm.sh --run-id $RUN_ID"
      echo "The VM will NOT be stopped automatically."
      exit 2
    fi
    echo "  [$(date +%H:%M:%S)] Connection lost (attempt $POLL_FAILURES/$CONNECTION_RETRIES). Retrying in ${CONNECTION_RETRY_WAIT}s..."
    sleep "$CONNECTION_RETRY_WAIT"
    continue
  }

  # Reset failure counter on successful contact
  POLL_FAILURES=0

  if [[ "$IS_RUNNING" == "stopped" ]]; then
    # PID is dead — terminal status should appear shortly
    echo "  [$(date +%H:%M:%S)] PID $REMOTE_PID stopped. Waiting for exit status..."
    sleep 5
    TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""
    if [[ -n "$TERMINAL" ]]; then
      break
    fi
    echo "  [$(date +%H:%M:%S)] No exit status written. Check logs manually."
    break
  fi

  # Print last log line for progress visibility
  LAST_LOG=$(ssh_cmd "
    tail -1 '$REMOTE_LOG' 2>/dev/null || echo 'waiting...'
  " 2>/dev/null || echo "polling...")
  echo "  [$(date +%H:%M:%S)] $LAST_LOG"
done

# ── check exit code ─────────────────────────────────────────────────────────

PIPELINE_EXIT=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null || echo "unknown")
PIPELINE_ERRORS=$(ssh_cmd "
  grep -c 'error\\|ERROR\\|Traceback\\|SystemExit' '$REMOTE_LOG' 2>/dev/null || echo 0
" 2>/dev/null || echo "unknown")

echo ""
echo "Pipeline finished."
echo "  Exit code:   $PIPELINE_EXIT"
echo "  Error lines: $PIPELINE_ERRORS"
echo "  Run ID:      $RUN_ID"

# ── validate ─────────────────────────────────────────────────────────────────

if [[ "$CONFIG" == "full" ]]; then
  EXPECTED_ROLE="anchor"
  EXPECTED_SCENES="324"
else
  EXPECTED_ROLE="inference"
  EXPECTED_SCENES="21"
fi

case "$CONFIG" in
  full) OUTPUT_ROOT="gs://berlin-lst-data/dynamic/full" ;;
  inference_2026) OUTPUT_ROOT="gs://berlin-lst-data/dynamic/inference/2026" ;;
esac

echo "Validating $OUTPUT_ROOT ..."
uv run python scripts/validate_dynamic.py \
  --output-root "$OUTPUT_ROOT" \
  --expected-role "$EXPECTED_ROLE" \
  --expected-scenes "$EXPECTED_SCENES" \
  --check-bands \
  && VALIDATION_OK=0 || VALIDATION_OK=1

# ── stop VM ─────────────────────────────────────────────────────────────────

echo "Stopping VM..."
"$SCRIPTS_DIR/stop-vm.sh"

# ── report ───────────────────────────────────────────────────────────────────

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: $CONFIG completed and validated."
  echo "  Run ID: $RUN_ID"
else
  echo "FAIL: $CONFIG validation failed or pipeline had errors."
  echo "  Run ID:     $RUN_ID"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Check VM-side files with: scripts/status-dynamic-vm.sh --run-id $RUN_ID"
  exit 1
fi