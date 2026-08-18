#!/usr/bin/env bash
# Run the scene feature-stack pipeline (full, 324 anchors) on the
# On-Demand berlin-lst-vm.
#
# Lifecycle: start VM → deploy committed branch → launch full run →
# poll the run marker → validate the published feature stacks locally →
# stop VM. Products (24-band COGs, masks, sidecars, ledger) live in GCS
# under gs://berlin-lst-data/features/v1/, not on the VM disk.
#
# Every remote command uses ssh-vm.sh (strict host-key verification) and
# the fail-closed lifecycle scripts from .opencode/skills/google-access/.
# The VM is always stopped in this script — including on pipeline failure.
#
# Usage:
#   scripts/run_features_vm.sh [branch]
#   scripts/run_features_vm.sh main

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
VM_SCRIPTS="$(cd "$(dirname "$0")/../.opencode/skills/google-access/scripts" && pwd)"
source "$VM_SCRIPTS/vm-identity.sh"

APP_DIR="/workspace/app"
FEATURES_ROOT="gs://berlin-lst-data/features/v1"

CONNECTION_RETRIES=5
CONNECTION_RETRY_WAIT=30

# ── args ─────────────────────────────────────────────────────────────

BRANCH="${1:-main}"

WRAP_RUN_ID="features-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$APP_DIR/logs/runs/$WRAP_RUN_ID"
MARKER="$LOG_DIR/marker.json"
STATUS_FILE="$LOG_DIR/exit_status"
REMOTE_LOG="$LOG_DIR/nohup.log"
REMOTE_PID_FILE="$LOG_DIR/pid"

echo "Feature stacks | Branch: $BRANCH | Run: $WRAP_RUN_ID"

# ── SSH helper ───────────────────────────────────────────────────────

ssh_cmd() {
  "$VM_SCRIPTS/ssh-vm.sh" -- "$@"
}

# ── preflight: canonical root must be empty or reconcileable ──────────
# A pre-existing ledger is legitimate only if the config hash matches
# (reconcile skips done rows); a foreign schema would be caught by the
# ledger schema check on open. Warn-only — the run itself fails closed.
REMOTE_HAS_ROOT=$(uv run python -c \
  "from berlin_lst_downscaling.data.io import exists; print(exists('$FEATURES_ROOT/_state/features/ledger.parquet'))" \
  2>/dev/null || echo "unknown")
echo "Canonical root $FEATURES_ROOT — existing ledger: $REMOTE_HAS_ROOT"

# ── start VM + deploy ────────────────────────────────────────────────

echo "Starting VM..."
"$VM_SCRIPTS/start-vm.sh"

echo "Pushing branch $BRANCH to origin..."
git push origin "$BRANCH" --quiet

echo "Deploying code on VM..."
# ``git checkout`` alone does not move a stale local branch; fast-forward
# the VM workspace to the pushed origin ref so the new code actually runs.
ssh_cmd "
  cd $APP_DIR && \
  git fetch origin && \
  git checkout $BRANCH && \
  git reset --hard origin/$BRANCH && \
  uv sync --frozen --quiet
"

# ── marker + launch ──────────────────────────────────────────────────

echo "Creating run marker: $WRAP_RUN_ID"
ssh_cmd "
  mkdir -p '$LOG_DIR' && \
  cat > '$MARKER' <<MARKER_JSON
{
  \"run_id\": \"$WRAP_RUN_ID\",
  \"config\": \"full\",
  \"branch\": \"$BRANCH\",
  \"started\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"pid\": 0,
  \"log\": \"$REMOTE_LOG\",
  \"status_file\": \"$STATUS_FILE\"
}
MARKER_JSON
"

echo "Launching feature-stack run on VM..."
# The detached wrapper writes the pipeline's real exit code to STATUS_FILE
# from inside the process (wait on a sibling subshell would return 127).
# The isolated runner spawns one subprocess per scene so memory is
# released between scenes (a full-grid scene peaks at ~7-8 GB; the
# in-process run OOM-killed the 16 GB VM at scene 4).
ssh_cmd "
  cd $APP_DIR && \
  nohup sh -c 'uv run python scripts/run_features_isolated.py --config-name full; rc=\$?; echo \"\$rc\" > $STATUS_FILE' \
    > $REMOTE_LOG 2>&1 &
  PID=\$! && echo \$PID > $REMOTE_PID_FILE
"

REMOTE_PID=$(ssh_cmd "cat '$REMOTE_PID_FILE'" 2>/dev/null || echo "unknown")
echo "Remote PID: $REMOTE_PID"

if [[ "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
  ssh_cmd "
    sed -i 's/\"pid\": 0/\"pid\": $REMOTE_PID/' '$MARKER'
  " 2>/dev/null || true
fi

# ── poll for completion ──────────────────────────────────────────────

echo "Polling for feature-stack completion ($WRAP_RUN_ID)..."
POLL_FAILURES=0

while true; do
  sleep 60

  TERMINAL=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null) || TERMINAL=""

  if [[ -n "$TERMINAL" ]]; then
    if [[ "$TERMINAL" == "0" ]]; then
      echo "  [$(date +%H:%M:%S)] Feature-stack run completed successfully."
    else
      echo "  [$(date +%H:%M:%S)] Feature-stack run exited with code $TERMINAL."
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
    echo "  [$(date +%H:%M:%S)] No exit status written. Check logs manually."
    break
  fi

  LAST_LOG=$(ssh_cmd "
    tail -1 '$REMOTE_LOG' 2>/dev/null || echo 'waiting...'
  " 2>/dev/null || echo "polling...")
  echo "  [$(date +%H:%M:%S)] $LAST_LOG"
done

# ── exit code + run id discovery ─────────────────────────────────────

PIPELINE_EXIT=$(ssh_cmd "cat '$STATUS_FILE'" 2>/dev/null || echo "unknown")
echo ""
echo "Feature-stack run finished."
echo "  Exit code:   $PIPELINE_EXIT"
echo "  Run ID:      $WRAP_RUN_ID"

RUN_ID=$(ssh_cmd "
  grep -o '$FEATURES_ROOT/logs/features/isolated_summary_[0-9T]*' '$REMOTE_LOG' 2>/dev/null \
    | tail -1 | sed 's#.*isolated_summary_##; s#\.json##'
" 2>/dev/null || echo "")

if [[ -z "$RUN_ID" ]]; then
  echo "WARNING: could not discover run id from remote log (validation still covers the full root)."
fi

# ── validate published feature stacks (local, read-only) ──────────────

VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating $FEATURES_ROOT ..."
  uv run python scripts/validate_feature_stacks.py \
    --root "$FEATURES_ROOT" \
    --aoi data/boundaries/aoi_10m.tif \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

# ── stop VM (always) ─────────────────────────────────────────────────

echo "Stopping VM..."
"$VM_SCRIPTS/stop-vm.sh"

# ── report ───────────────────────────────────────────────────────────

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: feature stacks published and validated."
  echo "  Root:   $FEATURES_ROOT"
  echo "  Run ID: ${RUN_ID:-unknown}"
else
  echo "FAIL: feature-stack run or validation failed."
  echo "  Root:       $FEATURES_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  echo "  Check VM-side files with: status-vm.sh"
  exit 1
fi
