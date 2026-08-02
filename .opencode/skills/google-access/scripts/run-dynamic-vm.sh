#!/usr/bin/env bash
# Run a Dynamic pipeline config on the On-Demand berlin-lst-vm.
#
# Lifecycle: start VM → deploy code → launch pipeline → poll → validate → stop VM.
# The boot disk is retained after stop; pipeline products live in GCS.
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
ZONE="europe-west3-a"
NAME="berlin-lst-vm"
PROJECT="masterarbeit-berlin-lst-v2"
APP_DIR="/workspace/app"
MANIFEST_URI="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"

# ── args ──────────────────────────────────────────────────────────────
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

echo "Config: $CONFIG | Branch: $BRANCH"

# ── start VM ──────────────────────────────────────────────────────────
echo "Starting VM..."
"$SCRIPTS_DIR/start-vm.sh"

# ── push branch so VM can fetch ──────────────────────────────────────
echo "Pushing branch $BRANCH to origin..."
git push origin "$BRANCH" --quiet

# ── deploy code on VM ────────────────────────────────────────────────
echo "Deploying code on VM..."
rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
  cd $APP_DIR && \
  git fetch origin && \
  git checkout $BRANCH && \
  uv sync --frozen --quiet
"

# ── launch pipeline ──────────────────────────────────────────────────
LOG_DIR="$APP_DIR/logs/dynamic"
MARKER="$LOG_DIR/${CONFIG}_exit_code"

# Clean stale marker
rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
  rm -f '$MARKER'
"

echo "Launching pipeline on VM..."
rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
  cd $APP_DIR && \
  nohup uv run python scripts/run_dynamic.py \\
    --config-name $CONFIG \\
    manifest_uri=$MANIFEST_URI \\
    > $LOG_DIR/${CONFIG}_nohup.log 2>&1 &
  echo \$! > $LOG_DIR/${CONFIG}_pid
"

# Get the PID for log tracking
REMOTE_PID=$(rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="cat $LOG_DIR/${CONFIG}_pid" 2>/dev/null || echo "unknown")
echo "Remote PID: $REMOTE_PID"

# ── poll for completion ──────────────────────────────────────────────
echo "Polling for pipeline completion..."
while true; do
  sleep 60

  # Check if the process is still running
  IS_RUNNING=$(rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
    if kill -0 $REMOTE_PID 2>/dev/null; then echo running; else echo stopped; fi
  " 2>/dev/null || echo "unknown")

  if [[ "$IS_RUNNING" == "stopped" ]]; then
    break
  fi

  # Print last log line for progress visibility
  LAST_LOG=$(rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
    tail -1 '$LOG_DIR/${CONFIG}_nohup.log' 2>/dev/null || echo 'waiting...'
  " 2>/dev/null || echo "polling...")
  echo "  [$(date +%H:%M:%S)] $LAST_LOG"
done

# ── check exit code ──────────────────────────────────────────────────
# The pipeline writes its own log; check the nohup output for errors
PIPELINE_FAILED=$(rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="
  grep -c 'error\\|ERROR\\|Traceback\\|SystemExit' '$LOG_DIR/${CONFIG}_nohup.log' 2>/dev/null || echo 0
" 2>/dev/null || echo "0")

echo "Pipeline finished. Error lines in log: $PIPELINE_FAILED"

# ── validate ─────────────────────────────────────────────────────────
if [[ "$CONFIG" == "full" ]]; then
  EXPECTED_ROLE="anchor"
  EXPECTED_SCENES="324"
else
  EXPECTED_ROLE="inference"
  EXPECTED_SCENES="21"
fi

# Map config name to GCS root
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

# ── stop VM ──────────────────────────────────────────────────────────
echo "Stopping VM..."
"$SCRIPTS_DIR/stop-vm.sh"

# ── report ────────────────────────────────────────────────────────────
if [[ "$VALIDATION_OK" -eq 0 ]]; then
  echo "SUCCESS: $CONFIG completed and validated."
else
  echo "FAIL: $CONFIG validation failed or pipeline had errors."
  echo "Check VM-side log: $LOG_DIR/${CONFIG}_nohup.log (VM is stopped, disk retained)"
  exit 1
fi
