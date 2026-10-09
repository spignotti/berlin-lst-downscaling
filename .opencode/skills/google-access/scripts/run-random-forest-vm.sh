#!/usr/bin/env bash
# Run a committed RF smoke or full baseline, validate, secure artifacts, and stop.
# Usage: run-random-forest-vm.sh <smoke|full> <branch> <hourly-rate-usd>
# Full mode requires operator inspection of smoke memory/throughput first.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

MODE="${1:-}"
BRANCH="${2:-}"
HOURLY_RATE="${3:-}"
case "$MODE" in
  smoke) CONFIG="random_forest_smoke"; RUN_SECONDS=1800; TOTAL_SECONDS=3600 ;;
  full) CONFIG="random_forest_full"; RUN_SECONDS=21600; TOTAL_SECONDS=25200 ;;
  *) echo "Usage: $0 <smoke|full> <branch> <hourly-rate-usd>" >&2; exit 1 ;;
esac
if [[ ! "$HOURLY_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! awk -v rate="$HOURLY_RATE" 'BEGIN { exit !(rate > 0 && rate * 8 <= 10) }'; then
  echo "ERROR: require a verified hourly rate > 0 with 8-hour exposure <= 10 USD." >&2
  exit 1
fi

DEADLINE=$((SECONDS + TOTAL_SECONDS))
remaining_seconds() {
  local remaining=$((DEADLINE - SECONDS))
  if (( remaining <= 0 )); then
    echo "ERROR: total RF lifecycle deadline exhausted; inspect the run before further work." >&2
    return 1
  fi
  echo "$remaining"
}

bounded_command() {
  local remaining
  remaining=$(remaining_seconds) || return 124
  python3 - "$remaining" "$@" <<'PY_TIMEOUT'
import subprocess
import sys
try:
    result = subprocess.run(sys.argv[2:], timeout=int(sys.argv[1]), check=False)
except subprocess.TimeoutExpired:
    raise SystemExit(124) from None
raise SystemExit(result.returncode)
PY_TIMEOUT
}

ssh_cmd() {
  bounded_command "$VM_SCRIPTS/ssh-vm.sh" -- "$@"
}

PIPELINE_LABEL="Random Forest $MODE (stages 1–4)"
MARKER_CONFIG="$CONFIG"
POLL_MAX_SECONDS=$((RUN_SECONDS + 900))
vm_preflight_pushed
assert_vm_identity TERMINATED STOPPED
if [[ "$(_vm_field 'machineType.basename()')" != "$VM_MACHINE" ]]; then
  echo "ERROR: unexpected machine type; re-check cost and memory bounds." >&2
  exit 1
fi
vm_init_run "random-forest-$MODE"
OUTPUT_REL="data/runs/random-forest/$WRAP_RUN_ID"
REMOTE_OUTPUT="$APP_DIR/$OUTPUT_REL"
REPO_ROOT="$(cd "$VM_SCRIPTS/../../../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/data/runs/random-forest/$WRAP_RUN_ID"

# Keep this launcher's wrapper log and marker under the pipeline output root.
LOG_DIR="$REMOTE_OUTPUT/logs/random_forest/launcher"
MARKER="$LOG_DIR/marker.json"
STATUS_FILE="$LOG_DIR/exit_status"
REMOTE_LOG="$LOG_DIR/nohup.log"
REMOTE_PID_FILE="$LOG_DIR/pid"
REMOTE_CMD="set -f; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s ${RUN_SECONDS}s uv run python scripts/runners/run_random_forest.py --config-name $CONFIG output_root=$OUTPUT_REL"

echo "$PIPELINE_LABEL | run=$WRAP_RUN_ID | maximum compute exposure: $(awk -v r="$HOURLY_RATE" 'BEGIN { printf "%.2f USD", r * 8 }')"
vm_start_and_wait_ssh
if ! REMOTE_STATE=$(ssh_cmd "if pgrep -f '[p]ython scripts/(runners|operators)/' >/dev/null; then echo occupied; elif test -n \"\$(git -C '$APP_DIR' status --porcelain)\"; then echo dirty; else echo idle; fi"); then
  echo "ERROR: remote preflight lost contact; preserving the VM." >&2
  leave_running=1
  exit 2
fi
if [[ "$REMOTE_STATE" != "idle" ]]; then
  echo "ERROR: remote workspace is $REMOTE_STATE; preserving the VM and existing work." >&2
  leave_running=1
  exit 2
fi
vm_push_deploy
REMAINING=$(remaining_seconds)
RESERVE=1200
if [[ "$MODE" == "full" ]]; then RESERVE=3600; fi
AVAILABLE=$((REMAINING - RESERVE))
if (( AVAILABLE < 60 )); then
  echo "ERROR: deployment consumed the run budget; refusing to launch." >&2
  exit 1
fi
if (( RUN_SECONDS > AVAILABLE )); then RUN_SECONDS=$AVAILABLE; fi
REMOTE_CMD="set -f; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s ${RUN_SECONDS}s uv run python scripts/runners/run_random_forest.py --config-name $CONFIG output_root=$OUTPUT_REL"
vm_write_marker "$CONFIG"
vm_launch_detached
vm_poll
vm_finish

if [[ "$PIPELINE_EXIT" != "0" ]]; then
  ssh_cmd "tail -n 40 '$REMOTE_LOG'" || true
  vm_stop
  echo "FAIL: RF $MODE exited $PIPELINE_EXIT; outputs preserved at $REMOTE_OUTPUT." >&2
  exit 1
fi

REMAINING=$(remaining_seconds)
VALIDATION_SECONDS=$((REMAINING - 600))
if (( VALIDATION_SECONDS > 1800 )); then VALIDATION_SECONDS=1800; fi
if (( VALIDATION_SECONDS < 1 )); then
  echo "ERROR: no validation budget remains; outputs preserved at $REMOTE_OUTPUT." >&2
  vm_stop
  exit 1
fi
VALIDATION_RC=0
if VALIDATION_OUT=$(ssh_cmd "cd '$APP_DIR' && timeout --signal=TERM --kill-after=60s ${VALIDATION_SECONDS}s uv run python scripts/validators/validate_random_forest.py --report '$REMOTE_OUTPUT/random_forest_report.json' --max-patches 5" 2>&1); then
  printf '%s\n' "$VALIDATION_OUT"
else
  VALIDATION_RC=$?
  printf '%s\n' "$VALIDATION_OUT" >&2
fi

# An unsuccessful transfer leaves the VM up and its output intact for recovery.
leave_running=1
mkdir -p "$LOCAL_DIR"
printf '%s\n' "$VALIDATION_OUT" > "$LOCAL_DIR/validation.txt"
ARCHIVE="$APP_DIR/data/runs/random-forest/$WRAP_RUN_ID-artifact.tgz"
ssh_cmd "tar --exclude=./artifact.tgz -C '$REMOTE_OUTPUT' -czf '$ARCHIVE' ."
REMOTE_SHA=$(ssh_cmd "sha256sum '$ARCHIVE'" | cut -d' ' -f1)
ssh_cmd "cat '$ARCHIVE'" > "$LOCAL_DIR/artifact.tgz"
LOCAL_SHA=$(shasum -a 256 "$LOCAL_DIR/artifact.tgz" | cut -d' ' -f1)
if [[ -z "$REMOTE_SHA" || "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
  echo "FAIL: artifact transfer hash mismatch; VM and outputs preserved." >&2
  leave_running=1
  exit 2
fi
tar -xzf "$LOCAL_DIR/artifact.tgz" -C "$LOCAL_DIR"
printf '%s\n' "$REMOTE_SHA" > "$LOCAL_DIR/artifact.sha256"
leave_running=0

if [[ "$VALIDATION_RC" -eq 0 ]]; then
  ssh_cmd "rm -rf '$REMOTE_OUTPUT' && rm -f '$ARCHIVE'"
fi
vm_stop
if [[ "$VALIDATION_RC" -ne 0 ]]; then
  echo "FAIL: independent validator rejected RF artifacts; secured at $LOCAL_DIR." >&2
  exit 1
fi
echo "SUCCESS: RF $MODE validated and secured; VM stopped. Artifacts: $LOCAL_DIR"
