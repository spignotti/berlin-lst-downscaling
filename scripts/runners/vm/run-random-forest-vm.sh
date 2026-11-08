#!/usr/bin/env bash
# Run a committed RF smoke or full baseline, validate, secure artifacts, and stop.
# Usage: run-random-forest-vm.sh <smoke|full|v6-smoke|v6-full|v6-val-smoke|v6-val-full> <branch> <hourly-rate-usd>
#          [--naive-report PATH --naive-sha256 HEX]
# Every v6 mode requires the naive evidence binding (run the naive baseline_v6_val first and pass
# the secured baseline_report.json with the sha256 of the naive_evidence_manifest.sha256 next to it). Legacy smoke|full reject it and
# keep the legacy validator lookup. The evidence is verified locally before any VM call, uploaded
# with a remote sha256 check, and passed to the validator as an explicit --naive-report.
# Full mode requires operator inspection of smoke memory/throughput first.
# v6-smoke/v6-full open test 2025; until the v6 ablation matrix is frozen use
# v6-val-smoke/v6-val-full (validation 2024 only, never reads a test patch).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"
source "$(cd "$(dirname "$0")" && pwd)/naive-evidence-lib.sh"

MODE="${1:-}"
BRANCH="${2:-}"
HOURLY_RATE="${3:-}"
if (( $# >= 3 )); then shift 3; else shift $#; fi
NAIVE_REPORT=""
NAIVE_SHA256=""
NAIVE_USAGE="Usage: $0 <smoke|full|v6-smoke|v6-full|v6-val-smoke|v6-val-full> <branch> <hourly-rate-usd> [--naive-report PATH --naive-sha256 HEX]"
while (( $# > 0 )); do
  case "$1" in
    --naive-report|--naive-sha256)
      if (( $# < 2 )); then echo "$NAIVE_USAGE" >&2; exit 1; fi
      if [[ "$1" == "--naive-report" ]]; then NAIVE_REPORT="$2"; else NAIVE_SHA256="$2"; fi
      shift 2 ;;
    *) echo "$NAIVE_USAGE" >&2; exit 1 ;;
  esac
done
NAIVE_SPLITS=""
NAIVE_REPORT_SHA=""
NAIVE_MANIFEST=""
case "$MODE" in
  smoke) CONFIG="random_forest_smoke"; RUN_SECONDS=1800; TOTAL_SECONDS=3600 ;;
  full) CONFIG="random_forest_full"; RUN_SECONDS=21600; TOTAL_SECONDS=25200 ;;
  v6-smoke) CONFIG="random_forest_v6_smoke"; RUN_SECONDS=1800; TOTAL_SECONDS=3600; NAIVE_SPLITS="test,validation" ;;
  v6-full) CONFIG="random_forest_v6_full"; RUN_SECONDS=21600; TOTAL_SECONDS=25200; NAIVE_SPLITS="test,validation" ;;
  v6-val-smoke) CONFIG="random_forest_v6_val_smoke"; RUN_SECONDS=1800; TOTAL_SECONDS=3600; NAIVE_SPLITS="validation" ;;
  v6-val-full) CONFIG="random_forest_v6_val_full"; RUN_SECONDS=21600; TOTAL_SECONDS=25200; NAIVE_SPLITS="validation" ;;
  *) echo "$NAIVE_USAGE" >&2; exit 1 ;;
esac
if [[ -z "$NAIVE_SPLITS" && ( -n "$NAIVE_REPORT" || -n "$NAIVE_SHA256" ) ]]; then
  echo "ERROR: legacy mode '$MODE' does not take --naive-report/--naive-sha256." >&2
  exit 1
fi
if [[ -n "$NAIVE_SPLITS" && ( -z "$NAIVE_REPORT" || -z "$NAIVE_SHA256" ) ]]; then
  echo "ERROR: v6 modes require --naive-report and --naive-sha256 (run naive baseline_v6_val first)." >&2
  exit 1
fi
if [[ "$NAIVE_REPORT" == *"'"* ]]; then
  echo "ERROR: --naive-report must not contain a single quote." >&2
  exit 1
fi
if [[ ! "$HOURLY_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! awk -v rate="$HOURLY_RATE" 'BEGIN { exit !(rate > 0 && rate * 8 <= 10) }'; then
  echo "ERROR: require a verified hourly rate > 0 with 8-hour exposure <= 10 USD." >&2
  exit 1
fi

REPO_ROOT="$(cd "$VM_SCRIPTS/../../.." && pwd)"
# Verify the bound naive evidence before vm_preflight_pushed and before any gcloud call.
if [[ -n "$NAIVE_SPLITS" ]]; then
  if ! (cd "$REPO_ROOT" && uv run --locked python scripts/validators/verify_naive_binding.py \
      --report "$NAIVE_REPORT" --sha256 "$NAIVE_SHA256" --config-name "$CONFIG" \
      --expected-splits "$NAIVE_SPLITS"); then
    echo "ERROR: naive evidence binding rejected; VM not started." >&2
    exit 1
  fi
  NAIVE_MANIFEST="$(dirname "$NAIVE_REPORT")/naive_evidence_manifest.sha256"
  if ! NAIVE_REPORT_SHA=$(naive_manifest_report_sha "$NAIVE_MANIFEST" "$NAIVE_SHA256"); then
    echo "ERROR: naive evidence manifest changed after verification; VM not started." >&2
    exit 1
  fi
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

# Same deadline bound as bounded_command, but the timeout program is passed with -c so stdin
# stays attached to the caller (bounded_command reads its program from a heredoc on stdin).
UPLOAD_TIMEOUT_PY='import subprocess, sys
try:
    result = subprocess.run(sys.argv[2:], timeout=int(sys.argv[1]), check=False)
except subprocess.TimeoutExpired:
    raise SystemExit(124) from None
raise SystemExit(result.returncode)'
ssh_upload_cmd() {
  local remaining
  remaining=$(remaining_seconds) || return 124
  python3 -c "$UPLOAD_TIMEOUT_PY" "$remaining" "$VM_SCRIPTS/ssh-vm.sh" -- "$@"
}

PIPELINE_LABEL="Random Forest $MODE (stages 1–4)"
# shellcheck disable=SC2034  # launcher contract field (see vm-runner-common.sh)
MARKER_CONFIG="$CONFIG"
POLL_MAX_SECONDS=$((RUN_SECONDS + 900))
vm_preflight_pushed
assert_vm_identity TERMINATED STOPPED
if [[ "$(_vm_field 'machineType.basename()')" != "$VM_MACHINE" ]]; then
  echo "ERROR: unexpected machine type; re-check cost and memory bounds." >&2
  exit 1
fi
vm_init_run "random-forest-$MODE"
RUN_DIR="random-forest"
if [[ "$MODE" == v6-* ]]; then RUN_DIR="random-forest-v6"; fi
if [[ "$MODE" == v6-val-* ]]; then RUN_DIR="random-forest-v6-notest"; fi
OUTPUT_REL="data/runs/$RUN_DIR/$WRAP_RUN_ID"
REMOTE_OUTPUT="$APP_DIR/$OUTPUT_REL"
REMOTE_NAIVE_DIR="$APP_DIR/data/runs/$RUN_DIR/$WRAP_RUN_ID-naive"
REMOTE_NAIVE="$REMOTE_NAIVE_DIR/baseline_report.json"
LOCAL_DIR="$REPO_ROOT/data/runs/$RUN_DIR/$WRAP_RUN_ID"

# Keep this launcher's wrapper log and marker under the pipeline output root.
LOG_DIR="$REMOTE_OUTPUT/logs/random_forest/launcher"
MARKER="$LOG_DIR/marker.json"
STATUS_FILE="$LOG_DIR/exit_status"
REMOTE_LOG="$LOG_DIR/nohup.log"
REMOTE_PID_FILE="$LOG_DIR/pid"
REMOTE_CMD="set -f; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s ${RUN_SECONDS}s uv run python scripts/runners/run_random_forest.py --config-name $CONFIG output_root=$OUTPUT_REL"

echo "$PIPELINE_LABEL | run=$WRAP_RUN_ID | maximum compute exposure: $(awk -v r="$HOURLY_RATE" 'BEGIN { printf "%.2f USD", r * 8 }')"
vm_start_and_wait_ssh
# Probe fails closed: any git/app-dir error yields `broken`, never `idle`.
PROBE_CMD=$(vm_remote_idle_probe_cmd "$VM_HOME" "$VM_LOGIN_USER" "$VM_APP_ORIGIN_URL" "$VM_PIN_UID")
if ! REMOTE_STATE=$(ssh_cmd "$PROBE_CMD"); then
  echo "ERROR: remote preflight lost contact; preserving the VM." >&2
  leave_running=1
  exit 2
fi
if [[ "$REMOTE_STATE" != "idle" && "$REMOTE_STATE" != "absent" ]]; then
  echo "ERROR: remote workspace is $REMOTE_STATE; preserving the VM and existing work." >&2
  leave_running=1
  exit 2
fi
vm_push_deploy
REMAINING=$(remaining_seconds)
RESERVE=1200
if [[ "$MODE" == "full" || "$MODE" == "v6-full" || "$MODE" == "v6-val-full" ]]; then RESERVE=3600; fi
AVAILABLE=$((REMAINING - RESERVE))
if (( AVAILABLE < 60 )); then
  echo "ERROR: deployment consumed the run budget; refusing to launch." >&2
  exit 1
fi
if (( RUN_SECONDS > AVAILABLE )); then RUN_SECONDS=$AVAILABLE; fi
REMOTE_CMD="set -f; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s ${RUN_SECONDS}s uv run python scripts/runners/run_random_forest.py --config-name $CONFIG output_root=$OUTPUT_REL"
if [[ -n "$NAIVE_SPLITS" ]]; then
  if ! naive_evidence_push "$NAIVE_REPORT" "$NAIVE_REPORT_SHA" "$REMOTE_NAIVE"; then
    echo "ERROR: naive evidence transfer failed; refusing to launch." >&2
    exit 1
  fi
fi
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
NAIVE_ARG=""
VALIDATION_OUT=""
if [[ -n "$NAIVE_SPLITS" ]]; then
  if naive_evidence_check_remote "$NAIVE_REPORT_SHA" "$REMOTE_NAIVE"; then
    NAIVE_ARG="$REMOTE_NAIVE"
  else
    VALIDATION_RC=1
    VALIDATION_OUT="FAIL: bound naive evidence integrity check failed; validator not run"
  fi
fi
if [[ "$VALIDATION_RC" -eq 0 ]]; then
  if VALIDATION_OUT=$(ssh_cmd "$(rf_validation_remote_cmd "$APP_DIR" "$VALIDATION_SECONDS" "$REMOTE_OUTPUT/random_forest_report.json" "$NAIVE_ARG")" 2>&1); then
    printf '%s\n' "$VALIDATION_OUT"
  else
    VALIDATION_RC=$?
    printf '%s\n' "$VALIDATION_OUT" >&2
  fi
else
  printf '%s\n' "$VALIDATION_OUT" >&2
fi

# An unsuccessful transfer leaves the VM up and its output intact for recovery.
leave_running=1
mkdir -p "$LOCAL_DIR"
if [[ -n "$NAIVE_SPLITS" ]]; then
  printf '%s  %s\n%s  %s\n' "$NAIVE_SHA256" "$NAIVE_MANIFEST" "$NAIVE_REPORT_SHA" "$NAIVE_REPORT" > "$LOCAL_DIR/naive_evidence.sha256"
fi
printf '%s\n' "$VALIDATION_OUT" > "$LOCAL_DIR/validation.txt"
ARCHIVE="$APP_DIR/data/runs/$RUN_DIR/$WRAP_RUN_ID-artifact.tgz"
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
PROVENANCE_RC=0
python3 "$REPO_ROOT/scripts/validators/validate_run_provenance.py" \
  --report "$LOCAL_DIR/random_forest_report.json" --expected-sha "$DEPLOYED_SHA" \
  --context-dir "$LOCAL_DIR/logs/random_forest" || PROVENANCE_RC=$?
leave_running=0

if [[ "$VALIDATION_RC" -eq 0 && "$PROVENANCE_RC" -eq 0 ]]; then
  ssh_cmd "rm -rf '$REMOTE_OUTPUT' && rm -f '$ARCHIVE' && rm -rf '$REMOTE_NAIVE_DIR'"
fi
vm_stop
if [[ "$PROVENANCE_RC" -ne 0 ]]; then
  echo "FAIL: RF provenance missing/mismatched; secured at $LOCAL_DIR." >&2
  exit 1
fi
if [[ "$VALIDATION_RC" -ne 0 ]]; then
  echo "FAIL: independent validator rejected RF artifacts; secured at $LOCAL_DIR." >&2
  exit 1
fi
echo "SUCCESS: RF $MODE validated and secured; VM stopped. Artifacts: $LOCAL_DIR"
