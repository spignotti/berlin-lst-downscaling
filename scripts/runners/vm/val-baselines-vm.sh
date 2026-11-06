#!/usr/bin/env bash
# Box sequencer for the val-only baselines (validation 2024; test 2025 stays closed).
#
# Usage: scripts/operators/with-gcp-operator.sh -- \
#          scripts/runners/vm/val-baselines-vm.sh <naive|rf-smoke|rf-full|plan> <branch> <hourly-rate-usd> \
#          [--naive-report PATH --naive-sha256 HEX] [--execute]
#
# Order: naive -> rf-smoke -> (human inspects smoke memory/throughput) -> rf-full,
# all on the same VM and never concurrently. The rf steps need the binding (secured
# baseline_report.json path and sha256) printed by the naive step. One step runs per invocation and
# nothing chains automatically. The default prints the plan only: no gcloud,
# no git, no VM start. --execute (human Run OK) runs exactly one step through
# the existing launcher; the caller wraps this whole script with
# with-gcp-operator.sh.
#
# The RF launcher gate is "8-hour exposure <= 10 USD at the given rate"; it is
# an exposure gate, not a hard total budget. On connection loss or poll expiry
# the launchers leave the VM RUNNING by design (fail-closed, to protect unsaved
# artifacts), so cost continues until a human stops it. This script adds no
# auto-stop.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  echo "Usage: $0 <naive|rf-smoke|rf-full|plan> <branch> <hourly-rate-usd> [--naive-report PATH --naive-sha256 HEX] [--execute]" >&2
  exit 1
}

STEP="${1:-}"
BRANCH="${2:-}"
RATE="${3:-}"
EXECUTE=false
NAIVE_REPORT=""
NAIVE_SHA256=""
if [[ $# -ge 4 ]]; then
  set -- "${@:4}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --execute) EXECUTE=true; shift ;;
      --naive-report|--naive-sha256)
        if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 1; fi
        if [[ "$1" == "--naive-report" ]]; then NAIVE_REPORT="$2"; else NAIVE_SHA256="$2"; fi
        shift 2 ;;
      *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
  done
fi

case "$STEP" in
  smoke|full|v6-smoke|v6-full|baseline_full|baseline_v6_full|test*)
    echo "ERROR: '$STEP' opens test 2025; this readiness path is val-only." >&2
    exit 1 ;;
  rf-smoke|rf-full|naive|plan) ;;
  *) usage ;;
esac

[[ -n "$BRANCH" && -n "$RATE" ]] || usage
if [[ ! "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ || "$BRANCH" == -* ]]; then
  echo "ERROR: invalid branch name." >&2
  exit 1
fi
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! awk -v rate="$RATE" 'BEGIN { exit !(rate > 0 && rate * 8 <= 10) }'; then
  echo "ERROR: require a verified hourly rate > 0 with 8-hour exposure <= 10 USD." >&2
  exit 1
fi
case "$STEP" in
  rf-smoke|rf-full)
    if [[ -z "$NAIVE_REPORT" || -z "$NAIVE_SHA256" ]]; then
      echo "ERROR: $STEP requires --naive-report and --naive-sha256 (run the naive step first)." >&2
      exit 1
    fi
    if [[ ! "$NAIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ERROR: --naive-sha256 must be 64 lowercase hex characters." >&2
      exit 1
    fi ;;
  *)
    if [[ -n "$NAIVE_REPORT" || -n "$NAIVE_SHA256" ]]; then
      echo "ERROR: '$STEP' does not take --naive-report/--naive-sha256." >&2
      exit 1
    fi ;;
esac

print_step() {
  local n="$1" name="$2" cmdline="$3" hours="$4" config="$5"
  echo "Val-only baselines (validation 2024; test 2025 closed) — step $n/3: $name"
  echo "  Command: scripts/operators/with-gcp-operator.sh -- scripts/runners/vm/$cmdline"
  echo "  Exposure at $RATE USD/h: ~$(awk -v r="$RATE" -v h="$hours" 'BEGIN { printf "%.2f", r * h }') USD for $hours h window (not a hard cap; connection loss leaves VM RUNNING)"
  echo "  Reference rate 0.168788 USD/h (europe-west3 n2-highmem-2, On-Demand list price) — re-verify before a paid run."
  echo "  Preflight (read-only, run first): scripts/runners/vm/status-vm.sh; scripts/runners/vm/ssh-vm.sh --check;"
  echo "    uv run python scripts/validators/preflight_v6_reads.py --config-name $config --limit 8"
  echo "  Order: naive → rf-smoke → inspect → rf-full; never concurrent."
}

if [[ "$STEP" == "plan" ]]; then
  if [[ "$EXECUTE" == "true" ]]; then
    echo "ERROR: 'plan' cannot be combined with --execute." >&2
    exit 1
  fi
  BIND_ARGS="--naive-report <secured baseline_report.json> --naive-sha256 <sha256 printed by naive>"
  print_step 1 naive "run-baseline-vm.sh $BRANCH baseline_v6_val" 2 baseline_v6_val
  print_step 2 rf-smoke "run-random-forest-vm.sh v6-val-smoke $BRANCH $RATE $BIND_ARGS" 1 random_forest_v6_val_smoke
  print_step 3 rf-full "run-random-forest-vm.sh v6-val-full $BRANCH $RATE $BIND_ARGS" 8 random_forest_v6_val_full
  echo "[dry-run] nothing started. Re-run a single step with --execute after a human Run OK."
  exit 0
fi

case "$STEP" in
  rf-smoke) CMD=(run-random-forest-vm.sh v6-val-smoke "$BRANCH" "$RATE" --naive-report "$NAIVE_REPORT" --naive-sha256 "$NAIVE_SHA256"); HOURS=1; N=2; CONFIG=random_forest_v6_val_smoke ;;
  rf-full) CMD=(run-random-forest-vm.sh v6-val-full "$BRANCH" "$RATE" --naive-report "$NAIVE_REPORT" --naive-sha256 "$NAIVE_SHA256"); HOURS=8; N=3; CONFIG=random_forest_v6_val_full ;;
  naive) CMD=(run-baseline-vm.sh "$BRANCH" baseline_v6_val); HOURS=2; N=1; CONFIG=baseline_v6_val ;;
esac

print_step "$N" "$STEP" "${CMD[*]}" "$HOURS" "$CONFIG"

if [[ "$EXECUTE" != "true" ]]; then
  echo "[dry-run] nothing started. Re-run with --execute after a human Run OK."
  exit 0
fi

exec 9>"${XDG_RUNTIME_DIR:-/tmp}/berlin-lst-val-baselines.lock"
if ! flock -n 9; then
  echo "ERROR: another val-baseline step is running on this box" >&2
  exit 1
fi

# shellcheck source=vm-identity.sh
source "$SCRIPTS_DIR/vm-identity.sh"
assert_vm_identity TERMINATED STOPPED

rc=0
"$SCRIPTS_DIR/${CMD[0]}" "${CMD[@]:1}" || rc=$?
exit "$rc"
