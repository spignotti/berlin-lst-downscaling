#!/usr/bin/env bash
# Box sequencer for the val-only baselines (validation 2024; test 2025 stays closed).
#
# Usage: scripts/operators/with-gcp-operator.sh -- \
#          scripts/runners/vm/val-baselines-vm.sh <rf-smoke|rf-full|naive|plan> <branch> <hourly-rate-usd> [--execute]
#
# Order: rf-smoke -> (human inspects smoke memory/throughput) -> rf-full -> naive,
# all on the same VM and never concurrently. One step runs per invocation and
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
  echo "Usage: $0 <rf-smoke|rf-full|naive|plan> <branch> <hourly-rate-usd> [--execute]" >&2
  exit 1
}

STEP="${1:-}"
BRANCH="${2:-}"
RATE="${3:-}"
EXECUTE=false
if [[ $# -ge 4 ]]; then
  if [[ "${4}" == "--execute" && $# -eq 4 ]]; then
    EXECUTE=true
  else
    echo "Unknown flag: ${4}" >&2
    exit 1
  fi
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

print_step() {
  local n="$1" name="$2" cmdline="$3" hours="$4" config="$5"
  echo "Val-only baselines (validation 2024; test 2025 closed) — step $n/3: $name"
  echo "  Command: scripts/operators/with-gcp-operator.sh -- scripts/runners/vm/$cmdline"
  echo "  Exposure at $RATE USD/h: ~$(awk -v r="$RATE" -v h="$hours" 'BEGIN { printf "%.2f", r * h }') USD for $hours h window (not a hard cap; connection loss leaves VM RUNNING)"
  echo "  Reference rate 0.168788 USD/h (europe-west3 n2-highmem-2, On-Demand list price) — re-verify before a paid run."
  echo "  Preflight (read-only, run first): scripts/runners/vm/status-vm.sh; scripts/runners/vm/ssh-vm.sh --check;"
  echo "    uv run python scripts/validators/preflight_v6_reads.py --config-name $config --limit 8"
  echo "  Order: rf-smoke → inspect → rf-full → naive; never concurrent."
}

if [[ "$STEP" == "plan" ]]; then
  if [[ "$EXECUTE" == "true" ]]; then
    echo "ERROR: 'plan' cannot be combined with --execute." >&2
    exit 1
  fi
  print_step 1 rf-smoke "run-random-forest-vm.sh v6-val-smoke $BRANCH $RATE" 1 random_forest_v6_val_smoke
  print_step 2 rf-full "run-random-forest-vm.sh v6-val-full $BRANCH $RATE" 8 random_forest_v6_val_full
  print_step 3 naive "run-baseline-vm.sh $BRANCH baseline_v6_val" 2 baseline_v6_val
  echo "[dry-run] nothing started. Re-run a single step with --execute after a human Run OK."
  exit 0
fi

case "$STEP" in
  rf-smoke) CMD=(run-random-forest-vm.sh v6-val-smoke "$BRANCH" "$RATE"); HOURS=1; N=1; CONFIG=random_forest_v6_val_smoke ;;
  rf-full) CMD=(run-random-forest-vm.sh v6-val-full "$BRANCH" "$RATE"); HOURS=8; N=2; CONFIG=random_forest_v6_val_full ;;
  naive) CMD=(run-baseline-vm.sh "$BRANCH" baseline_v6_val); HOURS=2; N=3; CONFIG=baseline_v6_val ;;
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
