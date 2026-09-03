#!/usr/bin/env bash
# Read-only TSV probe for Berlin LST cloud operations dashboard.
#
# Emits one TSV line per check: <label>\t<state>\t<value>\t<detail>
# States: ok | warn | fail | unknown
#
# Checks:
#   VM state, identity match, deletion protection, disk auto-delete
#   Bucket access (gs://berlin-lst-data)
#   rclone mount status (~/.mnt/berlin-lst)
#
# NEVER starts, stops, or mutates any cloud resource.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$SCRIPTS_DIR/../../.opencode/skills/google-access/scripts"

# Source pinned identity constants (VM_PROJECT, VM_ZONE, VM_NAME, VM_EXPECTED_ID, VM_SA).
# Do NOT source assert_vm_identity — that function exits on mismatch.
if [[ -f "$SKILL_DIR/vm-identity.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SKILL_DIR/vm-identity.sh"
else
  echo -e "VM identity\tunknown\t-\tvm-identity.sh not found"
  exit 0
fi

# ── helpers ──────────────────────────────────────────────────────────

_tmpdir=$(mktemp -d)
trap 'rm -rf "$_tmpdir"' EXIT

# Single-field gcloud describe (used in parallel below).
_vm_field_direct() {
  rtk gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$VM_PROJECT" \
    --format="value($1)" 2>/dev/null || echo ""
}

# ── VM checks (parallel field queries, ~5s) ─────────────────────────

(
  # Run all 4 gcloud queries in parallel within the subshell.
  _vm_field_direct "id"              >"$_tmpdir/vm_id"    &
  _vm_field_direct "status"          >"$_tmpdir/vm_state" &
  _vm_field_direct "disks[0].autoDelete" >"$_tmpdir/vm_disk" &
  _vm_field_direct "deletionProtection"  >"$_tmpdir/vm_prot" &
  wait

  act_id=$(cat "$_tmpdir/vm_id"    2>/dev/null || echo "")
  act_state=$(cat "$_tmpdir/vm_state" 2>/dev/null || echo "")
  act_disk_autodel=$(cat "$_tmpdir/vm_disk" 2>/dev/null || echo "")
  act_protection=$(cat "$_tmpdir/vm_prot"  2>/dev/null || echo "")

  if [[ -z "$act_id" ]]; then
    echo -e "VM\tfail\t-\tdoes not exist or unreachable"
  elif [[ "$act_id" != "$VM_EXPECTED_ID" ]]; then
    echo -e "VM\tfail\tMISMATCH\texpected $VM_EXPECTED_ID, got $act_id"
  else
    case "$act_state" in
      RUNNING)  echo -e "VM\tok\tRUNNING\t$VM_NAME" ;;
      STOPPED|TERMINATED) echo -e "VM\twarn\t$act_state\t$VM_NAME" ;;
      *)        echo -e "VM\twarn\t${act_state:-UNKNOWN}\t$VM_NAME" ;;
    esac
  fi

  if [[ -n "$act_protection" ]]; then
    if [[ "$act_protection" == "True" ]]; then
      echo -e "Protection\tok\tenabled\tdeletion protection on"
    else
      echo -e "Protection\twarn\toff\tinstance can be deleted"
    fi
  else
    echo -e "Protection\tunknown\t-\tcannot read"
  fi

  if [[ -n "$act_disk_autodel" ]]; then
    if [[ "$act_disk_autodel" == "True" ]]; then
      echo -e "Disk auto-del\twarn\ttrue\tboot disk at risk"
    else
      echo -e "Disk auto-del\tok\tfalse\tboot disk retained"
    fi
  else
    echo -e "Disk auto-del\tunknown\t-\tcannot read"
  fi
) >"$_tmpdir/vm" 2>/dev/null &
vm_pid=$!

# ── Bucket access (background) ──────────────────────────────────────

(
  if command -v gcloud >/dev/null 2>&1; then
    if gcloud storage ls "gs://berlin-lst-data/" --project="$VM_PROJECT" >/dev/null 2>&1; then
      echo -e "Bucket\tok\taccessible\tgs://berlin-lst-data"
    else
      echo -e "Bucket\tfail\tinaccessible\tgs://berlin-lst-data"
    fi
  else
    echo -e "Bucket\tunknown\t-\tgcloud not available"
  fi
) >"$_tmpdir/bucket" 2>/dev/null &
bucket_pid=$!

# ── rclone mount (inline, fast) ─────────────────────────────────────

MOUNT_POINT="$HOME/.mnt/berlin-lst"
if pgrep -f "rclone mount" >/dev/null 2>&1; then
  if [[ -d "$MOUNT_POINT" ]] && ls "$MOUNT_POINT" >/dev/null 2>&1; then
    echo -e "Mount\tok\t$MOUNT_POINT\trclone active"
  else
    echo -e "Mount\twarn\trunning but empty\t$MOUNT_POINT"
  fi
else
  echo -e "Mount\twarn\tnot running\trclone not active"
fi

# ── collect background results ──────────────────────────────────────

wait "$vm_pid" 2>/dev/null || true
wait "$bucket_pid" 2>/dev/null || true

cat "$_tmpdir/vm" 2>/dev/null || true
cat "$_tmpdir/bucket" 2>/dev/null || true
