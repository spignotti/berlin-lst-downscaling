#!/usr/bin/env bash
# Shared VM identity assertions for berlin-lst scripts.
#
# Source this file to assert that the pinned Compute Engine instance
# exists, is identified by the expected ID, and has its boot disk
# correctly attached.  Any mismatch is a hard STOP — no retry, no
# fallback, no silent resolution.
#
# Usage (in other scripts):
#   SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
#   source "$SCRIPTS_DIR/vm-identity.sh"
#   assert_vm_identity RUNNING        # fail unless RUNNING
#   assert_vm_identity TERMINATED     # fail unless TERMINATED

set -euo pipefail

# ── pinned identity — change only when the instance is knowingly re-created ──
VM_PROJECT="masterarbeit-berlin-lst-v2"
VM_ZONE="europe-west3-a"
VM_NAME="berlin-lst-vm"
VM_EXPECTED_ID="8456019039456721311"
VM_SA="masterarbeit-vertex@${VM_PROJECT}.iam.gserviceaccount.com"
VM_MACHINE="n2-highmem-2"
VM_DISK_DEVICE="persistent-disk-0"  # attachment device name; disk resource name is ${VM_NAME}
# Note: `gcloud compute instances set-disk-auto-delete` takes --disk=<resource name>
# (berlin-lst-vm), while --device-name takes the device name above.

# Known-hosts alias used by gcloud / OpenSSH
VM_KNOWN_HOSTS_ALIAS="compute.${VM_EXPECTED_ID}"

# ── describe helpers (read-only) ─────────────────────────────────────────────

_vm_describe() {
  # $1: value format expression
  rtk gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$VM_PROJECT" \
    --format="value($1)" 2>/dev/null
}

# ── single-field helper ──────────────────────────────────────────────────────

_vm_field() {
  # $1: value format expression
  rtk gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$VM_PROJECT" \
    --format="value($1)" 2>/dev/null
}

# ── core assertion ───────────────────────────────────────────────────────────

assert_vm_identity() {
  # Assert the pinned identity and return 0 only when the instance
  # matches AND is in one of the allowed states.
  #
  # Usage: assert_vm_identity [STATE...]
  #   e.g. assert_vm_identity RUNNING
  #        assert_vm_identity TERMINATED STOPPED
  #
  # Stops hard on any mismatch.

  local allowed_states=("$@")
  if [[ ${#allowed_states[@]} -eq 0 ]]; then
    allowed_states=("RUNNING" "TERMINATED" "STOPPED")
  fi

  # ── describe (individual fields — avoids tab-shift on empty values) ────
  local actual_id actual_state disk_source disk_auto_delete actual_sa _protection actual_machine

  actual_id=$(_vm_field "id") || true
  if [[ -z "$actual_id" || "$actual_id" == "(unset)" ]]; then
    echo "ERROR: Instance '$VM_NAME' does not exist or is inaccessible." >&2
    echo "  This script will NEVER create a new instance." >&2
    return 1
  fi

  actual_state=$(_vm_field "status")
  disk_source=$(_vm_field "disks[0].source.basename()")
  disk_auto_delete=$(_vm_field "disks[0].autoDelete")
  actual_sa=$(_vm_field "serviceAccounts[0].email")
  _protection=$(_vm_field "deletionProtection")
  actual_machine=$(_vm_field "machineType.basename()")

  # ── missing ──────────────────────────────────────────────────────────
  if [[ -z "$actual_id" || "$actual_id" == "(unset)" ]]; then
    echo "ERROR: Instance '$VM_NAME' does not exist or is inaccessible." >&2
    echo "  This script will NEVER create a new instance." >&2
    return 1
  fi

  # ── id ───────────────────────────────────────────────────────────────
  if [[ "$actual_id" != "$VM_EXPECTED_ID" ]]; then
    echo "ERROR: Instance ID mismatch." >&2
    echo "  Expected: $VM_EXPECTED_ID" >&2
    echo "  Actual:   $actual_id" >&2
    echo "  An instance with the same name but a different ID exists." >&2
    echo "  This usually means the original VM was deleted and recreated." >&2
    return 1
  fi

  # ── state ────────────────────────────────────────────────────────────
  local state_ok=false
  for allowed in "${allowed_states[@]}"; do
    if [[ "$actual_state" == "$allowed" ]]; then
      state_ok=true
      break
    fi
  done
  if [[ "$state_ok" != "true" ]]; then
    echo "ERROR: Instance is in unexpected state '$actual_state'." >&2
    echo "  Allowed: ${allowed_states[*]}" >&2
    return 1
  fi

  # ── boot disk ────────────────────────────────────────────────────────
  if [[ "$disk_source" == "(unset)" || -z "$disk_source" ]]; then
    echo "ERROR: Boot disk source not reported." >&2
    return 1
  fi

  # ── disk auto-delete risk ────────────────────────────────────────────
  if [[ "$disk_auto_delete" == "True" ]]; then
    echo "WARNING: Boot disk is set to auto-delete on instance deletion." >&2
    echo "  Deleting the instance would destroy the boot disk." >&2
  fi

  # ── export for callers ───────────────────────────────────────────────
  VM_ACTUAL_STATE="$actual_state"
  VM_INSTANCE_ID="$actual_id"
  VM_BOOT_DISK="$disk_source"
  VM_DISK_AUTO_DELETE="$disk_auto_delete"
}

# ── IP lookup ───────────────────────────────────────────────────────────────

vm_external_ip() {
  # Print external IP of the running VM, or empty string if not available.
  rtk gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$VM_PROJECT" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null \
    || true
}