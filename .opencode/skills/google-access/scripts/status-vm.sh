#!/usr/bin/env bash
# Read-only status probe for berlin-lst-vm.
#
# Reports identity, state, disk, protection, external IP, and SSH
# host-key readiness.  Never creates, starts, stops, deletes, or
# connects to the VM.  Never writes known_hosts.
#
# Usage: scripts/status-vm.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/vm-identity.sh"

echo "=== berlin-lst-vm status ==="
echo "Expected identity"
echo "  Project:        $VM_PROJECT"
echo "  Zone:           $VM_ZONE"
echo "  Name:           $VM_NAME"
echo "  Instance ID:    $VM_EXPECTED_ID"
echo "  Service account: $VM_SA"
echo ""

# ── describe (one value per call — avoids tab-shift on empty fields) ─────────

_field() {
  rtk gcloud compute instances describe "$VM_NAME" \
    --zone="$VM_ZONE" --project="$VM_PROJECT" \
    --format="value($1)" 2>/dev/null || echo "(error)"
}

act_id=$(_field "id")
act_state=$(_field "status")
act_machine=$(_field "machineType.basename()")
act_priv_ip=$(_field "networkInterfaces[0].networkIP")
act_ext_ip=$(_field "networkInterfaces[0].accessConfigs[0].natIP")
act_disk=$(_field "disks[0].source.basename()")
act_disk_autodel=$(_field "disks[0].autoDelete")
act_sa=$(_field "serviceAccounts[0].email")
act_protection=$(_field "deletionProtection")
act_created=$(_field "creationTimestamp")
act_start=$(_field "lastStartTimestamp")
act_stop=$(_field "lastStopTimestamp")

if [[ -z "$act_id" || "$act_id" == "(unset)" ]]; then
  echo "STATUS: MISSING — instance '$VM_NAME' does not exist."
  echo "  This script will NEVER create a new instance."
  exit 1
fi

echo "Observed identity"
echo "  Instance ID:    $act_id"
echo "  Status:         $act_state"
echo "  Machine type:   ${act_machine:-(unset)}"
echo "  Created:        ${act_created:-(unset)}"
echo "  Last start:     ${act_start:-(unset)}"
echo "  Last stop:      ${act_stop:-(unset)}"
echo ""
echo "Network"
echo "  Private IP:     ${act_priv_ip:-(unset)}"
echo "  External IP:    ${act_ext_ip:-(not assigned)}"
echo ""
echo "Disk"
echo "  Boot disk:      ${act_disk:-(unset)}"
echo "  Auto-delete:    ${act_disk_autodel:-(unset)}"
echo "  Delete protect: ${act_protection:-(unset)}"
echo ""
echo "Service account:  ${act_sa:-(unset)}"

# ── identity checks ──────────────────────────────────────────────────────────

echo ""
echo "=== Checks ==="

[[ "$act_id" == "$VM_EXPECTED_ID" ]] && echo "  Instance ID:    OK" || echo "  Instance ID:    MISMATCH (expected $VM_EXPECTED_ID, got $act_id)"
[[ "$act_sa" == "$VM_SA" ]]          && echo "  Service account: OK"          || echo "  Service account: MISMATCH (expected $VM_SA, got $act_sa)"
[[ "$act_disk_autodel" != "True" ]] && echo "  Disk auto-del:  OK (false)"    || echo "  Disk auto-del:  RISK — auto-delete=true; delete would destroy disk"
[[ "$act_protection" == "True" ]]   && echo "  Instance protect: OK (true)"  || echo "  Instance protect: OFF — instance can be deleted"

# ── known-hosts readiness ────────────────────────────────────────────────────

echo ""
echo "SSH host-key readiness"
echo "  Known-hosts alias:   $VM_KNOWN_HOSTS_ALIAS"
if [[ "$act_state" == "RUNNING" ]]; then
  IP="$act_ext_ip"
  if [[ -z "$IP" || "$IP" == "(unset)" ]]; then
    echo "  External IP:         NOT AVAILABLE"
    echo "  Host-key status:     CANNOT CHECK — no external IP"
  else
    GHS="$HOME/.ssh/google_compute_known_hosts"
    if [[ -f "$GHS" ]] && rg -q "compute\\.${VM_EXPECTED_ID} " "$GHS"; then
      KEY_LINE=$(rg "compute\\.${VM_EXPECTED_ID} " "$GHS" | head -1)
      echo "  Host-key entry:      FOUND"
      echo "  Key:                 $KEY_LINE"
      echo "  (Use fingerprint comparison before connecting.)"
    else
      echo "  Host-key entry:      NOT FOUND in $GHS"
      echo "  Status:              HOST KEY PROVISIONING REQUIRED (manual/ceremony only)"
    fi
  fi
else
  GHS="$HOME/.ssh/google_compute_known_hosts"
  if [[ -f "$GHS" ]] && rg -q "compute\\.${VM_EXPECTED_ID} " "$GHS"; then
    echo "  Host-key entry:      PRESENT (waiting for VM start to verify)"
  else
    echo "  Host-key entry:      NOT FOUND — will need provisioning before first connection"
  fi
fi

# ── summary ─────────────────────────────────────────────────────────────────

echo ""
if [[ "$act_id" != "$VM_EXPECTED_ID" ]]; then
  echo "VERDICT: IDENTITY MISMATCH"
  exit 1
elif [[ "$act_disk_autodel" == "True" && "$act_protection" != "True" ]]; then
  echo "VERDICT: AT RISK (disk auto-delete=true, no instance protection)"
  exit 0
else
  echo "VERDICT: OK"
  exit 0
fi