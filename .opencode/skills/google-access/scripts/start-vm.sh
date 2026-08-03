#!/usr/bin/env bash
# Start the berlin-lst-vm On-Demand VM.
#
# Fails closed if the VM is missing, has been replaced with a
# different instance, or the identity assertions fail.
# NEVER creates a new VM.
#
# Usage: scripts/start-vm.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    *)         echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/vm-identity.sh"

assert_vm_identity RUNNING TERMINATED STOPPED

case "$VM_ACTUAL_STATE" in
  RUNNING)
    echo "Already running."
    ;;
  TERMINATED|STOPPED)
    echo "Starting $VM_NAME (ID: $VM_INSTANCE_ID)..."
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "[dry-run] Would execute: gcloud compute instances start $VM_NAME --zone=$VM_ZONE --project=$VM_PROJECT"
      exit 0
    fi
    rtk gcloud compute instances start "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT"
    ;;
esac

echo ""
echo "Waiting for external IP..."
for i in {1..30}; do
  IP=$(vm_external_ip)
  if [[ -n "$IP" && "$IP" != "(unset)" ]]; then
    echo "External IP: $IP"
    echo "SSH with: scripts/ssh-vm.sh [-- command]"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for external IP."
exit 1