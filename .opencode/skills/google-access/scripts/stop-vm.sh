#!/usr/bin/env bash
# Stop the berlin-lst-vm On-Demand VM (keeps boot disk; resumable).
#
# Fails closed if the pinned identity / disk does not match.
#
# Usage: scripts/stop-vm.sh [--dry-run]

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
    echo "Stopping $VM_NAME (ID: $VM_INSTANCE_ID)..."
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "[dry-run] Would execute: gcloud compute instances stop $VM_NAME --zone=$VM_ZONE --project=$VM_PROJECT"
    else
      rtk gcloud compute instances stop "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT"
    fi
    ;;
  TERMINATED|STOPPED)
    echo "Already stopped."
    ;;
esac