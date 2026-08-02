#!/usr/bin/env bash
# Stop the berlin-lst-vm On-Demand VM (keeps boot disk; resumable).
#
# Lifecycle: start → run pipeline → stop. Disk is retained between runs.
# Cost while stopped: ~$1/month for 50-GB pd-balanced.
#
# Usage: scripts/stop-vm.sh

set -euo pipefail

ZONE="europe-west3-a"
NAME="berlin-lst-vm"
PROJECT="masterarbeit-berlin-lst-v2"

CUR=$(rtk gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" --format='get(status)' 2>/dev/null || echo "MISSING")
echo "Current state: $CUR"

case "$CUR" in
  RUNNING)
    echo "Stopping..."
    rtk gcloud compute instances stop "$NAME" --zone="$ZONE" --project="$PROJECT"
    ;;
  TERMINATED|STOPPED)
    echo "Already stopped." ;;
  MISSING)
    echo "VM does not exist. Use scripts/start-vm.sh to create it." ;;
  *)
    echo "Unexpected state: $CUR — exiting."
    exit 1 ;;
esac
