#!/usr/bin/env bash
# Start the berlin-lst-vm On-Demand VM (creates it if it does not exist).
#
# Lifecycle: start → run pipeline → stop. Disk is retained between runs.
# Uses n2-highmem-2 (16 GB RAM) with a 50-GB pd-balanced boot disk.
#
# Usage: scripts/start-vm.sh

set -euo pipefail

ZONE="europe-west3-a"
NAME="berlin-lst-vm"
PROJECT="masterarbeit-berlin-lst-v2"
SA="masterarbeit-vertex@${PROJECT}.iam.gserviceaccount.com"
MACHINE="n2-highmem-2"
DISK_SIZE="50"
DISK_TYPE="pd-balanced"

status() { rtk gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" --format='get(status)' 2>/dev/null || echo "MISSING"; }

CUR=$(status)
echo "Current state: $CUR"

case "$CUR" in
  RUNNING)
    echo "Already running." ;;
  TERMINATED|STOPPED)
    echo "Starting..."
    rtk gcloud compute instances start "$NAME" --zone="$ZONE" --project="$PROJECT"
    ;;
  MISSING)
    echo "VM does not exist. Creating On-Demand ${MACHINE}..."
    rtk gcloud compute instances create "$NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --image-family=debian-12 --image-project=debian-cloud \
      --boot-disk-size="$DISK_SIZE" --boot-disk-type="$DISK_TYPE" \
      --service-account="$SA" \
      --scopes=cloud-platform \
      --no-autodelete-boot-disk \
      --labels=purpose=berlin-lst-runner,owner=silas
    ;;
  *)
    echo "Unexpected state: $CUR — exiting."
    exit 1 ;;
esac

echo
echo "Waiting for external IP..."
for i in {1..30}; do
  IP=$(rtk gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)
  if [[ -n "$IP" && "$IP" != "(unset)" ]]; then
    echo "External IP: $IP"
    echo "SSH with: scripts/ssh-vm.sh"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for external IP."
exit 1
