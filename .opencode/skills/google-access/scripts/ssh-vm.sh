#!/usr/bin/env bash
# SSH into the berlin-lst-vm VM. Forwards any args to gcloud compute ssh.
#
# Usage: scripts/ssh-vm.sh [-- command-to-run]
#        scripts/ssh-vm.sh                 # interactive shell

set -euo pipefail

ZONE="europe-west3-a"
NAME="berlin-lst-vm"
PROJECT="masterarbeit-berlin-lst-v2"

exec rtk gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" "$@"