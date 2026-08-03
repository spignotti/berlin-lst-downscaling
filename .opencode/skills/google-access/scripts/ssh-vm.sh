#!/usr/bin/env bash
# SSH into berlin-lst-vm using strict host-key verification.
#
# Connects with direct OpenSSH rather than `gcloud compute ssh`.
# Requires:
#   - VM identity matches pinned instance ID
#   - A host-key entry exists in google_compute_known_hosts for that ID
#   - An SSH identity file is readable at ~/.ssh/google_compute_engine
#   - The VM is RUNNING with an external IP
#
# NEVER writes known-hosts, generates keys, or provisions metadata keys.
#
# Usage: scripts/ssh-vm.sh [--check] [-- command-to-run]
#        scripts/ssh-vm.sh                 # interactive shell
#        scripts/ssh-vm.sh --check         # validate readiness, no connection

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/vm-identity.sh"

SSH_USER="silas"
SSH_PORT=22
SSH_TIMEOUT=10
SSH_IDENTITY_FILE="$HOME/.ssh/google_compute_engine"

# ── parse args ───────────────────────────────────────────────────────────────

CHECK_ONLY=false
SSH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   CHECK_ONLY=true; shift ;;
    --)        shift; SSH_ARGS=("$@"); break ;;
    -*)        echo "Unknown flag: $1" >&2; exit 1 ;;
    *)         SSH_ARGS=("$@"); break ;;
  esac
done

# ── identity & state ────────────────────────────────────────────────────────

# --check allows any state; connection requires RUNNING
if [[ "$CHECK_ONLY" == "true" ]]; then
  assert_vm_identity RUNNING TERMINATED STOPPED
else
  assert_vm_identity RUNNING
fi

IP=$(vm_external_ip)
if [[ "$CHECK_ONLY" == "true" ]]; then
  IP="${IP:-(not available — VM is $VM_ACTUAL_STATE)}"
elif [[ -z "$IP" || "$IP" == "(unset)" ]]; then
  echo "ERROR: VM is RUNNING but has no external IP." >&2
  exit 1
fi

# ── host-key check ──────────────────────────────────────────────────────────

GHS="$HOME/.ssh/google_compute_known_hosts"
if [[ ! -f "$GHS" ]]; then
  echo "ERROR: Known-hosts file not found: $GHS" >&2
  echo "  Run the host-key ceremony before attempting SSH." >&2
  exit 1
fi

# ssh-keygen -F works with both plain and hashed known-hosts entries
if ! ssh-keygen -F "$VM_KNOWN_HOSTS_ALIAS" -f "$GHS" >/dev/null 2>&1; then
  echo "ERROR: No host-key entry for alias '$VM_KNOWN_HOSTS_ALIAS'." >&2
  echo "  Expected an entry in: $GHS" >&2
  echo "  Run the host-key ceremony before attempting SSH." >&2
  exit 1
fi

# ── identity file check ─────────────────────────────────────────────────────

if [[ ! -r "$SSH_IDENTITY_FILE" ]]; then
  echo "ERROR: SSH identity file not readable: $SSH_IDENTITY_FILE" >&2
  echo "  Provide the key file at this path." >&2
  exit 1
fi

# ── --check: report readiness without connecting ─────────────────────────────

if [[ "$CHECK_ONLY" == "true" ]]; then
  KEY_LINE=$(ssh-keygen -F "$VM_KNOWN_HOSTS_ALIAS" -f "$GHS" 2>/dev/null | grep -v "^#" | head -1)
  echo "SSH readiness: OK"
  echo "  Instance:     $VM_NAME (ID: $VM_INSTANCE_ID)"
  echo "  State:        $VM_ACTUAL_STATE"
  echo "  IP:           $IP"
  echo "  User:         $SSH_USER"
  echo "  Identity:     $SSH_IDENTITY_FILE"
  echo "  Host key:     $KEY_LINE"
  exit 0
fi

# ── connect ─────────────────────────────────────────────────────────────────

exec ssh \
  -i "$SSH_IDENTITY_FILE" \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$GHS" \
  -o HostKeyAlias="$VM_KNOWN_HOSTS_ALIAS" \
  -o PasswordAuthentication=no \
  -o BatchMode=yes \
  -o ConnectTimeout="$SSH_TIMEOUT" \
  -p "$SSH_PORT" \
  "${SSH_USER}@${IP}" \
  "${SSH_ARGS[@]}"