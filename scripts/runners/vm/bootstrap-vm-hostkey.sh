#!/usr/bin/env bash
# Box VM host-key bootstrap (serial-console trust path).
#
# Usage: scripts/operators/with-gcp-operator.sh -- \
#          scripts/runners/vm/bootstrap-vm-hostkey.sh [--execute] [--help]
#
# Box-only operator helper. The default is a dry-run plan that runs read-only
# `describe` calls only: no `instances start`, no metadata write, no serial
# read, no SSH.
#
# --execute exists only for a later human Run OK; it is never the CI/default
# path. It starts a TERMINATED/STOPPED VM (cost: VM-hours at the verified
# hourly rate) and leaves it RUNNING for `ssh-vm.sh --check`. The human stops
# it with stop-vm.sh.
#
# Trust basis: the authenticated GCP API (serial port output). This helper
# never scans host keys over the network, relaxes host-key checking, or trusts on first connect,
# and it never opens a network SSH connection.
#
# Metadata policy: an existing `startup-script` is backed up to a 0600 local
# file and restored afterwards. If `startup-script-url` exists the helper
# refuses (it would also run at boot). Cleanup runs on success and on failure
# (EXIT trap). After cleanup the metadata is re-read and must equal the
# pre-run state; on mismatch the helper fails hard and keeps the backup path.
#
# Output: statuses and fingerprints only. Full serial output is never written
# to disk or stdout.
#
# It does not generate or register OS Login keys. BERLIN_LST_SSH_IDENTITY_FILE
# applies only to the later `ssh-vm.sh --check`.
#
# Exit codes: 0 ok, 1 usage/pin/refusal, 2 malformed key, 3 metadata restore
# not verified, 4 changed host key, 5 serial timeout.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=vm-identity.sh
source "$SCRIPTS_DIR/vm-identity.sh"
# shellcheck source=hostkey-lib.sh
source "$SCRIPTS_DIR/hostkey-lib.sh"

EXECUTE=false
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=true ;;
    --help|-h) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

GHS="$HOME/.ssh/google_compute_known_hosts"
POLL_INTERVAL="${BERLIN_LST_HOSTKEY_POLL_INTERVAL:-10}"
POLL_TRIES="${BERLIN_LST_HOSTKEY_POLL_TRIES:-30}"
if [[ ! "$POLL_INTERVAL" =~ ^[0-9]+$ || ! "$POLL_TRIES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: poll interval and tries must be non-negative integers." >&2
  exit 1
fi

assert_vm_identity RUNNING TERMINATED STOPPED

WORK="$(mktemp -d)"
chmod 700 "$WORK"
trap 'rm -rf "$WORK"' EXIT

gcloud compute instances describe "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" \
  --format=json > "$WORK/meta-before.json"
KEYS="$(hostkey_metadata_keys "$WORK/meta-before.json")"
if grep -qx startup-script-url <<<"$KEYS"; then
  echo "ERROR: instance has startup-script-url metadata; it would also run at boot. Refusing." >&2
  exit 1
fi
HAS_STARTUP=0
if grep -qx startup-script <<<"$KEYS"; then HAS_STARTUP=1; fi

ALIAS_STATE="absent"
if [[ -f "$GHS" ]] && [[ -n "$(ssh-keygen -F "$VM_KNOWN_HOSTS_ALIAS" -f "$GHS" 2>/dev/null)" ]]; then
  ALIAS_STATE="present"
fi

if [[ "$EXECUTE" != "true" ]]; then
  STARTUP_NOTE="no"
  if [[ $HAS_STARTUP -eq 1 ]]; then STARTUP_NOTE="yes (will be backed up and restored)"; fi
  cat <<PLAN
[dry-run] Host-key bootstrap plan for $VM_NAME (ID $VM_EXPECTED_ID, $VM_ZONE, $VM_PROJECT)
  State: $VM_ACTUAL_STATE; existing startup-script: $STARTUP_NOTE
  known_hosts alias $VM_KNOWN_HOSTS_ALIAS: $ALIAS_STATE  ($GHS)
  1. add temporary startup-script (prints /etc/ssh/ssh_host_*_key.pub to serial)
  2. gcloud compute instances start (VM must be TERMINATED/STOPPED)
  3. read serial port 1 via gcloud get-serial-port-output; parse nonce-tagged public keys only
  4. remove temporary startup-script; restore backup if any; verify metadata
  5. validate keys with ssh-keygen; pin into $GHS as $VM_KNOWN_HOSTS_ALIAS; fail on changed key
  6. leave VM RUNNING; next: ssh-vm.sh --check, then stop-vm.sh
Re-run with --execute only after a human Run OK.
PLAN
  exit 0
fi

if [[ "$VM_ACTUAL_STATE" == "RUNNING" ]]; then
  echo "ERROR: VM is RUNNING; startup-script only runs at boot. Stop it first (stop-vm.sh) or follow the manual ceremony." >&2
  exit 1
fi

PERSIST_BACKUP=""
if [[ $HAS_STARTUP -eq 1 ]]; then
  hostkey_metadata_value "$WORK/meta-before.json" startup-script > "$WORK/startup-script.backup"
  STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/berlin-lst/hostkey-bootstrap"
  (umask 077; mkdir -p "$STATE_DIR")
  chmod 700 "$STATE_DIR"
  PERSIST_BACKUP="$STATE_DIR/startup-script.$(date -u +%Y%m%dT%H%M%SZ).bak"
  (umask 077; cp "$WORK/startup-script.backup" "$PERSIST_BACKUP")
  chmod 600 "$PERSIST_BACKUP"
  echo "existing startup-script backed up to $PERSIST_BACKUP"
fi

NONCE="$(hostkey_new_nonce)"
hostkey_startup_script "$NONCE" > "$WORK/startup.sh"

METADATA_WRITTEN=0
VM_STARTED=0

# Restore and verify metadata. Returns non-zero when not verified.
hostkey_restore_metadata() {
  local ok=1
  if [[ $HAS_STARTUP -eq 1 ]]; then
    gcloud compute instances add-metadata "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" \
      --metadata-from-file=startup-script="$WORK/startup-script.backup" >/dev/null 2>&1 || ok=0
  else
    gcloud compute instances remove-metadata "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" \
      --keys=startup-script >/dev/null 2>&1 || ok=0
  fi
  if [[ $ok -eq 1 ]]; then
    gcloud compute instances describe "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" \
      --format=json > "$WORK/meta-after.json" 2>/dev/null || ok=0
  fi
  if [[ $ok -eq 1 ]]; then
    local after_keys
    after_keys="$(hostkey_metadata_keys "$WORK/meta-after.json")" || ok=0
    if [[ $HAS_STARTUP -eq 1 ]]; then
      hostkey_metadata_value "$WORK/meta-after.json" startup-script > "$WORK/startup-script.after" 2>/dev/null || ok=0
      if [[ $ok -eq 1 ]] && ! cmp -s "$WORK/startup-script.after" "$WORK/startup-script.backup"; then ok=0; fi
    elif grep -qx startup-script <<<"$after_keys"; then
      ok=0
    fi
  fi
  if [[ $ok -ne 1 ]]; then
    echo "ERROR: metadata restore NOT verified; backup kept at ${PERSIST_BACKUP:-<none: no prior startup-script>}" >&2
    return 1
  fi
  if [[ $HAS_STARTUP -eq 1 ]]; then
    echo "metadata restored: startup-script restored"
    rm -f "$PERSIST_BACKUP"
  else
    echo "metadata restored: startup-script removed"
  fi
}

hostkey_cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ $METADATA_WRITTEN -eq 1 ]]; then
    if hostkey_restore_metadata; then
      METADATA_WRITTEN=0
    else
      rc=3
    fi
  fi
  if [[ $VM_STARTED -eq 1 ]]; then
    echo "VM left RUNNING (fail-closed); inspect then run scripts/runners/vm/stop-vm.sh"
  fi
  rm -rf "$WORK"
  exit "$rc"
}
trap hostkey_cleanup EXIT

# Set before the call: a client-side failure after a server-side apply must still restore.
METADATA_WRITTEN=1
gcloud compute instances add-metadata "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" \
  --metadata-from-file=startup-script="$WORK/startup.sh" >/dev/null

gcloud compute instances start "$VM_NAME" --zone="$VM_ZONE" --project="$VM_PROJECT" >/dev/null
VM_STARTED=1

FOUND=0
for ((i = 1; i <= POLL_TRIES; i++)); do
  rc=0
  set +e
  gcloud compute instances get-serial-port-output "$VM_NAME" --zone="$VM_ZONE" \
    --project="$VM_PROJECT" --port=1 2>/dev/null | hostkey_parse_serial "$NONCE" > "$WORK/keys.txt"
  ps=("${PIPESTATUS[@]}")
  gc_rc=${ps[0]}
  rc=${ps[1]}
  set -e
  if [[ $rc -eq 0 && $gc_rc -eq 0 ]]; then FOUND=1; break; fi
  if [[ $rc -eq 2 ]]; then exit 2; fi
  sleep "$POLL_INTERVAL"
done
if [[ $FOUND -ne 1 ]]; then
  echo "ERROR: host keys not seen on serial console after $POLL_TRIES polls" >&2
  exit 5
fi

if hostkey_restore_metadata; then
  METADATA_WRITTEN=0
else
  METADATA_WRITTEN=0
  exit 3
fi

hostkey_validate_keys "$WORK/keys.txt"
hostkey_merge_known_hosts "$GHS" "$VM_KNOWN_HOSTS_ALIAS" "$WORK/keys.txt"

echo "OK: host key pinned for $VM_KNOWN_HOSTS_ALIAS. Next (human): ssh-vm.sh --check; optional read-only login; stop-vm.sh"
