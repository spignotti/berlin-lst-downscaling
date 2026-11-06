#!/usr/bin/env bash
# Pure helpers for bootstrap-vm-hostkey.sh. No network, no gcloud.
# They never print private material (only public keys and fingerprints).
# shellcheck disable=SC2034
set -euo pipefail

HOSTKEY_ALLOWED_TYPES="ssh-ed25519 ecdsa-sha2-nistp256 ecdsa-sha2-nistp384 ecdsa-sha2-nistp521 ssh-rsa"

hostkey_new_nonce() {
  od -An -N8 -tx1 /dev/urandom | tr -d ' \n'
}

hostkey_startup_script() {
  local nonce="$1"
  cat <<SCRIPT
#!/bin/bash
# berlin-lst temporary host-key publisher (removed by bootstrap-vm-hostkey.sh)
for f in /etc/ssh/ssh_host_*_key.pub; do
  [ -f "\$f" ] || continue
  printf 'BERLIN-LST-HOSTKEY-${nonce} %s\n' "\$(cut -d' ' -f1,2 "\$f")" > /dev/ttyS0
done
printf 'BERLIN-LST-HOSTKEY-${nonce}-END\n' > /dev/ttyS0
SCRIPT
}

hostkey_parse_serial() {
  local nonce="$1"
  local token="BERLIN-LST-HOSTKEY-${nonce} "
  local end_token="BERLIN-LST-HOSTKEY-${nonce}-END"
  local types_re="${HOSTKEY_ALLOWED_TYPES// /|}"
  local re="^(${types_re}) ([A-Za-z0-9+/]+={0,3})$"
  local line payload seen_end=0 count=0 existing
  local -a keys=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ "$line" == *"$end_token"* ]]; then
      seen_end=1
      continue
    fi
    [[ "$line" == *"$token"* ]] || continue
    payload="${line#*"$token"}"
    if [[ ! "$payload" =~ $re ]]; then
      echo "ERROR: malformed host-key line on serial console" >&2
      return 2
    fi
    existing=0
    for k in "${keys[@]+"${keys[@]}"}"; do
      if [[ "$k" == "$payload" ]]; then existing=1; break; fi
    done
    if [[ $existing -eq 0 ]]; then
      keys+=("$payload")
      count=$((count + 1))
    fi
  done
  if [[ $seen_end -eq 0 ]]; then
    return 3
  fi
  if [[ $count -eq 0 ]]; then
    echo "ERROR: no host keys published before the end marker" >&2
    return 2
  fi
  printf '%s\n' "${keys[@]}"
}

hostkey_validate_keys() {
  local keysfile="$1"
  local alias_name="${2:-$VM_KNOWN_HOSTS_ALIAS}"
  local tmp type blob out rc
  tmp="$(mktemp)"
  while read -r type blob; do
    [[ -n "$type" ]] || continue
    printf '%s %s %s\n' "$alias_name" "$type" "$blob" > "$tmp"
    rc=0
    out="$(ssh-keygen -lf "$tmp" 2>/dev/null)" || rc=$?
    if [[ $rc -ne 0 ]]; then
      rm -f "$tmp"
      echo "ERROR: host key failed ssh-keygen validation" >&2
      return 2
    fi
    echo "  $type $(awk '{print $2}' <<<"$out")"
  done < "$keysfile"
  rm -f "$tmp"
}

_hostkey_fp() {
  # $1: alias, $2: type, $3: blob
  local tmp fp
  tmp="$(mktemp)"
  printf '%s %s %s\n' "$1" "$2" "$3" > "$tmp"
  fp="$(ssh-keygen -lf "$tmp" 2>/dev/null | awk '{print $2}')" || fp=""
  rm -f "$tmp"
  echo "${fp:-unknown}"
}

hostkey_merge_known_hosts() {
  local kh="$1" alias_name="$2" keysfile="$3"
  local dir created_dir=0
  dir="$(dirname "$kh")"
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
    chmod 700 "$dir"
    created_dir=1
  fi
  if [[ ! -e "$kh" ]]; then
    touch "$kh"
    chmod 600 "$kh"
  fi

  local etype eblob found ktype kblob fp_old fp_new
  local existing
  existing="$(ssh-keygen -F "$alias_name" -f "$kh" 2>/dev/null | grep -v '^#' || true)"
  if [[ -n "$existing" ]]; then
    while read -r _ etype eblob _; do
      [[ -n "$etype" ]] || continue
      found=0
      while read -r ktype kblob; do
        if [[ "$ktype" == "$etype" && "$kblob" == "$eblob" ]]; then found=1; break; fi
      done < "$keysfile"
      if [[ $found -eq 0 ]]; then
        fp_old="$(_hostkey_fp "$alias_name" "$etype" "$eblob")"
        fp_new="absent"
        while read -r ktype kblob; do
          if [[ "$ktype" == "$etype" ]]; then fp_new="$(_hostkey_fp "$alias_name" "$ktype" "$kblob")"; fi
        done < "$keysfile"
        echo "ERROR: changed host key for $alias_name ($etype): existing $fp_old vs serial $fp_new" >&2
        return 4
      fi
    done <<<"$existing"
  fi

  local -a new_lines=()
  local present
  while read -r ktype kblob; do
    [[ -n "$ktype" ]] || continue
    present=0
    if [[ -n "$existing" ]]; then
      while read -r _ etype eblob _; do
        if [[ "$ktype" == "$etype" && "$kblob" == "$eblob" ]]; then present=1; break; fi
      done <<<"$existing"
    fi
    if [[ $present -eq 0 ]]; then
      new_lines+=("$alias_name $ktype $kblob")
    fi
  done < "$keysfile"

  if [[ ${#new_lines[@]} -eq 0 ]]; then
    echo "known_hosts already pinned; no change"
    return 0
  fi
  if [[ -s "$kh" ]]; then
    local bak
    bak="$kh.bak.$(date -u +%Y%m%dT%H%M%SZ)"
    cp "$kh" "$bak"
    chmod 600 "$bak"
  fi
  # Keep a trailing newline invariant before appending.
  if [[ -s "$kh" && -n "$(tail -c1 "$kh")" ]]; then
    printf '\n' >> "$kh"
  fi
  printf '%s\n' "${new_lines[@]}" >> "$kh"
  echo "appended ${#new_lines[@]} key(s) to $kh"
  : "$created_dir"
}

hostkey_metadata_keys() {
  python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
for item in (d.get("metadata") or {}).get("items") or []:
    print(item["key"])
' "$1"
}

hostkey_metadata_value() {
  python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
for item in (d.get("metadata") or {}).get("items") or []:
    if item["key"] == sys.argv[2]:
        sys.stdout.write(item.get("value", ""))
        sys.exit(0)
sys.exit(1)
' "$1" "$2"
}
