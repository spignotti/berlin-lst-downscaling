#!/usr/bin/env bash
# Pure helpers that print POSIX sh text to run on the VM over ssh. No network, no gcloud.
# The VM app dir may be owned by a different uid than the OS Login user, so plain git
# refuses with "dubious ownership". Trust is granted per git process with
# `-c safe.directory=<exact realpath>`, never `*`. Nothing here writes ~/.gitconfig or
# /etc/gitconfig, edits any global git config, escalates privileges, or changes ownership
# or ACLs. The trust is applied only after the realpath/symlink/.git/owner checks pass.
# A manual ownership change stays a separate optional ops step (docs/vm-operations.md).
set -euo pipefail

vm_remote_git_preamble() {
  local dir="${1-}"
  if [[ ! "$dir" =~ ^/[A-Za-z0-9._/-]+$ || "$dir" == *..* || "$dir" == *//* || "$dir" == */ ]]; then
    echo "ERROR: unsafe app dir literal: $dir" >&2
    return 1
  fi
  local text
  text=$(
    cat <<'SH'
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_CEILING_DIRECTORIES
_vm_exp='__APP_DIR__'
_vm_fail() { echo "ERROR: remote app dir check failed: $*" >&2; exit 3; }
[ -d "$_vm_exp" ] || _vm_fail "missing or not a directory: $_vm_exp"
[ ! -L "$_vm_exp" ] || _vm_fail "is a symlink: $_vm_exp"
APP_REAL=$(cd "$_vm_exp" && pwd -P) || _vm_fail "cannot resolve $_vm_exp"
[ "$APP_REAL" = "$_vm_exp" ] || _vm_fail "realpath $APP_REAL differs from $_vm_exp"
{ [ -d "$APP_REAL/.git" ] && [ ! -L "$APP_REAL/.git" ]; } || _vm_fail ".git is not a real directory in $APP_REAL"
_vm_own=$(ls -ldn "$APP_REAL" | awk '{print $3}')
_vm_gown=$(ls -ldn "$APP_REAL/.git" | awk '{print $3}')
case "$_vm_own" in ''|*[!0-9]*) _vm_fail "cannot read owner of $APP_REAL";; esac
[ "$_vm_own" = "$_vm_gown" ] || _vm_fail "owner mismatch: $APP_REAL uid $_vm_own, .git uid $_vm_gown"
[ "$_vm_own" != 0 ] || _vm_fail "$APP_REAL is root-owned; refusing to trust it"
[ -w "$APP_REAL/.git" ] || _vm_fail ".git in $APP_REAL is not writable by $(id -un)"
rgit() { git -c "safe.directory=$APP_REAL" -C "$APP_REAL" "$@"; }
_vm_top=$(rgit rev-parse --show-toplevel) || _vm_fail "git rev-parse failed in $APP_REAL"
[ "$_vm_top" = "$APP_REAL" ] || _vm_fail "git toplevel $_vm_top != $APP_REAL"
_vm_gd=$(rgit rev-parse --absolute-git-dir) || _vm_fail "git rev-parse --absolute-git-dir failed"
[ "$_vm_gd" = "$APP_REAL/.git" ] || _vm_fail "git dir $_vm_gd != $APP_REAL/.git"
[ "$_vm_own" = "$(id -u)" ] || echo "NOTE: $APP_REAL is owned by uid $_vm_own (login uid $(id -u)); git trust is process-scoped to this path only." >&2
SH
  )
  printf '%s\n' "${text//__APP_DIR__/$dir}"
}

vm_remote_idle_probe_cmd() {
  local preamble head tail
  preamble=$(vm_remote_git_preamble "${1-}") || return 1
  head=$(
    cat <<'SH'
if pgrep -f '[p]ython scripts/(runners|operators)/' >/dev/null; then echo occupied; exit 0; fi
_vm_state=$( (
SH
  )
  tail=$(
    cat <<'SH'
_vm_st=$(rgit status --porcelain) || exit 4
if [ -n "$_vm_st" ]; then echo dirty; else echo idle; fi
) ) || _vm_state=broken
case "$_vm_state" in idle|dirty) ;; *) _vm_state=broken;; esac
echo "$_vm_state"
SH
  )
  printf '%s\n%s\n%s\n' "$head" "$preamble" "$tail"
}
