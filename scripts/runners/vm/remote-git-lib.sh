#!/usr/bin/env bash
# Pure helpers that print POSIX sh text to run on the VM over ssh. No network, no gcloud.
# Git's ownership protection stays fully on: the app dir is owned by the login uid of the
# SSH session (id -u), pinned from the OS Login profile name and an optional uid pin, and
# git runs with no trust override and no git config change at all. The identity and
# owner checks run before any git or uv call. Nothing here touches the legacy workspace
# app dir, escalates privileges, or changes ownership or ACLs.
set -euo pipefail

# Validate <home> <user> <origin> <pin_uid>; print an error and return 1 on any violation.
_vm_lib_check_args() {
  local home="${1-}" user="${2-}" origin="${3-}" pin="${4-}"
  if [[ ! "$home" =~ ^/[A-Za-z0-9._/-]+$ || "$home" == *..* || "$home" == *//* || "$home" == */ ]]; then
    echo "ERROR: unsafe home literal: $home" >&2
    return 1
  fi
  if [[ ! "$user" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: unsafe user literal: $user" >&2
    return 1
  fi
  if [[ -z "$origin" || ! "$origin" =~ ^[A-Za-z0-9._/:@+-]+$ ]]; then
    echo "ERROR: unsafe origin literal: $origin" >&2
    return 1
  fi
  if [[ -n "$pin" && ! "$pin" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: unsafe uid pin literal: $pin" >&2
    return 1
  fi
}

# Shared remote text: identity, owner and layout checks. Never mutates anything. Exit 3 on failure.
vm_remote_app_preamble() {
  _vm_lib_check_args "${1-}" "${2-}" "${3-}" "${4-}" || return 1
  local text
  text=$(
    cat <<'SH'
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CEILING_DIRECTORIES GIT_CONFIG_PARAMETERS GIT_CONFIG_COUNT GIT_DISCOVERY_ACROSS_FILESYSTEM
_vm_home='__HOME__'
_vm_user='__USER__'
_vm_origin='__ORIGIN__'
_vm_pin='__PIN__'
APP_DIR="$_vm_home/berlin-lst-app"
_vm_fail() { echo "ERROR: remote app dir check failed: $*" >&2; exit 3; }
_vm_uid=$(id -u) || _vm_fail "id -u failed"
case "$_vm_uid" in ''|*[!0-9]*) _vm_fail "login uid not numeric: $_vm_uid";; esac
[ "$_vm_uid" != 0 ] || _vm_fail "login uid is 0; refusing to run as root"
[ "$(id -un)" = "$_vm_user" ] || _vm_fail "login user $(id -un) != expected $_vm_user"
if [ -n "$_vm_pin" ]; then [ "$_vm_pin" = "$_vm_uid" ] || _vm_fail "login uid $_vm_uid != pinned uid $_vm_pin"; fi
_vm_owner() { stat -c %u -- "$1"; }
_vm_own_dir() {
  [ -d "$1" ] || _vm_fail "$2 is not a directory: $1"
  [ ! -L "$1" ] || _vm_fail "$2 is a symlink: $1"
  _vm_o=$(_vm_owner "$1") || _vm_fail "cannot read owner of $2: $1"
  case "$_vm_o" in ''|*[!0-9]*) _vm_fail "owner of $2 not numeric: $1";; esac
  [ "$_vm_o" = "$_vm_uid" ] || _vm_fail "$2 $1 owned by uid $_vm_o, expected login uid $_vm_uid"
  [ -w "$1" ] || _vm_fail "$2 $1 not writable by uid $_vm_uid"
}
[ "${HOME-}" = "$_vm_home" ] || _vm_fail "HOME '${HOME-}' != expected $_vm_home"
_vm_own_dir "$_vm_home" home
[ "$(cd "$_vm_home" && pwd -P)" = "$_vm_home" ] || _vm_fail "realpath of $_vm_home differs (symlinked parent)"
rgit() { git -C "$APP_DIR" "$@"; }
_vm_check_app() {
  _vm_own_dir "$APP_DIR" "app dir"
  [ "$(cd "$APP_DIR" && pwd -P)" = "$APP_DIR" ] || _vm_fail "realpath of $APP_DIR differs"
  _vm_own_dir "$APP_DIR/.git" .git
  for _vm_sub in .venv data logs; do
    if [ -e "$APP_DIR/$_vm_sub" ] || [ -L "$APP_DIR/$_vm_sub" ]; then _vm_own_dir "$APP_DIR/$_vm_sub" "$_vm_sub"; fi
  done
  _vm_top=$(rgit rev-parse --show-toplevel) || _vm_fail "git rev-parse --show-toplevel failed"
  [ "$_vm_top" = "$APP_DIR" ] || _vm_fail "git toplevel $_vm_top != $APP_DIR"
  _vm_gd=$(rgit rev-parse --absolute-git-dir) || _vm_fail "git rev-parse --absolute-git-dir failed"
  [ "$_vm_gd" = "$APP_DIR/.git" ] || _vm_fail "git dir $_vm_gd != $APP_DIR/.git"
  _vm_url=$(rgit remote get-url origin) || _vm_fail "git remote get-url origin failed"
  [ "$_vm_url" = "$_vm_origin" ] || _vm_fail "origin $_vm_url != pinned $_vm_origin"
}
if [ -e "$APP_DIR" ] || [ -L "$APP_DIR" ]; then APP_STATE=present; _vm_check_app; else APP_STATE=absent; fi
SH
  )
  text="${text//__HOME__/$1}"
  text="${text//__USER__/$2}"
  text="${text//__ORIGIN__/$3}"
  text="${text//__PIN__/${4-}}"
  printf '%s\n' "$text"
}

vm_remote_idle_probe_cmd() {
  local preamble head tail
  preamble=$(vm_remote_app_preamble "${1-}" "${2-}" "${3-}" "${4-}") || return 1
  head=$(
    cat <<'SH'
if pgrep -f '[p]ython scripts/(runners|operators)/' >/dev/null; then echo occupied; exit 0; fi
_vm_state=$( (
SH
  )
  tail=$(
    cat <<'SH'
if [ "$APP_STATE" = absent ]; then echo absent; exit 0; fi
_st=$(rgit status --porcelain) || exit 4
if [ -n "$_st" ]; then echo dirty; else echo idle; fi
) ) || _vm_state=broken
case "$_vm_state" in idle|dirty|absent) ;; *) _vm_state=broken;; esac
echo "$_vm_state"
SH
  )
  printf '%s\n%s\n%s\n' "$head" "$preamble" "$tail"
}

vm_remote_deploy_cmd() {
  local branch="${5-}" sha="${6-}"
  _vm_lib_check_args "${1-}" "${2-}" "${3-}" "${4-}" || return 1
  if [[ ! "$branch" =~ ^[A-Za-z0-9._/-]+$ || "$branch" == -* ]]; then
    echo "ERROR: unsafe branch literal: $branch" >&2
    return 1
  fi
  if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: unsafe sha literal: $sha" >&2
    return 1
  fi
  local preamble body
  preamble=$(vm_remote_app_preamble "$1" "$2" "$3" "$4") || return 1
  body=$(
    cat <<'SH'
_vm_branch='__BRANCH__'
_vm_sha='__SHA__'
umask 077
command -v uv >/dev/null 2>&1 || _vm_fail "uv not on PATH for $(id -un)"
uv --version >/dev/null 2>&1 || _vm_fail "uv --version failed"
if [ "$APP_STATE" = absent ]; then
  _vm_tmp=$(mktemp -d "$_vm_home/.berlin-lst-app.clone.XXXXXX") || _vm_fail mktemp
  git clone --quiet -- "$_vm_origin" "$_vm_tmp/app" || { rm -rf -- "$_vm_tmp"; _vm_fail "clone failed"; }
  mv -T -- "$_vm_tmp/app" "$APP_DIR" || { rm -rf -- "$_vm_tmp"; _vm_fail "mv failed"; }
  rmdir -- "$_vm_tmp"
  _vm_check_app
fi
_vm_dirty_guard() {
  _vm_st=$(rgit status --porcelain) || _vm_fail "git status failed"
  if [ -n "$_vm_st" ]; then
    printf '%s\n' "$_vm_st"
    echo "ERROR: $APP_DIR has uncommitted or untracked changes; refusing checkout/reset" >&2
    exit 6
  fi
}
_vm_dirty_guard
rgit fetch --quiet origin "$_vm_branch:refs/remotes/origin/$_vm_branch" || _vm_fail "git fetch failed"
rgit checkout -q -B "$_vm_branch" "refs/remotes/origin/$_vm_branch" || _vm_fail "git checkout failed"
rgit reset -q --hard "refs/remotes/origin/$_vm_branch" || _vm_fail "git reset failed"
[ "$(rgit rev-parse HEAD)" = "$_vm_sha" ] || _vm_fail "HEAD != pinned sha"
cd "$APP_DIR" && uv sync --frozen --quiet || _vm_fail "uv sync failed"
_vm_dirty_guard
SH
  )
  body="${body//__BRANCH__/$branch}"
  body="${body//__SHA__/$sha}"
  printf '%s\n%s\n' "$preamble" "$body"
}

vm_remote_provenance_cmd() {
  local preamble tail
  preamble=$(vm_remote_app_preamble "${1-}" "${2-}" "${3-}" "${4-}") || return 1
  tail=$(
    cat <<'SH'
[ "$APP_STATE" = present ] || { echo "ERROR: app dir missing" >&2; exit 3; }
cd "$APP_DIR"
_h=$(git rev-parse HEAD) || exit 5
_s=$(git status --porcelain) || exit 5
[ -z "$_s" ] || exit 5
echo "$_h"
SH
  )
  printf '%s\n%s\n' "$preamble" "$tail"
}

vm_remote_app_cd_cmd() {
  local preamble tail
  preamble=$(vm_remote_app_preamble "${1-}" "${2-}" "${3-}" "${4-}") || return 1
  tail=$(
    cat <<'SH'
[ "$APP_STATE" = present ] || { echo "ERROR: app dir missing; deploy first" >&2; exit 3; }
cd "$APP_DIR"
SH
  )
  printf '%s\n%s\n' "$preamble" "$tail"
}
