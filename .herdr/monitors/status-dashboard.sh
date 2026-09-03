#!/usr/bin/env bash
# Dependency-free Herdr status dashboard.
#
# Usage:
#   status-monitor.sh --title <title> --checks <executable> [--interval <seconds>] [--once]
#
# The checks executable must emit TSV on stdout:
#   <label>\t<state>\t<value>\t<detail>
#
# State values: ok | warn | fail | unknown
#
# ONE CHECK per line. Lines not matching TSV are ignored.
# The dashboard invokes the checks executable once per refresh cycle.
set -euo pipefail

# ── arguments ────────────────────────────────────────────────────────
TITLE="Status"
CHECKS=""
INTERVAL=30
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)    TITLE="$2"; shift 2 ;;
    --checks)   CHECKS="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once)     ONCE=1; shift ;;
    *)          shift ;;
  esac
done

if [[ -z "$CHECKS" ]]; then
  echo "Usage: $0 --title <title> --checks <executable> [--interval <seconds>] [--once]" >&2
  exit 1
fi

if [[ ! -x "$CHECKS" ]]; then
  echo "Error: checks executable not found or not executable: $CHECKS" >&2
  exit 1
fi

# ── terminal setup ───────────────────────────────────────────────────
_DIM="" _BOLD="" _RESET="" _GREEN="" _YELLOW="" _RED="" _WHITE="" _CYAN=""
HAS_COLOR=0

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  colors=$(tput colors 2>/dev/null || echo 0)
  if [[ "$colors" -ge 8 ]]; then
    HAS_COLOR=1
    _BOLD=$(tput bold)
    _DIM=$(tput dim)
    _RESET=$(tput sgr0)
    _GREEN=$(tput setaf 2)
    _YELLOW=$(tput setaf 3)
    _RED=$(tput setaf 1)
    _WHITE=$(tput setaf 7)
    _CYAN=$(tput setaf 6)
  fi
fi

clear_screen() {
  if [[ "$HAS_COLOR" -eq 1 ]]; then
    tput home 2>/dev/null || printf '\033[H'
    tput ed 2>/dev/null || printf '\033[J'
  else
    printf '\033[2J\033[H'
  fi
}

# ── signal handling ──────────────────────────────────────────────────
cleanup() {
  if [[ "$HAS_COLOR" -eq 1 ]]; then
    tput sgr0 2>/dev/null || true
  fi
  exit 0
}
trap cleanup INT TERM

# ── sanitization ─────────────────────────────────────────────────────
sanitize() {
  # Strip control characters, ANSI escape sequences, and truncate to 80 chars
  local input="$1"
  input=$(printf '%s' "$input" | tr -d '\000-\037' | sed 's/\[[0-9;]*[a-zA-Z]//g')
  printf '%.80s' "$input"
}

# ── state rendering ──────────────────────────────────────────────────
render_indicator() {
  local state="$1"
  case "$state" in
    ok)      printf "${_GREEN}\xe2\x97\x8f${_RESET} OK" ;;   # ●
    warn)    printf "${_YELLOW}\xe2\x97\x8f${_RESET} WARN" ;; # ●
    fail)    printf "${_RED}\xe2\x97\x8f${_RESET} FAIL" ;;   # ●
    *)       printf "${_WHITE}\xe2\x97\x8c${_RESET} ?" ;;    # ◌
  esac
}

render_overall() {
  local worst="ok"
  local state
  while IFS=$'\t' read -r _label state _value _detail; do
    [[ -z "$state" ]] && continue
    case "$state" in
      fail)    worst="fail"; break ;;
      warn)
        if [[ "$worst" != "fail" ]]; then
          worst="warn"
        fi
        ;;
      ok)      [[ "$worst" == "ok" ]] && worst="ok" ;;
      *)       [[ "$worst" == "ok" ]] && worst="unknown" ;;
    esac
  done <<< "$CHECK_OUTPUT"

  case "$worst" in
    ok)      printf "${_GREEN}ALL OK${_RESET}" ;;
    warn)    printf "${_YELLOW}DEGRADED${_RESET}" ;;
    fail)    printf "${_RED}DOWN${_RESET}" ;;
    *)       printf "${_WHITE}UNKNOWN${_RESET}" ;;
  esac
}

render_rows() {
  local label state value detail
  local max_label=0

  # First pass: find max label width for alignment
  while IFS=$'\t' read -r label _state _value _detail; do
    [[ -z "$label" ]] && continue
    local len=${#label}
    (( len > max_label )) && max_label=$len
  done <<< "$CHECK_OUTPUT"

  # Second pass: render rows
  while IFS=$'\t' read -r label state value detail; do
    [[ -z "$label" ]] && continue
    label=$(sanitize "$label")
    value=$(sanitize "$value")
    detail=$(sanitize "$detail")
    printf "  "
    render_indicator "$state"
    printf "  %-${max_label}s" "$label"
    if [[ -n "$value" ]]; then
      if [[ "$HAS_COLOR" -eq 1 ]]; then
        printf "  ${_CYAN}%s${_RESET}" "$value"
      else
        printf "  %s" "$value"
      fi
    fi
    if [[ -n "$detail" ]]; then
      if [[ "$HAS_COLOR" -eq 1 ]]; then
        printf "  ${_DIM}%s${_RESET}" "$detail"
      else
        printf "  (%s)" "$detail"
      fi
    fi
    printf "\n"
  done <<< "$CHECK_OUTPUT"
}

render() {
  local now
  now=$(date '+%Y-%m-%d %H:%M:%S')

  # Header
  printf "\n"
  if [[ "$HAS_COLOR" -eq 1 ]]; then
    printf "  ${_BOLD}%s${_RESET}  " "$TITLE"
    if [[ -n "$CHECK_OUTPUT" ]]; then
      render_overall
    else
      printf "${_WHITE}UNKNOWN${_RESET}"
    fi
    printf "  ${_DIM}%s${_RESET}\n" "$now"
  else
    printf "  %s  %s  %s\n" "$TITLE" "$now"
  fi
  printf "  ──────────────────────────────────────────────\n"

  # Check rows
  if [[ -n "$CHECK_OUTPUT" ]]; then
    render_rows
  else
    printf "  %s\n" "(no checks returned output)"
  fi

  printf "\n"
  if [[ "$HAS_COLOR" -eq 1 ]]; then
    printf "  ${_DIM}refresh in %ds  |  ^C to quit${_RESET}\n" "$INTERVAL"
  else
    printf "  refresh in %ds  |  ^C to quit\n" "$INTERVAL"
  fi
}

# ── main loop ────────────────────────────────────────────────────────
CHECK_OUTPUT=""
CHECK_TIMEOUT=15

run_checks() {
  local raw
  # Run checks with a timeout to prevent hung processes from blocking the dashboard
  if command -v timeout >/dev/null 2>&1; then
    raw=$(timeout "$CHECK_TIMEOUT" "$CHECKS" 2>/dev/null) || true
  else
    # Portable fallback: run in background, kill after timeout
    local tmpfile
    tmpfile=$(mktemp)
    "$CHECKS" >"$tmpfile" 2>/dev/null &
    local pid=$!
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 1
      elapsed=$((elapsed + 1))
      if [[ "$elapsed" -ge "$CHECK_TIMEOUT" ]]; then
        kill "$pid" 2>/dev/null || true
        break
      fi
    done
    raw=$(cat "$tmpfile" 2>/dev/null || true)
    rm -f "$tmpfile"
  fi
  # Only keep lines that look like TSV (have at least one tab)
  CHECK_OUTPUT=$(printf '%s' "$raw" | grep -F $'\t') || true
}

while true; do
  run_checks
  clear_screen
  render
  [[ "$ONCE" -eq 1 ]] && exit 0
  sleep "$INTERVAL"
done
