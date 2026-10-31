#!/usr/bin/env bash
# Shared Infisical access for operator scripts. Source it; do not execute it.
#
# Auth precedence (first that applies wins):
#   1. INFISICAL_THESIS_CLIENT_ID + INFISICAL_THESIS_CLIENT_SECRET: Universal Auth
#      login for a short-lived token. The client secret reaches the CLI through
#      its environment, never argv, and the token is never exported or printed.
#   2. INFISICAL_TOKEN already set by the caller.
#   3. The caller's interactive `infisical login` session.
#
# Defaults (overridable): INFISICAL_DOMAIN=https://eu.infisical.com,
# INFISICAL_THESIS_PROJECT_ID=<thesis project>, INFISICAL_THESIS_ENV=dev.
#
# Functions:
#   infisical_thesis_init            check the CLI and authenticate
#   infisical_thesis_get NAME PATH   write the plain value of NAME to stdout

INFISICAL_THESIS_PROJECT_ID_DEFAULT="5da7dfb7-954d-4736-ba2e-4471ade9d766"

infisical_thesis_init() {
  export INFISICAL_DOMAIN="${INFISICAL_DOMAIN:-https://eu.infisical.com}"
  export INFISICAL_DISABLE_UPDATE_CHECK=true
  INFISICAL_THESIS_PROJECT_ID="${INFISICAL_THESIS_PROJECT_ID:-$INFISICAL_THESIS_PROJECT_ID_DEFAULT}"
  INFISICAL_THESIS_ENV="${INFISICAL_THESIS_ENV:-dev}"
  _INFISICAL_THESIS_TOKEN=""

  if ! command -v infisical >/dev/null 2>&1; then
    echo "ERROR: infisical CLI not found on PATH." >&2
    return 1
  fi

  local client_id="${INFISICAL_THESIS_CLIENT_ID:-}"
  local client_secret="${INFISICAL_THESIS_CLIENT_SECRET:-}"
  if [[ -n "$client_id" && -z "$client_secret" ]]; then
    echo "ERROR: INFISICAL_THESIS_CLIENT_ID is set but INFISICAL_THESIS_CLIENT_SECRET is missing." >&2
    return 1
  fi
  if [[ -z "$client_id" && -n "$client_secret" ]]; then
    echo "ERROR: INFISICAL_THESIS_CLIENT_SECRET is set but INFISICAL_THESIS_CLIENT_ID is missing." >&2
    return 1
  fi

  if [[ -n "$client_id" ]]; then
    INFISICAL_THESIS_AUTH_MODE="universal-auth"
    if ! _INFISICAL_THESIS_TOKEN="$(
      INFISICAL_UNIVERSAL_AUTH_CLIENT_ID="$client_id" \
        INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET="$client_secret" \
        infisical login --method=universal-auth --plain --silent </dev/null
    )" || [[ -z "$_INFISICAL_THESIS_TOKEN" ]]; then
      _INFISICAL_THESIS_TOKEN=""
      echo "ERROR: Infisical Universal Auth login failed ($INFISICAL_DOMAIN)." >&2
      echo "  Check INFISICAL_THESIS_CLIENT_ID / INFISICAL_THESIS_CLIENT_SECRET." >&2
      return 1
    fi
  elif [[ -n "${INFISICAL_TOKEN:-}" ]]; then
    INFISICAL_THESIS_AUTH_MODE="INFISICAL_TOKEN"
  else
    INFISICAL_THESIS_AUTH_MODE="infisical login session"
  fi
}

infisical_thesis_get() {
  local name="$1" path="$2"
  local args=(secrets get "$name" --projectId="$INFISICAL_THESIS_PROJECT_ID"
    --env="$INFISICAL_THESIS_ENV" --path="$path" --plain --silent)
  local rc=0
  if [[ -n "${_INFISICAL_THESIS_TOKEN:-}" ]]; then
    INFISICAL_TOKEN="$_INFISICAL_THESIS_TOKEN" infisical "${args[@]}" </dev/null || rc=$?
  else
    infisical "${args[@]}" </dev/null || rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: could not read $name from Infisical (env $INFISICAL_THESIS_ENV, path $path," >&2
    echo "  auth: $INFISICAL_THESIS_AUTH_MODE, domain: $INFISICAL_DOMAIN). Authenticate with one of:" >&2
    echo "  - INFISICAL_THESIS_CLIENT_ID and INFISICAL_THESIS_CLIENT_SECRET (Universal Auth)" >&2
    echo "  - INFISICAL_TOKEN" >&2
    echo "  - infisical login --domain https://eu.infisical.com" >&2
    return 1
  fi
}
