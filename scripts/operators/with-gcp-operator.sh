#!/usr/bin/env bash
# Run one command as the operator service account, with its key from Infisical.
#
# Usage:
#   scripts/operators/with-gcp-operator.sh [--activate] -- <command> [args...]
#   scripts/operators/with-gcp-operator.sh --check
#
# Fetches GCP_OPERATOR_SA_KEY (thesis project, env dev, folder /operator) into a
# 0600 temp file, checks that it is a service-account key, and runs <command>
# with GOOGLE_APPLICATION_CREDENTIALS and CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE
# pointing at it, so gcloud and the Google client libraries use the operator SA
# for this command only; the caller's gcloud config is not changed.
#
#   --activate  instead of the gcloud override, run
#               `gcloud auth activate-service-account --key-file` once. This is a
#               persistent gcloud login: gcloud keeps its own copy of the
#               credential after the temp file is gone. The command is optional.
#   --check     fetch and validate the key, print only the SA client_email.
#
# The temp file is removed on every exit (success, failure, INT, TERM), after the
# command has finished. The command's exit code is propagated. Infisical auth:
# see _infisical_auth.sh.

set +x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/operators/_infisical_auth.sh
source "$SCRIPT_DIR/_infisical_auth.sh"

SECRET_NAME="GCP_OPERATOR_SA_KEY"
SECRET_PATH="/operator"

usage() {
  sed -n '4,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
}

ACTIVATE=false
CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --activate) ACTIVATE=true; shift ;;
    --check)    CHECK_ONLY=true; shift ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; break ;;
    *)          echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$CHECK_ONLY" == "false" && "$ACTIVATE" == "false" && $# -eq 0 ]]; then
  echo "ERROR: no command given." >&2
  usage
  exit 2
fi

KEY_FILE=""
cleanup() {
  if [[ -n "$KEY_FILE" && -e "$KEY_FILE" ]]; then
    if command -v shred >/dev/null 2>&1; then
      shred -u "$KEY_FILE" 2>/dev/null || rm -f "$KEY_FILE"
    else
      rm -f "$KEY_FILE"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to validate the key." >&2
  exit 1
fi

infisical_thesis_init

KEY_FILE="$(umask 077 && mktemp "${TMPDIR:-/tmp}/gcp-operator-key.XXXXXX")"
chmod 600 "$KEY_FILE"

infisical_thesis_get "$SECRET_NAME" "$SECRET_PATH" >"$KEY_FILE"

if [[ ! -s "$KEY_FILE" ]]; then
  echo "ERROR: Infisical returned an empty $SECRET_NAME." >&2
  exit 1
fi

# Prints only client_email; never the key or a parse error that could quote it.
if ! CLIENT_EMAIL="$(
  python3 -I -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        key = json.load(fh)
except Exception:
    sys.exit(1)
if not isinstance(key, dict) or key.get("type") != "service_account":
    sys.exit(1)
email = key.get("client_email")
if not isinstance(email, str) or not email:
    sys.exit(1)
print(email)
' "$KEY_FILE" 2>/dev/null
)"; then
  echo "ERROR: $SECRET_NAME is not a service-account JSON key (content not shown)." >&2
  exit 1
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "$CLIENT_EMAIL"
  exit 0
fi

export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
if [[ "$ACTIVATE" == "true" ]]; then
  gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
  echo "gcloud: activated $CLIENT_EMAIL (persistent gcloud login)." >&2
  [[ $# -eq 0 ]] && exit 0
else
  export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$KEY_FILE"
fi

rc=0
"$@" || rc=$?
exit "$rc"
