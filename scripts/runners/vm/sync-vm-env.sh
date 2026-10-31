#!/usr/bin/env bash
# Write the VM app .env from Infisical (thesis project, env dev, folder /data).
#
# Fetches EARTHDATA_TOKEN, EARTHDATA_USERNAME and EARTHDATA_PASSWORD and replaces
# /workspace/app/.env on berlin-lst-vm with exactly these names (mode 0600). The
# content travels over ssh-vm.sh stdin only: values never appear on argv, in the
# terminal, or in logs. Infisical auth: see scripts/operators/_infisical_auth.sh.
# gcloud must be able to describe the VM (wrap in with-gcp-operator.sh if needed).
#
# Usage: scripts/runners/vm/sync-vm-env.sh [--check]
#   --check   list the names that would be written; no fetch, no connection.

set +x
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/operators/_infisical_auth.sh
source "$SCRIPTS_DIR/../../operators/_infisical_auth.sh"

SECRET_PATH="/data"
SECRET_NAMES=(EARTHDATA_TOKEN EARTHDATA_USERNAME EARTHDATA_PASSWORD)
APP_DIR="/workspace/app"  # same as vm-runner-common.sh

case "${1:-}" in
  "") CHECK_ONLY=false ;;
  --check) CHECK_ONLY=true ;;
  *) echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "Would write $APP_DIR/.env (mode 0600) on the VM from Infisical env ${INFISICAL_THESIS_ENV:-dev}, path $SECRET_PATH:"
  printf '  %s\n' "${SECRET_NAMES[@]}"
  exit 0
fi

WORK_DIR=""
cleanup() {
  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    if command -v shred >/dev/null 2>&1; then
      find "$WORK_DIR" -type f -exec shred -u {} + 2>/dev/null || true
    fi
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

infisical_thesis_init

WORK_DIR="$(umask 077 && mktemp -d "${TMPDIR:-/tmp}/sync-vm-env.XXXXXX")"
ENV_FILE="$WORK_DIR/env"
(umask 077 && : >"$ENV_FILE")

for name in "${SECRET_NAMES[@]}"; do
  value_file="$WORK_DIR/$name"
  infisical_thesis_get "$name" "$SECRET_PATH" >"$value_file"
  # --plain ends the value with one newline. Values are written single-quoted
  # (literal for the dotenv reader), so a value must be one line without `'`.
  value="$(cat "$value_file"; printf x)"
  value="${value%x}"
  value="${value%$'\n'}"
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *"'"* ]]; then
    echo "ERROR: $name from Infisical is empty, spans several lines, or contains a single quote (value not shown)." >&2
    exit 1
  fi
  printf "%s='%s'\n" "$name" "$value" >>"$ENV_FILE"
  unset value
done

# One remote string; ssh passes it to the VM's login shell.
REMOTE_CMD="set -eu; umask 077; cd $APP_DIR; tmp=\$(mktemp .env.XXXXXX); trap 'rm -f \"\$tmp\"' EXIT; cat >\"\$tmp\"; chmod 600 \"\$tmp\"; mv -f \"\$tmp\" .env"
"$SCRIPTS_DIR/ssh-vm.sh" -- "$REMOTE_CMD" <"$ENV_FILE"

echo "Wrote $APP_DIR/.env (mode 0600) on the VM: ${SECRET_NAMES[*]}"
