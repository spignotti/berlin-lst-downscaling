#!/usr/bin/env bash
# Shared run logging for the run-*-vm.sh launchers.
#
# Source this file right after `set -euo pipefail`, before anything else:
#
#   source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
#
# On first source it re-runs the calling launcher with the same arguments
# and tees stdout+stderr to the gitignored repo directory
#   .runs/logs/<run_id>.log
# When the launcher exits (success, failure, or interrupt) every local file
# of this run (.runs/logs/<run_id>.*, including the remote VM log fetched by
# vm_preserve_logs) is uploaded to
#   ${LOG_GCS_PREFIX}/<YYYY-MM>/
# The local files are removed after a successful upload and kept, with a
# warning, when the upload fails. The launcher's exit code is preserved.
# Inside the re-run (RUNLOG_ACTIVE=1) sourcing is a no-op apart from the
# exported RUNLOG_* variables.
#
# Environment:
#   LOG_GCS_PREFIX  upload prefix (default gs://berlin-lst-training-data/logs/vm-runs)
#
# Uploads use `gcloud storage cp` with the operator's ADC; no credentials
# are read here.

if [[ "${RUNLOG_ACTIVE:-}" != "1" ]]; then
  RUNLOG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  RUNLOG_DIR="$RUNLOG_REPO_ROOT/.runs/logs"
  RUNLOG_MONTH="$(date -u +%Y-%m)"
  RUNLOG_ID="$(basename "$0" .sh)-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  RUNLOG_FILE="$RUNLOG_DIR/$RUNLOG_ID.log"
  RUNLOG_DEST="${LOG_GCS_PREFIX:-gs://berlin-lst-training-data/logs/vm-runs}"
  RUNLOG_DEST="${RUNLOG_DEST%/}/$RUNLOG_MONTH/"
  mkdir -p "$RUNLOG_DIR"
  export RUNLOG_ACTIVE=1 RUNLOG_DIR RUNLOG_ID RUNLOG_FILE

  # Handled (not ignored) signals are reset to the default in the child, so
  # Ctrl-C still stops the launcher; this shell only survives to upload.
  trap 'echo "WARNING: launcher interrupted; uploading the partial log." >&2' INT TERM HUP

  set +e
  # tee -i ignores SIGINT so the tail of an interrupted run is still written.
  "$BASH" "$0" "$@" 2>&1 | tee -i "$RUNLOG_FILE"
  runlog_rc=${PIPESTATUS[0]}

  runlog_files=("$RUNLOG_DIR/$RUNLOG_ID".*)
  if runlog_err=$(gcloud storage cp "${runlog_files[@]}" "$RUNLOG_DEST" 2>&1); then
    rm -f "${runlog_files[@]}"
    echo "Run log uploaded: ${RUNLOG_DEST}${RUNLOG_ID}.log"
  else
    echo "WARNING: run log upload to $RUNLOG_DEST failed; local copy kept:" >&2
    printf '%s\n' "$runlog_err" | tail -n 5 >&2
    printf '  %s\n' "${runlog_files[@]}" >&2
  fi
  exit "$runlog_rc"
fi
