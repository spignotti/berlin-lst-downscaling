#!/usr/bin/env bash
# Run the full naive prior-expand baseline over the validation and test splits
# on the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy committed branch
# → launch scripts/runners/run_baseline.py for validation+test → poll the run
# marker → validate the report independently while the VM is up → copy the
# report and run logs back to the workstation → remove this run's ephemeral
# output → stop the VM.
#
# The baseline reads the published patch index, ARD COGs and feature stacks
# and writes only this run's local output root on the VM. Nothing is written
# to GCS, and no canonical product or retained-QA prefix is touched.
#
# Safety:
#   - Poll expiry or connection loss leaves the VM RUNNING for operator
#     inspection (shared fail-closed lifecycle); the run is never relaunched
#     or killed automatically.
#   - If the run artifact cannot be secured locally, the VM is left RUNNING so
#     the output of an expensive run is not lost. Once secured, the VM is
#     stopped even when the run's evidence checks fail.
#
# Usage (run from the repository root, like the sibling launchers):
#   run-baseline-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Full naive baseline (validation+test)"
MARKER_CONFIG="baseline_full"
POLL_MAX_SECONDS=7200
# Independent-validator coverage: patches recomputed per split, with the block
# means re-derived by a different path than the reader (whole-COG accumulation).
VALIDATOR_PATCHES=25

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TMP_BASE="${TMPDIR:-$REPO_ROOT/.tmp}"

vm_init_run "baseline-full"

OUTPUT_REL="data/runs/baseline-full/$WRAP_RUN_ID"
REMOTE_OUTPUT="$APP_DIR/$OUTPUT_REL"
REMOTE_REPORT="$REMOTE_OUTPUT/baseline_report.json"
LOCAL_DIR="${TMP_BASE%/}/opencode/baseline-full/$WRAP_RUN_ID"
# `set -f` (noglob) keeps the Hydra list override safe inside the detached
# shell. No single quote may appear in REMOTE_CMD (see vm_launch_detached).
REMOTE_CMD="set -f; uv run python scripts/runners/run_baseline.py --config-name baseline_full splits=[validation,test] output_root=$OUTPUT_REL"

echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Validation/test only; output root: $OUTPUT_REL"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

if [[ "$PIPELINE_EXIT" != "0" ]]; then
  echo "Remote log tail:"
  ssh_cmd "tail -n 40 '$REMOTE_LOG'" || true
  vm_stop
  echo "FAIL: $PIPELINE_LABEL did not complete cleanly (exit $PIPELINE_EXIT)."
  echo "  Run ID:     $WRAP_RUN_ID"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi

# ── independent validation + artifact transfer (VM still up) ─────────
TRANSFER_OK=1
EVIDENCE_OK=1

echo "Running independent baseline validation on the VM..."
VALIDATION_OUT=""
if VALIDATION_OUT=$(ssh_cmd "cd $APP_DIR && uv run python scripts/validators/validate_baseline.py --report '$REMOTE_REPORT' --max-patches $VALIDATOR_PATCHES" 2>&1); then
  VALIDATION_RC=0
else
  VALIDATION_RC=$?
fi
printf '%s\n' "$VALIDATION_OUT"

echo "Copying the report and run logs from the VM..."
mkdir -p "$LOCAL_DIR"
printf '%s\n' "$VALIDATION_OUT" > "$LOCAL_DIR/validation.txt"
ssh_cmd "tar -C '$REMOTE_OUTPUT' -czf - ." > "$LOCAL_DIR/artifact.tgz"
tar -xzf "$LOCAL_DIR/artifact.tgz" -C "$LOCAL_DIR"
rm -f "$LOCAL_DIR/artifact.tgz"

# Trust the local copy only after it matches the remote byte-for-byte.
REMOTE_REPORT_SHA=$(ssh_cmd "sha256sum '$REMOTE_REPORT'" | cut -d' ' -f1)
LOCAL_REPORT_SHA=$(shasum -a 256 "$LOCAL_DIR/baseline_report.json" | cut -d' ' -f1)
if [[ -s "$LOCAL_DIR/baseline_report.json" && -n "$REMOTE_REPORT_SHA" && "$LOCAL_REPORT_SHA" == "$REMOTE_REPORT_SHA" ]]; then
  TRANSFER_OK=0
fi

if [[ "$TRANSFER_OK" -eq 0 ]]; then
  REPORT_METHOD=$(jq -r '.method' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo INVALID)
  REPORT_SPLITS=$(jq -r '.splits | keys | join(",")' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo INVALID)
  REPORT_REV=$(jq -r '.git_revision' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo INVALID)
  REPORT_EXCLUSIONS=$(jq -r '[.exclusions, (.splits[] | .exclusions)] | map(length) | add' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo INVALID)
  VALID_CELLS_VALIDATION=$(jq -r '.splits.validation.valid_cells // 0' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo 0)
  VALID_CELLS_TEST=$(jq -r '.splits.test.valid_cells // 0' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || echo 0)

  echo "Report: method=$REPORT_METHOD | splits=$REPORT_SPLITS | git_revision=$REPORT_REV"
  jq -r '.splits | to_entries[] | "  \(.key): patches \(.value.evaluated_patches)/\(.value.requested_patches) | cells \(.value.valid_cells) | MAE \(.value.mae) | SSIM \(.value.ssim)"' "$LOCAL_DIR/baseline_report.json" 2>/dev/null || true
  echo "  exclusions (top-level + per split): $REPORT_EXCLUSIONS"

  # A full-run anchor needs: validator success (the amended validator proves
  # every missing row is a contract-permitted exclusion, so a nonzero count is
  # acceptable only when it passes), the exact split set, the deployed SHA, and
  # positive valid cells per split.
  if [[ "$VALIDATION_RC" -eq 0 \
    && "$REPORT_METHOD" == "naive_prior_expand" \
    && "$REPORT_SPLITS" == "test,validation" \
    && "$REPORT_REV" == "$DEPLOYED_SHA" \
    && "$VALID_CELLS_VALIDATION" -gt 0 \
    && "$VALID_CELLS_TEST" -gt 0 ]]; then
    EVIDENCE_OK=0
  fi
fi

# ── cleanup + stop, or preserve the VM ───────────────────────────────
if [[ "$TRANSFER_OK" -eq 0 ]]; then
  if [[ "$EVIDENCE_OK" -eq 0 ]]; then
    ssh_cmd "rm -rf '$REMOTE_OUTPUT'" || true
  fi
  vm_stop
  if [[ "$EVIDENCE_OK" -eq 0 ]]; then
    echo "SUCCESS: full validation/test baseline completed, independently validated, and secured."
    echo "  Run ID:       $WRAP_RUN_ID"
    echo "  Deployed SHA: $DEPLOYED_SHA"
    echo "  Artifact:     $LOCAL_DIR"
    echo "  Remote log:   $REMOTE_LOG"
  else
    echo "FAIL: baseline completed but its evidence checks did not pass."
    echo "  Run ID:       $WRAP_RUN_ID"
    echo "  Artifact:     $LOCAL_DIR (secured before the VM was stopped)"
    echo "  Remote log:   $REMOTE_LOG"
    exit 1
  fi
else
  echo "FAIL: could not secure the run artifact from the VM; leaving the VM RUNNING for inspection."
  echo "  Run ID:      $WRAP_RUN_ID"
  echo "  Remote path: $REMOTE_OUTPUT"
  echo "  Remote log:  $REMOTE_LOG"
  exit 1
fi
