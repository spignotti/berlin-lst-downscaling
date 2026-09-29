#!/usr/bin/env bash
# Validate a retained full-baseline report on the On-Demand berlin-lst-vm.
#
# Lifecycle (see vm-runner-common.sh): start VM → deploy the committed branch
# → run scripts/validators/validate_baseline.py against a report a previous
# run already produced → poll → pin the retained report hash → capture the
# validator evidence and retrieve the artifact → remove only that run's
# ephemeral output → stop the VM.
#
# This launcher NEVER runs the baseline: it re-validates an existing report, so
# an expensive full-split run is not repeated. It writes nothing to GCS and
# touches no canonical product or retained-QA prefix.
#
# Safety: poll expiry or connection loss leaves the VM RUNNING for operator
# inspection (shared fail-closed lifecycle). A failed or unproven validation,
# or a report whose bytes differ from the expected hash, stops the VM but
# leaves the retained output in place and accepts no artifact.
#
# Usage (run from the repository root, like the sibling launchers):
#   run-baseline-validation-vm.sh <branch> <run-id> <expected-report-sha256>
#
# Example:
#   run-baseline-validation-vm.sh feat/39-full-baseline-val-test-report \
#     baseline-full-20260929T084243Z-29A5F946 \
#     5a0daca809055195028c6688da7f337b4448723038c07bc4b30c24c9353d62d6

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-}"
RUN_ID_ARG="${2:-}"
EXPECT_SHA="${3:-}"

if [[ -z "$BRANCH" || -z "$RUN_ID_ARG" || -z "$EXPECT_SHA" ]]; then
  echo "Usage: $0 <branch> <run-id> <expected-report-sha256>" >&2
  exit 1
fi

PIPELINE_LABEL="Baseline report validation"
MARKER_CONFIG="baseline_validation"
POLL_MAX_SECONDS=3600
VALIDATOR_PATCHES=25

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TMP_BASE="${TMPDIR:-$REPO_ROOT/.tmp}"

vm_init_run "baseline-validate"

OUTPUT_REL="data/runs/baseline-full/$RUN_ID_ARG"
REMOTE_OUTPUT="$APP_DIR/$OUTPUT_REL"
REMOTE_REPORT="$REMOTE_OUTPUT/baseline_report.json"
LOCAL_DIR="${TMP_BASE%/}/opencode/baseline-full/$RUN_ID_ARG"
# `set -f` (noglob) keeps the command safe inside the detached shell. No single
# quote may appear in REMOTE_CMD (see vm_launch_detached).
REMOTE_CMD="set -f; uv run python scripts/validators/validate_baseline.py --report $OUTPUT_REL/baseline_report.json --max-patches $VALIDATOR_PATCHES"

echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Report:     $OUTPUT_REL/baseline_report.json"
echo "  Expect sha: $EXPECT_SHA"

# ── start VM + deploy ────────────────────────────────────────────────
vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"

# ── launch + poll ────────────────────────────────────────────────────
vm_launch_detached
vm_poll
vm_finish

# ── verify the retained report + capture validation evidence ─────────
SHA_OK=1
VALIDATOR_OK=1
TRANSFER_OK=1
EVIDENCE_OK=1

REMOTE_REPORT_SHA=$(ssh_cmd "sha256sum '$REMOTE_REPORT'" | cut -d' ' -f1)
echo "  Retained report sha256: $REMOTE_REPORT_SHA"
if [[ "$REMOTE_REPORT_SHA" == "$EXPECT_SHA" ]]; then
  SHA_OK=0
else
  echo "ERROR: retained report sha256 does not match the expected value."
fi

mkdir -p "$LOCAL_DIR"
ssh_cmd "grep -nE 'selection |gap |OK: baseline report valid|FAIL|Traceback' '$REMOTE_LOG' || true" \
  | tee "$LOCAL_DIR/validation.txt" || true

if [[ "$PIPELINE_EXIT" == "0" ]] \
  && ssh_cmd "grep -q 'OK: baseline report valid' '$REMOTE_LOG'"; then
  VALIDATOR_OK=0
fi

if [[ "$SHA_OK" -eq 0 && "$VALIDATOR_OK" -eq 0 ]]; then
  ssh_cmd "tar -C '$REMOTE_OUTPUT' -czf - ." > "$LOCAL_DIR/artifact.tgz"
  tar -xzf "$LOCAL_DIR/artifact.tgz" -C "$LOCAL_DIR"
  rm -f "$LOCAL_DIR/artifact.tgz"
  LOCAL_REPORT_SHA=$(shasum -a 256 "$LOCAL_DIR/baseline_report.json" | cut -d' ' -f1)
  if [[ -s "$LOCAL_DIR/baseline_report.json" && "$LOCAL_REPORT_SHA" == "$REMOTE_REPORT_SHA" ]]; then
    TRANSFER_OK=0
    EVIDENCE_OK=0
  fi
fi

# ── cleanup + stop ───────────────────────────────────────────────────
if [[ "$EVIDENCE_OK" -eq 0 ]]; then
  # The original report is safely secured locally; remove only this run's VM
  # output before stopping, while the VM is still reachable.
  ssh_cmd "rm -rf '$REMOTE_OUTPUT'" || true
fi
vm_stop

if [[ "$EVIDENCE_OK" -eq 0 ]]; then
  echo "SUCCESS: retained baseline report independently validated and secured."
  echo "  Run ID:          $WRAP_RUN_ID"
  echo "  Deployed SHA:    $DEPLOYED_SHA"
  echo "  Report sha256:   $REMOTE_REPORT_SHA"
  echo "  Artifact:        $LOCAL_DIR"
  echo "  Remote log:      $REMOTE_LOG"
else
  echo "FAIL: baseline report validation did not pass; retained output left in place."
  echo "  Run ID:        $WRAP_RUN_ID"
  echo "  Validator:     $([[ "$VALIDATOR_OK" -eq 0 ]] && echo passed || echo failed)"
  echo "  Report sha ok: $([[ "$SHA_OK" -eq 0 ]] && echo yes || echo no)"
  echo "  Artifact:      $LOCAL_DIR (evidence only, not accepted)"
  echo "  Remote path:   $REMOTE_OUTPUT"
  echo "  Remote log:    $REMOTE_LOG"
  exit 1
fi
