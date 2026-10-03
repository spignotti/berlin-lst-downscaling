#!/usr/bin/env bash
set -euo pipefail

project_root="$(git rev-parse --show-toplevel)"
tmp_parent="${TMPDIR%/}/opencode"
[[ -d "$tmp_parent" ]] || {
  printf 'Temporary parent does not exist: %s\n' "$tmp_parent" >&2
  exit 1
}
tmp_root="$(mktemp -d "$tmp_parent/vertex-efficiency-entrypoint.XXXXXX")"
trap 'rm -rf "$tmp_root"' EXIT

fake_bin="$tmp_root/bin"
workdir="$tmp_root/work"
trace="$tmp_root/trace.txt"
mkdir -p "$fake_bin" "$workdir"

cat > "$fake_bin/infisical" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == login ]]; then
  exit 0
fi
[[ "${1:-}" == run ]] || exit 2
shift
while [[ $# -gt 0 && "$1" != -- ]]; do shift; done
[[ $# -gt 0 ]] || exit 3
shift
exec "$@"
STUB

cat > "$fake_bin/uv" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == run && "${2:-}" == python ]] || exit 2
script="$3"
shift 3
root="data/runs/efficiency/$VERTEX_RUN_LABEL"
if [[ "$script" == scripts/validators/measure_training_efficiency.py ]]; then
  expected="--role baseline --slot 1 --run-label $VERTEX_RUN_LABEL --source-sha $VERTEX_SOURCE_SHA --image-digest $VERTEX_IMAGE_DIGEST --output-root $root --precision 32-true --workers 2 --hourly-rate-usd 1.0 --hourly-rate-source User-authorized Vertex Frankfurt range; USD 1.00/hour. --projected-exposure-usd 1.25000000 --projected-noncompute-total-usd 2.0 --run-learning-fit"
  actual="$(printf '%s ' "$@")"
  [[ "$actual" == "$expected " ]] || {
    printf 'Measurement argv mismatch.\nexpected: %s\nactual:   %s\n' "$expected" "$actual" >&2
    exit 4
  }
  [[ "$VERTEX_EFFICIENCY_SESSION" == stage1-efficiency-recovery-20261003 ]]
  [[ "$VERTEX_EFFICIENCY_OUTPUT_CLAIMED" == "$VERTEX_RUN_LABEL" ]]
  [[ -f "$root/logs/modeling/vertex-entrypoint.txt" ]]
  shopt -s nullglob
  root_entries=("$root"/*)
  log_entries=("$root/logs"/*)
  model_entries=("$root/logs/modeling"/*)
  [[ ${#root_entries[@]} -eq 1 && "${root_entries[0]}" == "$root/logs" ]]
  [[ ${#log_entries[@]} -eq 1 && "${log_entries[0]}" == "$root/logs/modeling" ]]
  [[ ${#model_entries[@]} -eq 1 && "${model_entries[0]}" == "$root/logs/modeling/vertex-entrypoint.txt" ]]
  mkdir -p "$root/learning/checkpoints"
  printf 'synthetic checkpoint\n' > "$root/learning/checkpoints/best-synthetic.ckpt"
  printf 'measure\n' >> "$ENTRYPOINT_CHECK_TRACE"
  exit 0
fi
if [[ "$script" == scripts/operators/vertex_evidence.py ]]; then
  expected_checkpoint="$root/learning/checkpoints/best-synthetic.ckpt"
  evidence_args="$(printf '%s ' "$@")"
  [[ "$evidence_args" == *"--profile efficiency "* ]]
  [[ "$evidence_args" == *"--checkpoint-path $expected_checkpoint "* ]]
  printf 'evidence\n' >> "$ENTRYPOINT_CHECK_TRACE"
  exit 0
fi
printf 'Unexpected uv command: %s\n' "$script" >&2
exit 5
STUB

chmod 700 "$fake_bin/infisical" "$fake_bin/uv"

run_entrypoint() {
  local label="$1"
  (
    cd "$workdir"
    env -i \
      PATH="$fake_bin:/usr/bin:/bin" \
      HOME="$tmp_root" \
      TMPDIR="$tmp_root" \
      ENTRYPOINT_CHECK_TRACE="$trace" \
      VERTEX_RUN_LABEL="$label" \
      VERTEX_SOURCE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
      VERTEX_IMAGE_DIGEST=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
      VERTEX_EVIDENCE_URI=gs://synthetic-bucket/qa/evidence.json \
      VERTEX_PROFILE=efficiency \
      VERTEX_OUTPUT_ROOT="data/runs/efficiency/$label" \
      VERTEX_EFFICIENCY_ROLE=baseline \
      VERTEX_EFFICIENCY_SLOT=1 \
      VERTEX_EFFICIENCY_SESSION=stage1-efficiency-recovery-20261003 \
      VERTEX_EFFICIENCY_WORKERS=2 \
      VERTEX_EFFICIENCY_PRECISION=32-true \
      VERTEX_EFFICIENCY_PIN_MEMORY=false \
      VERTEX_EFFICIENCY_RATE_USD=1.0 \
      VERTEX_EFFICIENCY_RATE_SOURCE='User-authorized Vertex Frankfurt range; USD 1.00/hour.' \
      VERTEX_EFFICIENCY_EXPOSURE_USD=1.25000000 \
      VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD=2.0 \
      INFISICAL_MACHINE_IDENTITY_ID=synthetic-identity \
      INFISICAL_PROJECT_ID=synthetic-project \
      INFISICAL_ENV=dev \
      INFISICAL_SECRET_PATH=/vertex \
      /bin/bash "$project_root/scripts/operators/vertex_entrypoint.sh"
  )
}

run_entrypoint stage1-efficiency-j1-self-check
[[ "$(<"$trace")" == $'measure\nevidence' ]]

stale_label=stage1-efficiency-j1-stale-self-check
stale_root="$workdir/data/runs/efficiency/$stale_label"
mkdir -p "$stale_root"
printf '{}\n' > "$stale_root/old-result.json"
if run_entrypoint "$stale_label" > "$tmp_root/stale-output.txt" 2>&1; then
  printf 'Stale efficiency output root was accepted.\n' >&2
  exit 1
fi
[[ "$(<"$trace")" == $'measure\nevidence' ]]
printf 'ENTRYPOINT SELF-CHECK OK: invocation wiring, claimed logs, checkpoint handoff, and stale-root rejection\n'
