#!/usr/bin/env bash
# Vertex run entrypoint (issues #38/#53).
#
# Injects WANDB_API_KEY from the Infisical EU vault using the worker's GCP
# identity, runs the launcher-selected modeling profile (smoke, probe,
# efficiency, or blocked full), and uploads its evidence record. Full/efficiency
# checkpoints are uploaded create-only before the manifest when present.
#
# No secret is ever printed: the short-lived Infisical access token and the W&B
# key live only in this process environment and are passed to the runner through
# the child environment. The key is never written to a file, a log, or an
# artifact.
#
# Required env (nonsecret, supplied by the guarded launcher):
#   VERTEX_RUN_LABEL, VERTEX_SOURCE_SHA, VERTEX_IMAGE_DIGEST, VERTEX_EVIDENCE_URI,
#   INFISICAL_MACHINE_IDENTITY_ID, INFISICAL_PROJECT_ID, INFISICAL_ENV
# Optional: INFISICAL_SECRET_PATH (default "/"), VERTEX_OUTPUT_ROOT,
#   VERTEX_CONFIG_NAME (default "vertex_smoke"), VERTEX_PROFILE (default "smoke")
set -euo pipefail

: "${VERTEX_RUN_LABEL:?VERTEX_RUN_LABEL is required}"
: "${VERTEX_SOURCE_SHA:?VERTEX_SOURCE_SHA is required}"
: "${VERTEX_IMAGE_DIGEST:?VERTEX_IMAGE_DIGEST is required}"
: "${VERTEX_EVIDENCE_URI:?VERTEX_EVIDENCE_URI is required}"
: "${INFISICAL_MACHINE_IDENTITY_ID:?INFISICAL_MACHINE_IDENTITY_ID is required}"
: "${INFISICAL_PROJECT_ID:?INFISICAL_PROJECT_ID is required}"
: "${INFISICAL_ENV:?INFISICAL_ENV is required}"

# EU Cloud: a machine identity login does not persist the instance, so the
# domain must be set explicitly or every command would go to US Cloud.
export INFISICAL_DOMAIN="${INFISICAL_DOMAIN:-https://eu.infisical.com}"
export INFISICAL_DISABLE_UPDATE_CHECK=true

OUTPUT_ROOT="${VERTEX_OUTPUT_ROOT:-data/runs/smoke/$VERTEX_RUN_LABEL}"
CONFIG_NAME="${VERTEX_CONFIG_NAME:-vertex_smoke}"
PROFILE="${VERTEX_PROFILE:-smoke}"
if [[ "$PROFILE" == "efficiency" ]]; then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export GDAL_NUM_THREADS=1
fi
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/logs/modeling"
RESULT_FILE="$OUTPUT_ROOT/logs/modeling/vertex-entrypoint.txt"

# Exchange the worker's GCP identity token for a short-lived Infisical access
# token. --plain --silent prints only the token; nothing is echoed.
export INFISICAL_TOKEN
INFISICAL_TOKEN="$(infisical login --method=gcp-id-token \
  --machine-identity-id="$INFISICAL_MACHINE_IDENTITY_ID" --plain --silent)"

if [[ "$PROFILE" == "efficiency" ]]; then
  : "${VERTEX_EFFICIENCY_ROLE:?VERTEX_EFFICIENCY_ROLE is required}"
  : "${VERTEX_EFFICIENCY_SLOT:?VERTEX_EFFICIENCY_SLOT is required}"
  : "${VERTEX_EFFICIENCY_WORKERS:?VERTEX_EFFICIENCY_WORKERS is required}"
  : "${VERTEX_EFFICIENCY_PRECISION:?VERTEX_EFFICIENCY_PRECISION is required}"
  : "${VERTEX_EFFICIENCY_PIN_MEMORY:?VERTEX_EFFICIENCY_PIN_MEMORY is required}"
  : "${VERTEX_EFFICIENCY_RATE_USD:?VERTEX_EFFICIENCY_RATE_USD is required}"
  : "${VERTEX_EFFICIENCY_RATE_SOURCE:?VERTEX_EFFICIENCY_RATE_SOURCE is required}"
  : "${VERTEX_EFFICIENCY_EXPOSURE_USD:?VERTEX_EFFICIENCY_EXPOSURE_USD is required}"
  : "${VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD:?VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD is required}"
  run_command=(uv run python scripts/validators/measure_training_efficiency.py
    --role "$VERTEX_EFFICIENCY_ROLE"
    --slot "$VERTEX_EFFICIENCY_SLOT"
    --run-label "$VERTEX_RUN_LABEL"
    --source-sha "$VERTEX_SOURCE_SHA"
    --image-digest "$VERTEX_IMAGE_DIGEST"
    --output-root "$OUTPUT_ROOT"
    --precision "$VERTEX_EFFICIENCY_PRECISION"
    --workers "$VERTEX_EFFICIENCY_WORKERS"
    --hourly-rate-usd "$VERTEX_EFFICIENCY_RATE_USD"
    --hourly-rate-source "$VERTEX_EFFICIENCY_RATE_SOURCE"
    --projected-exposure-usd "$VERTEX_EFFICIENCY_EXPOSURE_USD"
    --projected-noncompute-total-usd "$VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD")
  if [[ "$VERTEX_EFFICIENCY_PIN_MEMORY" == "true" ]]; then
    run_command+=(--pin-memory)
  fi
  if [[ "$VERTEX_EFFICIENCY_ROLE" == "baseline" || "$VERTEX_EFFICIENCY_ROLE" == "final" ]]; then
    run_command+=(--run-learning-fit)
  fi
else
  run_command=(uv run python scripts/runners/run_modeling.py \
    --config-name "$CONFIG_NAME" "output_root=$OUTPUT_ROOT")
fi

rc=0
set +e
infisical run --projectId="$INFISICAL_PROJECT_ID" --env="$INFISICAL_ENV" \
  --path="${INFISICAL_SECRET_PATH:-/}" -- \
  "${run_command[@]}" 2>&1 | tee "$RESULT_FILE"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 && "$PROFILE" != "efficiency" ]]; then
  echo "vertex $PROFILE run failed (rc=$rc); no evidence uploaded"
  exit "$rc"
fi

# Full and efficiency runs retain a selected checkpoint when one exists. Upload
# is create-only and precedes the manifest that references it.
ckpt_args=()
if [[ "$PROFILE" == "full" ]]; then
  shopt -s nullglob
  ckpts=("$OUTPUT_ROOT"/checkpoints/best-*.ckpt)
  shopt -u nullglob
  if [[ "${#ckpts[@]}" -ne 1 ]]; then
    echo "expected exactly one best checkpoint under $OUTPUT_ROOT/checkpoints, found ${#ckpts[@]}"
    exit 1
  fi
  ckpt_args=(--checkpoint-path "${ckpts[0]}" --checkpoint-uri "${VERTEX_EVIDENCE_URI%/*}/best.ckpt")
elif [[ "$PROFILE" == "efficiency" && "$rc" -eq 0 && ( "$VERTEX_EFFICIENCY_ROLE" == "baseline" || "$VERTEX_EFFICIENCY_ROLE" == "final" ) ]]; then
  shopt -s nullglob
  ckpts=("$OUTPUT_ROOT"/learning/checkpoints/best-*.ckpt)
  shopt -u nullglob
  if [[ "${#ckpts[@]}" -ne 1 ]]; then
    echo "expected exactly one recovery checkpoint under $OUTPUT_ROOT/learning/checkpoints, found ${#ckpts[@]}"
    rc=1
  else
    ckpt_args=(--checkpoint-path "${ckpts[0]}" --checkpoint-uri "${VERTEX_EVIDENCE_URI%/*}/best.ckpt")
  fi
fi

if [[ "$PROFILE" == "efficiency" ]]; then
  uv run python scripts/operators/vertex_evidence.py \
    --run-root "$OUTPUT_ROOT" \
    --evidence-uri "$VERTEX_EVIDENCE_URI" \
    --run-label "$VERTEX_RUN_LABEL" \
    --source-sha "$VERTEX_SOURCE_SHA" \
    --image-digest "$VERTEX_IMAGE_DIGEST" \
    --profile "$PROFILE" \
    --result-file "$RESULT_FILE" \
    --efficiency-slot "$VERTEX_EFFICIENCY_SLOT" \
    --efficiency-role "$VERTEX_EFFICIENCY_ROLE" \
    "${ckpt_args[@]}"
else
  uv run python scripts/operators/vertex_evidence.py \
    --run-root "$OUTPUT_ROOT" \
    --evidence-uri "$VERTEX_EVIDENCE_URI" \
    --run-label "$VERTEX_RUN_LABEL" \
    --source-sha "$VERTEX_SOURCE_SHA" \
    --image-digest "$VERTEX_IMAGE_DIGEST" \
    --profile "$PROFILE" \
    --result-file "$RESULT_FILE" \
    "${ckpt_args[@]}"
fi

if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
