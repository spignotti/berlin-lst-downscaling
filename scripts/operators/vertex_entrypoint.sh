#!/usr/bin/env bash
# Vertex acceptance-run entrypoint (issue #38).
#
# Injects WANDB_API_KEY from the Infisical EU vault using the worker's GCP
# identity, runs the bounded modeling smoke, and uploads a metadata-only
# evidence record on success.
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
mkdir -p "$OUTPUT_ROOT"
RESULT_FILE="$OUTPUT_ROOT/result.txt"

# Exchange the worker's GCP identity token for a short-lived Infisical access
# token. --plain --silent prints only the token; nothing is echoed.
export INFISICAL_TOKEN
INFISICAL_TOKEN="$(infisical login --method=gcp-id-token \
  --machine-identity-id="$INFISICAL_MACHINE_IDENTITY_ID" --plain --silent)"

rc=0
set +e
infisical run --projectId="$INFISICAL_PROJECT_ID" --env="$INFISICAL_ENV" \
  --path="${INFISICAL_SECRET_PATH:-/}" -- \
  uv run python scripts/runners/run_modeling.py \
  --config-name "$CONFIG_NAME" "output_root=$OUTPUT_ROOT" 2>&1 | tee "$RESULT_FILE"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "vertex $PROFILE run failed (rc=$rc); no evidence uploaded"
  exit "$rc"
fi

uv run python scripts/operators/vertex_evidence.py \
  --run-root "$OUTPUT_ROOT" \
  --evidence-uri "$VERTEX_EVIDENCE_URI" \
  --run-label "$VERTEX_RUN_LABEL" \
  --source-sha "$VERTEX_SOURCE_SHA" \
  --image-digest "$VERTEX_IMAGE_DIGEST" \
  --profile "$PROFILE" \
  --result-file "$RESULT_FILE"
