#!/bin/bash
# Operator-only watcher for the three authorized validation-only TESSERA seeds.
# Run only with the owner's approval. No region, GPU, or scientific substitution.
set -u
set -o pipefail

REPO="${TESSERA_REPO:-$HOME/code/berlin-lst-downscaling}"
DIR="${TESSERA_STATE_DIR:-$HOME/Library/Application Support/tessera-seeds}"
LOG="${TESSERA_LOG:-$HOME/Library/Logs/tessera-seeds.log}"
STATUS="$DIR/status.txt"
LOCK="$DIR/retry.lock"
# One authorized sequence = one image digest, its source commit and a fresh label.
# None has a default, so a retired image or label cannot be reused by accident.
IMAGE="${TESSERA_IMAGE:-}"
SHA="${TESSERA_SOURCE_SHA:-}"
LABEL="${TESSERA_LABEL:-}"
TIMEOUT="${TESSERA_TIMEOUT_SECONDS:-17400}"
# Window-read image of the cancelled 2026-11-13 sequence (about an hour per epoch).
RETIRED_DIGEST="sha256:e5aa497c215290343842011ca58b3da99dc22562e9795453b2bf5ed80a3e2c9f"

mkdir -p "$DIR" "$(dirname "$LOG")" || exit 1
exec >>"$LOG" 2>&1
echo "=== tick $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [[ -f "$DIR/STOP" ]]; then
  echo "stopped: $(cat "$DIR/STOP")"
  exit 0
fi
config_error() {
  printf 'configuration error: %s\n' "$1" >"$STATUS"
  echo "configuration error: $1; no submission"
  exit 2
}
[[ "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] ||
  config_error "TESSERA_IMAGE must be pinned by digest"
[[ "$IMAGE" != *"@$RETIRED_DIGEST" ]] ||
  config_error "TESSERA_IMAGE is the retired window-read image"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] ||
  config_error "TESSERA_SOURCE_SHA must be the image's 40-character commit"
[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] ||
  config_error "TESSERA_LABEL must be a fresh lowercase run-label prefix"
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] ||
  config_error "TESSERA_TIMEOUT_SECONDS must be a positive integer"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

stop() {
  printf '%s\n' "$1" >"$DIR/STOP"
  printf '%s\n' "$1" >"$STATUS"
  echo "STOP: $1"
  exit 1
}

operator() {
  case "${TESSERA_AUTH:-operator}" in
    operator) "$REPO/scripts/operators/with-gcp-operator.sh" -- "$@" ;;
    user-adc) "$@" ;;
    *) echo "unknown TESSERA_AUTH" >&2; return 1 ;;
  esac
}

latest_job() {
  operator gcloud ai custom-jobs list \
    --region=europe-west3 --project=berlin-lst-training \
    --filter="displayName=$1" \
    --format='json(name,state,error.code,error.message,createTime)' |
    jq -er 'if type != "array" then error("expected jobs array")
      elif length == 0 then ["NONE", "", ""]
      elif any(.[]; .state == "JOB_STATE_QUEUED" or .state == "JOB_STATE_PENDING"
        or .state == "JOB_STATE_RUNNING" or .state == "JOB_STATE_UPDATING"
        or .state == "JOB_STATE_CANCELLING") then ["ACTIVE", "", ""]
      elif any(.[]; .state == "JOB_STATE_SUCCEEDED") then
        ["JOB_STATE_SUCCEEDED", "", ""]
      elif any(.[]; .state != "JOB_STATE_FAILED"
        or .error.code != 14
        or ((.error.message // "") | contains("Resources are insufficient") | not))
      then ["STOP", "", ""]
      else sort_by(.createTime) | last |
        [.state, .error.message // "", .name // ""]
      end | @tsv'
}

cd "$REPO" || stop "repository unavailable"
for seed in 0 1 2; do
  display="tessera-${LABEL}-seed${seed}-ep40-cos-notest"
  if ! row="$(latest_job "$display")"; then
    stop "seed $seed lookup failed; no submission"
  fi
  state="${row%%$'\t'*}"
  echo "seed $seed state=$state"
  case "$state" in
    JOB_STATE_SUCCEEDED) continue ;;
    ACTIVE)
      printf 'seed %s active\n' "$seed" >"$STATUS"
      exit 0
      ;;
    NONE|JOB_STATE_FAILED) ;;
    *) stop "seed $seed has a non-capacity failure or unknown state; inspect Vertex" ;;
  esac
  printf 'submitting seed %s\n' "$seed" >"$STATUS"
  if operator uv run --group operators \
    python scripts/operators/launch_vertex_modeling.py \
    --mode tessera --seed "$seed" --max-epochs 40 --lr-schedule cosine --skip-test \
    --patch-cache prebuilt \
    --image-uri "$IMAGE" --source-sha "$SHA" --run-label "$LABEL" \
    --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
    --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
    --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
    --infisical-env dev --infisical-path /vertex \
    --hourly-rate-usd 1.00 --timeout-seconds "$TIMEOUT" --max-wait-seconds 600 \
    --max-exposure-usd 5
  then
    continue
  fi
  if ! row="$(latest_job "$display")"; then
    stop "seed $seed post-submit lookup failed; reconcile before retry"
  fi
  state="${row%%$'\t'*}"
  case "$state" in
    JOB_STATE_FAILED)
      printf 'no capacity, seed %s; retry next tick\n' "$seed" >"$STATUS"
      exit 0
      ;;
    ACTIVE)
      printf 'seed %s active; reconnect next tick\n' "$seed" >"$STATUS"
      exit 0
      ;;
    *) stop "seed $seed launcher failed; reconcile before retry" ;;
  esac
done
printf 'all three seeds succeeded\n' >"$STATUS"
