#!/usr/bin/env bash
# v6 ablation matrix: 9 arms x seeds 0/1/2 = 27 Vertex jobs, submitted one at a time through
# scripts/operators/launch_vertex_modeling.py (cosine lr schedule, prebuilt patch cache).
# The submit paths spend money: run them only with the project owner's explicit OK.
#
# Usage:  launch_v6_matrix.sh preflight [ARM SEED]          # launcher --preflight only, submits nothing
#         launch_v6_matrix.sh submit ARM SEED               # one job (blocks until terminal)
#         launch_v6_matrix.sh submit-all [--from ARM SEED]  # sequential jobs in fixed order, resumable
#         launch_v6_matrix.sh state ARM SEED                # existing jobs for the pair (read-only)
#         launch_v6_matrix.sh help | -h | --help            # this text, touches nothing
#
# Environment (V6M_*):
#   Required (no default):
#     V6M_IMAGE_URI     runner image pinned by digest (repo/name@sha256:<64 hex>)
#     V6M_SOURCE_SHA    40 lowercase hex; the checkout HEAD must equal it and the tree must be clean
#     V6M_LABEL_PREFIX  run-label prefix, ^[a-z0-9][a-z0-9-]*$
#     V6M_RUN_DIR       (submit, submit-all) existing directory OUTSIDE the repo for logs/temp files
#   Optional (default):
#     V6M_PROJECT (berlin-lst-training)  V6M_REGION (europe-west3)
#     V6M_SERVICE_ACCOUNT (berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com)
#     V6M_INFISICAL_IDENTITY (7ba603e5-b94d-42d1-bc57-d64658dde09d)   public identifier
#     V6M_INFISICAL_PROJECT (5da7dfb7-954d-4736-ba2e-4471ade9d766)    public identifier
#     V6M_HOURLY_RATE_USD (1.00)   V6M_TIMEOUT_SECONDS (14400)   V6M_MAX_WAIT_SECONDS (1800)
#     V6M_MAX_EPOCHS (40)          V6M_POLL_SECONDS (60)         V6M_GCLOUD_TIMEOUT (120)
#     V6M_LAUNCHER_GRACE (120)     V6M_CREATE_WAIT (900)         V6M_SUBMIT_ATTEMPTS (3)
#     V6M_MAX_POLL_ERR_SECONDS (21600)   V6M_LOOKUP_BACKOFF_SECONDS (30)
#     V6M_GCLOUD (gcloud)          V6M_RESUBMIT_FAILED (0; 1 = operator opt-in, see below)
#
# Hardening:
#  - Idempotent: before submitting, existing jobs are looked up by exact displayName. SUCCEEDED
#    -> skip; QUEUED/PENDING/RUNNING/... -> wait on it; other terminal states -> stop the loop
#    (no automatic resubmit). V6M_RESUBMIT_FAILED=1 allows one resubmit, only when every
#    existing job for the pair is FAILED.
#  - If the launcher exits non-zero before reporting a final state but the job exists, the job is
#    polled via gcloud until terminal; transient status errors are tolerated for a long time.
#    The launcher's queue allowance (max-wait) is enforced the same way (cancel + stop).
#  - A submission that fails before a job exists is retried (attempts, backoff), with a
#    displayName lookup before every attempt so no duplicate is created.
#  - The launcher runs in the background; as soon as its job exists (id in its output or found by
#    displayName) this script polls the job itself. On a terminal state the launcher gets a grace
#    period and is then killed (process tree). A launcher showing no job after CREATE_WAIT is killed.
#  - Every gcloud call runs under `timeout` and with --project.
#  - Only SUCCEEDED continues the loop; anything else stops it cleanly with a STOP log line.
# Needs bash 4+ and GNU date (`date -d`), i.e. a Linux operator machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
SELF="$(basename "$0")"

declare -A MODE=( [L1]=v6-l1 [L2]=v6-l2 [L3]=v6-l3 [L4]=v6-l4 [L5]=v6-l5 [LOO-M]=v6-loo-morph
  [LOO-S]=v6-loo-shadow [NO-NDVI-NDWI]=v6-no-ndvi-ndwi [NO-RED-NIR-IDX]=v6-no-red-nir-index )
ARMS=(L1 L2 L3 L4 L5 LOO-M LOO-S NO-NDVI-NDWI NO-RED-NIR-IDX)

usage() {
  cat <<USAGE
usage: $SELF preflight [ARM SEED]
       $SELF submit ARM SEED
       $SELF submit-all [--from ARM SEED]
       $SELF state ARM SEED
       $SELF help

ARM: ${ARMS[*]}    SEED: 0 | 1 | 2

preflight/submit/submit-all need a clean checkout at V6M_SOURCE_SHA. submit/submit-all spend money.

Required env: V6M_IMAGE_URI (repo/name@sha256:<64 hex>), V6M_SOURCE_SHA (40 hex),
  V6M_LABEL_PREFIX ([a-z0-9][a-z0-9-]*), V6M_RUN_DIR (submit/submit-all; existing dir outside the repo).
  state needs only V6M_LABEL_PREFIX; preflight needs no V6M_RUN_DIR.
Optional env (default): V6M_PROJECT (berlin-lst-training), V6M_REGION (europe-west3),
  V6M_SERVICE_ACCOUNT, V6M_INFISICAL_IDENTITY, V6M_INFISICAL_PROJECT, V6M_HOURLY_RATE_USD (1.00),
  V6M_TIMEOUT_SECONDS (14400), V6M_MAX_WAIT_SECONDS (1800), V6M_MAX_EPOCHS (40), V6M_POLL_SECONDS (60),
  V6M_GCLOUD_TIMEOUT (120), V6M_LAUNCHER_GRACE (120), V6M_CREATE_WAIT (900), V6M_SUBMIT_ATTEMPTS (3),
  V6M_MAX_POLL_ERR_SECONDS (21600), V6M_LOOKUP_BACKOFF_SECONDS (30), V6M_GCLOUD (gcloud),
  V6M_RESUBMIT_FAILED (0; 1 = resubmit once when every existing job is FAILED).
USAGE
}

die() { echo "$SELF: $*" >&2; exit 2; }

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $SELF: $*"; }

pos_int() {  # name value
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got '$2'"
}

load_config() {  # needs: $1 = "run" if V6M_RUN_DIR is required
  PROJECT="${V6M_PROJECT:-berlin-lst-training}"
  REGION="${V6M_REGION:-europe-west3}"
  SA="${V6M_SERVICE_ACCOUNT:-berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com}"
  INF_ID="${V6M_INFISICAL_IDENTITY:-7ba603e5-b94d-42d1-bc57-d64658dde09d}"
  INF_PROJECT="${V6M_INFISICAL_PROJECT:-5da7dfb7-954d-4736-ba2e-4471ade9d766}"
  RATE="${V6M_HOURLY_RATE_USD:-1.00}"
  TIMEOUT="${V6M_TIMEOUT_SECONDS:-14400}"
  MAX_WAIT="${V6M_MAX_WAIT_SECONDS:-1800}"
  MAX_EPOCHS="${V6M_MAX_EPOCHS:-40}"
  POLL="${V6M_POLL_SECONDS:-60}"
  GCLOUD_TIMEOUT="${V6M_GCLOUD_TIMEOUT:-120}"
  LAUNCHER_GRACE="${V6M_LAUNCHER_GRACE:-120}"
  CREATE_WAIT="${V6M_CREATE_WAIT:-900}"
  SUBMIT_ATTEMPTS="${V6M_SUBMIT_ATTEMPTS:-3}"
  MAX_POLL_ERR_SECONDS="${V6M_MAX_POLL_ERR_SECONDS:-21600}"
  BACKOFF="${V6M_LOOKUP_BACKOFF_SECONDS:-30}"
  GCLOUD="${V6M_GCLOUD:-gcloud}"
  RESUBMIT_FAILED="${V6M_RESUBMIT_FAILED:-0}"
  local n
  for n in TIMEOUT MAX_WAIT MAX_EPOCHS POLL GCLOUD_TIMEOUT LAUNCHER_GRACE CREATE_WAIT \
           SUBMIT_ATTEMPTS MAX_POLL_ERR_SECONDS; do
    pos_int "V6M_$n" "${!n}"
  done
  [[ "$BACKOFF" =~ ^[0-9]+$ ]] || die "V6M_LOOKUP_BACKOFF_SECONDS must be a non-negative integer"
  [[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ && "$RATE" =~ [1-9] ]] || die "V6M_HOURLY_RATE_USD must be a positive decimal"
  [[ "$RESUBMIT_FAILED" =~ ^[01]$ ]] || die "V6M_RESUBMIT_FAILED must be 0 or 1"
  [[ -n "$PROJECT" && -n "$REGION" && -n "$SA" && -n "$INF_ID" && -n "$INF_PROJECT" ]] \
    || die "V6M_PROJECT/REGION/SERVICE_ACCOUNT/INFISICAL_* must not be empty"
  LABEL_PREFIX="${V6M_LABEL_PREFIX:-}"
  [[ "$LABEL_PREFIX" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "V6M_LABEL_PREFIX missing or invalid"
  if [[ "${1:-}" == image ]] || [[ "${1:-}" == run ]]; then
    IMAGE="${V6M_IMAGE_URI:-}"
    [[ "$IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] || die "V6M_IMAGE_URI missing or not a digest-pinned image"
    SHA="${V6M_SOURCE_SHA:-}"
    [[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || die "V6M_SOURCE_SHA missing or not 40 lowercase hex"
  fi
  if [[ "${1:-}" == run ]]; then
    local rd="${V6M_RUN_DIR:-}"
    [[ -n "$rd" && -d "$rd" ]] || die "V6M_RUN_DIR must be an existing directory"
    RUNDIR="$(cd "$rd" && pwd -P)"
    [[ "$RUNDIR" != "$REPO_ROOT" && "$RUNDIR" != "$REPO_ROOT"/* ]] || die "V6M_RUN_DIR must be outside the repo"
  fi
}

check_repo() {
  [[ "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" == "$SHA" \
     && -z "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]] \
    || { echo "$SELF: repo not clean at $SHA" >&2; exit 1; }
}

check_pair() {  # arm seed
  [[ -n "${1:-}" && -n "${MODE[$1]:-}" ]] || die "unknown ARM '${1:-}' (expected one of: ${ARMS[*]})"
  [[ "${2:-}" =~ ^[012]$ ]] || die "SEED must be 0, 1 or 2, got '${2:-}'"
}

lower() { echo "$1" | tr 'A-Z' 'a-z'; }

display_name() {  # arm seed -> exact Vertex displayName (mode-runlabel, as built by the launcher)
  local arm=$1 seed=$2
  echo "${MODE[$arm]}-${LABEL_PREFIX}-$(lower "$arm")-seed${seed}-ep${MAX_EPOCHS}-cos"
}

run() {  # arm seed [--preflight]
  local arm=$1 seed=$2; shift 2
  local label="${LABEL_PREFIX}-$(lower "$arm")"
  cd "$REPO_ROOT"
  # operator SA key from Infisical /operator into a 0600 temp file for this command only
  "$REPO_ROOT/scripts/operators/with-gcp-operator.sh" -- \
    uv run --group operators python scripts/operators/launch_vertex_modeling.py \
    --mode "${MODE[$arm]}" --seed "$seed" --max-epochs "$MAX_EPOCHS" --lr-schedule cosine \
    --patch-cache prebuilt \
    --image-uri "$IMAGE" --source-sha "$SHA" --run-label "$label" \
    --service-account "$SA" \
    --infisical-identity "$INF_ID" --infisical-project "$INF_PROJECT" \
    --infisical-env dev --infisical-path /vertex \
    --hourly-rate-usd "$RATE" --timeout-seconds "$TIMEOUT" --max-wait-seconds "$MAX_WAIT" "$@"
}

gc() { timeout "$GCLOUD_TIMEOUT" "$GCLOUD" "$@" --project "$PROJECT"; }

# Existing jobs for a displayName: lines "id<TAB>state". Retries; returns 1 if lookup impossible.
lookup() {
  local dn=$1 i out
  for i in 1 2 3 4 5 6; do
    if out=$(gc ai custom-jobs list --region "$REGION" --filter="displayName=\"$dn\"" \
               --format='value(name.basename(),state)' 2>/dev/null); then
      printf '%s' "$out"; return 0
    fi
    sleep $(( BACKOFF * i ))
  done
  return 1
}

# Poll one job until terminal; echoes final state on stdout (last line). Logs go to stderr.
poll_job() {
  local id=$1 out state ct st err_since="" last="" now ct_s
  while true; do
    if out=$(gc ai custom-jobs describe "$id" --region "$REGION" \
               --format='value(state,createTime,startTime)' 2>/dev/null) && [[ -n "$out" ]]; then
      err_since=""
      read -r state ct st <<<"$out"
      [[ "$state" != "$last" ]] && { log "job $id state: $state" >&2; last=$state; }
      case "$state" in
        JOB_STATE_SUCCEEDED|JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED|JOB_STATE_PARTIALLY_SUCCEEDED)
          if [[ "$state" == JOB_STATE_SUCCEEDED && -n "${st:-}" ]] &&
             (( $(date -d "$st" +%s) > $(date -d "$ct" +%s) + MAX_WAIT )); then
            echo "QUEUE_DEADLINE_MISSED"; return 0
          fi
          echo "$state"; return 0 ;;
      esac
      now=$(date +%s); ct_s=$(date -d "$ct" +%s)
      if [[ -z "${st:-}" && "$state" != JOB_STATE_RUNNING ]] && (( now > ct_s + MAX_WAIT )); then
        # same rule as the launcher: not started within max-wait -> cancel, never replace
        log "job $id not started within ${MAX_WAIT}s of creation; requesting cancellation (launcher queue rule)" >&2
        gc ai custom-jobs cancel "$id" --region "$REGION" >&2 2>&1 || log "cancel request for $id returned an error; inspect manually" >&2
        echo "QUEUE_DEADLINE_MISSED"; return 0
      fi
      if (( now > ct_s + MAX_WAIT + TIMEOUT + 3600 )); then
        log "job $id still $state long after server timeout; giving up waiting" >&2
        echo "WAIT_EXPIRED"; return 0
      fi
    else
      now=$(date +%s); [[ -z "$err_since" ]] && err_since=$now
      log "status query for $id failed (transient?), retrying" >&2
      if (( now - err_since > MAX_POLL_ERR_SECONDS )); then
        log "status of $id unreadable for >${MAX_POLL_ERR_SECONDS}s; giving up waiting" >&2
        echo "STATUS_UNKNOWN"; return 0
      fi
    fi
    sleep "$POLL"
  done
}

kill_tree() {  # TERM a process and all descendants (children first)
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill -TERM "$1" 2>/dev/null || true
}

stop_launcher() {  # grace period, then kill a still-running launcher tree
  local lpid=$1 i
  for i in $(seq 1 $(( LAUNCHER_GRACE / 5 ))); do kill -0 "$lpid" 2>/dev/null || break; sleep 5; done
  if kill -0 "$lpid" 2>/dev/null; then
    log "launcher (pid $lpid) still alive ${LAUNCHER_GRACE}s after terminal state; killing it"
    kill_tree "$lpid"; sleep 2
  fi
  wait "$lpid" 2>/dev/null || true
}

# One pair, idempotent. Returns 0 only if the pair ends SUCCEEDED.
do_pair() {
  local arm=$1 seed=$2 dn existing prior_ids id state attempt rc tmp pairlog lpid t0
  dn=$(display_name "$arm" "$seed")
  pairlog="$RUNDIR/submit-${arm}-seed${seed}.log"
  for attempt in $(seq 1 "$SUBMIT_ATTEMPTS"); do
    existing=$(lookup "$dn") || { log "STOP: cannot list jobs for $dn (idempotency check impossible)"; return 1; }
    if [[ -n "$existing" ]]; then
      if grep -q $'\tJOB_STATE_SUCCEEDED$' <<<"$existing"; then
        log "skip $arm seed$seed: SUCCEEDED job exists ($(grep $'\tJOB_STATE_SUCCEEDED$' <<<"$existing" | cut -f1 | head -1))"; return 0
      fi
      id=$(grep -E $'\tJOB_STATE_(QUEUED|PENDING|RUNNING|CANCELLING|PAUSED|UPDATING)$' <<<"$existing" | cut -f1 | head -1 || true)
      if [[ -n "$id" ]]; then
        log "$arm seed$seed: active job $id exists, waiting on it"
        state=$(poll_job "$id" | tail -1)
        log "$arm seed$seed: job $id final: $state"
        [[ "$state" == JOB_STATE_SUCCEEDED ]] && return 0
        log "STOP: $arm seed$seed ended $state (job $id); no automatic resubmit"; return 1
      fi
      if [[ "$RESUBMIT_FAILED" == 1 ]] && ! grep -qv $'\tJOB_STATE_FAILED$' <<<"$existing"; then
        log "$arm seed$seed: only FAILED jobs exist ($(tr '\n\t' '; ' <<<"$existing")); V6M_RESUBMIT_FAILED=1 set by operator, resubmitting"
      else
        log "STOP: $arm seed$seed has only non-successful jobs: $(tr '\n\t' '; ' <<<"$existing"); no automatic resubmit"; return 1
      fi
    fi
    # jobs that existed before this attempt (FAILED ones under V6M_RESUBMIT_FAILED=1) are never
    # mistaken for the job this attempt creates
    prior_ids=$(cut -f1 <<<"$existing")
    log "submit $arm seed$seed ($dn), attempt $attempt/$SUBMIT_ATTEMPTS"
    tmp=$(mktemp "$RUNDIR/.attempt.XXXXXX"); rc=""
    ( run "$arm" "$seed" 2>&1 | tee "$tmp" | tee -a "$pairlog" ) &
    lpid=$!
    id=""; t0=$(date +%s)
    # wait until the job exists (id printed or found by name) or the launcher exits
    while [[ -z "$id" ]]; do
      id=$(sed -n 's#^submitted job: .*/customJobs/\([0-9]*\)$#\1#p' "$tmp" | tail -1)
      [[ -n "$id" ]] && break
      if ! kill -0 "$lpid" 2>/dev/null; then
        rc=0; wait "$lpid" || rc=$?
        id=$(sed -n 's#^submitted job: .*/customJobs/\([0-9]*\)$#\1#p' "$tmp" | tail -1)
        [[ -n "$id" ]] && break
        log "launcher exited rc=$rc before a job id appeared"
        break
      fi
      if (( $(date +%s) - t0 > CREATE_WAIT )); then
        log "launcher shows no job after ${CREATE_WAIT}s; killing it"
        kill_tree "$lpid"; wait "$lpid" 2>/dev/null || true
        break
      fi
      sleep 10
    done
    if [[ -z "$id" ]]; then
      sleep "$BACKOFF"
      existing=$(lookup "$dn") || { rm -f "$tmp"; log "STOP: job lookup failed for $dn after launcher ended"; return 1; }
      id=$(cut -f1 <<<"$existing" | grep -vxF -e "$prior_ids" -e "" | head -1 || true)
    fi
    if [[ -n "$id" ]]; then
      log "$arm seed$seed: job $id exists; polling it independently of the launcher"
      state=$(poll_job "$id" | tail -1)
      log "$arm seed$seed: job $id final: $state"
      stop_launcher "$lpid"
      rm -f "$tmp"
      [[ "$state" == JOB_STATE_SUCCEEDED ]] && return 0
      log "STOP: $arm seed$seed ended $state (job $id); no automatic resubmit"; return 1
    fi
    rm -f "$tmp"
    rc=${rc:-?}
    log "launcher rc=$rc and no job exists for $dn; submission failed before create"
    (( attempt < SUBMIT_ATTEMPTS )) && sleep $(( 4 * BACKOFF * attempt * attempt ))
  done
  log "STOP: $arm seed$seed submission failed $SUBMIT_ATTEMPTS times"; return 1
}

submit_all() {
  local from_arm="" from_seed="" started=1 a s n=0
  if [[ "${1:-}" == --from ]]; then
    from_arm=${2:-}; from_seed=${3:-}; started=0
    check_pair "$from_arm" "$from_seed"
  elif [[ $# -gt 0 ]]; then
    die "unexpected argument '$1'"
  fi
  log "submit-all start (pid $$) from ${from_arm:-L1} seed${from_seed:-0}"
  for a in "${ARMS[@]}"; do for s in 0 1 2; do
    [[ $started == 0 && "$a" == "$from_arm" && "$s" == "$from_seed" ]] && started=1
    [[ $started == 0 ]] && continue
    if ! do_pair "$a" "$s"; then log "submit-all stopped at $a seed$s"; exit 1; fi
    n=$((n+1))
  done; done
  log "submit-all done: $n pair(s) SUCCEEDED/skipped in this invocation"
}

case "${1:-}" in
  ""|help|-h|--help) usage; exit 0 ;;
  preflight)
    [[ $# -eq 1 || $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config image
    [[ $# -eq 3 ]] && check_pair "$2" "$3"
    check_repo
    if [[ $# -eq 3 ]]; then run "$2" "$3" --preflight
    else for a in "${ARMS[@]}"; do for s in 0 1 2; do run "$a" "$s" --preflight; done; done; fi ;;
  submit)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config run; check_pair "$2" "$3"; check_repo
    do_pair "$2" "$3" ;;
  state)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config; check_pair "$2" "$3"
    lookup "$(display_name "$2" "$3")"; echo ;;
  submit-all)
    shift
    load_config run
    if [[ "${1:-}" == --from ]]; then check_pair "${2:-}" "${3:-}"; fi
    check_repo
    submit_all "$@" ;;
  *) usage >&2; exit 2 ;;
esac
