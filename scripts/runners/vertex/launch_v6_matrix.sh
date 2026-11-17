#!/usr/bin/env bash
# v6 ablation matrix: 9 arms x seeds 0/1/2 = 27 Vertex jobs, submitted one at a time through
# scripts/operators/launch_vertex_modeling.py (cosine lr schedule, prebuilt patch cache).
# The submit paths spend money: run them only with the project owner's explicit OK.
#
# Usage:  launch_v6_matrix.sh preflight [ARM SEED]          # launcher --preflight only, submits nothing
#         launch_v6_matrix.sh submit ARM SEED               # one job (blocks until terminal)
#         launch_v6_matrix.sh submit-all [--from ARM SEED]  # sequential jobs in fixed order, resumable
#         launch_v6_matrix.sh state ARM SEED                # existing jobs for the pair (read-only)
#         launch_v6_matrix.sh holdout --max-total-usd USD [--arms A,B|all] [--seeds 0,1,2]
#                             [--assignments v1|cv5-fold0,...] [--max-parallel N] [--include-stage1]
#                             [--dry-run | --preflight]     # spatial holdout batch (docs/spatial-holdout.md)
#         launch_v6_matrix.sh help | -h | --help            # this text, touches nothing
#
# `holdout` submits the spatial holdout runs (mode <matrix mode>-spatial-holdout, --skip-test, job-local
# cache, 10800 s timeout, 600 s queue allowance) of the given arms and seeds, seeds first, with at most
# N (default 4, max 4) jobs in flight. It is idempotent by displayName and resumable (run it again with
# the same arguments). Before the first submission it looks up every run, and refuses (exit 2) when the
# summed per-job exposure caps of the runs it would start exceed --max-total-usd; a pair whose only jobs
# are not resubmittable (see V6M_RESUBMIT_*) refuses the whole batch (exit 1). --dry-run makes no cloud
# call and prices every requested run. A run that is cancelled by the queue rule or whose submission is
# refused for capacity or quota ends with code 3: the batch goes on with the remaining runs. Any other
# failure stops new submissions; jobs in flight are waited for, never cancelled. Exit codes: 0 all runs
# succeeded or were skipped, 1 failure, 2 refused, 3 only capacity failures.
# L1 and L4 seed 0 (stage 1, already SUCCEEDED under another label) are skipped unless --include-stage1;
# that applies to the assignment v1 only.
# --assignments (default v1) selects the spatial assignment of every run: v1, or the cross-validation
# folds cv5-fold0 ... cv5-fold4, which exist for arms L1 and L4 at seed 0 only. A fold run carries its
# assignment in the run label and the displayName (<label>-<arm>-cv5-fold0-seed0-...), so it never
# collides with a v1 run or another fold. The image must be built from a commit that knows the folds.
# Jobs already active are watched first and hold slots; their pairs never submit. One batch per
# V6M_RUN_DIR (lock directory holdout.lock). A cancel that fails (CANCEL_UNCONFIRMED) or a worker that
# dies without a result is a failure, not a capacity miss.
#
# Environment (V6M_*):
#   Required (no default):
#     V6M_IMAGE_URI     runner image pinned by digest (repo/name@sha256:<64 hex>)
#     V6M_SOURCE_SHA    40 lowercase hex; the checkout HEAD must equal it and the tree must be clean
#     V6M_LABEL_PREFIX  run-label prefix, ^[a-z0-9][a-z0-9-]*$
#     V6M_RUN_DIR       (submit, submit-all) existing directory OUTSIDE the repo for logs/temp files
#   Optional (default):
#     V6M_PROJECT (berlin-lst-training)  V6M_REGION (europe-west3)   must equal the launcher pin
#     V6M_SERVICE_ACCOUNT (berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com)
#     V6M_INFISICAL_IDENTITY (7ba603e5-b94d-42d1-bc57-d64658dde09d)   public identifier
#     V6M_INFISICAL_PROJECT (5da7dfb7-954d-4736-ba2e-4471ade9d766)    public identifier
#     V6M_HOURLY_RATE_USD (1.00)   V6M_TIMEOUT_SECONDS (14400)   V6M_MAX_WAIT_SECONDS (1800)
#     V6M_MAX_EPOCHS (40)          V6M_POLL_SECONDS (60)         V6M_GCLOUD_TIMEOUT (120)
#     V6M_LAUNCHER_GRACE (120)     V6M_CREATE_WAIT (900)         V6M_SUBMIT_ATTEMPTS (3)
#     V6M_MAX_POLL_ERR_SECONDS (21600)   V6M_LOOKUP_BACKOFF_SECONDS (30)
#     V6M_GCLOUD (gcloud)          V6M_RESUBMIT_FAILED (0; 1 = operator opt-in, see below)
#     V6M_RESUBMIT_CANCELLED (0; 1 = like V6M_RESUBMIT_FAILED for CANCELLED jobs, e.g. after a queue miss)
#     V6M_BATCH_TICK_SECONDS (5)   (holdout) V6M_TIMEOUT_SECONDS defaults to 10800 and V6M_MAX_WAIT_SECONDS
#                                  to 600 there; the launcher refuses more than 10800 s and USD 3.25 per job
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
#  - Every gcloud call (lookup, poll, cancel) runs through scripts/operators/with-gcp-operator.sh
#    (operator SA, per-command credential override, never --activate), under `timeout`, with
#    --project/--region. The launcher gets the same --project/--region; V6M_PROJECT/V6M_REGION
#    must equal its pin (berlin-lst-training/europe-west3), otherwise the script exits 2.
#  - Queue rule: a job that has not started within MAX_WAIT of creation is cancelled and the loop
#    stops (final token QUEUE_DEADLINE_MISSED). A job that reached a terminal state is never
#    cancelled and always reports its real Vertex state; a late start only adds a
#    QUEUE_DEADLINE_VIOLATION log line, so a late-start SUCCEEDED still continues the loop.
#  - Only JOB_STATE_SUCCEEDED continues the loop; anything else stops it cleanly with a STOP log line.
# Needs bash 4+, GNU `timeout` and GNU `date -d`, i.e. a Linux operator machine (on macOS: brew install
# coreutils and put its gnubin directory on PATH); every cloud subcommand checks this first and exits 2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
SELF="$(basename "$0")"

declare -A MODE=( [L1]=v6-l1 [L2]=v6-l2 [L3]=v6-l3 [L4]=v6-l4 [L5]=v6-l5 [LOO-M]=v6-loo-morph
  [LOO-S]=v6-loo-shadow [NO-NDVI-NDWI]=v6-no-ndvi-ndwi [NO-RED-NIR-IDX]=v6-no-red-nir-index )
ARMS=(L1 L2 L3 L4 L5 LOO-M LOO-S NO-NDVI-NDWI NO-RED-NIR-IDX)
# pinned in scripts/operators/launch_vertex_modeling.py (PROJECT/REGION)
PIN_PROJECT=berlin-lst-training
PIN_REGION=europe-west3
# pinned in scripts/operators/launch_vertex_modeling.py (SPATIAL_HOLDOUT_*); a test compares them
HOLDOUT_TIMEOUT_SECONDS=10800
HOLDOUT_EXPOSURE_CENTS=325
# pinned in src/berlin_lst_downscaling/modeling/spatial_holdout.py (SPATIAL_HOLDOUT_CV_*); a test compares them
CV_ASSIGNMENTS=(cv5-fold0 cv5-fold1 cv5-fold2 cv5-fold3 cv5-fold4)
CV_ARMS=(L1 L4)
VARIANT=matrix  # matrix | holdout (the `holdout` subcommand)
OPERATOR_WRAPPER="$REPO_ROOT/scripts/operators/with-gcp-operator.sh"

usage() {
  cat <<USAGE
usage: $SELF preflight [ARM SEED]
       $SELF submit ARM SEED
       $SELF submit-all [--from ARM SEED]
       $SELF state ARM SEED
       $SELF holdout --max-total-usd USD [--arms A,B,...|all] [--seeds 0,1,2] [--assignments v1|cv5-fold0,...] [--max-parallel N] [--include-stage1] [--dry-run|--preflight]
       $SELF help

ARM: ${ARMS[*]}    SEED: 0 | 1 | 2

preflight/submit/submit-all need a clean checkout at V6M_SOURCE_SHA. submit/submit-all spend money.

Required env: V6M_IMAGE_URI (repo/name@sha256:<64 hex>), V6M_SOURCE_SHA (40 hex),
  V6M_LABEL_PREFIX ([a-z0-9][a-z0-9-]*), V6M_RUN_DIR (submit/submit-all; existing dir outside the repo).
  state needs only V6M_LABEL_PREFIX; preflight needs no V6M_RUN_DIR.
Optional env (default): V6M_PROJECT (berlin-lst-training), V6M_REGION (europe-west3); both must
  equal the launcher pin.
  V6M_SERVICE_ACCOUNT, V6M_INFISICAL_IDENTITY, V6M_INFISICAL_PROJECT, V6M_HOURLY_RATE_USD (1.00),
  V6M_TIMEOUT_SECONDS (14400), V6M_MAX_WAIT_SECONDS (1800), V6M_MAX_EPOCHS (40), V6M_POLL_SECONDS (60),
  V6M_GCLOUD_TIMEOUT (120), V6M_LAUNCHER_GRACE (120), V6M_CREATE_WAIT (900), V6M_SUBMIT_ATTEMPTS (3),
  V6M_MAX_POLL_ERR_SECONDS (21600), V6M_LOOKUP_BACKOFF_SECONDS (30), V6M_GCLOUD (gcloud),
  V6M_RESUBMIT_FAILED (0; 1 = resubmit once when every existing job is FAILED).
USAGE
}

die() { echo "$SELF: $*" >&2; exit 2; }

# GNU tools the cloud paths rely on (V6M_TIMEOUT_BIN / V6M_DATE_BIN name them for a test or a gnubin-less PATH)
TIMEOUT_BIN="${V6M_TIMEOUT_BIN:-timeout}"
DATE_BIN="${V6M_DATE_BIN:-date}"
GNU_FIX="on macOS run: brew install coreutils, then put \"\$(brew --prefix coreutils)/libexec/gnubin\" first on PATH"
require_gnu_tools() {  # every subcommand that touches the cloud; fails at once instead of retrying for minutes
  "$TIMEOUT_BIN" 5 true >/dev/null 2>&1 \
    || die "GNU 'timeout' is missing or not working ($TIMEOUT_BIN); $GNU_FIX"
  "$DATE_BIN" -d 2026-01-01T00:00:00Z +%s >/dev/null 2>&1 \
    || die "'$DATE_BIN -d' is not GNU date (BSD date has no -d); $GNU_FIX"
}

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
  if [[ "$VARIANT" == holdout ]]; then
    TIMEOUT="${V6M_TIMEOUT_SECONDS:-$HOLDOUT_TIMEOUT_SECONDS}"
    MAX_WAIT="${V6M_MAX_WAIT_SECONDS:-600}"
  else
    TIMEOUT="${V6M_TIMEOUT_SECONDS:-14400}"
    MAX_WAIT="${V6M_MAX_WAIT_SECONDS:-1800}"
  fi
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
  RESUBMIT_CANCELLED="${V6M_RESUBMIT_CANCELLED:-0}"
  TICK="${V6M_BATCH_TICK_SECONDS:-5}"
  local n
  for n in TIMEOUT MAX_WAIT MAX_EPOCHS POLL GCLOUD_TIMEOUT LAUNCHER_GRACE CREATE_WAIT \
           SUBMIT_ATTEMPTS MAX_POLL_ERR_SECONDS; do
    pos_int "V6M_$n" "${!n}"
  done
  [[ "$BACKOFF" =~ ^[0-9]+$ ]] || die "V6M_LOOKUP_BACKOFF_SECONDS must be a non-negative integer"
  [[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ && "$RATE" =~ [1-9] ]] || die "V6M_HOURLY_RATE_USD must be a positive decimal"
  [[ "$RESUBMIT_FAILED" =~ ^[01]$ ]] || die "V6M_RESUBMIT_FAILED must be 0 or 1"
  [[ "$RESUBMIT_CANCELLED" =~ ^[01]$ ]] || die "V6M_RESUBMIT_CANCELLED must be 0 or 1"
  pos_int V6M_BATCH_TICK_SECONDS "$TICK"
  if [[ "$VARIANT" == holdout ]]; then
    # the launcher refuses these too; failing here keeps a misconfigured batch off the cloud
    (( TIMEOUT <= HOLDOUT_TIMEOUT_SECONDS )) \
      || die "V6M_TIMEOUT_SECONDS $TIMEOUT exceeds the holdout cap $HOLDOUT_TIMEOUT_SECONDS"
    awk -v r="$RATE" -v t="$TIMEOUT" -v w="$MAX_WAIT" -v cap="$HOLDOUT_EXPOSURE_CENTS" \
      'BEGIN { exit !(r <= 1.0 && r * (t + w) / 3600 * 100 <= cap + 1e-9) }' \
      || die "holdout rate/timeout/queue allowance exceed the \$1.00/h cap or the per-job exposure cap"
  fi
  [[ -n "$PROJECT" && -n "$REGION" && -n "$SA" && -n "$INF_ID" && -n "$INF_PROJECT" ]] \
    || die "V6M_PROJECT/REGION/SERVICE_ACCOUNT/INFISICAL_* must not be empty"
  [[ "$PROJECT" == "$PIN_PROJECT" && "$REGION" == "$PIN_REGION" ]] \
    || die "V6M_PROJECT/V6M_REGION ($PROJECT/$REGION) must equal the launcher pin $PIN_PROJECT/$PIN_REGION"
  [[ -f "$OPERATOR_WRAPPER" && -x "$OPERATOR_WRAPPER" ]] \
    || die "operator wrapper missing or not executable: $OPERATOR_WRAPPER"
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

# the run-label part of a fold run: "" for v1, "-cv5-foldK" otherwise
asg_suffix() { [[ "${1:-v1}" == v1 ]] || printf -- '-%s' "$1"; }

display_name() {  # arm seed [assignment] -> exact Vertex displayName (mode-runlabel, as built by the launcher)
  local arm=$1 seed=$2 asg=${3:-v1}
  if [[ "$VARIANT" == holdout ]]; then
    echo "${MODE[$arm]}-spatial-holdout-${LABEL_PREFIX}-$(lower "$arm")$(asg_suffix "$asg")-seed${seed}-ep${MAX_EPOCHS}-cos-notest"
  else
    echo "${MODE[$arm]}-${LABEL_PREFIX}-$(lower "$arm")-seed${seed}-ep${MAX_EPOCHS}-cos"
  fi
}

run() {  # arm seed assignment [--preflight]
  local arm=$1 seed=$2 asg=$3; shift 3
  local label="${LABEL_PREFIX}-$(lower "$arm")" mode="${MODE[$arm]}"
  local -a variant_args=(--patch-cache prebuilt)
  if [[ "$VARIANT" == holdout ]]; then  # job-local cache is the launcher default; it refuses prebuilt
    mode="$mode-spatial-holdout"; variant_args=(--skip-test)
    # the launcher appends the assignment to the run label (and refuses all but L1/L4 at seed 0)
    [[ "$asg" == v1 ]] || variant_args+=(--assignment "$asg")
  fi
  cd "$REPO_ROOT"
  # operator SA key from Infisical /operator into a 0600 temp file for this command only
  "$OPERATOR_WRAPPER" -- \
    uv run --group operators python scripts/operators/launch_vertex_modeling.py \
    --mode "$mode" --seed "$seed" --max-epochs "$MAX_EPOCHS" --lr-schedule cosine \
    "${variant_args[@]}" \
    --image-uri "$IMAGE" --source-sha "$SHA" --run-label "$label" \
    --service-account "$SA" --project "$PROJECT" --region "$REGION" \
    --infisical-identity "$INF_ID" --infisical-project "$INF_PROJECT" \
    --infisical-env dev --infisical-path /vertex \
    --hourly-rate-usd "$RATE" --timeout-seconds "$TIMEOUT" --max-wait-seconds "$MAX_WAIT" "$@"
}

# gcloud as the operator SA for this command only (same wrapper as run(), no --activate);
# `timeout` covers the credential fetch as well as the gcloud call.
gc() { "$TIMEOUT_BIN" -k 10 "$GCLOUD_TIMEOUT" "$OPERATOR_WRAPPER" -- "$GCLOUD" "$@" --project "$PROJECT"; }

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

# Poll one job until terminal; echoes the real Vertex terminal state on stdout (last line), or
# QUEUE_DEADLINE_MISSED (not started within MAX_WAIT, cancel requested), CANCEL_UNCONFIRMED (same,
# but the cancel call failed), WAIT_EXPIRED, STATUS_UNKNOWN. A terminal job is never cancelled. Logs go to stderr.
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
          if [[ -n "${st:-}" ]] && (( $("$DATE_BIN" -d "$st" +%s) > $("$DATE_BIN" -d "$ct" +%s) + MAX_WAIT )); then
            log "QUEUE_DEADLINE_VIOLATION: job $id started at $st, more than ${MAX_WAIT}s after creation ($ct); terminal state $state is reported as is, no cancel" >&2
          fi
          echo "$state"; return 0 ;;
      esac
      now=$(date +%s); ct_s=$("$DATE_BIN" -d "$ct" +%s)
      if [[ -z "${st:-}" && "$state" != JOB_STATE_RUNNING ]] && (( now > ct_s + MAX_WAIT )); then
        # same rule as the launcher: not started within max-wait -> cancel, never replace
        log "QUEUE_DEADLINE_VIOLATION: job $id ($state) not started within ${MAX_WAIT}s of creation; requesting cancellation (launcher queue rule)" >&2
        if ! gc ai custom-jobs cancel "$id" --region "$REGION" >&2 2>&1; then
          # the old job may still start and run: not a capacity miss, nothing may replace it
          log "cancel request for $id FAILED; job $id may still start, inspect it manually" >&2
          echo "CANCEL_UNCONFIRMED"; return 0
        fi
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

# 0 when every listed job is in a state the operator opted to resubmit (V6M_RESUBMIT_*)
resubmittable() {  # lookup lines
  local pat=""
  [[ "$RESUBMIT_FAILED" == 1 ]] && pat=$'\tJOB_STATE_FAILED$'
  [[ "$RESUBMIT_CANCELLED" == 1 ]] && pat+="${pat:+|}"$'\tJOB_STATE_CANCELLED$'
  [[ -n "$pat" ]] && ! grep -qvE "$pat" <<<"$1"
}

# Submission refusals that mean "no capacity" (batch: continue with the other runs)
CAPACITY_RE='RESOURCE_EXHAUSTED|quota|stockout|insufficient capacity|no available capacity'

# One pair, idempotent. Returns 0 only if the pair ends SUCCEEDED; 3 when the pair got no
# capacity (queue rule cancel, or a submission refused for capacity); 1 for any other outcome.
do_pair() {  # arm seed [assignment]
  local arm=$1 seed=$2 asg=${3:-v1} dn existing prior_ids id state attempt rc tmp pairlog lpid t0 capacity=0
  dn=$(display_name "$arm" "$seed" "$asg")
  pairlog="$RUNDIR/submit-${arm}-seed${seed}$(asg_suffix "$asg").log"
  for attempt in $(seq 1 "$SUBMIT_ATTEMPTS"); do
    existing=$(lookup "$dn") || { log "STOP: cannot list jobs for $dn (idempotency check impossible)"; return 1; }
    # a pair planned as "attach" (its job was active when the batch was priced) never submits
    if [[ "${ATTACH_ONLY:-0}" == 1 ]] \
       && ! grep -qE $'\tJOB_STATE_(SUCCEEDED|QUEUED|PENDING|RUNNING|CANCELLING|PAUSED|UPDATING)$' <<<"$existing"; then
      log "STOP: $arm seed$seed was planned as attached to an active job, which is no longer active ($(tr '\n\t' '; ' <<<"$existing")); nothing was priced for a new job"
      return 1
    fi
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
        log "STOP: $arm seed$seed ended $state (job $id); no automatic resubmit"
        [[ "$state" == QUEUE_DEADLINE_MISSED ]] && return 3
        return 1
      fi
      if resubmittable "$existing"; then
        log "$arm seed$seed: only resubmittable jobs exist ($(tr '\n\t' '; ' <<<"$existing")); V6M_RESUBMIT_* set by operator, resubmitting"
      else
        log "STOP: $arm seed$seed has only non-successful jobs: $(tr '\n\t' '; ' <<<"$existing"); no automatic resubmit"; return 1
      fi
    fi
    # jobs that existed before this attempt (FAILED ones under V6M_RESUBMIT_FAILED=1) are never
    # mistaken for the job this attempt creates
    prior_ids=$(cut -f1 <<<"$existing")
    log "submit $arm seed$seed ($dn), attempt $attempt/$SUBMIT_ATTEMPTS"
    tmp=$(mktemp "$RUNDIR/.attempt.XXXXXX"); rc=""
    ( run "$arm" "$seed" "$asg" 2>&1 | tee "$tmp" | tee -a "$pairlog" ) &
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
      sleep 2
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
      log "STOP: $arm seed$seed ended $state (job $id); no automatic resubmit"
      [[ "$state" == QUEUE_DEADLINE_MISSED ]] && return 3
      return 1
    fi
    grep -qiE "$CAPACITY_RE" "$tmp" 2>/dev/null && capacity=1
    rm -f "$tmp"
    rc=${rc:-?}
    log "launcher rc=$rc and no job exists for $dn; submission failed before create"
    (( attempt < SUBMIT_ATTEMPTS )) && sleep $(( 4 * BACKOFF * attempt * attempt ))
  done
  log "STOP: $arm seed$seed submission failed $SUBMIT_ATTEMPTS times"
  (( capacity )) && return 3
  return 1
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

to_cents() {  # "12.5" -> REPLY=1250 (at most two decimals; no subshell, so die ends the script)
  [[ "$1" =~ ^([0-9]+)(\.([0-9]{1,2}))?$ ]] || die "not a USD amount with at most two decimals: '$1'"
  local frac="${BASH_REMATCH[3]:-0}"
  [[ ${#frac} -eq 1 ]] && frac="${frac}0"
  REPLY=$(( 10#${BASH_REMATCH[1]} * 100 + 10#$frac ))
}
fmt_cents() { printf '%d.%02d' $(( $1 / 100 )) $(( $1 % 100 )); }

# Spatial holdout batch: the given arms x seeds (seeds first) with at most --max-parallel jobs in flight.
# "L1/seed0" for v1, "L1/seed0/cv5-fold0" for a fold run; ptext: "L1 seed0" / "L1 seed0 cv5-fold0"
pname() { printf '%s/seed%s%s' "$1" "$2" "$([[ "${3:-v1}" == v1 ]] || printf '/%s' "$3")"; }
ptext() { printf '%s seed%s%s' "$1" "$2" "$([[ "${3:-v1}" == v1 ]] || printf ' %s' "$3")"; }

holdout_batch() {
  local arms_arg=all seeds_arg=0,1,2 asg_arg=v1 par=4 max_total="" dry=0 pre=0 stage1=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --arms) arms_arg=${2:-}; shift 2 ;;
      --seeds) seeds_arg=${2:-}; shift 2 ;;
      --assignments) asg_arg=${2:-}; shift 2 ;;
      --max-parallel) par=${2:-}; shift 2 ;;
      --max-total-usd) max_total=${2:-}; shift 2 ;;
      --dry-run) dry=1; shift ;;
      --preflight) pre=1; shift ;;
      --include-stage1) stage1=1; shift ;;
      *) die "unexpected argument '$1'" ;;
    esac
  done
  [[ "$par" =~ ^[1-4]$ ]] || die "--max-parallel must be 1, 2, 3 or 4, got '$par'"
  [[ $dry == 0 || $pre == 0 ]] || die "--dry-run and --preflight exclude each other"
  [[ -n "$max_total" || $pre == 1 ]] || die "--max-total-usd is required (summed per-job exposure caps must stay within it)"
  local -a arms=() seeds=() asgs=() plan=()
  if [[ "${arms_arg,,}" == all ]]; then arms=("${ARMS[@]}"); else IFS=, read -r -a arms <<<"${arms_arg^^}"; fi
  IFS=, read -r -a seeds <<<"$seeds_arg"
  IFS=, read -r -a asgs <<<"$asg_arg"
  [[ ${#arms[@]} -gt 0 && ${#seeds[@]} -gt 0 && ${#asgs[@]} -gt 0 ]] || die "--arms, --seeds and --assignments must not be empty"
  local a s g n pair
  for a in "${arms[@]}"; do for s in "${seeds[@]}"; do check_pair "$a" "$s"; done; done
  for n in "${arms[@]}"; do [[ $(printf '%s\n' "${arms[@]}" | grep -cxF -- "$n") == 1 ]] || die "duplicate arm '$n'"; done
  for n in "${seeds[@]}"; do [[ $(printf '%s\n' "${seeds[@]}" | grep -cxF -- "$n") == 1 ]] || die "duplicate seed '$n'"; done
  for n in "${asgs[@]}"; do [[ $(printf '%s\n' "${asgs[@]}" | grep -cxF -- "$n") == 1 ]] || die "duplicate assignment '$n'"; done
  for g in "${asgs[@]}"; do
    [[ "$g" == v1 ]] && continue
    [[ " ${CV_ASSIGNMENTS[*]} " == *" $g "* ]] || die "unknown assignment '$g' (expected v1 or one of: ${CV_ASSIGNMENTS[*]})"
    # the folds exist for two arms and one seed; the launcher and the config guard refuse the rest too
    for a in "${arms[@]}"; do
      [[ " ${CV_ARMS[*]} " == *" $a "* ]] || die "assignment $g is defined for arms ${CV_ARMS[*]} only, not $a"
    done
    for s in "${seeds[@]}"; do [[ "$s" == 0 ]] || die "assignment $g is defined for seed 0 only, not seed $s"; done
  done
  for s in "${seeds[@]}"; do for g in "${asgs[@]}"; do for a in "${arms[@]}"; do
    if [[ $g == v1 && $stage1 == 0 && "$s" == 0 && ( "$a" == L1 || "$a" == L4 ) ]]; then
      # stage 1 ran these two under another label prefix; their displayNames would not match
      echo "NOTE $a seed0 is a stage-1 run (job ${MODE[$a]}-spatial-holdout-v6sh-99a71cf-$(lower "$a")-seed0-ep${MAX_EPOCHS}-cos-notest, SUCCEEDED): skipped; --include-stage1 runs it again"
      continue
    fi
    plan+=("$a $s $g")
  done; done; done
  [[ ${#plan[@]} -gt 0 ]] || die "nothing to run: only stage-1 runs were requested (see --include-stage1)"
  local cap=$HOLDOUT_EXPOSURE_CENTS limit=0 total=0
  if [[ -n "$max_total" ]]; then to_cents "$max_total"; limit=$REPLY; fi

  if [[ $dry == 1 ]]; then  # no cloud call: every requested run is priced, none is looked up
    log "holdout dry run: ${#plan[@]} run(s), at most $par in flight; no cloud call, no idempotency lookup"
    for pair in "${plan[@]}"; do
      read -r a s g <<<"$pair"
      printf 'PLAN %-14s seed%s label=%s-%s%s jobname=%s mode=%s-spatial-holdout cap=USD %s\n' \
        "$a" "$s" "$LABEL_PREFIX" "$(lower "$a")" "$(asg_suffix "$g")" "$(display_name "$a" "$s" "$g")" "${MODE[$a]}" "$(fmt_cents $cap)"
      total=$(( total + cap ))
    done
    log "summed exposure caps of all ${#plan[@]} run(s): USD $(fmt_cents $total); limit USD $(fmt_cents $limit)"
    (( total <= limit )) || { log "REFUSED: summed exposure caps exceed --max-total-usd"; exit 2; }
    return 0
  fi
  if [[ $pre == 1 ]]; then
    check_repo
    for pair in "${plan[@]}"; do read -r a s g <<<"$pair"; run "$a" "$s" "$g" --preflight; done
    return 0
  fi

  # one batch at a time per run directory: two concurrent batches would submit the same runs
  HOLDOUT_LOCK="$RUNDIR/holdout.lock"
  if ! mkdir "$HOLDOUT_LOCK" 2>/dev/null; then
    die "another holdout batch holds the lock $HOLDOUT_LOCK (pid $(cat "$HOLDOUT_LOCK/pid" 2>/dev/null || echo unknown)); if none is running, remove that directory"
  fi
  echo $$ >"$HOLDOUT_LOCK/pid"
  trap 'rm -rf "$HOLDOUT_LOCK"' EXIT
  # look every run up first: what would be started decides the total, nothing is submitted before
  check_repo
  local dn existing status blocked=0 skipped=0
  local -a queue=() attached=()
  local -A is_attach=()
  for pair in "${plan[@]}"; do
    read -r a s g <<<"$pair"
    dn=$(display_name "$a" "$s" "$g")
    existing=$(lookup "$dn") || { log "STOP: cannot list jobs for $dn (idempotency check impossible)"; exit 1; }
    if grep -q $'\tJOB_STATE_SUCCEEDED$' <<<"$existing"; then status=skip
    elif grep -qE $'\tJOB_STATE_(QUEUED|PENDING|RUNNING|CANCELLING|PAUSED|UPDATING)$' <<<"$existing"; then status=attach
    elif [[ -z "$existing" ]]; then status=start
    elif resubmittable "$existing"; then status=restart
    else status=BLOCKED; blocked=$((blocked + 1)); fi
    printf 'PLAN %-14s seed%s %-8s jobname=%s\n' "$a" "$s" "$status" "$dn"
    case "$status" in
      skip) skipped=$((skipped + 1)) ;;
      attach) attached+=("$pair"); is_attach["$pair"]=1 ;;
      *) queue+=("$pair"); [[ $status == BLOCKED ]] || total=$(( total + cap )) ;;
    esac
  done
  # jobs that are already active are watched first and hold their slots; they can never submit
  queue=("${attached[@]}" "${queue[@]}")
  log "plan: ${#attached[@]} active job(s) to watch first, ${#queue[@]} run(s) in all, $skipped already SUCCEEDED; to submit: summed exposure caps USD $(fmt_cents $total), limit USD $(fmt_cents $limit)"
  (( blocked == 0 )) || { log "REFUSED: $blocked pair(s) have only non-successful jobs (see V6M_RESUBMIT_FAILED, V6M_RESUBMIT_CANCELLED)"; exit 1; }
  (( total <= limit )) || { log "REFUSED: summed exposure caps exceed --max-total-usd"; exit 2; }

  local -A inflight=()  # pid -> "arm seed"
  local -a ok=() capacity_miss=() failed=() unstarted=()
  local i=0 stop=0 pid rc rcf st
  trap 'for pid in "${!inflight[@]}"; do kill_tree "$pid"; done; exit 130' INT TERM
  while (( (stop == 0 && i < ${#queue[@]}) || ${#inflight[@]} > 0 )); do
    while (( stop == 0 && i < ${#queue[@]} && ${#inflight[@]} < par )); do
      read -r a s g <<<"${queue[i]}"; i=$((i + 1))
      rcf="$RUNDIR/.holdout-rc-${a}-seed${s}$(asg_suffix "$g")"; rm -f "$rcf"
      ( ATTACH_ONLY=${is_attach["$a $s $g"]:-0}; rc=0; do_pair "$a" "$s" "$g" || rc=$?; echo "$rc" >"$rcf.tmp"; mv "$rcf.tmp" "$rcf" ) &
      inflight[$!]="$a $s $g"
      log "started $(ptext "$a" "$s" "$g") (in flight ${#inflight[@]}/$par)"
    done
    for pid in "${!inflight[@]}"; do
      read -r a s g <<<"${inflight[$pid]}"
      rcf="$RUNDIR/.holdout-rc-${a}-seed${s}$(asg_suffix "$g")"
      if [[ ! -f "$rcf" ]]; then
        # no result yet: still running, or dead without one (a finished child stays a zombie until waited for)
        st=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        [[ -z "$st" || "$st" == Z* ]] || continue
        wait "$pid" 2>/dev/null || true
        if [[ ! -f "$rcf" ]]; then
          log "$(ptext "$a" "$s" "$g"): its worker (pid $pid) ended without a result; its job, if any, is still on Vertex: look it up by displayName $(display_name "$a" "$s" "$g")"
          echo 1 >"$rcf"
        fi
      fi
      wait "$pid" 2>/dev/null || true
      rc=$(<"$rcf"); rm -f "$rcf"
      unset "inflight[$pid]"
      case "$rc" in
        0) ok+=("$(pname "$a" "$s" "$g")") ;;
        3) capacity_miss+=("$(pname "$a" "$s" "$g")"); log "$(ptext "$a" "$s" "$g"): no capacity; continuing with the remaining runs" ;;
        *) failed+=("$(pname "$a" "$s" "$g")"); stop=1; log "STOP: $(ptext "$a" "$s" "$g") failed (rc $rc); no further run is started, jobs in flight are waited for" ;;
      esac
    done
    (( ${#inflight[@]} > 0 )) && sleep "$TICK"
  done
  trap - INT TERM
  while (( i < ${#queue[@]} )); do
    read -r a s g <<<"${queue[i]}"; unstarted+=("$(pname "$a" "$s" "$g")"); i=$((i + 1))
  done
  log "holdout batch: ${#ok[@]} succeeded, $skipped skipped (already SUCCEEDED), ${#capacity_miss[@]} without capacity [${capacity_miss[*]:-}], ${#failed[@]} failed [${failed[*]:-}], ${#unstarted[@]} not started [${unstarted[*]:-}]"
  (( ${#failed[@]} == 0 )) || exit 1
  (( ${#capacity_miss[@]} == 0 )) || { log "re-run the same command to retry the runs without capacity (V6M_RESUBMIT_CANCELLED=1 after a queue cancel)"; exit 3; }
}

case "${1:-}" in
  ""|help|-h|--help) usage; exit 0 ;;
  preflight)
    [[ $# -eq 1 || $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config image
    require_gnu_tools
    [[ $# -eq 3 ]] && check_pair "$2" "$3"
    check_repo
    if [[ $# -eq 3 ]]; then run "$2" "$3" v1 --preflight
    else for a in "${ARMS[@]}"; do for s in 0 1 2; do run "$a" "$s" v1 --preflight; done; done; fi ;;
  submit)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config run; check_pair "$2" "$3"; require_gnu_tools; check_repo
    do_pair "$2" "$3" || exit 1 ;;
  state)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    load_config; check_pair "$2" "$3"; require_gnu_tools
    lookup "$(display_name "$2" "$3")"; echo ;;
  submit-all)
    shift
    load_config run
    if [[ "${1:-}" == --from ]]; then check_pair "${2:-}" "${3:-}"; fi
    require_gnu_tools
    check_repo
    submit_all "$@" ;;
  holdout)
    shift
    VARIANT=holdout
    if [[ " $* " == *" --dry-run "* ]]; then load_config
    elif [[ " $* " == *" --preflight "* ]]; then load_config image
    else load_config run; fi
    [[ " $* " == *" --dry-run "* ]] || require_gnu_tools  # a dry run makes no cloud call
    holdout_batch "$@" ;;
  *) usage >&2; exit 2 ;;
esac
