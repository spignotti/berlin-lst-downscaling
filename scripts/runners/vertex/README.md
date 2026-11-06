# Vertex matrix runners

Operator wrappers for the v6 ablation matrix on Vertex AI. They call the existing
`scripts/operators/with-gcp-operator.sh` and `scripts/operators/launch_vertex_modeling.py`;
there is no second Vertex client here. Background on the Vertex path:
[`docs/vertex-gpu-training.md`](../../../docs/vertex-gpu-training.md).

Coding agents never run these scripts. A paid submit needs the project owner's explicit OK.

## `launch_v6_matrix.sh`

9 arms (`L1`…`L5`, `LOO-M`, `LOO-S`, `NO-NDVI-NDWI`, `NO-RED-NIR-IDX`) x seeds 0/1/2,
40 epochs (`V6M_MAX_EPOCHS`), cosine schedule, prebuilt patch cache.

```bash
scripts/runners/vertex/launch_v6_matrix.sh                     # help only (default)
scripts/runners/vertex/launch_v6_matrix.sh preflight [ARM SEED] # launcher --preflight, submits nothing
scripts/runners/vertex/launch_v6_matrix.sh state ARM SEED       # read-only job lookup by displayName
scripts/runners/vertex/launch_v6_matrix.sh submit ARM SEED      # PAID: one job, blocks until terminal
scripts/runners/vertex/launch_v6_matrix.sh submit-all [--from ARM SEED]  # PAID: fixed order, resumable
```

Run `preflight` first. Only `submit` and `submit-all` can create a job.

### Required environment

| Variable | Meaning |
| --- | --- |
| `V6M_IMAGE_URI` | image pinned by digest (`…@sha256:<64 hex>`) |
| `V6M_SOURCE_SHA` | 40-hex commit; the repo must be clean at exactly this HEAD |
| `V6M_LABEL_PREFIX` | run label prefix, e.g. `v6m-<short-sha>`; the arm and seed are appended |
| `V6M_RUN_DIR` | `submit`/`submit-all` only: existing directory outside the repo for per-pair logs |

A missing or malformed value exits 2 before any `gcloud` or launcher call. Optional
overrides (project, region, service account, Infisical IDs, rate, timeouts, poll
intervals, `V6M_GCLOUD`) are listed in the script header; defaults match
`docs/vertex-gpu-training.md`. `V6M_PROJECT` / `V6M_REGION` must equal the launcher
pin (`berlin-lst-training` / `europe-west3`); any other value exits 2 before any call.
The script passes the same `--project` / `--region` to the launcher and to every
`gcloud` call.

The script holds no secrets. Every launcher run and every `gcloud` call (lookup, poll,
cancel) goes through `scripts/operators/with-gcp-operator.sh -- …`: the operator SA key
comes from Infisical `/operator` into a temp file for that one command, via the
per-command credential override (never `--activate`, so the operator's `gcloud` login
is not changed). Each `gcloud` call fetches the key again, so a status poll costs one
Infisical read. A missing or non-executable wrapper exits 2; a wrapper that fails counts
as a failed `gcloud` call (retried within the usual bounds, then stop).

### Idempotency

Before every submit attempt the script lists jobs with the exact displayName
(`<mode>-<prefix>-<arm>-seed<seed>-ep<epochs>-cos`):

- `JOB_STATE_SUCCEEDED` exists: skip the pair.
- an active job (`QUEUED`, `PENDING`, `RUNNING`, …) exists: wait on it, never submit a second one.
- only `FAILED` / `CANCELLED` / other terminal jobs: stop with
  "no automatic resubmit". `V6M_RESUBMIT_FAILED=1` is an explicit operator opt-in that
  resubmits only when every existing job is `FAILED`.
- lookup impossible: stop.

After submit, the job is polled independently of the launcher process (transient
`gcloud` errors tolerated). Only `JOB_STATE_SUCCEEDED` continues `submit-all`.

### Queue deadline (`V6M_MAX_WAIT_SECONDS`)

- Not started within `MAX_WAIT` of creation (no `startTime`, not `RUNNING`): the script
  requests cancellation, logs `QUEUE_DEADLINE_VIOLATION`, reports `QUEUE_DEADLINE_MISSED`
  and stops. This cost guard is the same rule the launcher applies.
- A job in a terminal state is never cancelled, and the script reports its real Vertex
  state (`JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_CANCELLED`, …). A start more
  than `MAX_WAIT` after creation only adds a `QUEUE_DEADLINE_VIOLATION` log line.
- So a late-start `JOB_STATE_SUCCEEDED` counts as success: `submit-all` continues, and a
  later run skips that pair, as for any other SUCCEEDED job.

### Not shipped

No notification watchdog and no cache-verification helper ship here. A watchdog would
need a webhook secret, and a safe launch does not need one. The script needs bash 4+
and GNU `date` (a Linux operator machine).

### Epoch numbering

Evidence `best_epoch` / `stopped_epoch` and printed curves are 1-based; Lightning
`current_epoch` and `best-<epoch>-….ckpt` filenames are 0-based. See
[Epoch numbering](../../../docs/vertex-gpu-training.md#epoch-numbering-1-based-summaries-0-based-lightning).
