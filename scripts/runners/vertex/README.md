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
`docs/vertex-gpu-training.md`. The script holds no secrets: credentials come from
Infisical through `with-gcp-operator.sh`, and `gcloud` lookups use the operator's
existing `gcloud` login.

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
`gcloud` errors tolerated; the queue allowance `V6M_MAX_WAIT_SECONDS` is enforced by
cancel + stop). Only SUCCEEDED continues `submit-all`.

### Not shipped

No notification watchdog and no cache-verification helper ship here. A watchdog would
need a webhook secret, and a safe launch does not need one. The script needs bash 4+
and GNU `date` (a Linux operator machine).

### Epoch numbering

Evidence `best_epoch` / `stopped_epoch` and printed curves are 1-based; Lightning
`current_epoch` and `best-<epoch>-….ckpt` filenames are 0-based. See
[Epoch numbering](../../../docs/vertex-gpu-training.md#epoch-numbering-1-based-summaries-0-based-lightning).
