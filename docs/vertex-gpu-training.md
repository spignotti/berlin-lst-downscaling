# Vertex GPU training path

The repo trains on the CPU smoke VM (`berlin-lst-vm`) for pipeline and smoke
work, and on **Vertex AI Custom Training** for GPU work (`AGENTS.md`
§Compute Placement). This document is the launch recipe for the bounded GPU
acceptance smoke of the real-data modeling path (issue #38). It does **not**
cover the Stage-1 full temporal run, which remains a separate explicit
invocation.

## What the path is (and is not)

- It runs the **existing** runner (`scripts/runners/run_modeling.py`) with the
  `vertex_smoke` config through the `real` lifecycle: streamed patch reads from
  the published GCS roots, one epoch, at most four indexed refs per split, one
  GPU, W&B online, checkpoints on the disposable worker disk.
- It is a bounded **infrastructure and lifecycle** check, not a model-quality
  run. It proves device use, streamed I/O, online logging, checkpoint selection
  and reload, and clean teardown.
- It is deliberately a single, capped, on-demand job. There is no persistent
  GPU resource, no automatic retry, and no accelerator or region substitution.

## Region

- **Acceptance smoke: `europe-west4`.** `europe-west3` (Frankfurt) is where the
  bucket and the image live, but it has **no Vertex T4 training quota**, so the
  T4 job cannot start there. `europe-west4` has T4 training quota.
- Running in `europe-west4` while the bucket and Artifact Registry repository
  stay in `europe-west3` adds **cross-region transfer** (`$0.02/GiB` within
  Europe for both Cloud Storage and Artifact Registry) — small for the bounded
  smoke, but additive to the compute estimate.
- **This is a smoke-only choice.** The full Stage-1 run should return to
  `europe-west3` once its T4 quota is granted, to avoid repeated cross-region
  transfer at full split size. A T4 quota increase for `europe-west3` has been
  requested; until it is decided, the full run is blocked on capacity.

## Identities

| Role | Identity | Needs |
|------|----------|-------|
| Submitter (workstation) | the operator's user ADC | `roles/aiplatform.user`; Artifact Registry writer to push the image |
| Worker (Vertex) | `berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com` | read the published inputs; create (not rewrite) its own QA evidence prefix; Artifact Registry reader; Infisical identity via GCP ID token |
| Infisical | a machine identity using **GCP ID Token** auth | allowed service account = the worker SA; access to the `WANDB_API_KEY` secret only |

The worker must not reuse the VM's bucket-wide `objectAdmin` identity
(`berlin-lst-vertex@…`). Secrets never appear in the image, the job spec, the
command line, or logs.

## Secret ownership (human-only)

`WANDB_API_KEY` lives in the **Infisical EU vault** in a dedicated project
folder (`/vertex`, environment `dev`). The entrypoint sets `INFISICAL_DOMAIN`
(EU Cloud) explicitly because a machine-identity login does not persist the
instance. A human creates the machine identity, binds it to the worker service
account, and stores the key. Agents never read, print, or export secret values.

At runtime the worker (`scripts/operators/vertex_entrypoint.sh`) exchanges its
GCP identity token for a short-lived Infisical access token, then
`infisical run` injects `WANDB_API_KEY` into the runner's environment only. The
token and the key are never written to disk.

## Published roots (read-only)

```
patch_index_root: gs://berlin-lst-training-data/training/patch-index/v1
training_root:    gs://berlin-lst-training-data/training/v1
features_root:    gs://berlin-lst-training-data/features/v3
ard_root:         gs://berlin-lst-training-data/ard/full/2017-2026-cutoff-20260717T235959Z
```

## Build and push the image

The acceptance image was built with **Cloud Build** (the local Docker export
hung on this workstation) from code SHA `7ad2a9d`. Pass a throwaway config
(Cloud Build workers are linux/amd64, so no `--platform` is needed):

```bash
cat > /tmp/cloudbuild-vertex.yaml <<'YAML'
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, -f, Dockerfile.vertex, -t, ${_IMAGE}, .]
images:
  - ${_IMAGE}
YAML

gcloud builds submit --project=berlin-lst-training \
  --config=/tmp/cloudbuild-vertex.yaml \
  --substitutions=_IMAGE=europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex:<sha> .
```

Image digest in use:
`europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:1b803e57baaca5c8463b6ac4128b953b6097f1dc41f658107c05ff3c00fa57c7`

The launcher refuses an unpinned image. The image contains the worker code and
configs at build time; rebuild and record a new digest if any of
`Dockerfile.vertex`, `configs/modeling/`, `src/`, or `scripts/operators/` change.

## Launch, status, cancel, reconnect

```bash
# Validate bounds and print the plan without submitting:
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest> \
  --source-sha 7ad2a9d --run-label vertex-smoke-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity <identity-id> --infisical-project <project-id> \
  --infisical-env dev --infisical-path /vertex \
  --preflight

# Submit once (omitting --preflight):
#   ... same arguments, without --preflight

# Reconnect / inspect an existing job (do NOT resubmit):
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --status projects/<n>/locations/europe-west4/customJobs/<id>
```

Submission uses the low-level Vertex `JobServiceClient` with an explicit
`CustomJobSpec` (single worker pool, `service_account`, and `Scheduling` with
`timeout` and `disable_retries`). No `base_output_directory` is set, so no GCS
staging bucket is involved. Vertex rejects `max_wait_duration` unless the
strategy is `FLEX_START`, so the queue allowance is enforced client-side: the
launcher stops waiting past its budget but leaves the job for `--status`/cancel,
and nothing is billed while the job is QUEUED.

The launcher captures the job resource name from the create response before
polling, so a disconnected client reconnects with `--status <resource-name>`
instead of resubmitting. Cancel a runaway job with
`gcloud ai custom-jobs cancel <resource-name> --region=europe-west4`.

## Bounds and cost

- Machine: one on-demand `n1-standard-4` + one `NVIDIA_TESLA_T4` in
  `europe-west4` (smoke region).
- Server-side job `timeout` 2700 s (45 min) and `disable_retries` enforce the
  bounds; single worker pool; no persistent resource. The timeout is the hard
  ceiling on compute time.
- The launcher refuses to submit when the projected **compute** exposure
  (`hourly_rate × (timeout + max_wait)`) exceeds `--max-exposure-usd`
  (default $3.00). This is compute only: cross-region Cloud Storage reads and
  the image pull from `europe-west3` (`$0.02/GiB` within Europe) are additive,
  and the worker may read whole Landsat scenes during admission.
- Pass the verified regional rate with `--hourly-rate-usd`; the default $0.75/h
  is an **unverified estimate**. Google does not enforce a hard spending cap on
  a running job, so the timeout and the launcher's exposure check — not a
  budget — are the effective limits.

## Budget guardrails

- Existing: `berlin-lst-training budget`, €250/month, **net** (credits
  included), alerts at 50/90/100%. Kept unchanged.
- Added: a **€20/month gross** project-wide **alerts-only** budget (credits
  excluded) so alerting tracks gross cost while credit remains. This covers all
  services including Artifact Registry and Storage, but it is advisory only.
- Optional: a **Vertex AI spend cap** (Preview) for this project — an actual
  enforcement that pauses *new* Vertex usage once gross estimated cost reaches
  the target. It does **not** reliably stop an in-flight job and can overshoot
  due to reporting latency. Only set it if the account offers it in the Console.

## Retained evidence

Checkpoints and the local run directory stay on the worker disk and are **not**
retained. On success, the worker writes one small create-only object:

```
gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/evidence.json
```

Fields: `run_label`, `source_sha`, `image_digest`, `generated_at`, `run`
(pipeline/run_id/git commit/dirty), `data_scope` (mode, patches per split,
skips, exclusions, patch IDs), and a bounded `result_tail` of the runner's
stdout. Upload uses `if_generation_match=0`, so retained evidence is never
overwritten and the run label must never be reused.

## CPU VM versus Vertex

| | `berlin-lst-vm` (CPU) | Vertex Custom Job (GPU) |
|---|---|---|
| Purpose | pipeline stages, smoke, timing, baseline | managed GPU training |
| Lifetime | persistent, stopped between runs | ephemeral, released at job end |
| Launchers | `.opencode/skills/google-access/scripts/run-*-vm.sh` | `scripts/operators/launch_vertex_modeling.py` |
| Secrets | VM `.env` / metadata ADC | Infisical GCP ID token at runtime |

## Acceptance record

Filled in after the authorized run: job resource name, region, source SHA, image
digest, terminal state, measured duration, evidence URI, W&B run reference, and
limitations. Status: **pending** (no acceptance run has been performed yet).
