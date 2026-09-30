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

## Identities

| Role | Identity | Needs |
|------|----------|-------|
| Submitter (workstation) | the operator's user ADC | `roles/aiplatform.user`; Artifact Registry writer to push the image |
| Worker (Vertex) | `berlin-lst-vertex@berlin-lst-training.iam.gserviceaccount.com` | read the published inputs; create (not rewrite) its own QA evidence prefix; Artifact Registry reader; Infisical identity via GCP ID token |
| Infisical | a machine identity using **GCP ID Token** auth | allowed service account = the worker SA; access to the `WANDB_API_KEY` secret only |

The worker must not reuse the VM's bucket-wide `objectAdmin` identity. Secrets
never appear in the image, the job spec, the command line, or logs.

## Secret ownership (human-only)

`WANDB_API_KEY` lives in the **Infisical EU vault** (`INFISICAL_API_URL`
defaults to `https://eu.infisical.com`), in a dedicated folder — the default in
the launcher is `/vertex` on the `dev` environment, holding no other secret.
A human creates the machine identity, binds it to the worker service account,
and stores the key. Agents never read, print, or export secret values.

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

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.vertex \
  -t europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex \
  --push .
```

Record the pushed `@sha256:<digest>`; the launcher refuses an unpinned image.

## Launch, status, cancel, reconnect

```bash
# Validate bounds and print the plan without submitting:
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --image-uri <region>-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<digest> \
  --source-sha <git-sha> --run-label vertex-smoke-<utc>-<suffix> \
  --service-account berlin-lst-vertex@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity <identity-id> --infisical-project <project-id> \
  --preflight

# Submit once (omitting --preflight):
#   ... same arguments, without --preflight

# Reconnect / inspect an existing job (do NOT resubmit):
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --status <projects/.../locations/europe-west3/customJobs/...>
```

Submission uses the low-level Vertex `JobServiceClient` with an explicit
`CustomJobSpec` (single worker pool, `service_account`, and `Scheduling` with
`timeout`/`max_wait_duration`/`disable_retries`). No `base_output_directory` is
set, so no GCS staging bucket is involved and the worker writes only its local
output root plus the create-only QA evidence object.

The launcher captures the job resource name from the create response before
polling, so a disconnected client reconnects with `--status <resource-name>`
instead of resubmitting. If the client disconnects, the job keeps running
server-side; reconnect with `--status <resource-name>`. Cancel a runaway job
with `gcloud ai custom-jobs cancel <resource-name> --region=europe-west3`.

## Bounds and cost

- Machine: one on-demand `n1-standard-4` + one `NVIDIA_TESLA_T4` in
  `europe-west3`.
- Server-side job `timeout` 2700 s (45 min) and `disable_retries` enforce the
  bounds; single worker pool; no persistent resource. `max_wait_duration`
  (600 s) is also set, but it only takes effect under a dynamic-workload
  strategy (`DWS_FLEX_START`); under the default on-demand strategy it is
  dormant, and the client-side wait budget plus the exposure ceiling cover the
  queue window conservatively.
- The launcher refuses to submit when the projected exposure
  (`hourly_rate × (timeout + max_wait)`) exceeds `--max-exposure-usd`
  (default $3.00). Billing starts when resources are provisioned and runs until
  the job finishes, so the timeout is the hard ceiling.
- The default `--hourly-rate-usd 0.75` is an **estimate**; re-verify the live
  regional SKU before submitting. Google does not enforce a budget cap, so a
  billing alert is advisory only.

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

Filled in after the authorized run: job resource name, source SHA, image digest,
terminal state, measured duration, evidence URI, W&B run reference, and
limitations. Status: **pending** (no acceptance run has been performed yet).
