# Vertex GPU training path

The repo trains on the CPU smoke VM (`berlin-lst-vm`) for pipeline and smoke
work, and on **Vertex AI Custom Training** for GPU work (`AGENTS.md`
§Compute Placement). This document is the launch recipe for the cheap Stage-1
path: a bounded GPU smoke (`--mode smoke`) that proves job-local cache +
16-mixed, then the full temporal Stage-1 run (`--mode full`, issue #53).
Historical probes (#45/#47) and the closed efficiency gate
(`docs/stage1-efficiency.md`) stay documented below as prior evidence.

**Current path:** launcher modes are `smoke`, `probe`, `probe-lr3`, and `full`.
Efficiency and verification modes are closed. Frozen full-run runtime: job-local
cache, batch 4, workers 0, `pin_memory: false`, `16-mixed`.

## What the path is (and is not)

- It runs the **existing** runner (`scripts/runners/run_modeling.py`) with the
  selected Hydra profile through the `real` lifecycle on one on-demand T4,
  W&B online, checkpoints on the disposable worker disk, create-only evidence
  under the approved QA prefix.
- The smoke is a bounded **infrastructure and lifecycle** check for the cheap
  runtime (cache build, AMP, W&B, checkpoint reload). It is not a model-quality
  run.
- The full run is the Stage-1 ablation anchor (issue #53): 20 epochs over every
  published train/validation patch, then one-shot 2025 test scoring.
- Jobs are single, capped, and on-demand. There is no persistent GPU resource,
  no automatic retry, and no accelerator or region substitution.

## Region

- **Both the smoke and the full run use `europe-west3`** (Frankfurt), where the
  bucket and the image live and where the Vertex T4 **training** quota is now
  granted.
- History: the first submission attempts (`europe-west3`, then `europe-west4`)
  were rejected at admission with
  `429 aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus`. The
  Vertex *training* quota is separate from the Compute Engine `NVIDIA_T4_GPUS`
  quota (which read 1) and was 0 in both regions. A quota increase for
  `europe-west3` was requested and approved; `europe-west4` is not used.
- Because bucket, registry, and job share one region, there is **no
  cross-region** Cloud Storage or Artifact Registry transfer for this path.

## Identities

| Role | Identity | Needs |
|------|----------|-------|
| Submitter (workstation) | the operator's user ADC | `roles/aiplatform.user`; Artifact Registry writer to push the image |
| Worker (Vertex) | `berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com` | read the published inputs; create (not rewrite) its own QA evidence prefix; Artifact Registry reader; Infisical identity via GCP ID token |
| Infisical | a machine identity using **GCP ID Token** auth | allowed service account = the worker SA; access to the `WANDB_API_KEY` secret only |

The worker must not reuse the VM's bucket-wide `objectAdmin` identity
(`berlin-lst-vertex@…`). Secrets never appear in the image, the job spec, the
command line, or logs.

### Fixed, non-secret identifiers (reuse for every launch)

These are identifiers, not secret values; keep them here so a launch does not
have to look them up again. The vault value (`WANDB_API_KEY`) is never recorded.

| Field | Value |
|---|---|
| Worker service account | `berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com` |
| Infisical machine identity ID | `7ba603e5-b94d-42d1-bc57-d64658dde09d` |
| Infisical project ID | `5da7dfb7-954d-4736-ba2e-4471ade9d766` |
| Infisical environment / secret path | `dev` / `/vertex` |

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
hung on this workstation) from code SHA `ef80930`. Pass a throwaway config
(Cloud Build workers are linux/amd64, so no `--platform` is needed):

```bash
cat > /tmp/cloudbuild-vertex.yaml <<'YAML'
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, -f, Dockerfile.vertex, -t, '${_IMAGE}', .]
images:
  - '${_IMAGE}'
YAML

gcloud builds submit --project=berlin-lst-training \
  --config=/tmp/cloudbuild-vertex.yaml \
  --substitutions=_IMAGE=europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex:<sha> .
```

GPU-verified image digests:
- #38 acceptance smoke: `sha256:99af01061fac56967317d2d468c9dd3f13173e48052517434113dc0829b636ea`
- #53 cheap-runtime smoke (cache + 16-mixed, SHA `8bd094b`): `sha256:8772b5c8a6349448e820404c75008c59f9df8b63aa1099088e7deffeee90d887`
- #45 bounded Stage-1 probe (built from the probe tree): `sha256:78430e03643a328775897db9a99b296d70621db73b5019c0b72ea68bff74777e`

The image must expose the NVIDIA driver at runtime. The T4 device nodes are
attached, but a slim base does not set the driver library path, so
`Dockerfile.vertex` sets the standard NVIDIA container variables
(`NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`,
`LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`). Without them
`torch.cuda.is_available()` is `False` and Lightning raises
`No supported gpu backend found!`.

The launcher refuses an unpinned image. The image bakes the **worker runtime**
(`vertex_entrypoint.sh`, `vertex_evidence.py`, the runner, `src/`, and
`configs/`) at build time; rebuild and record a new digest if any of those
change. The launcher itself (`scripts/operators/launch_vertex_modeling.py`) runs
on the workstation, never inside the container, so a launcher-only edit does
**not** require a rebuild even though `scripts/` is copied into the image.

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
  --status projects/<n>/locations/europe-west3/customJobs/<id>
```

### Bounded Stage-1 probe (`--mode probe`)

Add `--mode probe` to run the `stage1_probe` profile (six epochs, scene-spread
128/64 cohort) instead of the four-ref, one-epoch smoke. It composes and asserts
the probe guards, defaults the server timeout to 10,800 s, and **requires** the
verified regional rate via `--hourly-rate-usd` (the built-in $0.75/h estimate is
not accepted for a multi-epoch run). The evidence prefix is unchanged; the probe
is distinguished by its `stage1-probe-*` run label. The projected Vertex compute
exposure must stay at or below `--max-exposure-usd` ($3.00 default).

```bash
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode probe \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<probe-digest> \
  --source-sha <clean-commit> --run-label stage1-probe-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 0.90 --timeout-seconds 10800 --max-wait-seconds 600 \
  --preflight
```

Validate a retained probe result with no source read or submission:

```bash
uv run python scripts/validators/validate_stage1_probe.py --self-check
uv run python scripts/validators/validate_stage1_probe.py --evidence <evidence.json>
```

### Stage-1 efficiency jobs (`--mode efficiency`)

The four roles measure the source/cache path, loader/transfer choices,
FP32/16-mixed compute and a matched learning guard. The fixed replacement
schedule is `1=baseline`, `2=cache`, `3=diagnostic`, `4=final`. The first J1
attempt failed before measurement setup because entrypoint log creation made the
measurer's output root non-empty. Its incomplete, create-only evidence and
original ledger are retained; it is never retried or overwritten. The replacement
uses fixed session `stage1-efficiency-recovery-20261003` and a separate ignored
ledger. The recovery allows four replacement submissions, at most five total
including the failed original. Full Stage-1 remains blocked regardless of
efficiency results.

Each replacement has a 2700-second server timeout and a maximum 1800-second
provisioning wait. If the resource has not entered `JOB_STATE_RUNNING` by that
deadline, the launcher cancels that exact resource once and waits for a terminal
state. A server `startTime` after `createTime + 1800 seconds` also counts as a
missed allowance, even if first observed later as `JOB_STATE_RUNNING`; if it is
already terminal, preserve that terminal state but still consume the slot and
stop. Ambiguous submit/cancel responses stop the sequence; inspect the saved
resource and never resubmit it. Automatic retries are disabled. The user-approved
rate input is `$1.00/h`; projected exposure is at most `$1.25` per replacement.
The conservative aggregate estimate reserves the failed attempt's `$0.9167`,
four replacements at `$1.25` each, and `$2.00` cumulative non-compute costs,
totalling about `$7.92` under the `$10` experiment ceiling. The `$2.00` reserve
supersedes the prior `$0.20` estimate and includes both image builds, storage,
and logging. These are estimates, not provider spending caps; the active
Billing-account rate and actual charges are not independently verified.
All replacement slots use one clean source SHA and one pinned image digest.

Before building or submitting, run the offline orchestration checks:

```bash
bash -n scripts/operators/vertex_entrypoint.sh
uv run --group operators python scripts/operators/launch_vertex_modeling.py --self-check
uv run python scripts/validators/check_efficiency_output_root.py
bash scripts/validators/check_vertex_efficiency_entrypoint.sh
uv run python scripts/validators/validate_training_efficiency.py --self-check
```

```bash
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode efficiency --efficiency-slot 1 --efficiency-role baseline \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<efficiency-digest> \
  --source-sha <clean-commit> --run-label stage1-efficiency-j1-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --efficiency-workers 2 --efficiency-precision 32-true \
  --hourly-rate-usd 1.00 \
  --hourly-rate-source 'User-authorized Vertex Frankfurt range; USD 1.00/hour.' \
  --timeout-seconds 2700 \
  --projected-noncompute-total-usd 2.00 \
  --max-wait-seconds 1800 --max-exposure-usd 1.25 --preflight
```

`--preflight` is read-only and does not reserve a slot. A real submission
reserves its slot immediately before `create_custom_job`. Preserve the original
ledger at `data/runs/.stage1-efficiency-control/slots.json`; do not reset either
ledger or reuse labels. After each replacement, the next slot remains blocked
until the job is `SUCCEEDED`, its evidence and checkpoint are downloaded, and
the independent validator marks the recovery-ledger slot validated. A worker
failure may retain incomplete `evidence.json`; it consumes the slot and stops
the sequence. The confirmed J1 checkpoint-location defect and J3 duplicate
loader-timing requirement may each be revalidated once with
`--record-revalidation --mark-ledger`, using the same hash-identical evidence
and no new Vertex job. The J3 timing comparison uses the validated J2 loader
measurement and verifies that J3 used its selected settings. The ledger keeps
both the initial failure and the revalidation verdict. No other failed
validation may be reopened.

J1 validation:

```bash
uv run --group operators python scripts/validators/validate_training_efficiency.py \
  --evidence <j1-evidence.json> \
  --checkpoint <downloaded-j1-best.ckpt> \
  --baseline docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json \
  --mark-ledger
```

J2 adds `--control-evidence <validated-j1-evidence.json>`. J3 supplies J1 and
J2 via `--control-evidence` and `--cache-evidence`. J4 supplies J1/J2/J3 via
`--control-evidence`, `--cache-evidence`, and `--diagnostic-evidence`, and both
`--checkpoint <downloaded-j4-best.ckpt>` and
`--control-checkpoint <downloaded-j1-best.ckpt>`. Failed validation with
`--mark-ledger` blocks later slots, except for the two authorized one-time
`--record-revalidation` cases above. The validator
also checks the recorded Vertex resource is `JOB_STATE_SUCCEEDED` before
marking a slot; after a client timeout, poll with `--status`, then validate the
downloaded evidence without resubmitting.

### Full Stage-1 run (`--mode full`)

Issue #53. Uses `configs/modeling/stage1_locked.yaml` with the frozen cheap
runtime from `docs/stage1-efficiency.md`. Fit admits train/validation only;
the 2025 test split is scored once after checkpoint freeze. Decision table and
readout live in `docs/stage1-full-results.md`. Run only after a successful
`--mode smoke` on the same image digest.

```bash
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode full \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<full-digest> \
  --source-sha <clean-commit> --run-label stage1-full-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 0.90 --preflight
```

### Tag-11 ablation full runs (`--mode stage2` … `stage5`, `isolate-shadows`, `isolate-era5`)

Issue #58 and issue #63. Same protocol, timeout, and cost ceilings as
`--mode full`, with Hydra profiles `stage2_locked` … `stage5_locked`,
`isolate_shadows_locked`, and `isolate_era5_locked` under the residual lock
(`docs/ablation-stage-configs.md`). Evidence reuses the `full` profile
(checkpoint + one-shot test). Rebuild the image from a commit that includes
those configs; do not retune between runs. Stage 5 must use an image that
contains the pad-free SSIM loss.

```bash
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode stage2 \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<ablation-digest> \
  --source-sha <clean-commit> --run-label stage2-full-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 1.00 --preflight
```

The client wait can outlast a session; if it expires the job keeps running
server-side, so reconnect with `--status <resource-name>` (never resubmit) and
cancel a runaway job with `gcloud ai custom-jobs cancel <resource-name>
--region=europe-west3`. The 48 h server timeout is the hard ceiling.

Validate a retained full-run result and re-derive its tier with no source read or
submission:

```bash
uv run python scripts/validators/validate_stage1_full.py --self-check
uv run python scripts/validators/validate_stage1_full.py --evidence <evidence.json>
# optional: also verify the downloaded checkpoint's bytes against its recorded sha256
uv run python scripts/validators/validate_stage1_full.py --evidence <evidence.json> \
  --checkpoint <downloaded best.ckpt>
```

Exit codes: `0` GO, `2` usable anchor, `3` NO-GO, `1` incomplete/undecidable.

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
`gcloud ai custom-jobs cancel <resource-name> --region=europe-west3`.

## Bounds and cost

- Machine: one on-demand `n1-standard-4` + one `NVIDIA_TESLA_T4` in
  `europe-west3`. The GPU is confirmed visible to the container (`torch 2.2.2+cu121`,
  `torch.cuda.is_available() == True`).
- Server-side job `timeout` 2700 s (45 min) and `disable_retries` enforce the
  bounds; single worker pool; no persistent resource. The timeout is the hard
  ceiling on compute time.
- The launcher refuses to submit when the projected **compute** exposure
  (`hourly_rate × (timeout + max_wait)`) exceeds `--max-exposure-usd`
  (default $3.00). The bucket and image are in the same region, so there is no
  cross-region transfer; only compute (and negligible same-region storage /
  logging) applies.
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
retained by the smoke and probe profiles. On success, the worker writes one
small create-only object:

```
gs://berlin-lst-training-data/qa/modeling/vertex-smoke/<run-label>/evidence.json
```

Fields: `run_label`, `source_sha`, `image_digest`, `generated_at`, `run`
(pipeline/run_id/git commit/dirty), `data_scope` (mode, patches per split,
skips, exclusions, patch IDs), and a bounded `result_tail` of the runner's
stdout. Upload uses `if_generation_match=0`, so retained evidence is never
overwritten and the run label must never be reused.

The `full` profile additionally retains, under the same prefix, the selected
checkpoint (`best.ckpt`, uploaded **first**) and `evidence.json` (uploaded
**last**) with the epoch curve, full-run summary, fit and test `data_scope`,
checkpoint SHA-256 and byte size. A manifest that references a checkpoint cannot
precede it; a missing or checksum-invalid checkpoint means the package is
incomplete and is not a GO.

## CPU VM versus Vertex

| | `berlin-lst-vm` (CPU) | Vertex Custom Job (GPU) |
|---|---|---|
| Purpose | pipeline stages, smoke, timing, baseline | managed GPU training |
| Lifetime | persistent, stopped between runs | ephemeral, released at job end |
| Launchers | `.opencode/skills/google-access/scripts/run-*-vm.sh` | `scripts/operators/launch_vertex_modeling.py` |
| Secrets | VM `.env` / metadata ADC | Infisical GCP ID token at runtime |

## Acceptance record

### #38 GPU path (2026-10-01)

Status: **passed**.

| Field | Value |
|---|---|
| Job | `projects/996559849187/locations/europe-west3/customJobs/3326855828958347264` |
| Region | `europe-west3` (T4, `n1-standard-4`) |
| Source SHA | `ef80930a202e82bc99cec12655e6d600004e9d1a` |
| Image digest | `sha256:99af01061fac56967317d2d468c9dd3f13173e48052517434113dc0829b636ea` |
| Terminal state | `JOB_STATE_SUCCEEDED` |
| Evidence | `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/vertex-smoke-20261001T000657Z-139d32/evidence.json` |
| W&B run | `comfy-energy-26` |

Verified the original streamed GPU lifecycle (4 refs/split, one epoch, W&B
online, create-only evidence).

### #53 cheap-runtime smoke (2026-10-05)

Status: **passed**. Proves job-local cache + `16-mixed` on the unlock image
before the full Stage-1 run.

| Field | Value |
|---|---|
| Job | `projects/996559849187/locations/europe-west3/customJobs/7397729461078065152` |
| Region | `europe-west3` (T4, `n1-standard-4`) |
| Source SHA | `8bd094b0758e661873bc338ef2ca6bfcad3f5883` |
| Image digest | `sha256:8772b5c8a6349448e820404c75008c59f9df8b63aa1099088e7deffeee90d887` |
| Run label | `vertex-smoke-20261005T101459Z-cache` |
| Terminal state | `JOB_STATE_SUCCEEDED` |
| Evidence | `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/vertex-smoke-20261005T101459Z-cache/evidence.json` |
| W&B run | `floral-gorge-31` — https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/jdqm4vfe |

Verified: train/validation only (4 refs each, 0 exclusions), C=10, workers 0,
`16-mixed`, one GPU epoch (`validation/mae_100m=302.0`), checkpoint selection
and reload, W&B online, create-only evidence, clean teardown. This is a
lifecycle check, not model quality. The full Stage-1 temporal run remains the
next explicit invocation under issue #53.
