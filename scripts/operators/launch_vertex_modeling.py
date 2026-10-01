"""Guarded launcher for the bounded Vertex GPU acceptance smoke (issue #38).

Submits exactly one on-demand ``n1-standard-4`` + ``NVIDIA_TESLA_T4`` Vertex
Custom Job in ``europe-west3`` that runs the bounded ``vertex_smoke`` config
through the existing Hydra/Lightning/W&B runner. Server-side job limits, a
single worker pool, no persistent resource, and no retries are set here so a
disconnected client cannot leave an unbounded or repeated GPU job.

``europe-west3`` is where the bucket and image live and now carries the approved
Vertex T4 *training* quota. A west4 attempt was rejected at admission
(``custom_model_training_nvidia_t4_gpus`` 429); see docs/vertex-gpu-training.md.

The launcher never handles a secret: it passes only non-secret identifiers to
the worker, which resolves ``WANDB_API_KEY`` from the Infisical vault using its
own GCP identity.

Usage (workstation, ADC for the submitter account):
    uv run --group operators python scripts/operators/launch_vertex_modeling.py \
        --image-uri <registry>/<repo>/modeling-vertex@sha256:<digest> \
        --source-sha <git-sha> --run-label vertex-smoke-<utc>-<suffix> \
        --service-account <worker-sa>@berlin-lst-training.iam.gserviceaccount.com
    # add --preflight to validate bounds and print the planned job without
    # submitting; use --status <resource-name> to inspect an existing job.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    CustomJob,
    CustomJobSpec,
    EnvVar,
    JobState,
    MachineSpec,
    Scheduling,
    WorkerPoolSpec,
)
from google.protobuf.duration_pb2 import Duration
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.run import (
    assert_stage1_probe,
    assert_vertex_smoke_bounds,
)

PROJECT = "berlin-lst-training"
# Acceptance-smoke region: europe-west3 (bucket + image region). Its Vertex T4
# *training* quota was granted after a west4 attempt failed at admission. See
# docs/vertex-gpu-training.md.
REGION = "europe-west3"
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
EVIDENCE_PREFIX = "gs://berlin-lst-training-data/qa/modeling/vertex-smoke"

# Server-side bounds (seconds). The job timeout is the run's hard cost ceiling.
DEFAULT_TIMEOUT_SECONDS = 2700
# The bounded Stage-1 probe runs six epochs over a scene-spread cohort, so it
# needs a larger ceiling than the one-epoch smoke. It stays under the same $3
# projected-exposure check.
PROBE_TIMEOUT_SECONDS = 10800
DEFAULT_MAX_WAIT_SECONDS = 600
# Approximate on-demand n1-standard-4 + T4 compute rate. An estimate only: pass
# the verified regional SKU with --hourly-rate-usd before submitting. The bucket
# and image share this region, so no cross-region transfer applies.
DEFAULT_HOURLY_RATE_USD = 0.75
DEFAULT_MAX_EXPOSURE_USD = 3.0
POLL_SECONDS = 20

# The two approved profiles. The probe is the bounded Stage-1 learning run; the
# smoke is the #38 GPU acceptance path and is left unchanged.
MODE_CONFIG_NAME = {"smoke": "vertex_smoke", "probe": "stage1_probe"}

_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TERMINAL = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_EXPIRED,
    JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}


def _config_dir() -> str:
    return str((Path(__file__).resolve().parents[2] / "configs" / "modeling").resolve())


def check_bounds(config_name: str) -> None:
    """Assert the shipped config still satisfies its own profile bounds."""
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(config_name=config_name)
    if config_name == MODE_CONFIG_NAME["probe"]:
        assert_stage1_probe(cfg)
    else:
        assert_vertex_smoke_bounds(cfg)


def _require_positive_finite(label: str, value: float) -> float:
    """Reject NaN/negative/zero numeric inputs that would defeat the cost guard."""
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"ERROR: {label} must be a finite positive number, got {value!r}")
    return value


def _validate_run_label(run_label: str) -> None:
    # fullmatch (not match): "$" also matches before a trailing newline.
    if not _LABEL_RE.fullmatch(run_label) or run_label in (".", ".."):
        raise SystemExit(f"ERROR: invalid run label: {run_label!r}")


def _validate_image_uri(image_uri: str) -> None:
    if "@sha256:" not in image_uri:
        raise SystemExit("ERROR: --image-uri must be pinned by digest (@sha256:...)")


def _evidence_exists(evidence_uri: str) -> bool:
    """True if the create-only evidence object already exists (label must be unique)."""
    bucket_name, _, object_name = evidence_uri[len("gs://") :].partition("/")
    try:
        from google.cloud import storage

        return storage.Client().bucket(bucket_name).blob(object_name).exists()
    except Exception as exc:
        # A failed probe must not silently allow a label reuse.
        raise SystemExit(
            f"ERROR: could not check the evidence prefix {evidence_uri}: {exc}"
        ) from exc


def _exposure_usd(hourly_rate: float, timeout: int, max_wait: int) -> float:
    # Billing starts when resources are provisioned, so include the queue window.
    return hourly_rate * (timeout + max_wait) / 3600.0


def _client(region: str) -> aiplatform_v1.JobServiceClient:
    return aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )


def _worker_pool_spec(image_uri: str, env: list[tuple[str, str]]) -> WorkerPoolSpec:
    return WorkerPoolSpec(
        replica_count=1,
        machine_spec=MachineSpec(
            machine_type=MACHINE_TYPE,
            accelerator_type=ACCELERATOR_TYPE,
            accelerator_count=1,
        ),
        container_spec=ContainerSpec(
            image_uri=image_uri,
            env=[EnvVar(name=name, value=value) for name, value in env],
        ),
    )


def _state(job: CustomJob) -> JobState:
    return JobState(job.state)


def _submit(
    client: aiplatform_v1.JobServiceClient,
    *,
    display_name: str,
    image_uri: str,
    env: list[tuple[str, str]],
    service_account: str,
    timeout_seconds: int,
    profile: str,
) -> str:
    """Create the job and return its full resource name.

    The create response is returned directly, so the resource name is available
    before any waiting — a disconnected client reconnects with ``--status``
    instead of resubmitting. No ``base_output_directory`` is set, so no GCS
    staging bucket is involved.

    Only ``timeout`` and ``disable_retries`` are set: Vertex rejects
    ``max_wait_duration`` unless the scheduling strategy is ``FLEX_START``, so
    the queue allowance is enforced client-side instead and nothing is billed
    while the job is still QUEUED.
    """
    custom_job = CustomJob(
        display_name=display_name,
        labels={"purpose": profile},
        job_spec=CustomJobSpec(
            worker_pool_specs=[_worker_pool_spec(image_uri, env)],
            service_account=service_account,
            scheduling=Scheduling(
                timeout=Duration(seconds=timeout_seconds),
                disable_retries=True,
            ),
        ),
    )
    created = client.create_custom_job(
        parent=f"projects/{PROJECT}/locations/{REGION}", custom_job=custom_job
    )
    return created.name


def _poll_until_terminal(
    client: aiplatform_v1.JobServiceClient, resource_name: str, deadline: float
) -> JobState:
    last = ""
    while True:
        job = client.get_custom_job(name=resource_name)
        state = _state(job)
        if state.name != last:
            print(f"  state: {state.name}", flush=True)
            last = state.name
        if state in _TERMINAL:
            return state
        if time.monotonic() > deadline:
            raise SystemExit(
                f"ERROR: client wait budget expired while {resource_name} was "
                f"{state.name}; the job is still running server-side. Query it with "
                f"--status {resource_name}"
            )
        time.sleep(POLL_SECONDS)


def _print_status(resource_name: str) -> int:
    # The pinned client targets REGION; a resource name from another location
    # would be queried against the wrong endpoint and fail server-side.
    if f"/locations/{REGION}/" not in resource_name:
        raise SystemExit(f"ERROR: job resource name is not in {REGION}: {resource_name}")
    job = _client(REGION).get_custom_job(name=resource_name)
    state = _state(job)
    print(f"job:   {resource_name}")
    print(f"state: {state.name}")
    return 0 if state == JobState.JOB_STATE_SUCCEEDED else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-uri")
    parser.add_argument("--source-sha")
    parser.add_argument("--run-label")
    parser.add_argument("--service-account")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG_NAME), default="smoke")
    parser.add_argument("--infisical-identity", help="Infisical machine identity ID (non-secret)")
    parser.add_argument("--infisical-project", help="Infisical project ID (non-secret)")
    parser.add_argument("--infisical-env", default="dev")
    parser.add_argument("--infisical-path", default="/vertex")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--region", default=REGION)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="server-side job timeout (default: 2700 smoke / 10800 probe)",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=DEFAULT_MAX_WAIT_SECONDS,
        help="client-side queue/provisioning wait budget (not sent to Vertex)",
    )
    parser.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=None,
        help="verified regional on-demand rate; required for --mode probe",
    )
    parser.add_argument("--max-exposure-usd", type=float, default=DEFAULT_MAX_EXPOSURE_USD)
    parser.add_argument("--evidence-prefix", default=EVIDENCE_PREFIX)
    parser.add_argument(
        "--preflight", action="store_true", help="validate and print the plan without submitting"
    )
    parser.add_argument("--status", help="print the state of an existing job and exit")
    args = parser.parse_args()

    if args.project != PROJECT or args.region != REGION:
        raise SystemExit(
            f"ERROR: this launcher is pinned to {PROJECT}/{REGION}, "
            f"got {args.project}/{args.region}"
        )

    if args.status:
        return _print_status(args.status)

    required = {
        "--image-uri": args.image_uri,
        "--source-sha": args.source_sha,
        "--run-label": args.run_label,
        "--service-account": args.service_account,
        "--infisical-identity": args.infisical_identity,
        "--infisical-project": args.infisical_project,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise SystemExit(f"ERROR: missing required options: {', '.join(missing)}")

    config_name = MODE_CONFIG_NAME[args.mode]
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else (PROBE_TIMEOUT_SECONDS if args.mode == "probe" else DEFAULT_TIMEOUT_SECONDS)
    )
    if args.mode == "probe" and args.hourly_rate_usd is None:
        raise SystemExit(
            "ERROR: --mode probe requires the verified regional rate via "
            "--hourly-rate-usd (the built-in estimate is not acceptable for a "
            "multi-epoch run)"
        )
    hourly_rate = (
        args.hourly_rate_usd
        if args.hourly_rate_usd is not None
        else DEFAULT_HOURLY_RATE_USD
    )
    _require_positive_finite("--timeout-seconds", float(timeout_seconds))
    _require_positive_finite("--hourly-rate-usd", float(hourly_rate))
    _require_positive_finite("--max-exposure-usd", float(args.max_exposure_usd))
    if not math.isfinite(float(args.max_wait_seconds)) or args.max_wait_seconds < 0:
        raise SystemExit("ERROR: --max-wait-seconds must be a finite non-negative number")

    _validate_image_uri(args.image_uri)
    _validate_run_label(args.run_label)
    check_bounds(config_name)

    exposure = _exposure_usd(hourly_rate, timeout_seconds, args.max_wait_seconds)
    if not math.isfinite(exposure) or exposure > args.max_exposure_usd:
        raise SystemExit(
            f"ERROR: projected exposure ${exposure:.2f} exceeds the "
            f"${args.max_exposure_usd:.2f} ceiling; refusing to submit"
        )

    evidence_uri = f"{args.evidence_prefix.rstrip('/')}/{args.run_label}/evidence.json"
    if _evidence_exists(evidence_uri):
        raise SystemExit(
            f"ERROR: evidence already exists at {evidence_uri}; the run label "
            "must never be reused (the create-only upload would fail after the run)"
        )
    env = [
        ("VERTEX_RUN_LABEL", args.run_label),
        ("VERTEX_SOURCE_SHA", args.source_sha),
        ("VERTEX_IMAGE_DIGEST", args.image_uri.split("@", 1)[1]),
        ("VERTEX_EVIDENCE_URI", evidence_uri),
        ("VERTEX_CONFIG_NAME", config_name),
        ("VERTEX_PROFILE", args.mode),
        ("VERTEX_OUTPUT_ROOT", f"data/runs/{args.mode}/{args.run_label}"),
        ("INFISICAL_MACHINE_IDENTITY_ID", args.infisical_identity),
        ("INFISICAL_PROJECT_ID", args.infisical_project),
        ("INFISICAL_ENV", args.infisical_env),
        ("INFISICAL_SECRET_PATH", args.infisical_path),
    ]

    display_name = f"{args.mode}-{args.run_label}"
    print("Planned Vertex Custom Job:")
    print(f"  project/region : {PROJECT}/{REGION}")
    print(f"  mode/config    : {args.mode} / {config_name}")
    print(f"  machine        : {MACHINE_TYPE} + 1x {ACCELERATOR_TYPE} (on-demand)")
    print(f"  image          : {args.image_uri}")
    print(f"  service account: {args.service_account}")
    print(f"  timeout        : {timeout_seconds}s  max-wait: {args.max_wait_seconds}s")
    print(f"  exposure       : ~${exposure:.2f} (estimate at ${hourly_rate}/h)")
    print(f"  evidence       : {evidence_uri}")

    if args.preflight:
        print("preflight only: no job submitted")
        return 0

    client = _client(REGION)
    resource_name = _submit(
        client,
        display_name=display_name,
        image_uri=args.image_uri,
        env=env,
        service_account=args.service_account,
        timeout_seconds=timeout_seconds,
        profile=args.mode,
    )
    print(f"submitted job: {resource_name}", flush=True)

    deadline = time.monotonic() + timeout_seconds + args.max_wait_seconds + 300
    final = _poll_until_terminal(client, resource_name, deadline)
    print(f"final state: {final.name}")
    if final == JobState.JOB_STATE_SUCCEEDED:
        print(f"SUCCESS: bounded vertex smoke completed. Evidence: {evidence_uri}")
        return 0
    print(f"FAIL: vertex smoke ended {final.name}. Query with --status {resource_name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
