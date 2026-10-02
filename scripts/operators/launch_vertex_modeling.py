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
import json
import math
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
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
    assert_stage1_efficiency,
    assert_stage1_full_bounds,
    assert_stage1_lock,
    assert_stage1_probe,
    assert_stage1_probe_lr3,
    assert_vertex_smoke_bounds,
)

PROJECT = "berlin-lst-training"
# Acceptance-smoke region: europe-west3 (bucket + image region). Its Vertex T4
# *training* quota was granted after a west4 attempt failed at admission. See
# docs/vertex-gpu-training.md.
REGION = "europe-west3"
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
# Both profiles write under the already-approved QA root: the probe keeps the
# `vertex-smoke` prefix (distinguished by its `stage1-probe-*` label) so it does
# not require a new IAM grant. The prefix is confined to APPROVED_EVIDENCE_ROOT,
# which keeps its trailing slash so `.../modeling-evil` cannot pass the check.
EVIDENCE_PREFIX = "gs://berlin-lst-training-data/qa/modeling/vertex-smoke"
APPROVED_EVIDENCE_ROOT = "gs://berlin-lst-training-data/qa/modeling/"

# Server-side bounds (seconds). The job timeout is the run's hard cost ceiling.
DEFAULT_TIMEOUT_SECONDS = 2700
# The bounded Stage-1 probe runs six epochs over a scene-spread cohort, so it
# needs a larger ceiling than the one-epoch smoke. It stays under the same $3
# projected-exposure check.
PROBE_TIMEOUT_SECONDS = 10800
# The full temporal Stage-1 run (issue #53) runs 20 unbounded epochs; the
# 48-hour server timeout is the hard ceiling. The projection assumes it is
# roughly 2x the heuristic full-run wall clock, not a measured runtime.
FULL_TIMEOUT_SECONDS = 172800
DEFAULT_MAX_WAIT_SECONDS = 600

# Full-run cost ceilings (issue #53). One on-demand n1-standard-4 + T4 for at
# most 48h; the launcher refuses to submit above $50 projected compute. The
# exposure is an admission estimate, never a provider spending cap.
FULL_MAX_EXPOSURE_USD = 50.0
# At $1.03/h the 48h+600s projection is $49.61, so a verified rate above this
# cannot stay under the $50 ceiling and must stop for a budget decision.
FULL_MAX_HOURLY_RATE_USD = 1.03
# Bounded efficiency jobs are capped at 45 minutes and one dollar projected
# each; the four-slot local ledger enforces the aggregate compute ceiling.
EFFICIENCY_TIMEOUT_SECONDS = 2700
EFFICIENCY_MAX_JOB_EXPOSURE_USD = 1.0
EFFICIENCY_MAX_HOURLY_RATE_USD = 1.03
EFFICIENCY_MAX_TOTAL_COMPUTE_USD = 3.78
EFFICIENCY_MAX_NONCOMPUTE_USD = 6.22
EFFICIENCY_MAX_TOTAL_USD = 10.0
EFFICIENCY_SESSION_ID = "stage1-efficiency-20261002"
# decision: keep the four-slot ledger in this git-ignored checkout because the
# project requires one sequential checkout and adding a cloud control object
# would create another external write. Alternative: Vertex-side reservations.
EFFICIENCY_LEDGER = (
    Path(__file__).resolve().parents[2] / "data/runs/.stage1-efficiency-control/slots.json"
)
EFFICIENCY_SLOT_ROLE = {1: "baseline", 2: "cache", 3: "diagnostic", 4: "final"}
# Approximate on-demand n1-standard-4 + T4 compute rate. An estimate only: pass
# the verified regional SKU with --hourly-rate-usd before submitting. The bucket
# and image share this region, so no cross-region transfer applies.
DEFAULT_HOURLY_RATE_USD = 0.75
DEFAULT_MAX_EXPOSURE_USD = 3.0
POLL_SECONDS = 20

# The approved profiles. `probe` and `probe-lr3` are the two Stage-1 recovery
# trials (issue #47) that share the residual method and differ only in learning
# rate; the smoke is the #38 GPU acceptance path and is left unchanged. `full`
# is the unbounded 20-epoch Stage-1 temporal run (issue #53).
MODE_CONFIG_NAME = {
    "smoke": "vertex_smoke",
    "probe": "stage1_probe",
    "probe-lr3": "stage1_probe_lr3",
    "full": "stage1_locked",
    "efficiency": "stage1_efficiency",
}

# Mode -> evidence profile. Both recovery trials emit `probe-residual` so the
# historical #45 `probe` evidence stays distinguishable from the recovery runs.
_PROBE_MODES = ("probe", "probe-lr3")
# Modes that require the verified regional rate before submitting.
_RATE_REQUIRED_MODES = ("probe", "probe-lr3", "full", "efficiency")
MODE_EVIDENCE_PROFILE = {
    "smoke": "smoke",
    "probe": "probe-residual",
    "probe-lr3": "probe-residual",
    "full": "full",
    "efficiency": "efficiency",
}

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


def check_bounds(config_name: str, *, efficiency_role: str | None = None) -> None:
    """Assert the shipped config still satisfies its own profile bounds."""
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        overrides = (
            [f"stage1_efficiency_role={efficiency_role}"] if efficiency_role is not None else []
        )
        cfg = compose(config_name=config_name, overrides=overrides)
    if config_name == MODE_CONFIG_NAME["probe"]:
        assert_stage1_probe(cfg)
    elif config_name == MODE_CONFIG_NAME["probe-lr3"]:
        assert_stage1_probe_lr3(cfg)
    elif config_name == MODE_CONFIG_NAME["full"]:
        assert_stage1_lock(cfg)
        assert_stage1_full_bounds(cfg)
    elif config_name == MODE_CONFIG_NAME["efficiency"]:
        assert_stage1_efficiency(cfg)
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
    if not re.search(r"@sha256:[0-9a-f]{64}$", image_uri):
        raise SystemExit("ERROR: --image-uri must be pinned by digest (@sha256:...)")


def _load_efficiency_ledger() -> list[dict]:
    if not EFFICIENCY_LEDGER.exists():
        return []
    try:
        payload = json.loads(EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: efficiency slot ledger is unreadable: {exc}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise SystemExit("ERROR: efficiency slot ledger has an invalid shape")
    return payload


def _check_efficiency_sequence(
    rows: list[dict],
    slot: int,
    role: str,
    noncompute_total: float,
    source_sha: str,
    image_digest: str,
    client: aiplatform_v1.JobServiceClient,
) -> None:
    if len(rows) != slot - 1 or [row.get("slot") for row in rows] != list(range(1, slot)):
        raise SystemExit(
            f"ERROR: efficiency jobs must run once, sequentially; slot {slot} expects "
            f"{slot - 1} prior reservations, found {len(rows)}"
        )
    if role != EFFICIENCY_SLOT_ROLE[slot]:
        raise SystemExit(
            f"ERROR: efficiency slot {slot} is reserved for role "
            f"{EFFICIENCY_SLOT_ROLE[slot]!r}, not {role!r}"
        )
    if any(row.get("session") != EFFICIENCY_SESSION_ID for row in rows):
        raise SystemExit("ERROR: efficiency ledger session ID mismatch")
    if any(row.get("source_sha") != source_sha for row in rows):
        raise SystemExit("ERROR: efficiency jobs must use the same committed source SHA")
    if any(row.get("image_digest") != image_digest for row in rows):
        raise SystemExit("ERROR: efficiency jobs must use the same pinned image digest")
    total = 0.0
    prior_noncompute = 0.0
    for row in rows:
        total += float(row.get("projected_exposure_usd", 0.0))
        prior_noncompute = max(
            prior_noncompute, float(row.get("projected_noncompute_total_usd", 0.0))
        )
        if row.get("state") != "validated":
            raise SystemExit(
                f"ERROR: prior efficiency slot {row.get('slot')} has not passed its "
                "independent evidence validator; do not submit the next slot"
            )
        resource_name = row.get("resource_name")
        if not resource_name:
            raise SystemExit(
                f"ERROR: previous efficiency slot {row.get('slot')} has an ambiguous "
                "submission with no job resource name; stop, do not resubmit"
            )
        previous = client.get_custom_job(name=str(resource_name))
        previous_state = _state(previous)
        if previous_state != JobState.JOB_STATE_SUCCEEDED:
            raise SystemExit(
                f"ERROR: previous efficiency slot {row.get('slot')} is {previous_state.name}; "
                "stop the sequence and inspect its evidence"
            )
    if total > EFFICIENCY_MAX_TOTAL_COMPUTE_USD:
        raise SystemExit(
            f"ERROR: previous efficiency reservations project ${total:.2f}, "
            f"above ${EFFICIENCY_MAX_TOTAL_COMPUTE_USD:.2f}"
        )
    if noncompute_total < prior_noncompute:
        raise SystemExit("ERROR: cumulative non-compute estimate cannot decrease between slots")
    if noncompute_total > EFFICIENCY_MAX_NONCOMPUTE_USD:
        raise SystemExit(
            "ERROR: non-compute estimate exceeds the remaining $6.22 experiment budget"
        )
    if total + noncompute_total > EFFICIENCY_MAX_TOTAL_USD:
        raise SystemExit("ERROR: aggregate projected efficiency cost exceeds $10")


@contextmanager
def _lock_efficiency_ledger() -> Iterator[None]:
    EFFICIENCY_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lock_path = EFFICIENCY_LEDGER.with_suffix(".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"ERROR: efficiency ledger is locked: {lock_path}") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _write_efficiency_ledger_locked(rows: list[dict]) -> None:
    temporary = EFFICIENCY_LEDGER.with_suffix(".partial")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, EFFICIENCY_LEDGER)


def _write_efficiency_ledger(rows: list[dict]) -> None:
    with _lock_efficiency_ledger():
        _write_efficiency_ledger_locked(rows)


def _reserve_efficiency_slot(
    *,
    slot: int,
    role: str,
    run_label: str,
    source_sha: str,
    image_digest: str,
    hourly_rate: float,
    exposure: float,
    noncompute_total: float,
) -> list[dict]:
    with _lock_efficiency_ledger():
        rows = _load_efficiency_ledger()
        if len(rows) != slot - 1 or [row.get("slot") for row in rows] != list(range(1, slot)):
            raise SystemExit(
                "ERROR: efficiency slot ledger changed after preflight; refusing submit"
            )
        if any(row.get("state") != "validated" for row in rows):
            raise SystemExit("ERROR: previous efficiency evidence is not validated")
        if any(
            row.get("session") != EFFICIENCY_SESSION_ID
            or row.get("source_sha") != source_sha
            or row.get("image_digest") != image_digest
            for row in rows
        ):
            raise SystemExit("ERROR: efficiency sequence source/image/session changed")
        prior_compute = sum(float(row.get("projected_exposure_usd", 0.0)) for row in rows)
        prior_other = max(
            (float(row.get("projected_noncompute_total_usd", 0.0)) for row in rows),
            default=0.0,
        )
        if (
            prior_compute + exposure > EFFICIENCY_MAX_TOTAL_COMPUTE_USD
            or noncompute_total < prior_other
            or noncompute_total > EFFICIENCY_MAX_NONCOMPUTE_USD
            or prior_compute + exposure + noncompute_total > EFFICIENCY_MAX_TOTAL_USD
        ):
            raise SystemExit("ERROR: locked efficiency reservation would exceed its budget gate")
        rows.append(
            {
                "session": EFFICIENCY_SESSION_ID,
                "slot": slot,
                "role": role,
                "run_label": run_label,
                "source_sha": source_sha,
                "image_digest": image_digest,
                "verified_hourly_rate_usd": hourly_rate,
                "projected_exposure_usd": exposure,
                "projected_noncompute_total_usd": noncompute_total,
                "state": "reserved",
                "resource_name": None,
            }
        )
        _write_efficiency_ledger_locked(rows)
    return rows


def _update_efficiency_slot(slot: int, **updates: object) -> None:
    rows = _load_efficiency_ledger()
    if not 1 <= slot <= len(rows) or int(rows[slot - 1].get("slot", -1)) != slot:
        raise SystemExit("ERROR: efficiency slot ledger lost the current reservation")
    rows[slot - 1].update(updates)
    _write_efficiency_ledger(rows)


def mark_efficiency_slot_validated(
    *, slot: int, run_label: str, evidence_sha256: str, verdict: str
) -> None:
    """Gate the next efficiency submission on independent evidence validation."""
    rows = _load_efficiency_ledger()
    if not 1 <= slot <= len(rows):
        raise SystemExit(f"ERROR: efficiency slot {slot} has no local reservation")
    row = rows[slot - 1]
    if row.get("slot") != slot or row.get("run_label") != run_label:
        raise SystemExit("ERROR: evidence label/slot does not match the local reservation")
    if row.get("state") != "awaiting_validation":
        raise SystemExit(
            f"ERROR: efficiency slot {slot} is {row.get('state')!r}, not awaiting_validation"
        )
    row.update(
        {
            "state": "validated" if verdict == "pass" else "validation_failed",
            "validation_verdict": verdict,
            "evidence_sha256": evidence_sha256,
        }
    )
    _write_efficiency_ledger(rows)


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
    parser.add_argument("--efficiency-slot", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--efficiency-role", choices=("baseline", "cache", "diagnostic", "final"))
    parser.add_argument("--efficiency-workers", type=int, choices=(0, 2, 4), default=2)
    parser.add_argument(
        "--efficiency-precision", choices=("32-true", "16-mixed"), default="32-true"
    )
    parser.add_argument("--efficiency-pin-memory", action="store_true")
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
        help="server-side timeout (2700s efficiency/smoke; 10800s probe; full blocked)",
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
        help="verified regional on-demand rate; required for multi-epoch/efficiency modes",
    )
    parser.add_argument(
        "--max-exposure-usd",
        type=float,
        default=None,
        help="projected exposure ceiling (default: $3 probe, $1 efficiency; full blocked)",
    )
    parser.add_argument(
        "--projected-noncompute-total-usd",
        type=float,
        default=None,
        help="cumulative Cloud Build/registry/storage/logging estimate; efficiency only",
    )
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

    if args.mode != "efficiency":
        raise SystemExit(
            "ERROR: paid Vertex submissions are temporarily restricted to the "
            "four-slot Stage-1 efficiency gate; full Stage-1 and other modes are blocked"
        )

    if args.mode == "efficiency":
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha or ""):
            raise SystemExit("ERROR: efficiency jobs require a full 40-character source SHA")
        if args.efficiency_slot is None or args.efficiency_role is None:
            raise SystemExit(
                "ERROR: --mode efficiency requires --efficiency-slot and --efficiency-role"
            )
        if args.run_label is None or not args.run_label.startswith(
            f"stage1-efficiency-j{args.efficiency_slot}-"
        ):
            raise SystemExit("ERROR: efficiency run-label must start stage1-efficiency-j<slot>-")
        if EFFICIENCY_SLOT_ROLE[args.efficiency_slot] != args.efficiency_role:
            raise SystemExit(
                f"ERROR: efficiency slot {args.efficiency_slot} requires "
                f"role {EFFICIENCY_SLOT_ROLE[args.efficiency_slot]}"
            )
        if args.efficiency_slot < 4 and args.efficiency_precision != "32-true":
            raise SystemExit("ERROR: 16-mixed training is only eligible for the J4 learning guard")
        if args.efficiency_slot < 3 and args.efficiency_pin_memory:
            raise SystemExit(
                "ERROR: pinned-memory selection is available only after J2 measures it"
            )
        if args.efficiency_slot == 1 and args.efficiency_workers != 2:
            raise SystemExit("ERROR: J1 control must use the current two-worker loader")
        if args.projected_noncompute_total_usd is None:
            raise SystemExit("ERROR: efficiency mode requires --projected-noncompute-total-usd")

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
    if args.timeout_seconds is not None:
        timeout_seconds = args.timeout_seconds
    elif args.mode == "full":
        timeout_seconds = FULL_TIMEOUT_SECONDS
    elif args.mode == "efficiency":
        timeout_seconds = EFFICIENCY_TIMEOUT_SECONDS
    elif args.mode in _PROBE_MODES:
        timeout_seconds = PROBE_TIMEOUT_SECONDS
    else:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if args.mode in _RATE_REQUIRED_MODES and args.hourly_rate_usd is None:
        raise SystemExit(
            "ERROR: this mode requires the verified regional rate via "
            "--hourly-rate-usd (the built-in estimate is not acceptable for a "
            "multi-epoch run)"
        )
    hourly_rate = (
        args.hourly_rate_usd if args.hourly_rate_usd is not None else DEFAULT_HOURLY_RATE_USD
    )
    max_exposure = (
        args.max_exposure_usd
        if args.max_exposure_usd is not None
        else (
            EFFICIENCY_MAX_JOB_EXPOSURE_USD
            if args.mode == "efficiency"
            else (FULL_MAX_EXPOSURE_USD if args.mode == "full" else DEFAULT_MAX_EXPOSURE_USD)
        )
    )
    _require_positive_finite("--timeout-seconds", float(timeout_seconds))
    _require_positive_finite("--hourly-rate-usd", float(hourly_rate))
    _require_positive_finite("--max-exposure-usd", float(max_exposure))
    if args.projected_noncompute_total_usd is not None and (
        not math.isfinite(args.projected_noncompute_total_usd)
        or args.projected_noncompute_total_usd < 0
    ):
        raise SystemExit("ERROR: non-compute estimate must be finite and non-negative")
    if not math.isfinite(float(args.max_wait_seconds)) or args.max_wait_seconds < 0:
        raise SystemExit("ERROR: --max-wait-seconds must be a finite non-negative number")
    if args.mode == "full":
        # issue #53: one bounded job, 48-hour server timeout, $50 projected
        # compute; the full run must not silently exceed either ceiling.
        if timeout_seconds > FULL_TIMEOUT_SECONDS:
            raise SystemExit(
                f"ERROR: --mode full server timeout {timeout_seconds}s exceeds the "
                f"{FULL_TIMEOUT_SECONDS}s ceiling"
            )

    if args.mode == "efficiency":
        if timeout_seconds > EFFICIENCY_TIMEOUT_SECONDS:
            raise SystemExit("ERROR: efficiency timeout cannot exceed 2700 seconds")
        if args.max_wait_seconds > DEFAULT_MAX_WAIT_SECONDS:
            raise SystemExit("ERROR: efficiency client allowance cannot exceed 600 seconds")
        if max_exposure > EFFICIENCY_MAX_JOB_EXPOSURE_USD:
            raise SystemExit("ERROR: efficiency per-job exposure cannot exceed $1.00")
        if hourly_rate > EFFICIENCY_MAX_HOURLY_RATE_USD:
            raise SystemExit("ERROR: efficiency verified rate exceeds $1.03/hour")
        if args.projected_noncompute_total_usd > EFFICIENCY_MAX_NONCOMPUTE_USD:
            raise SystemExit("ERROR: non-compute estimate exceeds the remaining $6.22 budget")

    _validate_image_uri(args.image_uri)
    _validate_run_label(args.run_label)
    prefix = args.evidence_prefix.rstrip("/") + "/"
    if not prefix.startswith(APPROVED_EVIDENCE_ROOT):
        raise SystemExit(
            f"ERROR: --evidence-prefix must stay under {APPROVED_EVIDENCE_ROOT}, "
            f"got {args.evidence_prefix!r}"
        )
    check_bounds(config_name, efficiency_role=args.efficiency_role)

    exposure = _exposure_usd(hourly_rate, timeout_seconds, args.max_wait_seconds)
    if not math.isfinite(exposure) or exposure > max_exposure:
        raise SystemExit(
            f"ERROR: projected exposure ${exposure:.2f} exceeds the "
            f"${max_exposure:.2f} ceiling; refusing to submit"
        )

    efficiency_rows: list[dict] = []
    client: aiplatform_v1.JobServiceClient | None = None
    if args.mode == "efficiency":
        client = _client(REGION)
        efficiency_rows = _load_efficiency_ledger()
        _check_efficiency_sequence(
            efficiency_rows,
            args.efficiency_slot,
            args.efficiency_role,
            args.projected_noncompute_total_usd,
            args.source_sha,
            args.image_uri.split("@", 1)[1],
            client,
        )
        if args.efficiency_slot >= 3:
            winner = efficiency_rows[1].get("selected_loader")
            if not isinstance(winner, dict):
                raise SystemExit("ERROR: validated J2 ledger has no selected loader settings")
            if args.efficiency_workers != winner.get("workers"):
                raise SystemExit("ERROR: J3/J4 worker count must match the validated J2 winner")
            if args.efficiency_pin_memory != winner.get("pin_memory"):
                raise SystemExit("ERROR: J3/J4 pin-memory must match the validated J2 winner")
        if args.efficiency_slot == 4 and args.efficiency_precision == "16-mixed":
            amp_gain = float(efficiency_rows[2].get("amp_projected_runtime_gain", 0.0))
            if amp_gain < 0.10:
                raise SystemExit(
                    f"ERROR: J3 measured projected AMP gain {amp_gain:.1%} < 10%; J4 must use FP32"
                )
        prior_exposure = sum(
            float(row.get("projected_exposure_usd", 0.0)) for row in efficiency_rows
        )
        if prior_exposure + exposure > EFFICIENCY_MAX_TOTAL_COMPUTE_USD:
            raise SystemExit(
                f"ERROR: four-slot projected compute would be "
                f"${prior_exposure + exposure:.2f}, above the $3.78 ceiling"
            )
        if (
            prior_exposure + exposure + args.projected_noncompute_total_usd
            > EFFICIENCY_MAX_TOTAL_USD
        ):
            raise SystemExit("ERROR: cumulative compute and other costs would exceed $10")

    evidence_uri = f"{args.evidence_prefix.rstrip('/')}/{args.run_label}/evidence.json"
    if _evidence_exists(evidence_uri):
        raise SystemExit(
            f"ERROR: evidence already exists at {evidence_uri}; the run label "
            "must never be reused (the create-only upload would fail after the run)"
        )
    run_prefix = evidence_uri.rsplit("/", 1)[0]
    if args.mode == "full" and _evidence_exists(f"{run_prefix}/best.ckpt"):
        # A previous full run may have died after the checkpoint upload but
        # before the manifest; that orphaned checkpoint would fail create-only
        # after burning a new paid job, so refuse the label here.
        raise SystemExit(
            f"ERROR: checkpoint already exists at {run_prefix}/best.ckpt; the run "
            "label must never be reused"
        )
    env = [
        ("VERTEX_RUN_LABEL", args.run_label),
        ("VERTEX_SOURCE_SHA", args.source_sha),
        ("VERTEX_IMAGE_DIGEST", args.image_uri.split("@", 1)[1]),
        ("VERTEX_EVIDENCE_URI", evidence_uri),
        ("VERTEX_CONFIG_NAME", config_name),
        ("VERTEX_PROFILE", MODE_EVIDENCE_PROFILE[args.mode]),
        ("VERTEX_OUTPUT_ROOT", f"data/runs/{args.mode}/{args.run_label}"),
        ("INFISICAL_MACHINE_IDENTITY_ID", args.infisical_identity),
        ("INFISICAL_PROJECT_ID", args.infisical_project),
        ("INFISICAL_ENV", args.infisical_env),
        ("INFISICAL_SECRET_PATH", args.infisical_path),
    ]
    if args.mode == "efficiency":
        env.extend(
            [
                ("VERTEX_EFFICIENCY_ROLE", args.efficiency_role),
                ("VERTEX_EFFICIENCY_SLOT", str(args.efficiency_slot)),
                ("VERTEX_EFFICIENCY_SESSION", EFFICIENCY_SESSION_ID),
                ("VERTEX_EFFICIENCY_WORKERS", str(args.efficiency_workers)),
                ("VERTEX_EFFICIENCY_PRECISION", args.efficiency_precision),
                ("VERTEX_EFFICIENCY_PIN_MEMORY", str(args.efficiency_pin_memory).lower()),
                ("VERTEX_EFFICIENCY_RATE_USD", str(hourly_rate)),
                ("VERTEX_EFFICIENCY_EXPOSURE_USD", f"{exposure:.8f}"),
                (
                    "VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD",
                    str(args.projected_noncompute_total_usd),
                ),
            ]
        )

    display_name = f"{args.mode}-{args.run_label}"
    print("Planned Vertex Custom Job:")
    print(f"  project/region : {PROJECT}/{REGION}")
    print(f"  mode/config    : {args.mode} / {config_name}")
    print(f"  machine        : {MACHINE_TYPE} + 1x {ACCELERATOR_TYPE} (on-demand)")
    print(f"  image          : {args.image_uri}")
    print(f"  service account: {args.service_account}")
    print(f"  timeout        : {timeout_seconds}s  max-wait: {args.max_wait_seconds}s")
    print(f"  exposure       : ~${exposure:.2f} (estimate at ${hourly_rate}/h)")
    if args.mode == "efficiency":
        print(
            f"  all-in estimate: ~${exposure + args.projected_noncompute_total_usd:.2f} "
            "(compute + cumulative other costs)"
        )
    print(f"  evidence       : {evidence_uri}")

    if args.preflight:
        print("preflight only: no job submitted")
        return 0

    if args.mode == "efficiency":
        efficiency_rows = _reserve_efficiency_slot(
            slot=args.efficiency_slot,
            role=args.efficiency_role,
            run_label=args.run_label,
            source_sha=args.source_sha,
            image_digest=args.image_uri.split("@", 1)[1],
            hourly_rate=hourly_rate,
            exposure=exposure,
            noncompute_total=args.projected_noncompute_total_usd,
        )
    if client is None:
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
    if args.mode == "efficiency":
        _update_efficiency_slot(
            args.efficiency_slot,
            state="submitted",
            resource_name=resource_name,
        )

    deadline = time.monotonic() + timeout_seconds + args.max_wait_seconds + 300
    final = _poll_until_terminal(client, resource_name, deadline)
    print(f"final state: {final.name}")
    if args.mode == "efficiency":
        _update_efficiency_slot(
            args.efficiency_slot,
            state=(
                "awaiting_validation"
                if final == JobState.JOB_STATE_SUCCEEDED
                else final.name.lower()
            ),
            terminal_state=final.name,
        )
    if final == JobState.JOB_STATE_SUCCEEDED:
        print(f"SUCCESS: bounded vertex {args.mode} completed. Evidence: {evidence_uri}")
        return 0
    print(f"FAIL: vertex {args.mode} ended {final.name}. Query with --status {resource_name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
