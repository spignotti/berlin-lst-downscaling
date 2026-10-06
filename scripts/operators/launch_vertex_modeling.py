"""Guarded launcher for Vertex GPU modeling jobs.

Submits exactly one on-demand ``n1-standard-4`` + ``NVIDIA_TESLA_T4`` Vertex
Custom Job in ``europe-west3``. Modes: ``smoke`` (cheap-runtime lifecycle),
``probe`` / ``probe-lr3`` (historical recovery probes), ``full`` (issue #53
Stage-1 temporal run), ``stage2``–``stage5`` (issue #58 Tag-11 ablation full
temporal runs), and ``isolate-shadows`` / ``isolate-era5`` (issue #63
single-family runs). All of those stay under the residual lock. Efficiency
and verification modes are closed.

Server-side job limits, a single worker pool, no persistent resource, and no
retries are set here so a disconnected client cannot leave an unbounded or
repeated GPU job. The launcher never handles a secret: it passes only
non-secret identifiers to the worker, which resolves ``WANDB_API_KEY`` from
Infisical using its own GCP identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import patch

from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    CustomJob,
    CustomJobSpec,
    DiskSpec,
    EnvVar,
    JobState,
    MachineSpec,
    Scheduling,
    WorkerPoolSpec,
)
from google.protobuf.duration_pb2 import Duration
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.efficiency_protocol import (
    EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD,
    EFFICIENCY_HISTORICAL_SUBMISSIONS,
    EFFICIENCY_MAX_HOURLY_RATE_USD,
    EFFICIENCY_MAX_JOB_EXPOSURE_USD,
    EFFICIENCY_MAX_NONCOMPUTE_USD,
    EFFICIENCY_MAX_SUBMISSIONS,
    EFFICIENCY_MAX_TOTAL_COMPUTE_USD,
    EFFICIENCY_MAX_TOTAL_USD,
    EFFICIENCY_MAX_WAIT_SECONDS,
    EFFICIENCY_REPLACEMENT_SUBMISSIONS,
    EFFICIENCY_SESSION_ID,
    EFFICIENCY_SLOT_ROLES,
    EFFICIENCY_TIMEOUT_SECONDS,
    HISTORICAL_EFFICIENCY_IMAGE_DIGEST,
    HISTORICAL_EFFICIENCY_RESOURCE_NAME,
    HISTORICAL_EFFICIENCY_RUN_LABEL,
    HISTORICAL_EFFICIENCY_SESSION_ID,
    HISTORICAL_EFFICIENCY_SOURCE_SHA,
)
from berlin_lst_downscaling.modeling.run import (
    assert_ablation_lock,
    assert_stage1_full_bounds,
    assert_stage1_lock,
    assert_stage1_probe,
    assert_stage1_probe_lr3,
    assert_stage1_runtime,
    assert_vertex_smoke_bounds,
)
from berlin_lst_downscaling.modeling.verification_protocol import (
    VERIFICATION_MAX_HOURLY_RATE_USD,
    VERIFICATION_MAX_JOB_EXPOSURE_USD,
    VERIFICATION_MAX_NONCOMPUTE_USD,
    VERIFICATION_MAX_WAIT_SECONDS,
    VERIFICATION_RUN_LABEL,
    VERIFICATION_SESSION_ID,
    VERIFICATION_TIMEOUT_SECONDS,
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
# The recovery ledger is isolated from the preserved original failed slot.
# decision: keep the four-slot ledger in this git-ignored checkout because the
# project requires one sequential checkout and adding a cloud control object
# would create another external write. Alternative: Vertex-side reservations.
_EFFICIENCY_CONTROL_ROOT = (
    Path(__file__).resolve().parents[2] / "data/runs/.stage1-efficiency-control"
)
EFFICIENCY_LEDGER = _EFFICIENCY_CONTROL_ROOT / EFFICIENCY_SESSION_ID / "slots.json"
HISTORICAL_EFFICIENCY_LEDGER = _EFFICIENCY_CONTROL_ROOT / "slots.json"
EFFICIENCY_SLOT_ROLE = EFFICIENCY_SLOT_ROLES
# Approximate on-demand n1-standard-4 + T4 compute rate. An estimate only: pass
# the verified regional SKU with --hourly-rate-usd before submitting. The bucket
# and image share this region, so no cross-region transfer applies.
DEFAULT_HOURLY_RATE_USD = 0.75
DEFAULT_MAX_EXPOSURE_USD = 3.0
POLL_SECONDS = 20
CANCEL_CONFIRMATION_SECONDS = 300

# The approved profiles. `probe` and `probe-lr3` are the two Stage-1 recovery
# trials (issue #47) that share the residual method and differ only in learning
# rate; the smoke is the #38 GPU acceptance path and is left unchanged. `full`
# is the unbounded 20-epoch Stage-1 temporal run (issue #53). `stage2`–`stage5`
# are the Tag-11 cumulative ablation full runs (issue #58). `isolate-shadows`
# and `isolate-era5` are the single-family runs (issue #63). All of them share
# the residual lock, timeout, and cost ceilings of `full`.
MODE_CONFIG_NAME = {
    "smoke": "vertex_smoke",
    "probe": "stage1_probe",
    "probe-lr3": "stage1_probe_lr3",
    "full": "stage1_locked",
    "stage2": "stage2_locked",
    "stage3": "stage3_locked",
    "stage4": "stage4_locked",
    "stage5": "stage5_locked",
    "isolate-shadows": "isolate_shadows_locked",
    "isolate-era5": "isolate_era5_locked",
}

# Mode -> evidence profile. Both recovery trials emit `probe-residual` so the
# historical #45 `probe` evidence stays distinguishable from the recovery runs.
# Ablation full runs reuse the Stage-1 `full` evidence schema (epoch curve,
# one-shot test, create-only checkpoint).
_PROBE_MODES = ("probe", "probe-lr3")
_FULL_TEMPORAL_MODES = (
    "full",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "isolate-shadows",
    "isolate-era5",
)
_ABLATION_MODES = (
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "isolate-shadows",
    "isolate-era5",
)
# Modes that require the verified regional rate before submitting.
_RATE_REQUIRED_MODES = ("probe", "probe-lr3", *_FULL_TEMPORAL_MODES)
MODE_EVIDENCE_PROFILE = {
    "smoke": "smoke",
    "probe": "probe-residual",
    "probe-lr3": "probe-residual",
    "full": "full",
    "stage2": "full",
    "stage3": "full",
    "stage4": "full",
    "stage5": "full",
    "isolate-shadows": "full",
    "isolate-era5": "full",
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
        assert_stage1_runtime(cfg)
    elif config_name in {MODE_CONFIG_NAME[m] for m in _ABLATION_MODES}:
        assert_ablation_lock(cfg)
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


def _verify_historical_efficiency_slot(
    client: aiplatform_v1.JobServiceClient,
) -> None:
    if not HISTORICAL_EFFICIENCY_LEDGER.is_file():
        raise SystemExit("ERROR: preserved original efficiency ledger is missing")
    try:
        rows = json.loads(HISTORICAL_EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: original efficiency ledger is unreadable: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise SystemExit("ERROR: original efficiency ledger does not contain exactly one slot")
    row = rows[0]
    expected = {
        "session": HISTORICAL_EFFICIENCY_SESSION_ID,
        "slot": 1,
        "role": "baseline",
        "run_label": HISTORICAL_EFFICIENCY_RUN_LABEL,
        "source_sha": HISTORICAL_EFFICIENCY_SOURCE_SHA,
        "image_digest": HISTORICAL_EFFICIENCY_IMAGE_DIGEST,
        "resource_name": HISTORICAL_EFFICIENCY_RESOURCE_NAME,
        "state": "job_state_failed",
        "terminal_state": JobState.JOB_STATE_FAILED.name,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise SystemExit(
            "ERROR: original failed efficiency slot does not match the preserved record"
        )
    try:
        reserved = float(row["projected_exposure_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("ERROR: original efficiency reservation has no valid exposure") from exc
    if (
        not math.isfinite(reserved)
        or abs(reserved - EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD) > 1e-7
    ):
        raise SystemExit("ERROR: original efficiency exposure differs from its approved reserve")
    historical_job = client.get_custom_job(name=HISTORICAL_EFFICIENCY_RESOURCE_NAME)
    if _state(historical_job) != JobState.JOB_STATE_FAILED:
        raise SystemExit("ERROR: historical efficiency job is not confirmed JOB_STATE_FAILED")


def _validate_projected_efficiency_budget(
    rows: list[dict], *, exposure: float, noncompute_total: float
) -> float:
    prior_compute = EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD + sum(
        float(row.get("projected_exposure_usd", 0.0)) for row in rows
    )
    projected_compute = prior_compute + exposure
    prior_noncompute = max(
        (float(row.get("projected_noncompute_total_usd", 0.0)) for row in rows),
        default=0.0,
    )
    if projected_compute > EFFICIENCY_MAX_TOTAL_COMPUTE_USD:
        raise SystemExit(
            f"ERROR: historical plus replacement compute projects ${projected_compute:.2f}, "
            f"above ${EFFICIENCY_MAX_TOTAL_COMPUTE_USD:.2f}"
        )
    if noncompute_total < prior_noncompute:
        raise SystemExit("ERROR: cumulative non-compute estimate cannot decrease between slots")
    if noncompute_total > EFFICIENCY_MAX_NONCOMPUTE_USD:
        raise SystemExit("ERROR: cumulative non-compute estimate cannot exceed $2.00")
    if projected_compute + noncompute_total > EFFICIENCY_MAX_TOTAL_USD:
        raise SystemExit("ERROR: historical and replacement costs would exceed $10")
    return projected_compute


def _check_efficiency_sequence(
    rows: list[dict],
    slot: int,
    role: str,
    exposure: float,
    noncompute_total: float,
    source_sha: str,
    image_digest: str,
    client: aiplatform_v1.JobServiceClient,
) -> float:
    _verify_historical_efficiency_slot(client)
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
    for row in rows:
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
    return _validate_projected_efficiency_budget(
        rows, exposure=exposure, noncompute_total=noncompute_total
    )


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
    hourly_rate_source: str,
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
        _validate_projected_efficiency_budget(
            rows, exposure=exposure, noncompute_total=noncompute_total
        )
        rows.append(
            {
                "session": EFFICIENCY_SESSION_ID,
                "slot": slot,
                "role": role,
                "run_label": run_label,
                "source_sha": source_sha,
                "image_digest": image_digest,
                "verified_hourly_rate_usd": hourly_rate,
                "hourly_rate_source": hourly_rate_source,
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


_VERIFICATION_CONTROL_ROOT = (
    Path(__file__).resolve().parents[2] / "data/runs/.stage1-verification-control"
)
VERIFICATION_LEDGER = _VERIFICATION_CONTROL_ROOT / VERIFICATION_SESSION_ID / "slot.json"


def _load_verification_ledger() -> list[dict]:
    if not VERIFICATION_LEDGER.exists():
        return []
    try:
        payload = json.loads(VERIFICATION_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: verification slot ledger is unreadable: {exc}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise SystemExit("ERROR: verification slot ledger has an invalid shape")
    return payload


@contextmanager
def _lock_verification_ledger() -> Iterator[None]:
    VERIFICATION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lock_path = VERIFICATION_LEDGER.with_suffix(".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"ERROR: verification ledger is locked: {lock_path}") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _write_verification_ledger_locked(rows: list[dict]) -> None:
    temporary = VERIFICATION_LEDGER.with_suffix(".partial")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, VERIFICATION_LEDGER)


def _reserve_verification_slot(
    *,
    run_label: str,
    source_sha: str,
    image_digest: str,
    hourly_rate: float,
    hourly_rate_source: str,
    exposure: float,
    noncompute_total: float,
) -> None:
    """Exclusively consume the single verification submission before create."""
    with _lock_verification_ledger():
        rows = _load_verification_ledger()
        if rows:
            raise SystemExit(
                "ERROR: the single verification slot is already consumed; "
                "no replacement submission is authorized"
            )
        rows.append(
            {
                "session": VERIFICATION_SESSION_ID,
                "run_label": run_label,
                "source_sha": source_sha,
                "image_digest": image_digest,
                "verified_hourly_rate_usd": hourly_rate,
                "hourly_rate_source": hourly_rate_source,
                "projected_exposure_usd": exposure,
                "projected_noncompute_total_usd": noncompute_total,
                "state": "reserved",
                "resource_name": None,
            }
        )
        _write_verification_ledger_locked(rows)


def _update_verification_slot(**updates: object) -> None:
    with _lock_verification_ledger():
        rows = _load_verification_ledger()
        if len(rows) != 1:
            raise SystemExit("ERROR: verification slot ledger lost the current reservation")
        rows[0].update(updates)
        _write_verification_ledger_locked(rows)


def _local_head_sha() -> str:
    """Read the local HEAD SHA without spawning a process.

    Follows ``.git/HEAD`` through loose refs and ``packed-refs``. Returns ""
    when the checkout layout is unreadable — an unverifiable HEAD never
    blocks, only a mismatch does.
    """
    try:
        git_dir = Path(__file__).resolve().parents[2] / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if re.fullmatch(r"[0-9a-f]{40}", head) else ""
        ref_path = git_dir.joinpath(*head[len("ref: ") :].split("/"))
        if ref_path.is_file():
            sha = ref_path.read_text(encoding="utf-8").strip()
            return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else ""
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name.strip() == head[len("ref: ") :] and re.fullmatch(r"[0-9a-f]{40}", sha):
                    return sha
    except OSError:
        pass
    return ""


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
        # Explicit 100 GB pd-ssd boot disk, matching the documented Vertex
        # recipe and the rate basis (the prior jobs report ~94 GiB usable).
        disk_spec=DiskSpec(boot_disk_type="pd-ssd", boot_disk_size_gb=100),
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

    Only ``timeout`` and ``disable_retries`` are set: Vertex only accepts
    ``max_wait_duration`` with ``FLEX_START``. The approved standard-scheduling
    queue allowance is enforced client-side, including a one-shot cancellation.
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
    # Bounded create RPC: no SDK retries, 60-second deadline. An ambiguous
    # response consumes the slot and is reconciled by inspection, never resubmitted.
    created = client.create_custom_job(
        parent=f"projects/{PROJECT}/locations/{REGION}",
        custom_job=custom_job,
        retry=None,
        timeout=60.0,
    )
    return created.name


def _poll_until_terminal(
    client: aiplatform_v1.JobServiceClient,
    resource_name: str,
    *,
    start_deadline: float,
    max_wait_seconds: int,
    overall_deadline: float,
) -> tuple[JobState, bool]:
    last = ""
    cancel_deadline: float | None = None
    started_within_deadline = False
    queue_deadline_missed = False
    while True:
        job = client.get_custom_job(name=resource_name)
        state = _state(job)
        if state.name != last:
            print(f"  state: {state.name}", flush=True)
            last = state.name
        now = time.monotonic()
        if job.start_time is not None and job.create_time is not None:
            started_within_deadline = job.start_time <= job.create_time + timedelta(
                seconds=max_wait_seconds
            )
            queue_deadline_missed |= not started_within_deadline
        elif not started_within_deadline and (
            job.start_time is not None or state == JobState.JOB_STATE_RUNNING
        ):
            if now < start_deadline:
                started_within_deadline = True
            else:
                queue_deadline_missed = True
        if state in _TERMINAL:
            if (
                state == JobState.JOB_STATE_SUCCEEDED
                and not started_within_deadline
                and now >= start_deadline
            ):
                queue_deadline_missed = True
            return state, queue_deadline_missed
        if not started_within_deadline and now >= start_deadline:
            queue_deadline_missed = True
        if queue_deadline_missed and cancel_deadline is None:
            try:
                client.cancel_custom_job(name=resource_name)
            except Exception as exc:
                raise SystemExit(
                    f"ERROR: queue deadline reached but cancellation response is ambiguous "
                    f"for {resource_name}; inspect status and do not resubmit: {exc}"
                ) from exc
            print(f"  queue allowance expired; cancellation requested for {resource_name}")
            cancel_deadline = now + CANCEL_CONFIRMATION_SECONDS
            continue
        if cancel_deadline is not None and now >= cancel_deadline:
            raise SystemExit(
                f"ERROR: cancellation was not confirmed for {resource_name}; "
                "inspect status and do not resubmit"
            )
        if now >= overall_deadline:
            raise SystemExit(
                f"ERROR: server timeout wait expired while {resource_name} was "
                f"{state.name}; inspect status and do not resubmit"
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


def _wait_self_check() -> int:
    class FakeJob:
        def __init__(
            self,
            state: JobState,
            start_time: object | None = None,
            create_time: object | None = None,
        ) -> None:
            self.state = state
            self.start_time = start_time
            self.create_time = create_time

    class QueueClient:
        def __init__(self) -> None:
            self.cancelled = False
            self.cancel_count = 0

        def get_custom_job(self, *, name: str) -> FakeJob:
            return FakeJob(
                JobState.JOB_STATE_CANCELLED if self.cancelled else JobState.JOB_STATE_PENDING
            )

        def cancel_custom_job(self, *, name: str) -> None:
            self.cancelled = True
            self.cancel_count += 1

    failures: list[str] = []
    queue_client = QueueClient()
    with (
        patch(__name__ + ".time.monotonic", side_effect=(0.0, 1800.0, 1801.0)),
        patch(__name__ + ".time.sleep"),
    ):
        state, queue_missed = _poll_until_terminal(
            queue_client,
            "fake-job",
            start_deadline=1800.0,
            max_wait_seconds=1800,
            overall_deadline=5100.0,
        )
    if state != JobState.JOB_STATE_CANCELLED or not queue_missed or queue_client.cancel_count != 1:
        failures.append("queued job was not cancelled exactly once")
    else:
        print("  PASS queue: one cancellation was requested and confirmed terminal")

    class LateStartClient(QueueClient):
        def get_custom_job(self, *, name: str) -> FakeJob:
            if self.cancelled:
                return FakeJob(JobState.JOB_STATE_CANCELLED)
            return FakeJob(
                JobState.JOB_STATE_RUNNING,
                start_time=datetime(2026, 10, 3, 12, 30, 1, tzinfo=UTC),
                create_time=datetime(2026, 10, 3, 12, 0, tzinfo=UTC),
            )

    late_client = LateStartClient()
    with (
        patch(__name__ + ".time.monotonic", side_effect=(1801.0, 1802.0)),
        patch(__name__ + ".time.sleep"),
    ):
        late_state, late_queue_missed = _poll_until_terminal(
            late_client,
            "late-start-job",
            start_deadline=1800.0,
            max_wait_seconds=1800,
            overall_deadline=5100.0,
        )
    if (
        late_state != JobState.JOB_STATE_CANCELLED
        or not late_queue_missed
        or late_client.cancel_count != 1
    ):
        failures.append("late-start job was not cancelled after the provisioning deadline")
    else:
        print("  PASS queue: late RUNNING transition is cancelled by server timestamps")

    class LateSuccessClient:
        def __init__(self) -> None:
            self.cancel_count = 0

        def get_custom_job(self, *, name: str) -> FakeJob:
            return FakeJob(
                JobState.JOB_STATE_SUCCEEDED,
                start_time=datetime(2026, 10, 3, 12, 30, 1, tzinfo=UTC),
                create_time=datetime(2026, 10, 3, 12, 0, tzinfo=UTC),
            )

        def cancel_custom_job(self, *, name: str) -> None:
            self.cancel_count += 1

    late_success_client = LateSuccessClient()
    with patch(__name__ + ".time.monotonic", return_value=2000.0):
        late_success_state, late_success_missed = _poll_until_terminal(
            cast(aiplatform_v1.JobServiceClient, late_success_client),
            "late-success-job",
            start_deadline=1800.0,
            max_wait_seconds=1800,
            overall_deadline=5100.0,
        )
    if (
        late_success_state != JobState.JOB_STATE_SUCCEEDED
        or not late_success_missed
        or late_success_client.cancel_count != 0
    ):
        failures.append("late terminal success was not marked as a consumed queue-deadline slot")
    else:
        print("  PASS queue: terminal success after a late start remains a consumed slot")

    source_sha = "a" * 40
    image_digest = "sha256:" + "b" * 64
    history_row = {
        "session": HISTORICAL_EFFICIENCY_SESSION_ID,
        "slot": 1,
        "role": "baseline",
        "run_label": HISTORICAL_EFFICIENCY_RUN_LABEL,
        "source_sha": HISTORICAL_EFFICIENCY_SOURCE_SHA,
        "image_digest": HISTORICAL_EFFICIENCY_IMAGE_DIGEST,
        "resource_name": HISTORICAL_EFFICIENCY_RESOURCE_NAME,
        "state": "job_state_failed",
        "terminal_state": JobState.JOB_STATE_FAILED.name,
        "projected_exposure_usd": EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD,
    }

    class HistoryClient:
        def get_custom_job(self, *, name: str) -> FakeJob:
            state = (
                JobState.JOB_STATE_FAILED
                if name == HISTORICAL_EFFICIENCY_RESOURCE_NAME
                else JobState.JOB_STATE_SUCCEEDED
            )
            return FakeJob(state, start_time=object())

    prior_rows = [
        {
            "slot": slot,
            "session": EFFICIENCY_SESSION_ID,
            "source_sha": source_sha,
            "image_digest": image_digest,
            "state": "validated",
            "resource_name": f"replacement-{slot}",
            "projected_exposure_usd": EFFICIENCY_MAX_JOB_EXPOSURE_USD,
            "projected_noncompute_total_usd": EFFICIENCY_MAX_NONCOMPUTE_USD,
        }
        for slot in range(1, EFFICIENCY_REPLACEMENT_SUBMISSIONS)
    ]
    with tempfile.TemporaryDirectory(
        dir=Path(os.environ["TMPDIR"]) / "opencode",
        prefix="efficiency-launcher-self-check-",
    ) as temporary:
        historical_ledger = Path(temporary) / "original-slots.json"
        historical_ledger.write_text(json.dumps([history_row]), encoding="utf-8")
        with patch(__name__ + ".HISTORICAL_EFFICIENCY_LEDGER", historical_ledger):
            client = HistoryClient()
            projected = _check_efficiency_sequence(
                prior_rows,
                EFFICIENCY_REPLACEMENT_SUBMISSIONS,
                "final",
                EFFICIENCY_MAX_JOB_EXPOSURE_USD,
                EFFICIENCY_MAX_NONCOMPUTE_USD,
                source_sha,
                image_digest,
                cast(aiplatform_v1.JobServiceClient, client),
            )
            if projected != EFFICIENCY_MAX_TOTAL_COMPUTE_USD:
                failures.append("cumulative compute did not include historical reservation")
            else:
                print("  PASS budget: four replacement slots plus historical reserve meet the cap")
            try:
                _check_efficiency_sequence(
                    prior_rows,
                    EFFICIENCY_REPLACEMENT_SUBMISSIONS,
                    "final",
                    EFFICIENCY_MAX_JOB_EXPOSURE_USD + 0.001,
                    EFFICIENCY_MAX_NONCOMPUTE_USD,
                    source_sha,
                    image_digest,
                    cast(aiplatform_v1.JobServiceClient, client),
                )
            except SystemExit as exc:
                if "above" not in str(exc):
                    failures.append(f"unexpected over-budget rejection: {exc}")
                else:
                    print("  PASS budget: exposure above the aggregate cap rejected")
            else:
                failures.append("aggregate compute above the cap was accepted")
            try:
                _check_efficiency_sequence(
                    [{"slot": 2}],
                    2,
                    "cache",
                    1.25,
                    EFFICIENCY_MAX_NONCOMPUTE_USD,
                    source_sha,
                    image_digest,
                    cast(aiplatform_v1.JobServiceClient, client),
                )
            except SystemExit as exc:
                if "sequentially" not in str(exc):
                    failures.append(f"unexpected duplicate-slot rejection: {exc}")
                else:
                    print("  PASS ledger: out-of-order/duplicate slot rejected")
            else:
                failures.append("out-of-order/duplicate slot was accepted")
        verification_ledger = Path(temporary) / "verification-slot.json"
        with patch(__name__ + ".VERIFICATION_LEDGER", verification_ledger):
            _reserve_verification_slot(
                run_label=VERIFICATION_RUN_LABEL,
                source_sha=source_sha,
                image_digest=image_digest,
                hourly_rate=1.0,
                hourly_rate_source="self-check review note",
                exposure=1.25,
                noncompute_total=2.0,
            )
            _update_verification_slot(state="submitted", resource_name="self-check-job")
            try:
                _reserve_verification_slot(
                    run_label=VERIFICATION_RUN_LABEL,
                    source_sha=source_sha,
                    image_digest=image_digest,
                    hourly_rate=1.0,
                    hourly_rate_source="self-check review note",
                    exposure=1.25,
                    noncompute_total=2.0,
                )
            except SystemExit as exc:
                if "already consumed" not in str(exc):
                    failures.append(f"unexpected verification double-reserve: {exc}")
                else:
                    print("  PASS ledger: verification double reservation rejected")
            else:
                failures.append("verification double reservation was accepted")
    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK OK: queue/cancel, recovery-ledger, and combined-budget checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-uri")
    parser.add_argument("--source-sha")
    parser.add_argument("--run-label")
    parser.add_argument("--service-account")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG_NAME), default="smoke")
    parser.add_argument(
        "--efficiency-slot",
        type=int,
        choices=tuple(sorted(EFFICIENCY_SLOT_ROLE)),
    )
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
        default=None,
        help="client-side queue/provisioning wait budget (not sent to Vertex)",
    )
    parser.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=None,
        help="verified regional on-demand rate; required for multi-epoch/efficiency modes",
    )
    parser.add_argument(
        "--hourly-rate-source",
        default=None,
        help="provenance note for the reviewed Vertex SKU/rate; efficiency-only",
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
    parser.add_argument("--self-check", action="store_true", help="run local wait/cancel checks")
    parser.add_argument("--status", help="print the state of an existing job and exit")
    args = parser.parse_args()

    if args.self_check:
        return _wait_self_check()

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
    if args.timeout_seconds is not None:
        timeout_seconds = args.timeout_seconds
    elif args.mode in _FULL_TEMPORAL_MODES:
        timeout_seconds = FULL_TIMEOUT_SECONDS
    elif args.mode == "efficiency":
        timeout_seconds = EFFICIENCY_TIMEOUT_SECONDS
    elif args.mode == "verification":
        timeout_seconds = VERIFICATION_TIMEOUT_SECONDS
    elif args.mode in _PROBE_MODES:
        timeout_seconds = PROBE_TIMEOUT_SECONDS
    else:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    max_wait_seconds = (
        args.max_wait_seconds
        if args.max_wait_seconds is not None
        else (
            EFFICIENCY_MAX_WAIT_SECONDS
            if args.mode == "efficiency"
            else (
                VERIFICATION_MAX_WAIT_SECONDS
                if args.mode == "verification"
                else DEFAULT_MAX_WAIT_SECONDS
            )
        )
    )
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
            else (
                VERIFICATION_MAX_JOB_EXPOSURE_USD
                if args.mode == "verification"
                else (
                    FULL_MAX_EXPOSURE_USD
                    if args.mode in _FULL_TEMPORAL_MODES
                    else DEFAULT_MAX_EXPOSURE_USD
                )
            )
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
    if not math.isfinite(float(max_wait_seconds)) or max_wait_seconds < 0:
        raise SystemExit("ERROR: --max-wait-seconds must be a finite non-negative number")
    if args.mode in _FULL_TEMPORAL_MODES:
        # issue #53 / #58: one bounded job, 48-hour server timeout, $50 projected
        # compute; a full temporal run must not silently exceed either ceiling.
        if timeout_seconds > FULL_TIMEOUT_SECONDS:
            raise SystemExit(
                f"ERROR: --mode {args.mode} server timeout {timeout_seconds}s exceeds the "
                f"{FULL_TIMEOUT_SECONDS}s ceiling"
            )
        if max_exposure > FULL_MAX_EXPOSURE_USD:
            raise SystemExit(
                f"ERROR: --mode {args.mode} exposure ceiling cannot exceed "
                f"${FULL_MAX_EXPOSURE_USD:.2f}"
            )
        if hourly_rate > FULL_MAX_HOURLY_RATE_USD:
            raise SystemExit(
                f"ERROR: --mode {args.mode} rate exceeds the "
                f"${FULL_MAX_HOURLY_RATE_USD:.2f}/hour cap"
            )

    if args.mode == "efficiency":
        if timeout_seconds > EFFICIENCY_TIMEOUT_SECONDS:
            raise SystemExit("ERROR: efficiency timeout cannot exceed 2700 seconds")
        if max_wait_seconds > EFFICIENCY_MAX_WAIT_SECONDS:
            raise SystemExit("ERROR: efficiency queue allowance cannot exceed 1800 seconds")
        if max_exposure > EFFICIENCY_MAX_JOB_EXPOSURE_USD:
            raise SystemExit("ERROR: efficiency per-job exposure cannot exceed $1.25")
        if hourly_rate > EFFICIENCY_MAX_HOURLY_RATE_USD:
            raise SystemExit("ERROR: efficiency rate exceeds the user-authorized $1.00/hour cap")
        if args.projected_noncompute_total_usd > EFFICIENCY_MAX_NONCOMPUTE_USD:
            raise SystemExit("ERROR: cumulative non-compute estimate cannot exceed $2.00")

    if args.mode == "verification":
        if args.projected_noncompute_total_usd is None:
            raise SystemExit("ERROR: verification mode requires --projected-noncompute-total-usd")
        if timeout_seconds > VERIFICATION_TIMEOUT_SECONDS:
            raise SystemExit("ERROR: verification timeout cannot exceed 2700 seconds")
        if max_wait_seconds > VERIFICATION_MAX_WAIT_SECONDS:
            raise SystemExit("ERROR: verification queue allowance cannot exceed 1800 seconds")
        if max_exposure > VERIFICATION_MAX_JOB_EXPOSURE_USD:
            raise SystemExit("ERROR: verification per-job exposure cannot exceed $1.25")
        if hourly_rate > VERIFICATION_MAX_HOURLY_RATE_USD:
            raise SystemExit("ERROR: verification rate exceeds the user-authorized $1.00/hour cap")
        if args.projected_noncompute_total_usd > VERIFICATION_MAX_NONCOMPUTE_USD:
            raise SystemExit("ERROR: verification non-compute estimate cannot exceed $2.00")

    _validate_image_uri(args.image_uri)
    _validate_run_label(args.run_label)
    prefix = args.evidence_prefix.rstrip("/") + "/"
    if not prefix.startswith(APPROVED_EVIDENCE_ROOT):
        raise SystemExit(
            f"ERROR: --evidence-prefix must stay under {APPROVED_EVIDENCE_ROOT}, "
            f"got {args.evidence_prefix!r}"
        )
    check_bounds(config_name, efficiency_role=args.efficiency_role)

    exposure = _exposure_usd(hourly_rate, timeout_seconds, max_wait_seconds)
    if not math.isfinite(exposure) or exposure > max_exposure:
        raise SystemExit(
            f"ERROR: projected exposure ${exposure:.2f} exceeds the "
            f"${max_exposure:.2f} ceiling; refusing to submit"
        )

    efficiency_rows: list[dict] = []
    projected_compute_total = EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD
    client: aiplatform_v1.JobServiceClient | None = None
    if args.mode == "efficiency":
        client = _client(REGION)
        efficiency_rows = _load_efficiency_ledger()
        projected_compute_total = _check_efficiency_sequence(
            efficiency_rows,
            args.efficiency_slot,
            args.efficiency_role,
            exposure,
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

    if args.mode == "verification":
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha or ""):
            raise SystemExit("ERROR: verification jobs require a full 40-character source SHA")
        head = _local_head_sha()
        if head and head != args.source_sha:
            raise SystemExit(
                f"ERROR: verification --source-sha {args.source_sha} is not the local "
                f"HEAD {head}; build the image from the submitted source first"
            )
        if args.run_label != VERIFICATION_RUN_LABEL:
            raise SystemExit(f"ERROR: verification run label is fixed: {VERIFICATION_RUN_LABEL!r}")
        if not args.hourly_rate_source or len(args.hourly_rate_source.strip()) < 20:
            raise SystemExit(
                "ERROR: verification mode requires a reviewed Vertex SKU/rate source note"
            )
        if _load_verification_ledger():
            raise SystemExit(
                "ERROR: the single verification slot is already consumed; "
                "no replacement submission is authorized"
            )

    evidence_uri = f"{args.evidence_prefix.rstrip('/')}/{args.run_label}/evidence.json"
    if _evidence_exists(evidence_uri):
        raise SystemExit(
            f"ERROR: evidence already exists at {evidence_uri}; the run label "
            "must never be reused (the create-only upload would fail after the run)"
        )
    run_prefix = evidence_uri.rsplit("/", 1)[0]
    _checkpoint_uploading = args.mode in _FULL_TEMPORAL_MODES or (
        args.mode == "efficiency" and args.efficiency_role in ("baseline", "final")
    )
    if _checkpoint_uploading and _evidence_exists(f"{run_prefix}/best.ckpt"):
        # A previous run may have died after the checkpoint upload but
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
                ("VERTEX_EFFICIENCY_RATE_SOURCE", args.hourly_rate_source),
                ("VERTEX_EFFICIENCY_EXPOSURE_USD", f"{exposure:.8f}"),
                (
                    "VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD",
                    str(args.projected_noncompute_total_usd),
                ),
            ]
        )
    if args.mode == "verification":
        env.extend(
            [
                ("VERTEX_VERIFICATION_SESSION", VERIFICATION_SESSION_ID),
                ("VERTEX_VERIFICATION_RATE_USD", str(hourly_rate)),
                ("VERTEX_VERIFICATION_RATE_SOURCE", args.hourly_rate_source),
                ("VERTEX_VERIFICATION_EXPOSURE_USD", f"{exposure:.8f}"),
                (
                    "VERTEX_VERIFICATION_NONCOMPUTE_TOTAL_USD",
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
    print(f"  timeout        : {timeout_seconds}s  max-wait: {max_wait_seconds}s")
    print(f"  exposure       : ~${exposure:.2f} (estimate at ${hourly_rate}/h)")
    if args.mode == "efficiency":
        print(
            f"  submissions    : {EFFICIENCY_HISTORICAL_SUBMISSIONS} preserved + "
            f"{EFFICIENCY_REPLACEMENT_SUBMISSIONS} replacement "
            f"({EFFICIENCY_MAX_SUBMISSIONS} total maximum)"
        )
        print(
            "  cumulative cost: "
            f"~${projected_compute_total + args.projected_noncompute_total_usd:.2f} "
            "including historical reservation and cumulative non-compute reserve"
        )
        print(
            "  sequence cap   : "
            f"~${EFFICIENCY_MAX_TOTAL_COMPUTE_USD + EFFICIENCY_MAX_NONCOMPUTE_USD:.2f} "
            "including preserved failure and all four replacement slots"
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
            hourly_rate_source=args.hourly_rate_source,
            exposure=exposure,
            noncompute_total=args.projected_noncompute_total_usd,
        )
    if args.mode == "verification":
        _reserve_verification_slot(
            run_label=args.run_label,
            source_sha=args.source_sha,
            image_digest=args.image_uri.split("@", 1)[1],
            hourly_rate=hourly_rate,
            hourly_rate_source=args.hourly_rate_source,
            exposure=exposure,
            noncompute_total=args.projected_noncompute_total_usd,
        )
    if client is None:
        client = _client(REGION)
    try:
        resource_name = _submit(
            client,
            display_name=display_name,
            image_uri=args.image_uri,
            env=env,
            service_account=args.service_account,
            timeout_seconds=timeout_seconds,
            profile=args.mode,
        )
    except Exception as exc:
        # An ambiguous create still consumes the slot: record and reconcile by
        # inspection, never resubmit.
        if args.mode == "verification":
            _update_verification_slot(state="create_ambiguous", failure=str(exc)[:500])
        raise
    print(f"submitted job: {resource_name}", flush=True)
    if args.mode == "efficiency":
        _update_efficiency_slot(
            args.efficiency_slot,
            state="submitted",
            resource_name=resource_name,
        )
    if args.mode == "verification":
        _update_verification_slot(state="submitted", resource_name=resource_name)

    submitted_at = time.monotonic()
    start_deadline = submitted_at + max_wait_seconds
    overall_deadline = start_deadline + timeout_seconds + CANCEL_CONFIRMATION_SECONDS
    final, queue_deadline_missed = _poll_until_terminal(
        client,
        resource_name,
        start_deadline=start_deadline,
        max_wait_seconds=max_wait_seconds,
        overall_deadline=overall_deadline,
    )
    print(f"final state: {final.name}")
    if args.mode == "efficiency":
        _update_efficiency_slot(
            args.efficiency_slot,
            state=(
                "queue_deadline_exceeded"
                if queue_deadline_missed
                else (
                    "awaiting_validation"
                    if final == JobState.JOB_STATE_SUCCEEDED
                    else final.name.lower()
                )
            ),
            terminal_state=final.name,
        )
    if args.mode == "verification":
        _update_verification_slot(
            state=(
                "queue_deadline_exceeded"
                if queue_deadline_missed
                else (
                    "awaiting_validation"
                    if final == JobState.JOB_STATE_SUCCEEDED
                    else final.name.lower()
                )
            ),
            terminal_state=final.name,
        )
    if queue_deadline_missed:
        print(
            f"FAIL: {resource_name} missed the provisioning deadline; "
            "its slot is consumed. "
            + (
                "Do not submit a replacement verification job."
                if args.mode == "verification"
                else "Do not continue the replacement sequence."
            )
        )
        return 1
    if final == JobState.JOB_STATE_SUCCEEDED:
        print(f"SUCCESS: bounded vertex {args.mode} completed. Evidence: {evidence_uri}")
        return 0
    print(f"FAIL: vertex {args.mode} ended {final.name}. Query with --status {resource_name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
