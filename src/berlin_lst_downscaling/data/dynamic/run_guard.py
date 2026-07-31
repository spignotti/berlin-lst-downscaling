"""GCS-based run guard for Dynamic pipeline single-writer enforcement.

Creates a bucket-global lock object to prevent concurrent Dynamic runs.
Uses GCS object generation preconditions for atomic acquire/release.

The lock record carries the owner's hostname, PID, and run_id.
Only same-host PID liveness is checked; cross-host stale locks are
reported but never auto-removed — the operator must decide.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

# Lock object key under the run root
_LOCK_SUFFIX = "_dynamic_run_lock.json"


@dataclass(frozen=True)
class RunGuardLease:
    """Opaque lease returned by acquire_run_guard.

    Pass this directly to release_run_guard — do not construct manually.
    """

    lock_uri: str
    generation: int


def _lock_uri(output_root: str) -> str:
    """Return the GCS URI for the run guard lock object."""
    return f"{output_root.rstrip('/')}/{_LOCK_SUFFIX}"


def acquire_run_guard(
    output_root: str,
    run_id: str,
    *,
    git_sha: str = "",
) -> RunGuardLease | None:
    """Try to acquire the Dynamic run guard.

    Returns a RunGuardLease on success, None if another run owns the lock.
    The lease carries the GCS generation needed for safe release.
    """
    from google.cloud import storage

    lock_path = _lock_uri(output_root)
    bucket_name, key = lock_path.removeprefix("gs://").split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    # Check if lock already exists
    if blob.exists():
        existing = json.loads(blob.download_as_text())
        owner_pid = existing.get("pid")
        owner_host = existing.get("host")
        owner_run = existing.get("run_id")
        owner_start = existing.get("start_utc", "")

        # Staleness check: lock older than 2 hours is always stale.
        # Handles subprocess crashes that bypass the finally-release path.
        lock_stale = False
        if owner_start:
            try:
                from datetime import datetime as _dt
                age_s = (_dt.now(UTC) - _dt.fromisoformat(owner_start)).total_seconds()
                lock_stale = age_s > 7200  # 2 hours
            except (ValueError, TypeError):
                pass

        if lock_stale:
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_stale_ttl",
                owner_run=owner_run,
                owner_host=owner_host,
                owner_pid=owner_pid,
                message="lock older than 2h; treating as stale",
            )
        else:
            # Same-host PID check: only meaningful when we share the process namespace
            same_host = owner_host == os.uname().nodename
            owner_alive = False
            if same_host:
                try:
                    os.kill(owner_pid, 0)
                    owner_alive = True
                except (ProcessLookupError, PermissionError):
                    owner_alive = False

            if owner_alive:
                log_event(
                    _logger,
                    logging.WARNING,
                    "run_guard_conflict",
                    owner_run=owner_run,
                    owner_host=owner_host,
                    owner_pid=owner_pid,
                )
                return None

            # Stale lock (dead on same host, or cross-host)
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_stale",
                owner_run=owner_run,
                owner_host=owner_host,
                owner_pid=owner_pid,
                same_host=same_host,
                message="stale lock; manual removal required",
            )
            return None

    # Acquire with generation precondition
    guard_data = json.dumps(
        {
            "run_id": run_id,
            "host": os.uname().nodename,
            "pid": os.getpid(),
            "start_utc": datetime.now(UTC).isoformat(),
            "git_sha": git_sha,
        },
        indent=2,
    )

    try:
        blob.upload_from_string(
            guard_data,
            content_type="application/json",
            if_generation_match=0,
        )
        gen = blob.generation
        if gen is None:
            raise RuntimeError("GCS blob generation is None after upload")
        log_event(
            _logger,
            logging.INFO,
            "run_guard_acquired",
            run_id=run_id,
            lock_uri=lock_path,
            generation=gen,
        )
        return RunGuardLease(lock_uri=lock_path, generation=gen)
    except Exception as exc:
        if "conditionNotMet" in str(exc) or "412" in str(exc):
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_race",
                message="another process acquired the guard first",
            )
            return None
        raise


def release_run_guard(output_root: str, lease: RunGuardLease) -> None:
    """Release the run guard by deleting the lock object.

    Uses the lease generation to prevent deleting a lock that was
    acquired by a different process after this lease was granted.
    """
    from google.cloud import storage

    bucket_name, key = lease.lock_uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)

    try:
        blob.delete(if_generation_match=lease.generation)
        log_event(
            _logger,
            logging.INFO,
            "run_guard_released",
            lock_uri=lease.lock_uri,
            generation=lease.generation,
        )
    except Exception as exc:
        if "conditionNotMet" in str(exc) or "412" in str(exc):
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_release_stale",
                lock_uri=lease.lock_uri,
                message="lock generation changed; another process may hold it",
            )
        else:
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_release_failed",
                lock_uri=lease.lock_uri,
                error=str(exc),
            )


def remove_stale_lock(output_root: str) -> bool:
    """Delete a stale lock unconditionally. Operator-only use.

    Returns True if the object existed and was deleted, False otherwise.
    """
    from google.cloud import storage

    lock_path = _lock_uri(output_root)
    bucket_name, key = lock_path.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)

    if not blob.exists():
        log_event(_logger, logging.INFO, "run_guard_absent", lock_uri=lock_path)
        return False

    content = json.loads(blob.download_as_text())
    blob.delete()
    log_event(
        _logger,
        logging.INFO,
        "run_guard_removed",
        lock_uri=lock_path,
        owner_run=content.get("run_id"),
        owner_host=content.get("host"),
        owner_pid=content.get("pid"),
    )
    return True


__all__ = [
    "RunGuardLease",
    "acquire_run_guard",
    "release_run_guard",
    "remove_stale_lock",
]
