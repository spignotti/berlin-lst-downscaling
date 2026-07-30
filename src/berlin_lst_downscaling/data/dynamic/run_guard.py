"""GCS-based run guard for Dynamic pipeline single-writer enforcement.

Creates a bucket-global lock object to prevent concurrent Dynamic runs.
Uses GCS object generation preconditions for atomic acquire/release.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

# Lock object key under the run root
_LOCK_SUFFIX = "_dynamic_run_lock.json"


def _lock_uri(output_root: str) -> str:
    """Return the GCS URI for the run guard lock object."""
    return f"{output_root.rstrip('/')}/{_LOCK_SUFFIX}"


def acquire_run_guard(
    output_root: str,
    run_id: str,
    *,
    git_sha: str = "",
) -> str | None:
    """Try to acquire the Dynamic run guard.

    Returns the lock URI on success, None if another run owns the lock.
    The lock records run_id, host, PID, start time and git SHA.
    """
    from google.cloud import storage

    lock_path = _lock_uri(output_root)
    bucket_name, key = lock_path.removeprefix("gs://").split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    # Check if lock already exists
    if blob.exists():
        # Read existing lock
        existing = json.loads(blob.download_as_text())
        owner_pid = existing.get("pid")
        owner_host = existing.get("host")
        owner_run = existing.get("run_id")

        # Check if the owning process is still alive (same host only)
        try:
            os.kill(owner_pid, 0)
            # Process alive — lock is held
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_conflict",
                owner_run=owner_run,
                owner_host=owner_host,
                owner_pid=owner_pid,
            )
            return None
        except (ProcessLookupError, PermissionError):
            # Process dead — stale lock, but do not auto-steal
            log_event(
                _logger,
                logging.WARNING,
                "run_guard_stale",
                owner_run=owner_run,
                owner_host=owner_host,
                owner_pid=owner_pid,
                message="stale lock detected; manual removal required",
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
        log_event(
            _logger,
            logging.INFO,
            "run_guard_acquired",
            run_id=run_id,
            lock_uri=lock_path,
        )
        return lock_path
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


def release_run_guard(output_root: str, lock_uri: str) -> None:
    """Release the run guard by deleting the lock object."""
    from google.cloud import storage

    bucket_name, key = lock_uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)

    try:
        blob.delete()
        log_event(_logger, logging.INFO, "run_guard_released", lock_uri=lock_uri)
    except Exception as exc:
        log_event(
            _logger,
            logging.WARNING,
            "run_guard_release_failed",
            lock_uri=lock_uri,
            error=str(exc),
        )


__all__ = [
    "acquire_run_guard",
    "release_run_guard",
]
