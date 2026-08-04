"""Guarded GCS operations for COG layout recovery.

Provides generation-pinned, create-only, and restore operations using
the public ``google-cloud-storage`` SDK.  All operations emit immutable
events and fail closed on any precondition violation.
"""

from __future__ import annotations

import logging
from typing import Any

from google.api_core.exceptions import PreconditionFailed

from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    EventType,
    ImmutableEvent,
    ObjectDescriptor,
    event_dir_for_uri,
    event_path_for,
    make_event,
)
from berlin_lst_downscaling.data.io.storage import (
    _gcs_client,
    _parse_gs_uri,
)

_logger = logging.getLogger(__name__)


# ── metadata snapshot ─────────────────────────────────────────────────


def snapshot_gcs_descriptor(uri: str) -> ObjectDescriptor:
    """Snapshot the full GCS metadata contract for *uri*.

    Returns an ``ObjectDescriptor``.  Raises ``FileNotFoundError`` if
    the object does not exist.
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    if not blob.exists():
        raise FileNotFoundError(f"Object does not exist: {uri}")

    blob.reload()
    return ObjectDescriptor(
        generation=blob.generation or 0,
        metageneration=blob.metageneration or 0,
        size=blob.size or 0,
        crc32c=blob.crc32c or "",
        content_type=blob.content_type or "",
        content_encoding=blob.content_encoding,
        cache_control=blob.cache_control,
        custom_metadata=dict(blob.metadata) if blob.metadata else {},
        storage_class=blob.storage_class or "",
        kms_key_name=blob.kms_key_name,
        md5_hash=blob.md5_hash,
        updated=blob.updated.isoformat() if blob.updated else None,
    )


def snapshot_soft_deleted_descriptor(
    uri: str,
    generation: int,
    *,
    restore_token: str | None = None,
) -> ObjectDescriptor:
    """Snapshot metadata for a soft-deleted GCS object.

    Requires the exact *generation*.  Raises ``FileNotFoundError`` if the
    soft-deleted generation does not exist or its retention expired.
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key, generation=generation)

    if not blob.exists(soft_deleted=True):
        raise FileNotFoundError(
            f"Soft-deleted generation {generation} not found: {uri}"
        )

    blob.reload(soft_deleted=True)
    return ObjectDescriptor(
        generation=blob.generation or 0,
        metageneration=blob.metageneration or 0,
        size=blob.size or 0,
        crc32c=blob.crc32c or "",
        content_type=blob.content_type or "",
        content_encoding=blob.content_encoding,
        cache_control=blob.cache_control,
        custom_metadata=dict(blob.metadata) if blob.metadata else {},
        storage_class=blob.storage_class or "",
        kms_key_name=blob.kms_key_name,
        md5_hash=blob.md5_hash,
        updated=blob.updated.isoformat() if blob.updated else None,
    )


# ── metadata verification ─────────────────────────────────────────────


def verify_descriptor(
    uri: str,
    expected: ObjectDescriptor,
    *,
    soft_deleted: bool = False,
    generation: int | None = None,
) -> list[str]:
    """Verify that a GCS object matches the expected descriptor.

    Returns a list of errors; empty means verified.  Permits generated
    differences: ``generation``, ``metageneration``, ``updated``.
    """
    errors: list[str] = []
    try:
        if soft_deleted:
            if generation is None:
                raise ValueError("generation required for soft_deleted=True")
            actual = snapshot_soft_deleted_descriptor(uri, generation=generation)
        else:
            actual = snapshot_gcs_descriptor(uri)
    except FileNotFoundError:
        return [f"Object not found: {uri}"]

    # payload identity
    if actual.crc32c != expected.crc32c:
        errors.append(
            f"CRC32C mismatch: expected {expected.crc32c}, got {actual.crc32c}"
        )
    if actual.size != expected.size:
        errors.append(
            f"Size mismatch: expected {expected.size}, got {actual.size}"
        )

    # editable metadata
    if actual.content_type != expected.content_type:
        errors.append(
            f"Content-Type mismatch: expected {expected.content_type!r}, "
            f"got {actual.content_type!r}"
        )
    if actual.content_encoding != expected.content_encoding:
        errors.append(
            f"Content-Encoding mismatch: expected {expected.content_encoding!r}, "
            f"got {actual.content_encoding!r}"
        )
    if actual.cache_control != expected.cache_control:
        errors.append(
            f"Cache-Control mismatch: expected {expected.cache_control!r}, "
            f"got {actual.cache_control!r}"
        )
    if actual.custom_metadata != expected.custom_metadata:
        errors.append(
            f"Custom metadata mismatch: expected {expected.custom_metadata!r}, "
            f"got {actual.custom_metadata!r}"
        )

    return errors


# ── direct event persistence ──────────────────────────────────────────


def save_event(
    recovery_root: str,
    event: ImmutableEvent,
) -> str:
    """Persist an immutable event as a create-only GCS object.

    On ``412 PreconditionFailed``, fetches the existing event and
    accepts only a byte-identical collision.  Returns the event path.

    Raises ``RuntimeError`` on non-identical collision or write failure.
    """
    import json

    path = event_path_for(
        recovery_root,
        event.uri,
        event.sequence,
        event.event_type,
    )

    content = json.dumps(event.to_dict(), indent=2, sort_keys=True).encode()
    bucket_name, key = _parse_gs_uri(path)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    try:
        blob.upload_from_string(
            content,
            content_type="application/json",
            if_generation_match=0,
            checksum="crc32c",
        )
        return path
    except PreconditionFailed:
        # Idempotent accept: fetch existing and compare
        existing = blob.download_as_bytes()
        if existing == content:
            _logger.debug("Event already exists (identical): %s", path)
            return path
        raise RuntimeError(
            f"Non-identical event collision at {path}: "
            f"existing={len(existing)} bytes, new={len(content)} bytes"
        ) from None


def load_events(recovery_root: str, uri: str) -> list[ImmutableEvent]:
    """Load all events for a canonical URI from the recovery store.

    Returns a sorted list of events.  Missing directory returns an
    empty list.
    """
    import json

    event_dir = event_dir_for_uri(recovery_root, uri)
    bucket_name, prefix = _parse_gs_uri(event_dir.rstrip("/") + "/")
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    events: list[ImmutableEvent] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith(".json"):
            continue
        try:
            data = blob.download_as_bytes()
            d = json.loads(data)
            events.append(ImmutableEvent.from_dict(d))
        except Exception as exc:
            _logger.warning("Failed to load event %s: %s", blob.name, exc)

    events.sort(key=lambda e: e.sequence)
    return events


# ── server-side copy ──────────────────────────────────────────────────


def copy_object_server_side(
    source_uri: str,
    dest_uri: str,
    *,
    source_generation: int | None = None,
    source_metageneration: int | None = None,
    dest_generation_match: int = 0,
) -> dict[str, Any]:
    """Server-side copy with generation guards.

    Uses ``Bucket.copy_blob`` for a single-API-call copy that
    preserves editable source metadata by default.

    Returns a dict with dest ``generation``, ``metageneration``,
    ``crc32c``, ``size``.

    Raises ``FileNotFoundError`` if source does not exist.
    Raises ``google.api_core.exceptions.PreconditionFailed`` on guard
    violation.
    """
    source_bucket_name, source_key = _parse_gs_uri(source_uri)
    dest_bucket_name, dest_key = _parse_gs_uri(dest_uri)

    client = _gcs_client()
    source_bucket = client.bucket(source_bucket_name)
    dest_bucket = client.bucket(dest_bucket_name)

    source_blob = source_bucket.blob(source_key, generation=source_generation)
    if not source_blob.exists():
        raise FileNotFoundError(f"Source does not exist: {source_uri}")

    result_blob = source_bucket.copy_blob(
        source_blob,
        dest_bucket,
        new_name=dest_key,
        if_generation_match=dest_generation_match,
        if_source_generation_match=source_generation,
        if_source_metageneration_match=source_metageneration,
    )

    return {
        "generation": result_blob.generation,
        "metageneration": result_blob.metageneration,
        "crc32c": result_blob.crc32c,
        "size": result_blob.size,
    }


def backup_current_live(
    uri: str,
    backup_root: str,
    *,
    run_id: str,
    config_hash: str,
    sequence: int,
) -> dict[str, Any]:
    """Backup the current live object to the recovery bucket.

    Creates a create-only server-side copy at
    ``<backup_root>/backups/current/<key>`` and emits
    ``BACKUP_VERIFIED``.

    Returns a dict with ``backup_uri``, ``generation``,
    ``metageneration``, ``crc32c``, ``size``.
    """
    descriptor = snapshot_gcs_descriptor(uri)
    bucket_name, key = _parse_gs_uri(uri)
    backup_uri = f"{backup_root.rstrip('/')}/backups/current/{key}"

    result = copy_object_server_side(
        source_uri=uri,
        dest_uri=backup_uri,
        source_generation=descriptor.generation,
        source_metageneration=descriptor.metageneration,
        dest_generation_match=0,
    )

    if result["crc32c"] != descriptor.crc32c:
        raise RuntimeError(
            f"CRC32C mismatch after backup: source={descriptor.crc32c}, "
            f"dest={result['crc32c']}"
        )

    event = make_event(
        uri=uri,
        run_id=run_id,
        sequence=sequence,
        event_type=EventType.BACKUP_VERIFIED,
        config_hash=config_hash,
        generation_before=descriptor.generation,
        generation_after=result["generation"],
        metageneration_before=descriptor.metageneration,
        metageneration_after=result["metageneration"],
        crc32c=result["crc32c"],
        descriptor=descriptor,
        details={"backup_uri": backup_uri},
    )
    save_event(backup_root, event)

    return {
        "backup_uri": backup_uri,
        "generation": result["generation"],
        "metageneration": result["metageneration"],
        "crc32c": result["crc32c"],
        "size": result["size"],
    }


# ── soft delete restore ───────────────────────────────────────────────


def restore_soft_deleted_object(
    uri: str,
    *,
    soft_deleted_generation: int,
    current_generation: int,
    current_metageneration: int,
    restore_token: str | None = None,
) -> ObjectDescriptor:
    """Restore a soft-deleted generation to the canonical path.

    The restore replaces the current live object and creates a new
    generation.  The displaced live generation becomes soft-deleted.

    Returns the ``ObjectDescriptor`` of the restored object.

    Raises ``PreconditionFailed`` if the live generation changed.
    Raises ``FileNotFoundError`` if the soft-deleted generation expired.
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    kw: dict[str, Any] = {
        "generation": soft_deleted_generation,
        "if_generation_match": current_generation,
        "if_metageneration_match": current_metageneration,
    }
    if restore_token:
        kw["restore_token"] = restore_token

    restored_blob = bucket.restore_blob(key, **kw)
    restored_blob.reload()

    return ObjectDescriptor(
        generation=restored_blob.generation or 0,
        metageneration=restored_blob.metageneration or 0,
        size=restored_blob.size or 0,
        crc32c=restored_blob.crc32c or "",
        content_type=restored_blob.content_type or "",
        content_encoding=restored_blob.content_encoding,
        cache_control=restored_blob.cache_control,
        custom_metadata=dict(restored_blob.metadata) if restored_blob.metadata else {},
        storage_class=restored_blob.storage_class or "",
        kms_key_name=restored_blob.kms_key_name,
        md5_hash=restored_blob.md5_hash,
        updated=restored_blob.updated.isoformat() if restored_blob.updated else None,
    )


# ── resumable server-side rewrite ─────────────────────────────────────


def rewrite_object_server_side(
    source_uri: str,
    dest_uri: str,
    *,
    source_generation: int,
    source_metageneration: int,
    dest_generation: int,
    dest_metageneration: int,
    dest_metadata: ObjectDescriptor | None = None,
) -> dict[str, Any]:
    """Resumable server-side rewrite with full generation guards.

    Sets explicit metadata on the destination blob before rewrite.
    Handles multi-request rewrite via token persistence.

    Returns a dict with ``generation``, ``metageneration``,
    ``crc32c``, ``size``, ``bytes_rewritten``.

    Raises ``PreconditionFailed`` on guard violation.
    """
    source_bucket_name, source_key = _parse_gs_uri(source_uri)
    dest_bucket_name, dest_key = _parse_gs_uri(dest_uri)

    client = _gcs_client()
    source_bucket = client.bucket(source_bucket_name)
    dest_bucket = client.bucket(dest_bucket_name)

    source_blob = source_bucket.blob(source_key, generation=source_generation)
    if not source_blob.exists():
        raise FileNotFoundError(f"Source does not exist: {source_uri}")

    dest_blob = dest_bucket.blob(dest_key)

    # set explicit metadata on destination before rewrite
    if dest_metadata is not None:
        dest_blob.content_type = dest_metadata.content_type
        dest_blob.content_encoding = dest_metadata.content_encoding
        dest_blob.cache_control = dest_metadata.cache_control
        dest_blob.metadata = (
            dict(dest_metadata.custom_metadata)
            if dest_metadata.custom_metadata
            else None
        )

    token: str | None = None
    total_bytes = 0

    while True:
        result = dest_blob.rewrite(
            source_blob,
            token=token,
            if_generation_match=dest_generation if token is None else None,
            if_metageneration_match=(
                dest_metageneration if token is None else None
            ),
            if_source_generation_match=source_generation,
            if_source_metageneration_match=source_metageneration,
        )
        token = result[0]
        total_bytes = result[1]
        done = result[2]

        if done:
            break

    dest_blob.reload()
    return {
        "generation": dest_blob.generation,
        "metageneration": dest_blob.metageneration,
        "crc32c": dest_blob.crc32c,
        "size": dest_blob.size,
        "bytes_rewritten": total_bytes,
    }




def rollback_to_payload(
    uri: str,
    backup_uri: str,
    *,
    recovery_root: str,
    run_id: str,
    config_hash: str,
    sequence: int,
    current_generation: int,
    current_metageneration: int,
    metadata_contract: ObjectDescriptor | None = None,
) -> dict[str, Any]:
    """Rollback from a verified backup payload to the canonical path.

    Emits ``ROLLBACK_INTENT`` before mutation and ``ROLLED_BACK`` on
    success.

    Returns a dict with ``generation``, ``metageneration``,
    ``crc32c``, ``size``.
    """
    intent = make_event(
        uri=uri,
        run_id=run_id,
        sequence=sequence,
        event_type=EventType.ROLLBACK_INTENT,
        config_hash=config_hash,
        generation_before=current_generation,
        metageneration_before=current_metageneration,
        details={"backup_uri": backup_uri},
    )
    save_event(recovery_root, intent)

    result = rewrite_object_server_side(
        source_uri=backup_uri,
        dest_uri=uri,
        source_generation=0,
        source_metageneration=0,
        dest_generation=current_generation,
        dest_metageneration=current_metageneration,
        dest_metadata=metadata_contract,
    )

    rolled = make_event(
        uri=uri,
        run_id=run_id,
        sequence=sequence + 1,
        event_type=EventType.ROLLED_BACK,
        config_hash=config_hash,
        generation_before=current_generation,
        generation_after=result["generation"],
        metageneration_before=current_metageneration,
        metageneration_after=result["metageneration"],
        crc32c=result["crc32c"],
        details={"backup_uri": backup_uri},
    )
    save_event(recovery_root, rolled)

    return result


# ── soft delete catalog ───────────────────────────────────────────────


def list_soft_deleted_generations(
    bucket_name: str,
    prefix: str = "",
    *,
    time_lo: Any | None = None,
    time_hi: Any | None = None,
) -> list[dict[str, Any]]:
    """List soft-deleted generations in *bucket_name* within a time window.

    Returns a list of dicts with ``name``, ``generation``,
    ``soft_delete_time``, ``hard_delete_time``, ``restore_token``.
    """
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    items: list[dict[str, Any]] = []

    for blob in client.list_blobs(bucket, prefix=prefix, soft_deleted=True):
        sdt = blob.soft_delete_time
        if time_lo and sdt and sdt < time_lo:
            continue
        if time_hi and sdt and sdt >= time_hi:
            continue
        items.append({
            "name": blob.name,
            "generation": blob.generation,
            "soft_delete_time": sdt.isoformat() if sdt else None,
            "hard_delete_time": (
                blob.hard_delete_time.isoformat() if blob.hard_delete_time else None
            ),
            "restore_token": getattr(blob, "restore_token", None),
        })

    return items


__all__ = [
    "snapshot_gcs_descriptor",
    "snapshot_soft_deleted_descriptor",
    "verify_descriptor",
    "save_event",
    "load_events",
    "copy_object_server_side",
    "backup_current_live",
    "restore_soft_deleted_object",
    "rewrite_object_server_side",
    "rollback_to_payload",
    "list_soft_deleted_generations",
]
