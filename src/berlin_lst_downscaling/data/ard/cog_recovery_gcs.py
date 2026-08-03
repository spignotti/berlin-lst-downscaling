"""Guarded GCS operations for COG layout recovery.

Provides generation-pinned, create-only, and restore operations using
the public google-cloud-storage SDK. All operations emit immutable
events and fail closed on any precondition violation.

This module does NOT implement the full recovery state machine — it
provides the building blocks that the recovery CLI uses.
"""

from __future__ import annotations

import logging
from typing import Any

from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    EventType,
    make_event,
    save_event,
)
from berlin_lst_downscaling.data.io.storage import (
    _gcs_client,
    _parse_gs_uri,
)

_logger = logging.getLogger(__name__)


# ── metadata operations ───────────────────────────────────────────────


def snapshot_gcs_metadata(uri: str) -> dict[str, Any]:
    """Snapshot GCS object metadata including generation, CRC32C, size.

    Returns a dict with keys: generation, metageneration, size, crc32c,
    content_type, metadata (custom), updated, etag.

    Raises FileNotFoundError if the object does not exist.
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    if not blob.exists():
        raise FileNotFoundError(f"Object does not exist: {uri}")

    blob.reload()
    return {
        "generation": blob.generation,
        "metageneration": blob.metageneration,
        "size": blob.size,
        "crc32c": blob.crc32c,
        "content_type": blob.content_type,
        "metadata": blob.metadata or {},
        "updated": blob.updated.isoformat() if blob.updated else None,
        "etag": blob.etag,
    }


def verify_gcs_metadata(
    uri: str,
    expected_generation: int,
    expected_crc32c: str,
) -> list[str]:
    """Verify that a GCS object matches expected generation and CRC32C.

    Returns a list of errors; empty means verified.
    """
    errors: list[str] = []

    try:
        actual = snapshot_gcs_metadata(uri)
    except FileNotFoundError:
        return [f"Object not found: {uri}"]

    if actual["generation"] != expected_generation:
        errors.append(
            f"Generation mismatch: expected {expected_generation}, "
            f"got {actual['generation']}"
        )

    if actual["crc32c"] != expected_crc32c:
        errors.append(
            f"CRC32C mismatch: expected {expected_crc32c}, "
            f"got {actual['crc32c']}"
        )

    return errors


# ── copy operations ───────────────────────────────────────────────────


def copy_with_guard(
    source_uri: str,
    dest_uri: str,
    *,
    source_generation: int | None = None,
    dest_generation_match: int = 0,  # 0 = create-only
    preserve_metadata: bool = True,
) -> dict[str, Any]:
    """Copy a GCS object with generation guards.

    Args:
        source_uri: Source gs:// URI
        dest_uri: Destination gs:// URI
        source_generation: If set, copy only this generation
        dest_generation_match: Generation precondition for dest (0 = create-only)
        preserve_metadata: If True, copy custom metadata

    Returns:
        Dict with dest generation, crc32c, size

    Raises:
        FileNotFoundError: Source does not exist
        google.api_core.exceptions.PreconditionFailed: Generation mismatch
    """
    source_bucket_name, source_key = _parse_gs_uri(source_uri)
    dest_bucket_name, dest_key = _parse_gs_uri(dest_uri)

    client = _gcs_client()
    source_bucket = client.bucket(source_bucket_name)
    dest_bucket = client.bucket(dest_bucket_name)

    # Get source blob
    source_blob = source_bucket.blob(source_key, generation=source_generation)
    if not source_blob.exists():
        raise FileNotFoundError(f"Source does not exist: {source_uri}")

    source_blob.reload()

    # Prepare dest blob
    dest_blob = dest_bucket.blob(dest_key)

    # Copy with generation guard
    if preserve_metadata:
        dest_blob.metadata = source_blob.metadata

    # Download source bytes and upload to dest
    source_bytes = source_blob.download_as_bytes()
    dest_blob.upload_from_string(
        source_bytes,
        content_type=source_blob.content_type,
        if_generation_match=dest_generation_match,
        checksum="crc32c",
    )

    # Verify
    dest_blob.reload()
    if dest_blob.crc32c != source_blob.crc32c:
        raise RuntimeError(
            f"CRC32C mismatch after copy: source={source_blob.crc32c}, "
            f"dest={dest_blob.crc32c}"
        )

    return {
        "generation": dest_blob.generation,
        "crc32c": dest_blob.crc32c,
        "size": dest_blob.size,
    }


def backup_current_live(
    uri: str,
    backup_root: str,
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Backup the current live object to the recovery bucket.

    Creates a create-only copy at:
        <backup_root>/backups/current/<key>

    Emits BACKUP_VERIFIED event on success.

    Returns:
        Dict with backup_uri, generation, crc32c, size
    """
    bucket_name, key = _parse_gs_uri(uri)
    backup_uri = f"{backup_root.rstrip('/')}/backups/current/{key}"

    # Snapshot current metadata
    metadata = snapshot_gcs_metadata(uri)

    # Copy to backup location
    result = copy_with_guard(
        source_uri=uri,
        dest_uri=backup_uri,
        source_generation=metadata["generation"],
        dest_generation_match=0,  # create-only
    )

    # Emit event
    event = make_event(
        uri=uri,
        sequence=0,
        event_type=EventType.BACKUP_VERIFIED,
        config_hash=config_hash,
        generation_before=metadata["generation"],
        generation_after=result["generation"],
        crc32c=result["crc32c"],
        details={"backup_uri": backup_uri},
    )
    save_event(backup_root, event)

    return {
        "backup_uri": backup_uri,
        "generation": result["generation"],
        "crc32c": result["crc32c"],
        "size": result["size"],
    }


# ── restore operations ────────────────────────────────────────────────


def restore_soft_deleted_original(
    uri: str,
    original_generation: int,
    expected_current_generation: int,
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Restore a soft-deleted original to the canonical path.

    Uses Bucket.restore_blob() with generation guard. The restore
    replaces the current live object and creates a new generation.

    Args:
        uri: Canonical gs:// URI
        original_generation: Generation of the soft-deleted original
        expected_current_generation: Current live generation (guard)

    Returns:
        Dict with new_generation, crc32c, size

    Raises:
        google.api_core.exceptions.PreconditionFailed: Current generation changed
        FileNotFoundError: Original generation not found (expired?)
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    # Verify current generation matches expected
    current_blob = bucket.blob(key)
    if not current_blob.exists():
        raise FileNotFoundError(f"Current object not found: {uri}")

    current_blob.reload()
    if current_blob.generation != expected_current_generation:
        raise RuntimeError(
            f"Current generation mismatch: expected {expected_current_generation}, "
            f"got {current_blob.generation}"
        )

    # Restore soft-deleted original
    # This replaces the current live object
    restored_blob = bucket.restore_blob(
        key,
        generation=original_generation,
        if_generation_match=expected_current_generation,
    )

    # Verify restored object
    restored_blob.reload()

    # Emit event
    event = make_event(
        uri=uri,
        sequence=1,
        event_type=EventType.ORIGINAL_RECOVERED,
        config_hash=config_hash,
        generation_before=expected_current_generation,
        generation_after=restored_blob.generation,
        crc32c=restored_blob.crc32c,
        details={"original_generation": original_generation},
    )
    save_event(f"gs://{bucket_name}", event)

    return {
        "new_generation": restored_blob.generation,
        "crc32c": restored_blob.crc32c,
        "size": restored_blob.size,
    }


def restore_from_backup(
    uri: str,
    backup_uri: str,
    expected_current_generation: int,
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Restore from a verified backup to the canonical path.

    Uses copy with generation guard to replace current live object.

    Args:
        uri: Canonical gs:// URI
        backup_uri: Backup gs:// URI
        expected_current_generation: Current live generation (guard)

    Returns:
        Dict with new_generation, crc32c, size
    """
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    # Verify current generation
    current_blob = bucket.blob(key)
    if not current_blob.exists():
        raise FileNotFoundError(f"Current object not found: {uri}")

    current_blob.reload()
    if current_blob.generation != expected_current_generation:
        raise RuntimeError(
            f"Current generation mismatch: expected {expected_current_generation}, "
            f"got {current_blob.generation}"
        )

    # Copy backup to canonical path
    result = copy_with_guard(
        source_uri=backup_uri,
        dest_uri=uri,
        dest_generation_match=expected_current_generation,
    )

    # Emit event
    event = make_event(
        uri=uri,
        sequence=2,
        event_type=EventType.ROLLED_BACK,
        config_hash=config_hash,
        generation_before=expected_current_generation,
        generation_after=result["generation"],
        crc32c=result["crc32c"],
        details={"backup_uri": backup_uri},
    )
    save_event(f"gs://{bucket_name}", event)

    return {
        "new_generation": result["generation"],
        "crc32c": result["crc32c"],
        "size": result["size"],
    }


# ── promote operations ────────────────────────────────────────────────


def promote_candidate(
    uri: str,
    candidate_uri: str,
    expected_current_generation: int,
    *,
    config_hash: str,
) -> dict[str, Any]:
    """Promote a staged candidate to the canonical path.

    Uses copy with generation guard to replace current live object.

    Args:
        uri: Canonical gs:// URI
        candidate_uri: Candidate gs:// URI in recovery bucket
        expected_current_generation: Current live generation (guard)

    Returns:
        Dict with new_generation, crc32c, size
    """
    # Emit promotion intent
    bucket_name, key = _parse_gs_uri(uri)
    event = make_event(
        uri=uri,
        sequence=3,
        event_type=EventType.PROMOTION_INTENT,
        config_hash=config_hash,
        generation_before=expected_current_generation,
        details={"candidate_uri": candidate_uri},
    )
    save_event(f"gs://{bucket_name}", event)

    # Copy candidate to canonical path
    result = copy_with_guard(
        source_uri=candidate_uri,
        dest_uri=uri,
        dest_generation_match=expected_current_generation,
    )

    # Emit promoted event
    event = make_event(
        uri=uri,
        sequence=4,
        event_type=EventType.PROMOTED,
        config_hash=config_hash,
        generation_before=expected_current_generation,
        generation_after=result["generation"],
        crc32c=result["crc32c"],
        details={"candidate_uri": candidate_uri},
    )
    save_event(f"gs://{bucket_name}", event)

    return {
        "new_generation": result["generation"],
        "crc32c": result["crc32c"],
        "size": result["size"],
    }


# ── scratch operations (for transaction smoke) ───────────────────────


def create_scratch_object(
    scratch_bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str = "image/tiff",
) -> dict[str, Any]:
    """Create a scratch object for testing.

    Returns dict with uri, generation, crc32c.
    """
    uri = f"gs://{scratch_bucket}/{key}"
    bucket_name, blob_key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_key)

    blob.upload_from_string(
        data,
        content_type=content_type,
        if_generation_match=0,  # create-only
        checksum="crc32c",
    )

    blob.reload()
    return {
        "uri": uri,
        "generation": blob.generation,
        "crc32c": blob.crc32c,
    }


def delete_scratch_object(scratch_bucket: str, key: str) -> None:
    """Delete a scratch object."""
    uri = f"gs://{scratch_bucket}/{key}"
    bucket_name, blob_key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_key)

    if blob.exists():
        blob.delete()


__all__ = [
    "snapshot_gcs_metadata",
    "verify_gcs_metadata",
    "copy_with_guard",
    "backup_current_live",
    "restore_soft_deleted_original",
    "restore_from_backup",
    "promote_candidate",
    "create_scratch_object",
    "delete_scratch_object",
]
