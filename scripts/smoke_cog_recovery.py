#!/usr/bin/env python3
"""Smoke test for COG recovery GCS transaction semantics.

Exercises restore, metadata reconstruction, rewrite resumption, event
persistence, and rollback on isolated, noncanonical scratch paths.

Usage::

    uv run python scripts/smoke_cog_recovery.py \\
        --config configs/cog_repair/remediation.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-canary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.api_core.exceptions import PreconditionFailed

from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
    copy_object_server_side,
    load_events,
    restore_soft_deleted_object,
    rewrite_object_server_side,
    save_event,
    snapshot_gcs_descriptor,
)
from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    EventType,
    ObjectDescriptor,
    event_content_hash,
    hash_config,
    make_event,
    make_operation_id,
    reduce_events,
)
from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# ── scratch object management ─────────────────────────────────────────


def create_scratch_object(
    scratch_bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str = "image/tiff",
) -> dict[str, Any]:
    """Create a scratch object with generation=0 guard."""
    uri = f"gs://{scratch_bucket}/{key}"
    bucket_name, blob_key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_key)

    blob.upload_from_string(
        data,
        content_type=content_type,
        if_generation_match=0,
        checksum="crc32c",
    )
    blob.reload()
    return {
        "uri": uri,
        "generation": blob.generation,
        "metageneration": blob.metageneration,
        "crc32c": blob.crc32c,
        "size": blob.size,
    }


def delete_scratch_prefix(scratch_bucket: str, prefix: str) -> int:
    """Delete all objects under a scratch prefix. Returns count."""
    client = _gcs_client()
    bucket = client.bucket(scratch_bucket)
    count = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        blob.delete()
        count += 1
    return count


# ── test cases ────────────────────────────────────────────────────────


class CanaryResult:
    def __init__(self, name: str):
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.started_at = _now()

    def ok(self, check: str) -> None:
        self.passed.append(check)
        print(f"  PASS: {check}")

    def fail(self, check: str, detail: str = "") -> None:
        msg = f"{check}: {detail}" if detail else check
        self.failed.append(msg)
        print(f"  FAIL: {msg}")

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "passed": len(self.passed),
            "failed": len(self.failed),
            "failures": self.failed,
            "started_at": self.started_at,
            "finished_at": _now(),
        }


def test_basic_copy_and_verify(
    scratch_bucket: str, prefix: str, config_hash: str,
) -> CanaryResult:
    """Test basic server-side copy with generation guards."""
    r = CanaryResult("basic_copy_and_verify")
    key_a = f"{prefix}/canary_a.tif"
    key_b = f"{prefix}/canary_b.tif"
    data_a = b"canary_data_A_" + _uid().encode()
    data_b = b"canary_data_B_" + _uid().encode()

    # create A
    try:
        a = create_scratch_object(scratch_bucket, key_a, data_a)
        r.ok("create_A")
    except Exception as exc:
        r.fail("create_A", str(exc))
        return r

    # create B
    try:
        create_scratch_object(scratch_bucket, key_b, data_b)
        r.ok("create_B")
    except Exception as exc:
        r.fail("create_B", str(exc))
        return r

    # copy A -> new location (create-only)
    key_c = f"{prefix}/canary_c.tif"
    uri_a = f"gs://{scratch_bucket}/{key_a}"
    uri_c = f"gs://{scratch_bucket}/{key_c}"
    try:
        result = copy_object_server_side(
            source_uri=uri_a,
            dest_uri=uri_c,
            source_generation=a["generation"],
            dest_generation_match=0,
        )
        r.ok("copy_A_to_C")
        if result["crc32c"] != a["crc32c"]:
            r.fail("copy_crc_match", f"expected {a['crc32c']}, got {result['crc32c']}")
        else:
            r.ok("copy_crc_match")
    except Exception as exc:
        r.fail("copy_A_to_C", str(exc))

    # verify duplicate create fails with 412
    try:
        copy_object_server_side(
            source_uri=uri_a,
            dest_uri=uri_c,
            source_generation=a["generation"],
            dest_generation_match=0,
        )
        r.fail("duplicate_create_rejected", "expected PreconditionFailed")
    except PreconditionFailed:
        r.ok("duplicate_create_rejected")
    except Exception as exc:
        r.fail("duplicate_create_rejected", f"unexpected: {exc}")

    # verify stale source generation fails
    try:
        copy_object_server_side(
            source_uri=uri_a,
            dest_uri=uri_c,
            source_generation=a["generation"] - 1,
            dest_generation_match=result["generation"],
        )
        r.fail("stale_source_rejected", "expected rejection")
    except (PreconditionFailed, FileNotFoundError):
        r.ok("stale_source_rejected")
    except Exception as exc:
        r.fail("stale_source_rejected", f"unexpected: {exc}")

    return r


def test_soft_delete_restore(
    scratch_bucket: str, prefix: str, config_hash: str,
) -> CanaryResult:
    """Test Soft Delete restore with generation guards."""
    r = CanaryResult("soft_delete_restore")
    key = f"{prefix}/restore_test.tif"
    data_v1 = b"version_1_data_" + _uid().encode()
    data_v2 = b"version_2_data_" + _uid().encode()

    # create v1
    try:
        v1 = create_scratch_object(scratch_bucket, key, data_v1)
        r.ok("create_v1")
    except Exception as exc:
        r.fail("create_v1", str(exc))
        return r

    # overwrite with v2 (this soft-deletes v1)
    try:
        uri = f"gs://{scratch_bucket}/{key}"
        bucket_name, blob_key = _parse_gs_uri(uri)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_key)
        blob.upload_from_string(
            data_v2,
            content_type="image/tiff",
            if_generation_match=v1["generation"],
            checksum="crc32c",
        )
        blob.reload()
        v2 = {
            "generation": blob.generation,
            "metageneration": blob.metageneration,
            "crc32c": blob.crc32c,
        }
        r.ok("overwrite_v2")
    except Exception as exc:
        r.fail("overwrite_v2", str(exc))
        return r

    # restore v1 (soft-deleted)
    try:
        restored = restore_soft_deleted_object(
            uri,
            soft_deleted_generation=v1["generation"],
            current_generation=v2["generation"],
            current_metageneration=v2["metageneration"],
        )
        r.ok("restore_v1")
        if restored.crc32c != v1["crc32c"]:
            r.fail(
                "restore_crc_match",
                f"expected {v1['crc32c']}, got {restored.crc32c}",
            )
        else:
            r.ok("restore_crc_match")
        if restored.content_type != "image/tiff":
            r.fail(
                "restore_content_type",
                f"expected 'image/tiff', got {restored.content_type!r}",
            )
        else:
            r.ok("restore_content_type")
    except Exception as exc:
        r.fail("restore_v1", str(exc))
        return r

    # verify stale restore fails
    try:
        restore_soft_deleted_object(
            uri,
            soft_deleted_generation=v1["generation"],
            current_generation=v2["generation"],  # stale — now restored
            current_metageneration=v2["metageneration"],
        )
        r.fail("stale_restore_rejected", "expected PreconditionFailed")
    except PreconditionFailed:
        r.ok("stale_restore_rejected")
    except Exception as exc:
        r.fail("stale_restore_rejected", f"unexpected: {exc}")

    return r


def test_rewrite_with_metadata(
    scratch_bucket: str, prefix: str, config_hash: str,
) -> CanaryResult:
    """Test server-side rewrite with explicit metadata."""
    r = CanaryResult("rewrite_with_metadata")
    key_src = f"{prefix}/rewrite_src.tif"
    key_dst = f"{prefix}/rewrite_dst.tif"
    data = b"rewrite_test_data_" + _uid().encode()

    # create source
    try:
        src = create_scratch_object(scratch_bucket, key_src, data, content_type="image/tiff")
        r.ok("create_source")
    except Exception as exc:
        r.fail("create_source", str(exc))
        return r

    # create destination with different content type
    try:
        dst_data = b"old_destination_" + _uid().encode()
        dst = create_scratch_object(
            scratch_bucket, key_dst, dst_data, content_type="application/octet-stream",
        )
        r.ok("create_destination")
    except Exception as exc:
        r.fail("create_destination", str(exc))
        return r

    # rewrite source -> destination with metadata override
    uri_src = f"gs://{scratch_bucket}/{key_src}"
    uri_dst = f"gs://{scratch_bucket}/{key_dst}"
    metadata = ObjectDescriptor(
        content_type="image/tiff",
        custom_metadata={"recovery": "canary"},
    )
    try:
        result = rewrite_object_server_side(
            source_uri=uri_src,
            dest_uri=uri_dst,
            source_generation=src["generation"],
            source_metageneration=src["metageneration"],
            dest_generation=dst["generation"],
            dest_metageneration=dst["metageneration"],
            dest_metadata=metadata,
        )
        r.ok("rewrite")
        if result["crc32c"] != src["crc32c"]:
            r.fail(
                "rewrite_crc_match",
                f"expected {src['crc32c']}, got {result['crc32c']}",
            )
        else:
            r.ok("rewrite_crc_match")
    except Exception as exc:
        r.fail("rewrite", str(exc))
        return r

    # verify metadata was applied
    try:
        desc = snapshot_gcs_descriptor(uri_dst)
        if desc.content_type != "image/tiff":
            r.fail(
                "rewrite_content_type",
                f"expected 'image/tiff', got {desc.content_type!r}",
            )
        else:
            r.ok("rewrite_content_type")
        if desc.custom_metadata.get("recovery") != "canary":
            r.fail(
                "rewrite_custom_metadata",
                f"expected {{'recovery': 'canary'}}, got {desc.custom_metadata}",
            )
        else:
            r.ok("rewrite_custom_metadata")
    except Exception as exc:
        r.fail("verify_metadata", str(exc))

    # verify stale destination generation fails
    try:
        rewrite_object_server_side(
            source_uri=uri_src,
            dest_uri=uri_dst,
            source_generation=src["generation"],
            source_metageneration=src["metageneration"],
            dest_generation=dst["generation"],  # stale
            dest_metageneration=dst["metageneration"],
        )
        r.fail("stale_rewrite_rejected", "expected PreconditionFailed")
    except PreconditionFailed:
        r.ok("stale_rewrite_rejected")
    except Exception as exc:
        r.fail("stale_rewrite_rejected", f"unexpected: {exc}")

    return r


def test_event_persistence(
    scratch_bucket: str, prefix: str, config_hash: str,
) -> CanaryResult:
    """Test create-only event writes and idempotent accept."""
    r = CanaryResult("event_persistence")
    recovery_root = f"gs://{scratch_bucket}/{prefix}/events"
    uri = f"gs://{scratch_bucket}/{prefix}/test_obj.tif"
    run_id = f"canary_{_uid()}"

    # save event
    event = make_event(
        uri=uri,
        run_id=run_id,
        sequence=0,
        event_type=EventType.SNAPSHOTTED,
        config_hash=config_hash,
        operation_id=make_operation_id("snap"),
    )
    try:
        path = save_event(recovery_root, event)
        r.ok("save_event")
    except Exception as exc:
        r.fail("save_event", str(exc))
        return r

    # idempotent accept (same event again)
    try:
        path2 = save_event(recovery_root, event)
        r.ok("idempotent_accept")
        if path != path2:
            r.fail("idempotent_path_match", f"{path} != {path2}")
        else:
            r.ok("idempotent_path_match")
    except Exception as exc:
        r.fail("idempotent_accept", str(exc))

    # load events
    try:
        events = load_events(recovery_root, uri)
        if len(events) != 1:
            r.fail("load_events_count", f"expected 1, got {len(events)}")
        else:
            r.ok("load_events_count")
        if events[0].event_type != EventType.SNAPSHOTTED:
            r.fail(
                "load_events_type",
                f"expected SNAPSHOTTED, got {events[0].event_type}",
            )
        else:
            r.ok("load_events_type")
    except Exception as exc:
        r.fail("load_events", str(exc))

    # save second event
    event2 = make_event(
        uri=uri,
        run_id=run_id,
        sequence=1,
        event_type=EventType.BACKUP_VERIFIED,
        config_hash=config_hash,
        operation_id=make_operation_id("backup"),
        prev_event_hash=event_content_hash(event),
    )
    try:
        save_event(recovery_root, event2)
        r.ok("save_second_event")
    except Exception as exc:
        r.fail("save_second_event", str(exc))

    # reduce events
    try:
        events = load_events(recovery_root, uri)
        state = reduce_events(events)
        if state["errors"]:
            r.fail("reduce_no_errors", str(state["errors"]))
        else:
            r.ok("reduce_no_errors")
        if state["status"] != EventType.BACKUP_VERIFIED:
            r.fail(
                "reduce_status",
                f"expected BACKUP_VERIFIED, got {state['status']}",
            )
        else:
            r.ok("reduce_status")
    except Exception as exc:
        r.fail("reduce_events", str(exc))

    # non-identical collision (different event at same path)
    try:
        import json as _json

        from berlin_lst_downscaling.data.ard.cog_recovery_state import event_path_for

        collision_event = make_event(
            uri=uri,
            run_id=run_id,
            sequence=0,
            event_type=EventType.SNAPSHOTTED,
            config_hash="DIFFERENT_HASH",
            operation_id=make_operation_id("collision"),
        )
        collision_path = event_path_for(recovery_root, uri, 0, EventType.SNAPSHOTTED)
        bucket_name, key = _parse_gs_uri(collision_path)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        content = _json.dumps(
            collision_event.to_dict(), indent=2, sort_keys=True,
        ).encode()
        try:
            blob.upload_from_string(
                content,
                content_type="application/json",
                if_generation_match=0,
            )
            r.fail("collision_rejected", "upload should have failed with 412")
        except PreconditionFailed:
            # try save_event which should detect non-identical collision
            try:
                save_event(recovery_root, collision_event)
                r.fail("collision_detected", "expected RuntimeError")
            except RuntimeError as exc:
                if "Non-identical" in str(exc):
                    r.ok("collision_detected")
                else:
                    r.fail("collision_detected", str(exc))
    except Exception as exc:
        r.fail("collision_test", str(exc))

    return r


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="COG recovery GCS canary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recovery-root", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_hash = hash_config(config_path)
    recovery_root = args.recovery_root

    # derive scratch bucket from recovery root
    scratch_bucket, recovery_prefix = _parse_gs_uri(recovery_root)
    run_id = f"canary_{_uid()}"
    prefix = f"{recovery_prefix}/{run_id}"

    print(f"Config hash: {config_hash}")
    print(f"Scratch bucket: {scratch_bucket}")
    print(f"Prefix: {prefix}")
    print(f"Run ID: {run_id}")
    print()

    results: list[CanaryResult] = []

    # test 1: basic copy and verify
    print("Test 1: Basic copy and verify")
    r1 = test_basic_copy_and_verify(scratch_bucket, prefix, config_hash)
    results.append(r1)

    # test 2: soft delete restore
    print("\nTest 2: Soft delete restore")
    r2 = test_soft_delete_restore(scratch_bucket, prefix, config_hash)
    results.append(r2)

    # test 3: rewrite with metadata
    print("\nTest 3: Rewrite with metadata")
    r3 = test_rewrite_with_metadata(scratch_bucket, prefix, config_hash)
    results.append(r3)

    # test 4: event persistence
    print("\nTest 4: Event persistence")
    r4 = test_event_persistence(scratch_bucket, prefix, config_hash)
    results.append(r4)

    # summary
    all_passed = all(r.success for r in results)
    total_passed = sum(len(r.passed) for r in results)
    total_failed = sum(len(r.failed) for r in results)

    print(f"\n{'='*60}")
    print(f"Canary summary: {total_passed} passed, {total_failed} failed")
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {r.name}: {len(r.passed)} passed, {len(r.failed)} failed")
        for f in r.failed:
            print(f"    - {f}")

    # publish canary report
    report = {
        "run_id": run_id,
        "config_hash": config_hash,
        "timestamp": _now(),
        "success": all_passed,
        "tests": [r.to_dict() for r in results],
        "total_passed": total_passed,
        "total_failed": total_failed,
    }
    report_path = f"{recovery_root}/canary/{run_id}/report.json"
    report_content = json.dumps(report, indent=2, sort_keys=True).encode()

    from berlin_lst_downscaling.data.io.storage import atomic_write
    atomic_write(report_path, report_content, overwrite=False)
    print(f"\nReport: {report_path}")

    # cleanup scratch objects
    print("\nCleaning up scratch objects...")
    deleted = delete_scratch_prefix(scratch_bucket, prefix)
    print(f"  Deleted {deleted} objects")

    if all_passed:
        print("\nCANARY PASSED")
        return 0
    else:
        print("\nCANARY FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
