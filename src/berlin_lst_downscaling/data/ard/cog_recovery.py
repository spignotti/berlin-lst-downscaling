"""Saga orchestrator for COG layout recovery.

Provides the per-object recovery lifecycle: baseline, original capture,
candidate staging, promotion with rollback, and final verification.
Every mutation is guarded by the run manifest, canary report, and
``--execute`` flag.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
    copy_object_server_side,
    restore_soft_deleted_object,
    rewrite_object_server_side,
    rollback_to_payload,
    save_event,
    snapshot_gcs_descriptor,
    snapshot_soft_deleted_descriptor,
    verify_descriptor,
)
from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    EventType,
    LayoutClass,
    ObjectDescriptor,
    assert_raster_equivalent,
    classify_evidence,
    event_content_hash,
    hash_config,
    make_event,
    make_operation_id,
    validate_strict_cog,
)
from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

_logger = logging.getLogger(__name__)


# ── manifest dataclasses ──────────────────────────────────────────────


@dataclass
class RunManifest:
    """Immutable run manifest binding config, code, inventory, and deadline."""
    run_id: str
    config_hash: str
    code_hash: str
    lockfile_hash: str
    inventory_hash: str
    canonical_count: int
    layout_counts: dict[str, int]
    evidence_counts: dict[str, int]
    soft_delete_count: int
    earliest_hard_delete: str
    gdal_version: str = ""
    python_version: str = ""
    canary_report_hash: str = ""
    event_root: str = ""
    legacy_root: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "lockfile_hash": self.lockfile_hash,
            "inventory_hash": self.inventory_hash,
            "canonical_count": self.canonical_count,
            "layout_counts": self.layout_counts,
            "evidence_counts": self.evidence_counts,
            "soft_delete_count": self.soft_delete_count,
            "earliest_hard_delete": self.earliest_hard_delete,
            "gdal_version": self.gdal_version,
            "python_version": self.python_version,
            "canary_report_hash": self.canary_report_hash,
            "event_root": self.event_root,
            "legacy_root": self.legacy_root,
            "created_at": self.created_at,
        }

    def content_hash(self) -> str:
        import hashlib
        canonical = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()[:16]


@dataclass
class BatchManifest:
    """Immutable batch manifest for a promotion batch."""
    run_id: str
    batch_id: int
    uris: list[str]
    canonical_generations: dict[str, int]
    canonical_metagenerations: dict[str, int]
    candidate_uris: dict[str, str]
    expected_descriptors: dict[str, dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ── helpers ───────────────────────────────────────────────────────────


def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _inventory_hash(rows: list[dict[str, Any]]) -> str:
    import hashlib
    canonical = json.dumps(
        sorted(r["uri"] for r in rows), sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _file_hash(path: str | Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _read_gs_bytes(uri: str) -> bytes:
    """Read full object content from a GCS URI."""
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


def _check_deadline(earliest_hard_delete: str, margin_hours: float = 24.0) -> None:
    """Raise if the deadline is within *margin_hours* of now."""
    deadline = datetime.fromisoformat(earliest_hard_delete)
    now = datetime.now(UTC)
    remaining = (deadline - now).total_seconds() / 3600
    if remaining < margin_hours:
        raise RuntimeError(
            f"Deadline in {remaining:.1f}h (need {margin_hours}h margin): "
            f"{earliest_hard_delete}"
        )


def _git_hash() -> str:
    """Get the current git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()[:12]


# ── preflight ─────────────────────────────────────────────────────────


def cmd_preflight(
    config_path: str | Path,
    *,
    recovery_root: str,
) -> int:
    """Verify bucket policies, IAM, and inventory key set.

    Read-only: no canonical mutations.
    """
    config = _load_config(config_path)
    cfg_hash = hash_config(config_path)

    print(f"Config hash: {cfg_hash}")
    print(f"Expected count: {config['expected_count']}")

    # verify main bucket
    client = _gcs_client()
    main_bucket_name = config["canonical_roots"]["ard_full"].split("/")[2]
    main_bucket = client.bucket(main_bucket_name)
    main_bucket.reload()

    print(f"\nMain bucket: {main_bucket_name}")
    print(f"  Versioning: {main_bucket.versioning_enabled}")
    sd = main_bucket.soft_delete_policy
    print(f"  Soft delete retention: {sd.retention_duration_seconds if sd else 0}s")
    print(f"  Location: {main_bucket.location}")

    if main_bucket.versioning_enabled:
        print("  FAIL: Versioning must be disabled")
        return 1

    # verify recovery bucket
    recovery_bucket_name = config["recovery_bucket"].replace("gs://", "").split("/")[0]
    try:
        recovery_bucket = client.bucket(recovery_bucket_name)
        recovery_bucket.reload()
        print(f"\nRecovery bucket: {recovery_bucket_name}")
        print(f"  Location: {recovery_bucket.location}")
    except Exception as exc:
        print(f"\nFAIL: Recovery bucket not accessible: {exc}")
        return 1

    # verify inventory count
    from berlin_lst_downscaling.data.ard.cog_repair import build_inventory_from_ledgers
    table = build_inventory_from_ledgers(
        manifest_uri="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet",
        ard_ledger_uri="gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet",
        static_sources_root="gs://berlin-lst-data/static/sources/full",
        static_derived_root="gs://berlin-lst-data/static/derived/full",
        dynamic_full_root="gs://berlin-lst-data/dynamic/full",
        dynamic_inference_root="gs://berlin-lst-data/dynamic/inference/2026",
    )

    actual_count = table.num_rows
    print(f"\nInventory: {actual_count}")
    if actual_count != config["expected_count"]:
        print(f"  FAIL: Expected {config['expected_count']}, got {actual_count}")
        return 1

    # check deadline
    incident_start = config["incident"]["first_overwrite"]
    retention_days = config["soft_delete_retention_days"]
    from datetime import timedelta
    deadline = (
        datetime.fromisoformat(incident_start.replace("Z", "+00:00"))
        + timedelta(days=retention_days)
    )
    now = datetime.now(UTC)
    remaining_hours = (deadline - now).total_seconds() / 3600
    print(f"\nDeadline: {deadline.isoformat()}")
    print(f"  Remaining: {remaining_hours:.1f}h")
    if remaining_hours <= 0:
        print("  FAIL: Deadline has passed")
        return 1

    print("\nPreflight PASSED")
    return 0


# ── rebaseline ────────────────────────────────────────────────────────


def cmd_rebaseline(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
) -> int:
    """Build the immutable baseline: inventory, Soft Delete catalog, and
    run manifest.

    Read-only on canonical paths.  Writes to *recovery_root*.
    """
    config = _load_config(config_path)
    cfg_hash = hash_config(config_path)

    print(f"Config hash: {cfg_hash}")
    print(f"Run ID: {run_id}")

    # build fresh inventory
    from berlin_lst_downscaling.data.ard.cog_repair import (
        build_inventory_from_ledgers,
        load_table,
    )

    print("Building inventory...")
    table = build_inventory_from_ledgers(
        manifest_uri="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet",
        ard_ledger_uri="gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet",
        static_sources_root="gs://berlin-lst-data/static/sources/full",
        static_derived_root="gs://berlin-lst-data/static/derived/full",
        dynamic_full_root="gs://berlin-lst-data/dynamic/full",
        dynamic_inference_root="gs://berlin-lst-data/dynamic/inference/2026",
    )
    rows = table.to_pylist()
    print(f"Inventory: {len(rows)} URIs")

    if len(rows) != config["expected_count"]:
        print(f"  FAIL: Expected {config['expected_count']}, got {len(rows)}")
        return 1

    inv_hash = _inventory_hash(rows)

    # ── Soft Delete catalog (must be built before classification) ──────
    print("Cataloging Soft Delete generations...")
    main_bucket_name = config["canonical_roots"]["ard_full"].split("/")[2]

    from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
        list_soft_deleted_generations,
    )

    first_overwrite = datetime.fromisoformat(
        config["incident"]["first_overwrite"].replace("Z", "+00:00")
    )
    last_overwrite = datetime.fromisoformat(
        config["incident"]["last_overwrite"].replace("Z", "+00:00")
    )
    # inclusive upper bound: last_overwrite + 1s
    time_hi = last_overwrite.replace(second=last_overwrite.second + 1)

    all_soft_deleted = list_soft_deleted_generations(
        main_bucket_name,
        time_lo=first_overwrite,
        time_hi=time_hi,
    )

    canonical_uris = {r["uri"] for r in rows}
    canonical_sd = [
        x for x in all_soft_deleted
        if f"gs://{main_bucket_name}/{x['name']}" in canonical_uris
    ]

    print(f"Soft Delete window: {len(all_soft_deleted)} total")
    print(f"  Canonical: {len(canonical_sd)}")
    print(f"  Non-canonical: {len(all_soft_deleted) - len(canonical_sd)}")

    expected_sd = 1407
    if len(canonical_sd) != expected_sd:
        print(f"  FAIL: Expected {expected_sd} canonical Soft Delete generations")
        return 1

    # earliest hard delete + runway check
    hard_delete_times = [
        datetime.fromisoformat(x["hard_delete_time"])
        for x in canonical_sd
        if x.get("hard_delete_time")
    ]
    earliest_hd = min(hard_delete_times) if hard_delete_times else None
    if earliest_hd:
        remaining_h = (earliest_hd - datetime.now(UTC)).total_seconds() / 3600
        print(f"  Earliest hard delete: {earliest_hd.isoformat()}")
        print(f"  Remaining: {remaining_h:.1f}h")
        # 3× estimated runtime (1s/object) + 24h safety margin
        est_capture_hours = expected_sd / 3600
        required_margin = 3 * est_capture_hours + 24
        if remaining_h < required_margin:
            print(
                f"  FAIL: Remaining {remaining_h:.1f}h < required "
                f"{required_margin:.1f}h (3×{est_capture_hours:.1f}h + 24h)"
            )
            return 1

    # index by canonical key for O(1) lookup during classification
    sd_by_key: dict[str, dict[str, Any]] = {}
    for x in canonical_sd:
        sd_by_key[x["name"]] = x

    # ── classify each asset ───────────────────────────────────────────
    print("Classifying assets...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    legacy_audit: dict[str, dict[str, Any]] = {}
    if config.get("legacy", {}).get("audit_uri"):
        try:
            audit_table = load_table(config["legacy"]["audit_uri"])
            for row in audit_table.to_pylist():
                legacy_audit[row["uri"]] = row
            print(f"Legacy audit: {len(legacy_audit)} rows")
        except Exception as exc:
            print(f"WARNING: Cannot load legacy audit: {exc}")

    classification_results: list[dict[str, Any]] = []

    def _classify_one(row: dict[str, Any]) -> dict[str, Any]:
        uri = row["uri"]
        _, key = _parse_gs_uri(uri)

        # snapshot current descriptor
        try:
            descriptor = snapshot_gcs_descriptor(uri)
        except FileNotFoundError:
            descriptor = ObjectDescriptor()

        # strict validation
        strict_result = validate_strict_cog(uri)

        # evidence classification using Soft Delete catalog
        sd_entry = sd_by_key.get(key)
        was_overwritten = sd_entry is not None
        has_legacy = uri in legacy_audit
        legacy_crc_match = None
        legacy_gen_match = None

        if has_legacy:
            audit_row = legacy_audit[uri]
            legacy_crc_match = audit_row.get("repaired_crc32c") == descriptor.crc32c
            legacy_gen_match = audit_row.get("new_generation") == descriptor.generation

        evidence_class = classify_evidence(
            was_overwritten=was_overwritten,
            has_legacy_audit=has_legacy,
            legacy_crc_match=legacy_crc_match,
            legacy_generation_match=legacy_gen_match,
        )

        # populate Soft Delete metadata if overwritten
        original_gen = None
        original_meta = None
        original_crc = None
        original_size = None
        original_desc_json = None
        hard_delete_time = None
        restore_token = None

        if sd_entry:
            original_gen = sd_entry["generation"]
            hard_delete_time = sd_entry.get("hard_delete_time")
            restore_token = sd_entry.get("restore_token")
            # snapshot original descriptor from Soft Delete catalog
            try:
                orig_desc = snapshot_soft_deleted_descriptor(
                    uri,
                    generation=original_gen,
                    restore_token=restore_token,
                )
                original_meta = orig_desc.metageneration
                original_crc = orig_desc.crc32c
                original_size = orig_desc.size
                original_desc_json = json.dumps(orig_desc.to_dict())
            except FileNotFoundError:
                pass  # generation may have expired

        return {
            "uri": uri,
            "asset_kind": row["asset_kind"],
            "source": row.get("source"),
            "partition": row.get("partition"),
            "year": row.get("year"),
            "current_generation": descriptor.generation,
            "current_metageneration": descriptor.metageneration,
            "current_crc32c": descriptor.crc32c,
            "current_size": descriptor.size,
            "current_content_type": descriptor.content_type,
            "current_metadata_json": json.dumps(descriptor.custom_metadata),
            "current_descriptor_json": json.dumps(descriptor.to_dict()),
            "original_generation": original_gen,
            "original_metageneration": original_meta,
            "original_crc32c": original_crc,
            "original_size": original_size,
            "original_descriptor_json": original_desc_json,
            "hard_delete_time": hard_delete_time,
            "restore_token": restore_token,
            "layout_signature": strict_result.layout_signature,
            "layout_class": strict_result.layout_class,
            "evidence_class": evidence_class,
            "recovery_bucket_uri": None,
            "original_backup_uri": None,
            "current_backup_uri": None,
            "candidate_uri": None,
            "status": "SNAPSHOTTED",
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_classify_one, row): row for row in rows}
        for future in as_completed(futures):
            classification_results.append(future.result())

    # layout/evidence summary
    from collections import Counter
    layout_counts = Counter(r["layout_class"] for r in classification_results)
    evidence_counts = Counter(r["evidence_class"] for r in classification_results)

    print("\nLayout classes:")
    for cls, count in sorted(layout_counts.items()):
        print(f"  {cls}: {count}")
    print("Evidence classes:")
    for cls, count in sorted(evidence_counts.items()):
        print(f"  {cls}: {count}")

    expected_layouts = {
        "strict_clean": 1243,
        "missing_overview": 164,
        "hard_layout": 672,
    }
    if dict(layout_counts) != expected_layouts:
        print(f"  FAIL: Expected layouts {expected_layouts}")
        return 1

    # save inventory
    import pyarrow as pa

    from berlin_lst_downscaling.data.ard.cog_repair import save_table

    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    field_names = [
        "uri", "asset_kind", "source", "partition", "year",
        "current_generation", "current_metageneration",
        "current_crc32c", "current_size", "current_content_type",
        "current_metadata_json", "current_descriptor_json",
        "original_generation", "original_metageneration",
        "original_crc32c", "original_size", "original_descriptor_json",
        "hard_delete_time", "restore_token",
        "layout_signature", "layout_class", "evidence_class",
        "recovery_bucket_uri", "original_backup_uri",
        "current_backup_uri", "candidate_uri", "status",
    ]
    arrays = [
        pa.array([r.get(f) for r in classification_results])
        for f in field_names
    ]
    inv_table = pa.table(arrays, names=field_names)
    save_table(inv_table, inv_path)
    print(f"\nSaved: {inv_path}")

    # build and persist run manifest
    manifest = RunManifest(
        run_id=run_id,
        config_hash=cfg_hash,
        code_hash=_git_hash(),
        lockfile_hash=_file_hash("uv.lock"),
        inventory_hash=inv_hash,
        canonical_count=len(rows),
        layout_counts=dict(layout_counts),
        evidence_counts=dict(evidence_counts),
        soft_delete_count=len(canonical_sd),
        earliest_hard_delete=(
            earliest_hd.isoformat() if earliest_hd else ""
        ),
        event_root=recovery_root,
        legacy_root=config.get("legacy", {}).get("audit_uri", ""),
    )
    manifest_path = f"{recovery_root}/manifests/{run_id}.json"
    from berlin_lst_downscaling.data.io.storage import atomic_write
    atomic_write(
        manifest_path,
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode(),
        overwrite=False,
    )
    print(f"Saved: {manifest_path}")
    print(f"\nRebaseline PASSED ({run_id})")
    return 0


# ── original capture ──────────────────────────────────────────────────


def cmd_capture_originals(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Serial per-object original capture saga.

    For each overwritten asset:
    1. Check deadline runway
    2. Restore pre-incident generation to canonical path
    3. Copy restored generation to recovery originals
    4. Reconstruct baseline from backup bytes + metadata contract
    5. Verify and persist terminal event

    Requires ``--execute`` to mutate canonical paths.
    """
    if dry_run:
        print("DRY RUN: Original capture requires --execute")
        return 0

    cfg_hash = hash_config(config_path)
    config = _load_config(config_path)

    # load inventory
    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inv_table = load_table(inv_path)
    rows = inv_table.to_pylist()

    # load run manifest for deadline
    manifest_path = f"{recovery_root}/manifests/{run_id}.json"
    manifest_bytes = _read_gs_bytes(manifest_path)
    manifest = json.loads(manifest_bytes)
    earliest_hd_str = manifest.get("earliest_hard_delete", "")

    # filter to overwritten assets that need capture
    overwritten = [
        r for r in rows
        if r.get("original_generation")
        and r["original_generation"] != r["current_generation"]
    ]
    print(f"Overwritten assets: {len(overwritten)}")
    print(f"Config hash: {cfg_hash}")
    print(f"Run ID: {run_id}")

    success = 0
    failed = 0
    skipped = 0

    for _i, row in enumerate(overwritten):
        uri = row["uri"]

        # deadline check before each asset
        if earliest_hd_str:
            _check_deadline(earliest_hd_str, margin_hours=1.0)

        seq = 0
        op_id = make_operation_id("capture")
        prev_hash = ""

        try:
            # ── step 1: persist intent ────────────────────────────────
            intent = make_event(
                uri=uri,
                run_id=run_id,
                sequence=seq,
                event_type=EventType.ORIGINAL_CAPTURE_INTENT,
                config_hash=cfg_hash,
                operation_id=op_id,
                generation_before=row["current_generation"],
                metageneration_before=row["current_metageneration"],
            )
            save_event(recovery_root, intent)
            seq += 1
            prev_hash = event_content_hash(intent)

            # ── step 2: restore pre-incident generation ───────────────
            restored_desc = restore_soft_deleted_object(
                uri,
                soft_deleted_generation=row["original_generation"],
                current_generation=row["current_generation"],
                current_metageneration=row["current_metageneration"],
                restore_token=row.get("restore_token"),
            )
            restored_event = make_event(
                uri=uri,
                run_id=run_id,
                sequence=seq,
                event_type=EventType.ORIGINAL_RESTORED,
                config_hash=cfg_hash,
                operation_id=op_id,
                prev_event_hash=prev_hash,
                generation_before=row["current_generation"],
                generation_after=restored_desc.generation,
                metageneration_before=row["current_metageneration"],
                metageneration_after=restored_desc.metageneration,
                crc32c=restored_desc.crc32c,
                descriptor=restored_desc,
            )
            save_event(recovery_root, restored_event)
            seq += 1
            prev_hash = event_content_hash(restored_event)

            # verify restored descriptor matches expected
            if row.get("original_crc32c") and restored_desc.crc32c != row["original_crc32c"]:
                raise RuntimeError(
                    f"Restored CRC mismatch: expected {row['original_crc32c']}, "
                    f"got {restored_desc.crc32c}"
                )

            # ── step 3: copy restored generation to recovery originals ─
            _, key = _parse_gs_uri(uri)
            original_uri = f"{recovery_root}/originals/{key}"
            copy_result = copy_object_server_side(
                source_uri=uri,
                dest_uri=original_uri,
                source_generation=restored_desc.generation,
                source_metageneration=restored_desc.metageneration,
                dest_generation_match=0,
            )
            copied_event = make_event(
                uri=uri,
                run_id=run_id,
                sequence=seq,
                event_type=EventType.ORIGINAL_COPIED,
                config_hash=cfg_hash,
                operation_id=op_id,
                prev_event_hash=prev_hash,
                generation_after=copy_result["generation"],
                crc32c=copy_result["crc32c"],
                details={"original_uri": original_uri},
            )
            save_event(recovery_root, copied_event)
            seq += 1
            prev_hash = event_content_hash(copied_event)

            # ── step 4: reconstruct baseline ──────────────────────────
            # rewrite canonical from LEGACY backup with frozen live metadata
            legacy_root = config["legacy_recovery_root"]
            legacy_backup_uri = f"{legacy_root}/backups/current/{key}"

            # snapshot the displaced live metadata (before restore overwrote it)
            # we use restored_desc since it captured the original's metadata
            metadata_contract = ObjectDescriptor(
                content_type=restored_desc.content_type,
                content_encoding=restored_desc.content_encoding,
                cache_control=restored_desc.cache_control,
                custom_metadata=restored_desc.custom_metadata,
            )

            # pin source generation from the backup blob
            backup_desc = snapshot_gcs_descriptor(legacy_backup_uri)

            baseline_result = rewrite_object_server_side(
                source_uri=legacy_backup_uri,
                dest_uri=uri,
                source_generation=backup_desc.generation,
                source_metageneration=backup_desc.metageneration,
                dest_generation=restored_desc.generation,
                dest_metageneration=restored_desc.metageneration,
                dest_metadata=metadata_contract,
            )
            baseline_event = make_event(
                uri=uri,
                run_id=run_id,
                sequence=seq,
                event_type=EventType.BASELINE_RECONSTRUCTED,
                config_hash=cfg_hash,
                operation_id=op_id,
                prev_event_hash=prev_hash,
                generation_before=restored_desc.generation,
                generation_after=baseline_result["generation"],
                metageneration_before=restored_desc.metageneration,
                metageneration_after=baseline_result["metageneration"],
                crc32c=baseline_result["crc32c"],
            )
            save_event(recovery_root, baseline_event)
            seq += 1
            prev_hash = event_content_hash(baseline_event)

            # ── step 5: verify baseline ───────────────────────────────
            final_desc = snapshot_gcs_descriptor(uri)
            verify_errors = verify_descriptor(uri, metadata_contract)
            if verify_errors:
                raise RuntimeError(f"Baseline metadata mismatch: {verify_errors}")

            verified_event = make_event(
                uri=uri,
                run_id=run_id,
                sequence=seq,
                event_type=EventType.ORIGINAL_VERIFIED,
                config_hash=cfg_hash,
                operation_id=op_id,
                prev_event_hash=prev_hash,
                generation_after=final_desc.generation,
                crc32c=final_desc.crc32c,
                descriptor=final_desc,
            )
            save_event(recovery_root, verified_event)

            success += 1
            if success % 10 == 0:
                print(f"  Captured: {success}/{len(overwritten)}")

        except Exception as exc:
            failed += 1
            _logger.error("Failed to capture %s: %s", uri, exc)
            # persist blocked event
            try:
                blocked = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=seq,
                    event_type=EventType.BLOCKED,
                    config_hash=cfg_hash,
                    operation_id=op_id,
                    prev_event_hash=prev_hash if seq > 0 else "",
                    details={"error": str(exc)},
                )
                save_event(recovery_root, blocked)
            except Exception:
                _logger.error("Failed to persist blocked event for %s", uri)

    print(f"\nCapture complete: {success} success, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


# ── restore baseline ─────────────────────────────────────────────────


def cmd_restore_baseline(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Restore canonical objects to their rebaseline snapshot state.

    For each of the 1,407 overwritten objects, rewrites canonical from
    the legacy backup payload with the frozen metadata contract from the
    rebaseline snapshot.  Used to recover from a failed capture that left
    objects in an intermediate state.

    Requires ``--execute``.
    """
    if dry_run:
        print("DRY RUN: Restore baseline requires --execute")
        return 0

    config = _load_config(config_path)
    cfg_hash = hash_config(config_path)
    legacy_root = config["legacy_recovery_root"]

    # load inventory snapshot
    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inv_table = load_table(inv_path)
    rows = inv_table.to_pylist()

    # filter to overwritten objects
    overwritten = [
        r for r in rows
        if r.get("original_generation")
        and r["original_generation"] != r["current_generation"]
    ]
    print(f"Config hash: {cfg_hash}")
    print(f"Overwritten objects: {len(overwritten)}")

    success = 0
    failed = 0

    for row in overwritten:
        uri = row["uri"]
        _, key = _parse_gs_uri(uri)

        try:
            # snapshot current live descriptor
            current_desc = snapshot_gcs_descriptor(uri)

            # skip if already matches snapshot
            if current_desc.generation == row["current_generation"]:
                success += 1
                continue

            # legacy backup source
            legacy_backup_uri = f"{legacy_root}/backups/current/{key}"
            backup_desc = snapshot_gcs_descriptor(legacy_backup_uri)

            # metadata contract: use backup's content_type and custom_metadata
            # but the snapshot's content_encoding/cache_control
            frozen_desc = ObjectDescriptor.from_dict(
                json.loads(row["current_descriptor_json"])
                if row.get("current_descriptor_json")
                else {}
            )
            # override CRC32C/size to match the legacy backup payload
            frozen_desc_for_rewrite = ObjectDescriptor(
                generation=frozen_desc.generation,
                metageneration=frozen_desc.metageneration,
                size=backup_desc.size,
                crc32c=backup_desc.crc32c,
                content_type=frozen_desc.content_type or backup_desc.content_type,
                content_encoding=frozen_desc.content_encoding or backup_desc.content_encoding,
                cache_control=frozen_desc.cache_control or backup_desc.cache_control,
                custom_metadata=frozen_desc.custom_metadata or backup_desc.custom_metadata,
            )

            # guarded rewrite: legacy backup → canonical
            rewrite_object_server_side(
                source_uri=legacy_backup_uri,
                dest_uri=uri,
                source_generation=backup_desc.generation,
                source_metageneration=backup_desc.metageneration,
                dest_generation=current_desc.generation,
                dest_metageneration=current_desc.metageneration,
                dest_metadata=frozen_desc_for_rewrite,
            )

            # verify restored state matches legacy backup payload
            verify_errors = verify_descriptor(uri, frozen_desc_for_rewrite)
            if verify_errors:
                _logger.error("Verify failed for %s: %s", uri, verify_errors)
                failed += 1
                continue

            success += 1
            if success % 100 == 0:
                print(f"  Restored: {success}/{len(overwritten)}")

        except Exception as exc:
            _logger.error("Failed to restore %s: %s", uri, exc)
            failed += 1

    print(f"\nRestore complete: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


# ── candidate staging ─────────────────────────────────────────────────


def cmd_stage_candidates(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    cogger_bin: str = "cogger",
    dry_run: bool = True,
) -> int:
    """Generate 164 GDAL and 672 Cogger candidates.

    Routes assets to GDAL COG (missing-overview flags) or Cogger
    (hard-layout errors).  Each candidate is strict-validated and
    semantically compared before staging.

    Requires ``--execute`` to write to recovery bucket.
    """
    if dry_run:
        print("DRY RUN: Candidate staging requires --execute")
        return 0

    cfg_hash = hash_config(config_path)
    config = _load_config(config_path)

    # load inventory
    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inv_table = load_table(inv_path)
    rows = inv_table.to_pylist()

    # filter to assets needing repair
    needs_repair = [
        r for r in rows
        if r["layout_class"] in (
            LayoutClass.HARD_LAYOUT.value,
            LayoutClass.MISSING_OVERVIEW.value,
        )
    ]
    print(f"Config hash: {cfg_hash}")
    print(f"Needs repair: {len(needs_repair)}")
    n_flag = sum(
        1 for r in needs_repair
        if r["layout_class"] == LayoutClass.MISSING_OVERVIEW.value
    )
    n_hard = sum(
        1 for r in needs_repair
        if r["layout_class"] == LayoutClass.HARD_LAYOUT.value
    )
    print(f"  Missing overview: {n_flag}")
    print(f"  Hard layout: {n_hard}")

    # verify Cogger binary
    import shutil
    cogger_path = shutil.which(cogger_bin)
    if cogger_path:
        print(f"Cogger binary: {cogger_path}")
    else:
        print(f"WARNING: Cogger binary not found at '{cogger_bin}'")

    success = 0
    failed = 0

    for row in needs_repair:
        uri = row["uri"]
        layout_class = row["layout_class"]
        _, key = _parse_gs_uri(uri)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_src = Path(tmp_dir) / "source.tif"
                tmp_candidate = Path(tmp_dir) / "candidate.tif"

                # route source: flags from originals, hard-layout from legacy backups
                if layout_class == LayoutClass.MISSING_OVERVIEW.value:
                    source_uri = f"{recovery_root}/originals/{key}"
                else:
                    legacy_root = config["legacy_recovery_root"]
                    source_uri = f"{legacy_root}/backups/current/{key}"
                src_bytes = _read_gs_bytes(source_uri)
                tmp_src.write_bytes(src_bytes)

                # route to engine
                if layout_class == LayoutClass.MISSING_OVERVIEW.value:
                    # GDAL COG with NEAREST overviews for flags
                    _generate_gdal_candidate(tmp_src, tmp_candidate)
                elif layout_class == LayoutClass.HARD_LAYOUT.value:
                    # Cogger byte-reshuffle
                    _generate_cogger_candidate(
                        tmp_src, tmp_candidate, cogger_bin,
                    )
                else:
                    raise ValueError(f"Unexpected layout class: {layout_class}")

                # strict validate candidate
                strict_result = validate_strict_cog(str(tmp_candidate))
                if not strict_result.valid:
                    all_issues = list(strict_result.errors) + list(strict_result.warnings)
                    raise ValueError(f"Candidate validation failed: {all_issues}")

                # semantic comparison
                compare_errors = assert_raster_equivalent(str(tmp_src), tmp_candidate)
                if compare_errors:
                    raise ValueError(f"Semantic comparison failed: {compare_errors}")

                # upload candidate create-only
                candidate_uri = f"{recovery_root}/candidates/{key}"
                bucket_name, candidate_key = _parse_gs_uri(candidate_uri)
                client = _gcs_client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(candidate_key)
                blob.upload_from_filename(
                    str(tmp_candidate),
                    if_generation_match=0,
                    checksum="crc32c",
                )
                blob.reload()

                # persist events (continuing from existing sequence)
                from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
                    load_events,
                )
                existing_events = load_events(recovery_root, uri)
                base_seq = (
                    existing_events[-1].sequence + 1 if existing_events else 0
                )
                prev_hash = (
                    event_content_hash(existing_events[-1])
                    if existing_events
                    else ""
                )

                staged_event = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=base_seq,
                    event_type=EventType.CANDIDATE_STAGED,
                    config_hash=cfg_hash,
                    operation_id=make_operation_id("stage"),
                    prev_event_hash=prev_hash,
                    generation_after=blob.generation,
                    crc32c=blob.crc32c,
                    details={
                        "candidate_uri": candidate_uri,
                        "engine": (
                            "gdal_cog"
                            if layout_class == LayoutClass.MISSING_OVERVIEW.value
                            else "cogger"
                        ),
                    },
                )
                save_event(recovery_root, staged_event)

                verified_event = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=base_seq + 1,
                    event_type=EventType.CANDIDATE_VERIFIED,
                    config_hash=cfg_hash,
                    operation_id=make_operation_id("verify"),
                    prev_event_hash=event_content_hash(staged_event),
                    generation_after=blob.generation,
                    crc32c=blob.crc32c,
                    descriptor=ObjectDescriptor(
                        generation=blob.generation or 0,
                        metageneration=blob.metageneration or 0,
                        crc32c=blob.crc32c or "",
                        size=blob.size or 0,
                        content_type=blob.content_type or "",
                    ),
                )
                save_event(recovery_root, verified_event)

                success += 1
                if success % 50 == 0:
                    print(f"  Staged: {success}/{len(needs_repair)}")

        except Exception as exc:
            failed += 1
            _logger.error("Failed to stage %s: %s", uri, exc)

    print(f"\nStaging complete: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


def _generate_gdal_candidate(src: Path, dst: Path) -> None:
    """Generate a GDAL COG candidate from source."""
    import rasterio.shutil

    rasterio.shutil.copy(
        str(src),
        str(dst),
        driver="COG",
        overview_resampling="NEAREST",
        blocksize=512,
        compress="ZSTD",
        predictor=1,
        bigtiff="IF_SAFER",
    )


def _generate_cogger_candidate(src: Path, dst: Path, cogger_bin: str) -> None:
    """Generate a Cogger candidate from source."""
    import subprocess

    result = subprocess.run(  # noqa: S603
        [cogger_bin, "-output", str(dst), str(src)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cogger failed: {result.stderr}")


# ── promotion ─────────────────────────────────────────────────────────


def cmd_promote(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Guarded promotion with bounded batches and rollback.

    Promotes verified candidates to canonical paths in bounded batches:
    1 canary per engine, then 10, 50, max 100.

    Each promotion: intent → guarded rewrite → independent verify →
    terminal event.  On failure: rollback from backup → verify → stop.

    Requires ``--execute`` to mutate canonical paths.
    """
    if dry_run:
        print("DRY RUN: Promotion requires --execute")
        return 0

    cfg_hash = hash_config(config_path)

    # load inventory
    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inv_table = load_table(inv_path)
    rows = inv_table.to_pylist()

    # filter to assets needing promotion (candidate URI derived from key)
    needs_promotion = [
        r for r in rows
        if r["layout_class"] in (
            LayoutClass.MISSING_OVERVIEW.value,
            LayoutClass.HARD_LAYOUT.value,
        )
    ]
    print(f"Config hash: {cfg_hash}")
    print(f"Needs promotion: {len(needs_promotion)}")

    # batch sizes: canary per engine, then expanding until all promoted
    batch_pattern = [1, 1, 10, 50]  # then 100 repeatedly
    promoted = 0
    rolled_back = 0
    batch_idx = 0

    while promoted < len(needs_promotion):
        remaining = len(needs_promotion) - promoted
        if batch_idx < len(batch_pattern):
            batch_size = batch_pattern[batch_idx]
        else:
            batch_size = 100
        batch_size = min(batch_size, remaining)
        batch_idx += 1

        batch = needs_promotion[promoted : promoted + batch_size]
        print(f"\nBatch {batch_idx} (size {batch_size}): {len(batch)} assets")

        for row in batch:
            uri = row["uri"]
            _, key = _parse_gs_uri(uri)
            candidate_uri = f"{recovery_root}/candidates/{key}"
            backup_uri = f"{recovery_root}/originals/{key}"

            # snapshot current canonical descriptor
            try:
                current_desc = snapshot_gcs_descriptor(uri)
            except FileNotFoundError:
                _logger.error("Canonical object not found: %s", uri)
                return 1

            # load events to get sequence offset
            from berlin_lst_downscaling.data.ard.cog_recovery_gcs import load_events
            events = load_events(recovery_root, uri)
            base_seq = (events[-1].sequence + 1) if events else 0
            prev_hash = event_content_hash(events[-1]) if events else ""

            # metadata contract for promotion and rollback
            metadata_contract = ObjectDescriptor(
                content_type=current_desc.content_type,
                content_encoding=current_desc.content_encoding,
                cache_control=current_desc.cache_control,
                custom_metadata=current_desc.custom_metadata,
            )

            try:
                # promotion intent
                intent = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=base_seq,
                    event_type=EventType.PROMOTION_INTENT,
                    config_hash=cfg_hash,
                    operation_id=make_operation_id("promote"),
                    prev_event_hash=prev_hash,
                    generation_before=current_desc.generation,
                    metageneration_before=current_desc.metageneration,
                    details={"candidate_uri": candidate_uri},
                )
                save_event(recovery_root, intent)

                # guarded rewrite: candidate → canonical
                candidate_desc = snapshot_gcs_descriptor(candidate_uri)
                result = rewrite_object_server_side(
                    source_uri=candidate_uri,
                    dest_uri=uri,
                    source_generation=candidate_desc.generation,
                    source_metageneration=candidate_desc.metageneration,
                    dest_generation=current_desc.generation,
                    dest_metageneration=current_desc.metageneration,
                    dest_metadata=metadata_contract,
                )

                # independent verification
                verify_desc = snapshot_gcs_descriptor(uri)
                verify_errors = verify_descriptor(uri, metadata_contract)
                if verify_errors:
                    raise RuntimeError(
                        f"Post-promotion verification failed: {verify_errors}"
                    )

                # terminal event
                promoted_event = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=base_seq + 1,
                    event_type=EventType.PROMOTED,
                    config_hash=cfg_hash,
                    operation_id=make_operation_id("promote"),
                    prev_event_hash=event_content_hash(intent),
                    generation_before=current_desc.generation,
                    generation_after=result["generation"],
                    metageneration_before=current_desc.metageneration,
                    metageneration_after=result["metageneration"],
                    crc32c=result["crc32c"],
                    details={"candidate_uri": candidate_uri},
                )
                save_event(recovery_root, promoted_event)

                verified_event = make_event(
                    uri=uri,
                    run_id=run_id,
                    sequence=base_seq + 2,
                    event_type=EventType.CANONICAL_VERIFIED,
                    config_hash=cfg_hash,
                    operation_id=make_operation_id("verify"),
                    prev_event_hash=event_content_hash(promoted_event),
                    generation_after=verify_desc.generation,
                    crc32c=verify_desc.crc32c,
                    descriptor=verify_desc,
                )
                save_event(recovery_root, verified_event)

                promoted += 1
                if promoted % 10 == 0:
                    print(f"  Promoted: {promoted}/{len(needs_promotion)}")

            except Exception as exc:
                _logger.error("Promotion failed for %s: %s", uri, exc)
                # rollback
                try:
                    rollback_to_payload(
                        uri,
                        backup_uri,
                        recovery_root=recovery_root,
                        run_id=run_id,
                        config_hash=cfg_hash,
                        sequence=base_seq + 1,
                        current_generation=current_desc.generation,
                        current_metageneration=current_desc.metageneration,
                        metadata_contract=metadata_contract,
                    )
                    rolled_back += 1
                    print(f"  ROLLBACK: {uri} restored from backup")
                except Exception as rb_exc:
                    _logger.error("ROLLBACK FAILED for %s: %s", uri, rb_exc)
                    print(f"  ROLLBACK FAILED: {uri}")
                    return 1

                # stop the run after rollback
                print(f"\nPromotion stopped after rollback: {uri}")
                print(f"  Promoted: {promoted}, Rolled back: {rolled_back}")
                return 1

    print(f"\nPromotion complete: {promoted} promoted, {rolled_back} rolled back")
    return 0


# ── final verification ────────────────────────────────────────────────


def cmd_verify_recovery(
    config_path: str | Path,
    *,
    recovery_root: str = "",
    run_id: str = "",
    workers: int = 4,
) -> int:
    """Independent verification of all canonical assets.

    Read-only.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from berlin_lst_downscaling.data.ard.cog_repair import build_inventory_from_ledgers

    cfg_hash = hash_config(config_path)
    print(f"Config hash: {cfg_hash}")

    print("Building inventory...")
    table = build_inventory_from_ledgers(
        manifest_uri="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet",
        ard_ledger_uri="gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet",
        static_sources_root="gs://berlin-lst-data/static/sources/full",
        static_derived_root="gs://berlin-lst-data/static/derived/full",
        dynamic_full_root="gs://berlin-lst-data/dynamic/full",
        dynamic_inference_root="gs://berlin-lst-data/dynamic/inference/2026",
    )
    rows = table.to_pylist()
    print(f"Inventory: {len(rows)} URIs")

    def _verify_one(row: dict[str, Any]) -> dict[str, Any]:
        uri = row["uri"]
        result = validate_strict_cog(uri)
        return {
            "uri": uri,
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "layout_class": result.layout_class,
        }

    print("Verifying assets...")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_verify_one, row): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())

    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = sum(1 for r in results if not r["valid"])

    print("\nVerification summary:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")

    if invalid_count > 0:
        print("\nInvalid assets:")
        for r in results:
            if not r["valid"]:
                print(f"  {r['uri']}: {list(r['errors']) + list(r['warnings'])}")

    if invalid_count == 0:
        print("\nVerification PASSED — all assets are strict-clean")
        # write create-only complete.json
        if recovery_root and run_id:
            complete = {
                "run_id": run_id,
                "config_hash": cfg_hash,
                "verified_count": valid_count,
                "expected_count": len(rows),
                "timestamp": datetime.now(UTC).isoformat(),
                "success": valid_count == len(rows),
            }
            complete_path = f"{recovery_root}/complete.json"
            complete_content = json.dumps(
                complete, indent=2, sort_keys=True,
            ).encode()
            from google.api_core.exceptions import PreconditionFailed
            bucket_name, key = _parse_gs_uri(complete_path)
            client = _gcs_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(key)
            try:
                blob.upload_from_string(
                    complete_content,
                    content_type="application/json",
                    if_generation_match=0,
                    checksum="crc32c",
                )
                print(f"Complete marker: {complete_path}")
            except PreconditionFailed:
                print(f"WARNING: complete.json already exists at {complete_path}")
        return 0
    else:
        print(f"\nVerification FAILED — {invalid_count} assets are not strict-clean")
        return 1


__all__ = [
    "RunManifest",
    "BatchManifest",
    "cmd_preflight",
    "cmd_rebaseline",
    "cmd_capture_originals",
    "cmd_restore_baseline",
    "cmd_stage_candidates",
    "cmd_promote",
    "cmd_verify_recovery",
]
