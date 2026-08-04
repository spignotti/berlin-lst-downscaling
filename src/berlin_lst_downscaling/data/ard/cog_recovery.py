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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
    snapshot_gcs_descriptor,
)
from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    EvidenceClass,
    ObjectDescriptor,
    classify_evidence,
    hash_config,
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

    # classify each asset
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
        # snapshot current descriptor
        try:
            descriptor = snapshot_gcs_descriptor(uri)
        except FileNotFoundError:
            descriptor = ObjectDescriptor()

        # strict validation
        strict_result = validate_strict_cog(uri)

        # evidence classification
        has_legacy = uri in legacy_audit
        was_overwritten = False  # determined by Soft Delete catalog later
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
            "original_generation": None,
            "original_metageneration": None,
            "original_crc32c": None,
            "original_size": None,
            "original_descriptor_json": None,
            "hard_delete_time": None,
            "restore_token": None,
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
            result = future.result()
            classification_results.append(result)

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

    # Soft Delete catalog
    print("\nCataloging Soft Delete generations...")
    main_bucket_name = config["canonical_roots"]["ard_full"].split("/")[2]
    canonical_prefixes = set()
    for root_key in config["canonical_roots"]:
        root_uri = config["canonical_roots"][root_key]
        _, prefix = _parse_gs_uri(root_uri)
        canonical_prefixes.add(prefix.rstrip("/") + "/")

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

    # filter to canonical objects only
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

    # earliest hard delete
    hard_delete_times = [
        datetime.fromisoformat(x["hard_delete_time"])
        for x in canonical_sd
        if x.get("hard_delete_time")
    ]
    earliest_hd = min(hard_delete_times) if hard_delete_times else None
    if earliest_hd:
        print(f"  Earliest hard delete: {earliest_hd.isoformat()}")
        remaining = (earliest_hd - datetime.now(UTC)).total_seconds() / 3600
        print(f"  Remaining: {remaining:.1f}h")

    # match soft-deleted generations to inventory
    sd_by_name: dict[str, list[dict[str, Any]]] = {}
    for x in canonical_sd:
        sd_by_name.setdefault(x["name"], []).append(x)

    for result in classification_results:
        bucket_name, key = _parse_gs_uri(result["uri"])
        sd_list = sd_by_name.get(key, [])
        if sd_list:
            sd = sd_list[0]  # should be exactly one per canonical name
            result["original_generation"] = sd["generation"]
            result["hard_delete_time"] = sd.get("hard_delete_time")
            result["restore_token"] = sd.get("restore_token")
            # update evidence class
            result["evidence_class"] = (
                EvidenceClass.UNAUDITED_OVERWRITE.value
                if result["evidence_class"] == EvidenceClass.UNTOUCHED.value
                and sd["generation"] != result["current_generation"]
                else result["evidence_class"]
            )

    # save inventory
    import pyarrow as pa

    from berlin_lst_downscaling.data.ard.cog_repair import save_table

    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    arrays = [
        pa.array([r.get(f) for r in classification_results])
        for f in [
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
    ]
    inv_table = pa.table(arrays)
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


# ── original capture (stub — requires canary) ─────────────────────────


def cmd_capture_originals(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Serial per-object original capture saga.

    Requires canary report and ``--execute`` to mutate canonical paths.
    """
    if dry_run:
        print("DRY RUN: Original capture requires --execute")
        return 0

    print("Original capture not yet implemented (requires canary)")
    return 1


# ── candidate staging (stub — requires canary) ────────────────────────


def cmd_stage_candidates(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Generate 164 GDAL and 672 Cogger candidates.

    Requires canary report and ``--execute``.
    """
    if dry_run:
        print("DRY RUN: Candidate staging requires --execute")
        return 0

    print("Candidate staging not yet implemented (requires canary)")
    return 1


# ── promotion (stub — requires verified candidates) ───────────────────


def cmd_promote(
    config_path: str | Path,
    *,
    recovery_root: str,
    run_id: str,
    dry_run: bool = True,
) -> int:
    """Guarded promotion with bounded batches and rollback.

    Requires verified candidates and ``--execute``.
    """
    if dry_run:
        print("DRY RUN: Promotion requires --execute")
        return 0

    print("Promotion not yet implemented (requires verified candidates)")
    return 1


# ── final verification ────────────────────────────────────────────────


def cmd_verify_recovery(
    config_path: str | Path,
    *,
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
    "cmd_stage_candidates",
    "cmd_promote",
    "cmd_verify_recovery",
]
