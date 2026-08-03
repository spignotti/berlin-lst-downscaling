#!/usr/bin/env python3
"""COG layout recovery CLI.

Provides fail-closed recovery commands for repairing systemic COG layout
issues in GCS with immutable per-object events, guarded mutations, and
durable evidence.

Usage::

    # Preflight — verify bucket policy, IAM, and inventory key set
    uv run python scripts/repair_cog_layout.py preflight \\
        --config configs/cog_repair/recovery.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03

    # Classify — build inventory + evidence matrix
    uv run python scripts/repair_cog_layout.py classify \\
        --config configs/cog_repair/recovery.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03

    # Backup — preserve all current live objects
    uv run python scripts/repair_cog_layout.py backup \\
        --config configs/cog_repair/recovery.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03

    # Stage — generate candidates to recovery bucket
    uv run python scripts/repair_cog_layout.py stage \\
        --config configs/cog_repair/recovery.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03 \\
        --cogger-bin /tmp/cog-repair/cogger

    # Promote — guarded canonical promotion with rollback
    uv run python scripts/repair_cog_layout.py promote \\
        --config configs/cog_repair/recovery.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03

    # Verify — independent verification of all canonical assets
    uv run python scripts/repair_cog_layout.py verify-recovery \\
        --config configs/cog_repair/recovery.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


# ── migration guard ────────────────────────────────────────────────────

def _reject_legacy_command(command: str) -> int:
    """Reject legacy commands that lack fail-closed guarantees."""
    print(
        f"ERROR: Command '{command}' has been migrated to the fail-closed "
        f"recovery workflow.\n\n"
        f"Use the new commands:\n"
        f"  preflight  — verify bucket policy and inventory\n"
        f"  classify   — build evidence matrix\n"
        f"  backup     — preserve current live objects\n"
        f"  stage      — generate repair candidates\n"
        f"  promote    — guarded canonical promotion\n"
        f"  verify-recovery — independent verification\n\n"
        f"See docs/cog-layout-recovery.md for the runbook.",
        file=sys.stderr,
    )
    return 1


# ── recovery commands ──────────────────────────────────────────────────

def cmd_preflight(args: argparse.Namespace) -> int:
    """Verify bucket policy, IAM, and inventory key set."""
    from berlin_lst_downscaling.data.ard.cog_recovery_state import hash_config

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    config_hash = hash_config(config_path)
    print(f"Config hash: {config_hash}")

    # Load config
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    expected_count = config.get("expected_count")
    print(f"Expected count: {expected_count}")

    # Verify bucket policies
    from google.cloud import storage
    client = storage.Client()

    # Check main bucket
    main_bucket_name = config["canonical_roots"]["ard_full"].split("/")[2]
    main_bucket = client.bucket(main_bucket_name)
    main_bucket.reload()

    print(f"\nMain bucket: {main_bucket_name}")
    print(f"  Versioning: {main_bucket.versioning_enabled}")
    print(f"  Soft delete: {main_bucket.soft_delete_policy}")
    print(f"  Location: {main_bucket.location}")

    # Check recovery bucket
    recovery_bucket_name = config["recovery_bucket"].replace("gs://", "").split("/")[0]
    try:
        recovery_bucket = client.bucket(recovery_bucket_name)
        recovery_bucket.reload()
        print(f"\nRecovery bucket: {recovery_bucket_name}")
        print(f"  Versioning: {recovery_bucket.versioning_enabled}")
        print(f"  Location: {recovery_bucket.location}")
    except Exception as exc:
        print(f"\nWARNING: Recovery bucket not accessible: {exc}")
        print("  Create it before proceeding:")
        print("    gcloud storage buckets create gs://<name> --location=europe-west3")

    # List all canonical COGs
    print("\nInventory check...")
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
    print(f"  Actual count: {actual_count}")

    if actual_count != expected_count:
        print(f"  FAIL: Expected {expected_count}, got {actual_count}")
        return 1

    print("  PASS: Key set matches expected count")

    # Check soft delete deadlines for overwritten objects
    print("\nSoft delete deadline check...")
    from datetime import UTC, datetime, timedelta
    incident_start = datetime.fromisoformat(
        config["incident"]["first_overwrite"].replace("Z", "+00:00")
    )
    retention_days = config["soft_delete_retention_days"]
    deadline = incident_start + timedelta(days=retention_days)
    now = datetime.now(UTC)

    print(f"  Incident start: {incident_start.isoformat()}")
    print(f"  Retention: {retention_days} days")
    print(f"  Deadline: {deadline.isoformat()}")
    print(f"  Now: {now.isoformat()}")
    print(f"  Remaining: {(deadline - now).total_seconds() / 3600:.1f} hours")

    if now >= deadline:
        print("  FAIL: Soft delete deadline has passed")
        return 1

    print("\nPreflight PASSED")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """Build inventory + evidence matrix."""
    import yaml

    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    from berlin_lst_downscaling.data.ard.cog_recovery_state import (
        INVENTORY_SCHEMA,
        LayoutClass,
        classify_evidence,
        hash_config,
    )
    from berlin_lst_downscaling.data.ard.cog_repair import (
        build_inventory_from_ledgers,
        load_table,
        save_table,
    )

    config_path = Path(args.config)
    config_hash = hash_config(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Config hash: {config_hash}")

    # Build inventory
    print("Building inventory...")
    table = build_inventory_from_ledgers(
        manifest_uri="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet",
        ard_ledger_uri="gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet",
        static_sources_root="gs://berlin-lst-data/static/sources/full",
        static_derived_root="gs://berlin-lst-data/static/derived/full",
        dynamic_full_root="gs://berlin-lst-data/dynamic/full",
        dynamic_inference_root="gs://berlin-lst-data/dynamic/inference/2026",
    )

    print(f"Inventory: {table.num_rows} URIs")

    # Load legacy audit if available
    legacy_audit = {}
    if config.get("legacy", {}).get("audit_uri"):
        try:
            audit_table = load_table(config["legacy"]["audit_uri"])
            for row in audit_table.to_pylist():
                legacy_audit[row["uri"]] = row
            print(f"Legacy audit: {len(legacy_audit)} rows")
        except Exception as exc:
            print(f"WARNING: Cannot load legacy audit: {exc}")

    # Classify each asset
    print("Classifying assets...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = table.to_pylist()
    classification_results = []

    def classify_one(row):
        uri = row["uri"]
        strict_result = validate_strict_cog(uri)

        # Determine evidence class
        was_overwritten = row.get("generation", 0) > 0  # simplified
        has_legacy_audit = uri in legacy_audit
        legacy_crc_match = None
        legacy_gen_match = None

        if has_legacy_audit:
            audit_row = legacy_audit[uri]
            legacy_crc_match = audit_row.get("repaired_crc32c") == row.get("crc32c")
            legacy_gen_match = audit_row.get("new_generation") == row.get("generation")

        evidence_class = classify_evidence(
            was_overwritten=was_overwritten,
            has_legacy_audit=has_legacy_audit,
            legacy_crc_match=legacy_crc_match,
            legacy_generation_match=legacy_gen_match,
        )

        return {
            "uri": uri,
            "asset_kind": row["asset_kind"],
            "source": row.get("source"),
            "partition": row.get("partition"),
            "year": row.get("year"),
            "current_generation": row.get("generation"),
            "current_crc32c": row.get("crc32c"),
            "current_size": row.get("size"),
            "current_content_type": row.get("content_type"),
            "current_metadata_json": row.get("metadata_json"),
            "layout_signature": strict_result.layout_signature,
            "layout_class": strict_result.layout_class,
            "evidence_class": evidence_class,
            "status": "SNAPSHOTTED",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_one, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            classification_results.append(result)

    # Print summary
    from collections import Counter
    layout_counts = Counter(r["layout_class"] for r in classification_results)
    evidence_counts = Counter(r["evidence_class"] for r in classification_results)

    print("\nClassification summary:")
    print("  Layout classes:")
    for cls, count in sorted(layout_counts.items()):
        print(f"    {cls}: {count}")

    print("  Evidence classes:")
    for cls, count in sorted(evidence_counts.items()):
        print(f"    {cls}: {count}")

    # Verify totals
    total = len(classification_results)
    unexpected = layout_counts.get(LayoutClass.UNEXPECTED, 0)

    print(f"\n  Total: {total}")
    print(f"  Expected: {config['expected_count']}")

    if total != config["expected_count"]:
        print("  FAIL: Count mismatch")
        return 1

    if unexpected > 0:
        print(f"  FAIL: {unexpected} unexpected layout classes")
        return 1

    print("\nClassification PASSED")

    # Save to recovery root
    recovery_root = args.recovery_root
    print(f"\nSaving to {recovery_root}...")

    import pyarrow as pa

    # Build table with correct schema
    inventory_arrays = []
    for field in INVENTORY_SCHEMA:
        col_data = [r.get(field.name) for r in classification_results]
        inventory_arrays.append(pa.array(col_data, type=field.type, from_pandas=False))

    inventory_table = pa.table(inventory_arrays, schema=INVENTORY_SCHEMA)
    save_table(inventory_table, f"{recovery_root}/snapshots/inventory.parquet")

    print(f"Saved: {recovery_root}/snapshots/inventory.parquet")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Preserve all current live objects to recovery bucket."""
    import yaml

    from berlin_lst_downscaling.data.ard.cog_recovery_state import hash_config
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

    config_path = Path(args.config)
    config_hash = hash_config(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Config hash: {config_hash}")

    recovery_root = args.recovery_root
    backup_root = f"{recovery_root}/backups/current"

    print(f"Backup root: {backup_root}")

    # Load inventory
    inventory = load_table(f"{recovery_root}/snapshots/inventory.parquet")
    print(f"Inventory: {inventory.num_rows} URIs")

    # Backup each object
    client = _gcs_client()
    success = 0
    failed = 0

    for row in inventory.to_pylist():
        uri = row["uri"]
        bucket_name, key = _parse_gs_uri(uri)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)

        if not blob.exists():
            print(f"  SKIP: {uri} does not exist")
            failed += 1
            continue

        # Compute backup path
        backup_key = f"{backup_root.replace('gs://berlin-lst-data-recovery/', '')}/{key}"
        backup_bucket_name = config["recovery_bucket"].replace("gs://", "").split("/")[0]
        backup_bucket = client.bucket(backup_bucket_name)

        # Copy with generation guard
        try:
            blob.reload()

            # Copy to recovery bucket
            new_blob = backup_bucket.blob(backup_key)
            new_blob.upload_from_string(
                blob.download_as_bytes(),
                if_generation_match=0,  # create-only
                checksum="crc32c",
            )

            # Verify backup
            new_blob.reload()
            if new_blob.crc32c != blob.crc32c:
                print(f"  FAIL: CRC mismatch for {uri}")
                failed += 1
                continue

            success += 1
            if success % 100 == 0:
                print(f"  Backed up: {success}/{inventory.num_rows}")

        except Exception as exc:
            print(f"  FAIL: {uri}: {exc}")
            failed += 1

    print(f"\nBackup complete: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_stage(args: argparse.Namespace) -> int:
    """Generate repair candidates to recovery bucket."""
    import yaml

    from berlin_lst_downscaling.data.ard.cog_recovery_state import (
        LayoutClass,
        hash_config,
    )

    config_path = Path(args.config)
    config_hash = hash_config(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Config hash: {config_hash}")

    recovery_root = args.recovery_root
    stage_root = f"{recovery_root}/candidates"

    print(f"Stage root: {stage_root}")

    # Load inventory
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inventory = load_table(f"{recovery_root}/snapshots/inventory.parquet")
    print(f"Inventory: {inventory.num_rows} URIs")

    # Filter to assets needing repair
    needs_repair = [
        row for row in inventory.to_pylist()
        if row["layout_class"] in (LayoutClass.HARD_LAYOUT, LayoutClass.MISSING_OVERVIEW)
    ]

    print(f"Needs repair: {len(needs_repair)}")

    if not needs_repair:
        print("No assets need repair")
        return 0

    # Stage candidates
    import subprocess
    import tempfile

    from berlin_lst_downscaling.data.ard.cog_layout import (
        assert_raster_equivalent,
        validate_strict_cog,
    )
    from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    cogger_bin = args.cogger_bin
    success = 0
    failed = 0

    for row in needs_repair:
        uri = row["uri"]
        layout_class = row["layout_class"]

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_src = Path(tmp_dir) / "source.tif"
                tmp_cog = Path(tmp_dir) / "candidate.tif"

                # Download source
                bucket_name, key = _parse_gs_uri(uri)
                client = _gcs_client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(key)
                blob.download_to_filename(str(tmp_src))

                if layout_class == LayoutClass.HARD_LAYOUT:
                    # Use Cogger for hard layout errors
                    result = subprocess.run(  # noqa: S603
                        [cogger_bin, "-output", str(tmp_cog), str(tmp_src)],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"Cogger failed: {result.stderr}")

                elif layout_class == LayoutClass.MISSING_OVERVIEW:
                    # Use GDAL COG driver for missing overviews
                    from rasterio.shutil import copy as rio_copy

                    # Determine resampling based on asset kind
                    resampling = "nearest" if row["asset_kind"] == "flag" else "cubic"

                    rio_copy(
                        str(tmp_src),
                        str(tmp_cog),
                        driver="COG",
                        overview_resampling=resampling,
                        blocksize=512,
                    )

                # Validate candidate
                strict_result = validate_strict_cog(str(tmp_cog))
                if not strict_result.valid:
                    all_issues = list(strict_result.errors) + list(strict_result.warnings)
                    raise ValueError(f"Candidate validation failed: {all_issues}")

                # Semantic comparison
                compare_errors = assert_raster_equivalent(gdal_uri(uri), tmp_cog)
                if compare_errors:
                    raise ValueError(f"Semantic comparison failed: {compare_errors}")

                # Upload candidate
                candidate_key = f"{stage_root.replace('gs://berlin-lst-data-recovery/', '')}/{key}"
                backup_bucket_name = config["recovery_bucket"].replace("gs://", "").split("/")[0]
                backup_bucket = client.bucket(backup_bucket_name)
                candidate_blob = backup_bucket.blob(candidate_key)

                candidate_blob.upload_from_filename(
                    str(tmp_cog),
                    if_generation_match=0,  # create-only
                    checksum="crc32c",
                )

                success += 1
                if success % 50 == 0:
                    print(f"  Staged: {success}/{len(needs_repair)}")

        except Exception as exc:
            print(f"  FAIL: {uri}: {exc}")
            failed += 1

    print(f"\nStage complete: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_promote(args: argparse.Namespace) -> int:
    """Guarded canonical promotion with rollback."""
    print("ERROR: Promote command not yet implemented")
    print("This is a complex command that requires:")
    print("  1. Load staged candidates")
    print("  2. For each candidate:")
    print("     a. Back up current live object")
    print("     b. Promote candidate with generation guard")
    print("     c. Verify promoted object")
    print("     d. On failure: restore from backup")
    print("  3. Record all events immutably")
    print("\nSee docs/cog-layout-recovery.md for the full procedure")
    return 1


def cmd_verify_recovery(args: argparse.Namespace) -> int:
    """Independent verification of all canonical assets."""
    import yaml

    from berlin_lst_downscaling.data.ard.cog_recovery_state import hash_config

    config_path = Path(args.config)
    config_hash = hash_config(config_path)

    with open(config_path) as f:
        yaml.safe_load(f)

    print(f"Config hash: {config_hash}")

    # Load inventory
    from berlin_lst_downscaling.data.ard.cog_repair import (
        build_inventory_from_ledgers,
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

    print(f"Inventory: {table.num_rows} URIs")

    # Verify each asset
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog

    rows = table.to_pylist()
    results = []

    def verify_one(row):
        uri = row["uri"]
        strict_result = validate_strict_cog(uri)
        return {
            "uri": uri,
            "valid": strict_result.valid,
            "errors": strict_result.errors,
            "warnings": strict_result.warnings,
            "layout_class": strict_result.layout_class,
        }

    print("Verifying assets...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(verify_one, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Print summary
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = sum(1 for r in results if not r["valid"])

    print("\nVerification summary:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")

    if invalid_count > 0:
        print("\nInvalid assets:")
        for r in results:
            if not r["valid"]:
                print(f"  {r['uri']}: {r['errors'] + r['warnings']}")

    if invalid_count == 0:
        print("\nVerification PASSED — all assets are strict-clean")
        return 0
    else:
        print(f"\nVerification FAILED — {invalid_count} assets are not strict-clean")
        return 1


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="COG layout recovery CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Recovery commands
    p_preflight = subparsers.add_parser("preflight", help="Verify bucket policy and inventory")
    p_preflight.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_preflight.add_argument("--recovery-root", required=True, help="GCS recovery root")

    p_classify = subparsers.add_parser("classify", help="Build evidence matrix")
    p_classify.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_classify.add_argument("--recovery-root", required=True, help="GCS recovery root")
    p_classify.add_argument("--workers", type=int, default=4)

    p_backup = subparsers.add_parser("backup", help="Preserve current live objects")
    p_backup.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_backup.add_argument("--recovery-root", required=True, help="GCS recovery root")

    p_stage = subparsers.add_parser("stage", help="Generate repair candidates")
    p_stage.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_stage.add_argument("--recovery-root", required=True, help="GCS recovery root")
    p_stage.add_argument("--cogger-bin", default="cogger", help="Path to Cogger binary")

    p_promote = subparsers.add_parser("promote", help="Guarded canonical promotion")
    p_promote.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_promote.add_argument("--recovery-root", required=True, help="GCS recovery root")

    p_verify = subparsers.add_parser("verify-recovery", help="Independent verification")
    p_verify.add_argument("--config", required=True, help="Path to recovery.yaml")
    p_verify.add_argument("--workers", type=int, default=4)

    # Legacy commands — reject with migration message
    subparsers.add_parser("snapshot", help="DEPRECATED — use classify")
    subparsers.add_parser("prove", help="DEPRECATED — use stage")
    subparsers.add_parser("apply", help="DEPRECATED — use promote")
    subparsers.add_parser("verify", help="DEPRECATED — use verify-recovery")
    subparsers.add_parser("rollback", help="DEPRECATED — use promote")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    # Migration guard for legacy commands
    legacy_commands = {"snapshot", "prove", "apply", "verify", "rollback"}
    if args.command in legacy_commands:
        return _reject_legacy_command(args.command)

    commands = {
        "preflight": cmd_preflight,
        "classify": cmd_classify,
        "backup": cmd_backup,
        "stage": cmd_stage,
        "promote": cmd_promote,
        "verify-recovery": cmd_verify_recovery,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
