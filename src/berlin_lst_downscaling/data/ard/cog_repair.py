"""COG layout repair harness for in-place GCS repair.

Provides snapshot, prove, apply, verify, and rollback operations for
repairing systemic COG layout issues without changing pixel values.
"""

from __future__ import annotations

import io
import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.ard.cog_layout import assert_raster_equivalent, validate_strict_cog
from berlin_lst_downscaling.data.io.storage import (
    _gcs_client,
    _parse_gs_uri,
    atomic_write,
    read_bytes,
)

_logger = logging.getLogger(__name__)

# ── schemas ───────────────────────────────────────────────────────────

ASSET_SCHEMA = pa.schema(
    [
        pa.field("uri", pa.string(), nullable=False),
        pa.field("asset_kind", pa.string(), nullable=False),  # "data" or "flag"
        pa.field("source", pa.string()),
        pa.field("partition", pa.string()),
        pa.field("year", pa.int32(), nullable=True),
        pa.field("generation", pa.int64()),
        pa.field("metageneration", pa.int64()),
        pa.field("size", pa.int64()),
        pa.field("crc32c", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("metadata_json", pa.string()),
        pa.field("pre_repair_errors", pa.list_(pa.string())),
        pa.field("status", pa.string()),  # "pending", "repaired", "verified", "failed"
    ]
)

AUDIT_SCHEMA = pa.schema(
    [
        pa.field("uri", pa.string(), nullable=False),
        pa.field("old_generation", pa.int64()),
        pa.field("new_generation", pa.int64()),
        pa.field("repaired_crc32c", pa.string()),
        pa.field("verified", pa.bool_()),
        pa.field("timestamp", pa.string()),
    ]
)

# ── inventory ──────────────────────────────────────────────────────────

def build_inventory_from_ledgers(
    manifest_uri: str,
    ard_ledger_uri: str,
    static_sources_root: str,
    static_derived_root: str,
    dynamic_full_root: str,
    dynamic_inference_root: str,
) -> pa.Table:
    """Build the frozen inventory from all published ledgers and manifest.

    This builds the complete inventory without requiring prior profiling output.
    """
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.profiling.inventory import (
        build_ard_assets,
        build_ard_flag_assets,
        build_static_assets,
        build_dynamic_assets,
    )

    # Build all assets
    assets = []
    assets.extend(build_ard_assets(manifest_uri, ard_ledger_uri))
    assets.extend(build_ard_flag_assets(ard_ledger_uri))
    assets.extend(build_static_assets(static_sources_root, static_derived_root))
    assets.extend(build_dynamic_assets(dynamic_full_root, dynamic_inference_root))

    rows: list[dict[str, Any]] = []
    for asset in assets:
        uri = asset.cog_uri
        if not uri:
            continue

        # Determine asset kind
        if asset.item_id.endswith("__flag"):
            asset_kind = "flag"
        else:
            asset_kind = "data"

        metadata = _get_gcs_metadata(uri)
        rows.append(
            {
                "uri": uri,
                "asset_kind": asset_kind,
                "source": asset.source.split("__")[0] if "__" in asset.source else asset.source,
                "partition": asset.partition,
                "year": asset.year,
                "generation": metadata.get("generation", 0),
                "metageneration": metadata.get("metageneration", 0),
                "size": metadata.get("size", 0),
                "crc32c": metadata.get("crc32c", ""),
                "content_type": metadata.get("content_type", ""),
                "metadata_json": json.dumps(metadata.get("metadata", {})),
                "pre_repair_errors": [],
                "status": "pending",
            }
        )

    # Create table row by row to avoid schema issues
    if not rows:
        return pa.table([], schema=ASSET_SCHEMA)

    # Convert None values to appropriate defaults
    for row in rows:
        row["year"] = row["year"] if row["year"] is not None else 0
        row["generation"] = row["generation"] if row["generation"] is not None else 0
        row["metageneration"] = row["metageneration"] if row["metageneration"] is not None else 0
        row["size"] = row["size"] if row["size"] is not None else 0

    # Build arrays from rows
    arrays = []
    for field in ASSET_SCHEMA:
        col_data = [row[field.name] for row in rows]
        arrays.append(pa.array(col_data, type=field.type, from_pandas=False))

    return pa.table(arrays, schema=ASSET_SCHEMA)


def build_inventory(
    profile_parquet_uri: str,
    ard_ledger_uri: str,
) -> pa.Table:
    """Build the frozen inventory from profiling output and ARD ledger.

    Returns a table with one row per COG URI (data + flags).
    """
    # Load profiling assets (data COGs)
    profiling_df = pq.read_table(io.BytesIO(read_bytes(profile_parquet_uri))).to_pandas()

    # Extract unique data COGs
    data_uris = profiling_df[profiling_df["scope"] == "asset"]["cog_uri"].unique().tolist()

    # Load ARD ledger for flag COGs
    ard_ledger = pq.read_table(io.BytesIO(read_bytes(ard_ledger_uri)))
    flag_uris = [
        ard_ledger.column("path_flag")[i].as_py()
        for i in range(ard_ledger.num_rows)
        if ard_ledger.column("path_flag")[i].as_py()
    ]

    rows: list[dict[str, Any]] = []

    # Process data COGs
    for uri in data_uris:
        row_data = profiling_df[
            (profiling_df["scope"] == "asset") & (profiling_df["cog_uri"] == uri)
        ].iloc[0]

        metadata = _get_gcs_metadata(uri)
        rows.append(
            {
                "uri": uri,
                "asset_kind": "data",
                "source": row_data.get("source"),
                "partition": row_data.get("partition"),
                "year": int(row_data["year"]) if row_data.get("year") and not (isinstance(row_data["year"], float) and math.isnan(row_data["year"])) else None,
                "generation": metadata.get("generation", 0),
                "metageneration": metadata.get("metageneration", 0),
                "size": metadata.get("size", 0),
                "crc32c": metadata.get("crc32c", ""),
                "content_type": metadata.get("content_type", ""),
                "metadata_json": json.dumps(metadata.get("metadata", {})),
                "pre_repair_errors": [],  # filled later
                "status": "pending",
            }
        )

    # Process flag COGs
    for uri in flag_uris:
        metadata = _get_gcs_metadata(uri)
        rows.append(
            {
                "uri": uri,
                "asset_kind": "flag",
                "source": None,
                "partition": None,
                "year": None,
                "generation": metadata.get("generation", 0),
                "metageneration": metadata.get("metageneration", 0),
                "size": metadata.get("size", 0),
                "crc32c": metadata.get("crc32c", ""),
                "md5": metadata.get("md5", ""),
                "content_type": metadata.get("content_type", ""),
                "metadata_json": json.dumps(metadata.get("metadata", {})),
                "pre_repair_errors": [],
                "status": "pending",
            }
        )

    return pa.Table.from_pylist(rows, schema=ASSET_SCHEMA)


def _get_gcs_metadata(uri: str) -> dict[str, Any]:
    """Get GCS object metadata including generation."""
    if not uri.startswith("gs://"):
        return {}

    try:
        bucket_name, key = _parse_gs_uri(uri)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(key)
        if blob is None:
            return {}

        return {
            "generation": blob.generation,
            "metageneration": blob.metageneration,
            "size": blob.size,
            "crc32c": blob.crc32c,
            "content_type": blob.content_type,
            "metadata": blob.metadata or {},
        }
    except Exception as exc:
        _logger.warning("Failed to get GCS metadata for %s: %s", uri, exc)
        return {}


def _set_pre_repair_errors(table: pa.Table) -> pa.Table:
    """Set pre_repair_errors for each pending row."""
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        if row["status"] == "pending":
            errors = validate_strict_cog(row["uri"])
            rows[i]["pre_repair_errors"] = errors
    return pa.table(rows, schema=table.schema)


# ── prove ──────────────────────────────────────────────────────────────

def prove_cogger(
    table: pa.Table,
    cogger_bin: str,
    uri: str | None = None,
    all_layout_signatures: bool = False,
) -> pa.Table:
    """Run Cogger proof on selected COGs without mutating GCS.

    Downloads each candidate to a local temp, runs Cogger, and validates.
    """
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    # Filter to pending rows
    rows = table.to_pylist()
    pending = [
        i for i, row in enumerate(rows)
        if row["status"] == "pending" and (uri is None or row["uri"] == uri)
    ]

    if not pending:
        return table

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx in pending:
            row = rows[idx]
            uri_val = row["uri"]
            tmp_src = Path(tmp_dir) / f"_src_{idx}.tif"
            tmp_cog = Path(tmp_dir) / f"_cog_{idx}.tif"

            try:
                # Download source
                _download_gcs(uri_val, tmp_src)

                # Run Cogger
                result = subprocess.run(  # noqa: S603
                    [cogger_bin, "-output", str(tmp_cog), str(tmp_src)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Cogger failed: {result.stderr}")

                # Validate strict COG
                strict_errors = validate_strict_cog(str(tmp_cog))
                if strict_errors:
                    raise ValueError(f"Strict validation failed: {strict_errors}")

                # Compare rasters
                compare_errors = assert_raster_equivalent(gdal_uri(uri_val), tmp_cog)
                if compare_errors:
                    raise ValueError(f"Raster comparison failed: {compare_errors}")

                rows[idx]["status"] = "proved"
                _logger.info("Cogger proof passed: %s", uri_val)

            except Exception as exc:
                rows[idx]["status"] = "failed"
                rows[idx]["pre_repair_errors"] = [f"Cogger proof failed: {exc}"]
                _logger.error("Cogger proof failed for %s: %s", uri_val, exc)

    # Create table arrays directly
    if not rows:
        return table

    arrays = []
    for field in table.schema:
        col_data = [row[field.name] for row in rows]
        arrays.append(pa.array(col_data, type=field.type, from_pandas=False))

    return pa.table(arrays, schema=table.schema)


# ── apply ──────────────────────────────────────────────────────────────

def apply_repair(
    table: pa.Table,
    cogger_bin: str,
    audit_root: str,
    *,
    uri: str | None = None,
    sources: str | None = None,
    workers: int = 1,
    checkpoint_every: int | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Apply Cogger repair to selected COGs and upload to GCS.

    Returns (asset_table, audit_table).
    """
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    rows = table.to_pylist()
    pending = [
        i for i, row in enumerate(rows)
        if row["status"] in ("pending", "proved")
        and (uri is None or row["uri"] == uri)
        and (sources is None or row["source"] in sources.split(","))
    ]

    if not pending:
        return table, pa.table([], schema=AUDIT_SCHEMA)

    audit_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for count, idx in enumerate(pending, 1):
            row = rows[idx]
            uri_val = row["uri"]
            tmp_src = Path(tmp_dir) / f"_src_{idx}.tif"
            tmp_cog = Path(tmp_dir) / f"_cog_{idx}.tif"

            try:
                # Download source
                _download_gcs(uri_val, tmp_src)

                # Run Cogger
                result = subprocess.run(  # noqa: S603
                    [cogger_bin, "-output", str(tmp_cog), str(tmp_src)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Cogger failed: {result.stderr}")

                # Validate strict COG
                strict_errors = validate_strict_cog(str(tmp_cog))
                if strict_errors:
                    raise ValueError(f"Strict validation failed: {strict_errors}")

                # Compare rasters
                compare_errors = assert_raster_equivalent(gdal_uri(uri_val), tmp_cog)
                if compare_errors:
                    raise ValueError(f"Raster comparison failed: {compare_errors}")

                # Upload to GCS with generation guard
                old_generation = row["generation"]
                _upload_cog_guarded(tmp_cog, uri_val, old_generation, row["metadata_json"])

                # Get new generation
                new_metadata = _get_gcs_metadata(uri_val)
                new_generation = new_metadata.get("generation", 0)

                rows[idx]["status"] = "repaired"
                audit_rows.append(
                    {
                        "uri": uri_val,
                        "old_generation": old_generation,
                        "new_generation": new_generation,
                        "repaired_crc32c": new_metadata.get("crc32c", ""),
                        "verified": False,
                        "timestamp": _now_iso(),
                    }
                )

                _logger.info(
                    "Repair applied: %s (gen %s -> %s)",
                    uri_val, old_generation, new_generation
                )

                # Checkpoint
                if checkpoint_every and count % checkpoint_every == 0:
                    _logger.info("Checkpoint at %d repairs", count)

            except Exception as exc:
                rows[idx]["status"] = "failed"
                rows[idx]["pre_repair_errors"] = [f"Repair failed: {exc}"]
                _logger.error("Repair failed for %s: %s", uri_val, exc)

    # Create table arrays directly
    if not rows:
        asset_table = pa.table([], schema=ASSET_SCHEMA)
    else:
        # Convert None values to defaults
        for row in rows:
            row["year"] = row["year"] if row["year"] is not None else 0
            row["generation"] = row["generation"] if row["generation"] is not None else 0
            row["metageneration"] = row["metageneration"] if row["metageneration"] is not None else 0
            row["size"] = row["size"] if row["size"] is not None else 0

        arrays = []
        for field in ASSET_SCHEMA:
            col_data = [row[field.name] for row in rows]
            arrays.append(pa.array(col_data, type=field.type, from_pandas=False))

        asset_table = pa.table(arrays, schema=ASSET_SCHEMA)

    # Create audit table
    if not audit_rows:
        audit_table = pa.table([], schema=AUDIT_SCHEMA)
    else:
        audit_arrays = []
        for field in AUDIT_SCHEMA:
            col_data = [row[field.name] for row in audit_rows]
            audit_arrays.append(pa.array(col_data, type=field.type, from_pandas=False))
        audit_table = pa.table(audit_arrays, schema=AUDIT_SCHEMA)

    return asset_table, audit_table


def _upload_cog_guarded(
    local_path: Path,
    uri: str,
    expected_generation: int,
    metadata_json: str,
) -> None:
    """Upload COG to GCS with generation guard."""
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)

    blob.upload_from_filename(
        str(local_path),
        if_generation_match=expected_generation,
        checksum="crc32c",
    )


# ── verify ─────────────────────────────────────────────────────────────

def verify_repair(
    table: pa.Table,
    *,
    uri: str | None = None,
    all: bool = False,
) -> tuple[pa.Table, list[str]]:
    """Verify repaired COGs are strict-clean.

    Returns (updated_table, errors).
    """
    rows = table.to_pylist()
    errors: list[str] = []

    targets = [
        i for i, row in enumerate(rows)
        if row["status"] == "repaired"
        and (uri is None or row["uri"] == uri)
    ]

    for idx in targets:
        row = rows[idx]
        uri_val = row["uri"]

        strict_errors = validate_strict_cog(uri_val)
        if strict_errors:
            rows[idx]["status"] = "failed"
            errors.append(f"Verify failed for {uri_val}: {strict_errors}")
        else:
            rows[idx]["status"] = "verified"

    return pa.table(rows, schema=ASSET_SCHEMA), errors


# ── rollback ───────────────────────────────────────────────────────────

def rollback_repair(
    asset_table: pa.Table,
    audit_table: pa.Table,
    *,
    status: str = "repaired",
    workers: int = 1,
) -> tuple[pa.Table, pa.Table]:
    """Rollback repaired COGs to their old generation.

    Only restores if current generation matches the recorded new generation.
    """
    rows = asset_table.to_pylist()
    audit_rows = audit_table.to_pylist()

    # Build audit lookup
    audit_lookup: dict[str, dict[str, Any]] = {}
    for audit_row in audit_rows:
        audit_lookup[audit_row["uri"]] = audit_row

    for i, row in enumerate(rows):
        if row["status"] != status:
            continue

        uri_val = row["uri"]
        audit = audit_lookup.get(uri_val)
        if audit is None:
            continue

        # Check current generation matches recorded new generation
        current_metadata = _get_gcs_metadata(uri_val)
        current_generation = current_metadata.get("generation", 0)

        if current_generation != audit["new_generation"]:
            _logger.warning(
                "Cannot rollback %s: current gen %s != recorded new gen %s",
                uri_val, current_generation, audit["new_generation"],
            )
            continue

        # Restore old generation
        bucket_name, key = _parse_gs_uri(uri_val)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)

        # Use copy to restore from soft-deleted old generation
        source_blob = bucket.blob(key, generation=audit["old_generation"])

        bucket.copy_blob(
            source_blob, bucket, key,
            if_generation_match=current_generation,
        )

        rows[i]["status"] = "rolled_back"
        _logger.info("Rolled back %s to generation %s", uri_val, audit["old_generation"])

    return pa.table(rows, schema=ASSET_SCHEMA), pa.table(audit_rows, schema=AUDIT_SCHEMA)


# ── helpers ────────────────────────────────────────────────────────────

def _download_gcs(uri: str, local_path: Path) -> None:
    """Download a GCS object to a local file."""
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    blob.download_to_filename(str(local_path))


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()


def save_table(table: pa.Table, uri: str) -> None:
    """Save a Parquet table to GCS or local path."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    atomic_write(uri, buf.getvalue(), overwrite=True)


def load_table(uri: str) -> pa.Table:
    """Load a Parquet table from GCS or local path."""
    if uri.startswith("gs://"):
        return pq.read_table(io.BytesIO(read_bytes(uri)))
    return pq.read_table(uri)


__all__ = [
    "build_inventory",
    "prove_cogger",
    "apply_repair",
    "verify_repair",
    "rollback_repair",
    "save_table",
    "load_table",
]
