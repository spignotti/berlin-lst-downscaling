"""Versioned Arrow schemas and metadata constants for the manifest bundle.

The manifest bundle consists of three artifacts:
  1. ``manifest.parquet`` — one unique executable scene per row.
  2. ``pairings.parquet`` — one Landsat→Sentinel-2 relation per anchor.
  3. ``manifest_report.json`` — publication gate with hashes, counts, policy.

Schema versioning:
  - v1/v2: legacy single-file manifest (retired).
  - v3: normalized bundle with separate pairings, enforced AOI metrics,
       explicit platform/cutoff, and policy metadata.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pyarrow as pa

# ── schema versions ───────────────────────────────────────────────────

SCHEMA_VERSION_MANIFEST = 3
SCHEMA_VERSION_PAIRINGS = 1
SCHEMA_VERSION_REPORT = 1

# ── allowed values ────────────────────────────────────────────────────

ALLOWED_LANDSAT_PLATFORMS = {"landsat-8", "landsat-9"}
ALLOWED_SOURCES = {"landsat-c2-l2", "sentinel-2-l2a", "ecostress"}
ALLOWED_ROLES = {"anchor", "predictor", "validation"}

# Required ECOSTRESS validation granules (2018-08-25 only)
ECOSTRESS_VALIDATION_IDS = frozenset({
    "ECOv002_L2T_LSTE_00770_009_32UQD_20180825T082058_0712_01",
    "ECOv002_L2T_LSTE_00770_009_33UUU_20180825T082058_0712_01",
    "ECOv002_L2T_LSTE_00770_009_33UVU_20180825T082058_0712_01",
    "ECOv002_L2T_LSTE_00771_005_32UQD_20180825T095710_0712_01",
    "ECOv002_L2T_LSTE_00771_005_33UUU_20180825T095710_0712_01",
    "ECOv002_L2T_LSTE_00771_005_33UVU_20180825T095710_0712_01",
})

# ── manifest.parquet schema v3 ────────────────────────────────────────

MANIFEST_SCHEMA = pa.schema([
    # Primary key (with `source`)
    pa.field("scene_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    # Classification
    pa.field("role", pa.string(), nullable=False),  # anchor | predictor | validation
    pa.field("platform", pa.string(), nullable=False),
    pa.field("year", pa.int32(), nullable=False),
    pa.field("acquisition_datetime", pa.timestamp("us", tz="UTC"), nullable=False),
    # STAC identity — exact item reference
    pa.field("item_href", pa.string(), nullable=True),  # required for PC STAC rows
    # AOI clear-fraction metrics (required for anchor/predictor)
    pa.field("aoi_clear_px", pa.int64(), nullable=True),
    pa.field("aoi_total_px", pa.int64(), nullable=True),
    pa.field("aoi_clear_frac", pa.float32(), nullable=True),
    # Diagnostic metadata (not used as gate)
    pa.field("cloud_cover", pa.float32(), nullable=True),
    pa.field("solar_azimuth", pa.float32(), nullable=True),
    pa.field("solar_elevation", pa.float32(), nullable=True),
])

# ── pairings.parquet schema v1 ────────────────────────────────────────

PAIRINGS_SCHEMA = pa.schema([
    pa.field("landsat_scene_id", pa.string(), nullable=False),
    pa.field("sentinel2_scene_id", pa.string(), nullable=False),
    pa.field("dt_seconds", pa.int64(), nullable=False),
    pa.field("landsat_clear_px", pa.int64(), nullable=False),
    pa.field("joint_clear_px", pa.int64(), nullable=False),
    pa.field("joint_clear_frac", pa.float32(), nullable=False),
    pa.field("score", pa.float32(), nullable=False),
])


# ── policy fingerprinting ─────────────────────────────────────────────


def policy_fingerprint(cfg: Any) -> str:
    """Return a stable SHA-256 fingerprint of the selection policy.

    Covers all parameters that change the manifest output: platforms,
    years, months, bbox, clear thresholds, window, and ECOSTRESS IDs.
    """
    import json

    payload = json.dumps({
        "platforms": sorted(cfg.get("platforms", ["landsat-8", "landsat-9"])),
        "years": sorted(cfg.get("years", [])),
        "months": sorted(cfg.get("months", [])),
        "bbox": list(cfg.get("bbox", [])),
        "landsat_min_clear_frac": (
            cfg.get("landsat", {}).get("anchor", {}).get("min_clear_frac", 0.05)
        ),
        "s2_min_clear_frac": cfg.get("sentinel2", {}).get("min_clear_frac", 0.05),
        "s2_window_days": cfg.get("sentinel2", {}).get("window_days", 3),
        "ecostress_ids": sorted(ECOSTRESS_VALIDATION_IDS),
    }, sort_keys=True)
    return sha256(payload.encode()).hexdigest()[:16]


def bundle_metadata(
    policy_hash: str,
    cutoff_utc: str,
    manifest_hash: str = "",
    pairings_hash: str = "",
) -> dict[str, str]:
    """Return Parquet metadata dict for manifest and pairings tables."""
    from datetime import UTC, datetime

    return {
        "schema_name": "berlin-lst-manifest",
        "schema_version": str(SCHEMA_VERSION_MANIFEST),
        "policy_sha256": policy_hash,
        "cutoff_utc": cutoff_utc,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_hash": manifest_hash,
        "pairings_hash": pairings_hash,
    }


def table_metadata(meta: dict[str, str]) -> dict[str, str]:
    """Wrap metadata dict for PyArrow table attachment.

    All values must be strings (PyArrow Parquet requirement).
    """
    return {k: str(v) for k, v in meta.items()}


__all__ = [
    "SCHEMA_VERSION_MANIFEST",
    "SCHEMA_VERSION_PAIRINGS",
    "SCHEMA_VERSION_REPORT",
    "ALLOWED_LANDSAT_PLATFORMS",
    "ALLOWED_SOURCES",
    "ALLOWED_ROLES",
    "ECOSTRESS_VALIDATION_IDS",
    "MANIFEST_SCHEMA",
    "PAIRINGS_SCHEMA",
    "policy_fingerprint",
    "bundle_metadata",
    "table_metadata",
]
