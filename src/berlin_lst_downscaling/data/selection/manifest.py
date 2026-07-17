"""Write manifest bundle per docs/ard-manifest-schema.md (v3).

The bundle consists of three artifacts:
  1. ``manifest.parquet`` — one unique executable scene per row.
  2. ``pairings.parquet`` — one Landsat→Sentinel-2 relation per anchor.
  3. ``manifest_report.json`` — publication gate with hashes, counts, policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.selection.schema import (
    MANIFEST_SCHEMA,
    PAIRINGS_SCHEMA,
    bundle_metadata,
    policy_fingerprint,
    table_metadata,
)


def write_bundle(
    coupled: list[dict],
    dropped: list[dict],
    ecostress_granules: list[dict],
    *,
    manifest_path: str,
    pairings_path: str,
    report_path: str,
    cutoff_utc: str,
    cfg: Any,
) -> BundleResult:
    """Write the full manifest bundle (manifest + pairings + report).

    Parameters
    ----------
    coupled :
        List of coupled pair dicts from ``couple_all``.
    dropped :
        List of dropped pair dicts from ``couple_all``.
    ecostress_granules :
        List of ECOSTRESS granule dicts (from allowlist resolution).
    manifest_path :
        Output path for manifest.parquet.
    pairings_path :
        Output path for pairings.parquet.
    report_path :
        Output path for manifest_report.json.
    cutoff_utc :
        ISO timestamp for 2026 data cutoff.
    cfg :
        Hydra config for policy fingerprinting.
    """
    p_hash = policy_fingerprint(cfg)
    now = datetime.now(UTC)

    # ── Build manifest rows ────────────────────────────────────────────
    manifest_rows: list[dict] = []
    # Track referenced S2 IDs — each S2 scene appears only once in manifest
    seen_s2: set[str] = set()

    for pair in coupled:
        anchor = pair["anchor"]
        s2 = pair["s2"]

        # Anchor row (unique per Landsat scene)
        manifest_rows.append({
            "scene_id": anchor["scene_id"],
            "source": "landsat-c2-l2",
            "role": "anchor",
            "platform": _extract_platform(anchor["scene_id"]),
            "year": anchor["year"],
            "acquisition_datetime": _naive_to_utc(anchor["datetime"]),
            "item_href": anchor.get("item_href"),
            "aoi_clear_px": anchor.get("aoi_clear_px"),
            "aoi_total_px": anchor.get("aoi_total_px"),
            "aoi_clear_frac": anchor.get("aoi_clear_frac"),
            "cloud_cover": anchor.get("cloud_cover"),
            "solar_azimuth": anchor.get("sun_azimuth"),
            "solar_elevation": anchor.get("sun_elevation"),
        })

        # S2 predictor row (deduplicated — one row per unique S2 scene)
        if s2["scene_id"] not in seen_s2:
            seen_s2.add(s2["scene_id"])
            manifest_rows.append({
                "scene_id": s2["scene_id"],
                "source": "sentinel-2-l2a",
                "role": "predictor",
                "platform": "sentinel-2",
                "year": s2["year"],
                "acquisition_datetime": _naive_to_utc(s2["datetime"]),
                "item_href": s2.get("item_href"),
                "aoi_clear_px": s2.get("aoi_clear_px"),
                "aoi_total_px": s2.get("aoi_total_px"),
                "aoi_clear_frac": s2.get("aoi_clear_frac"),
                "cloud_cover": s2.get("cloud_cover"),
                "solar_azimuth": None,
                "solar_elevation": None,
            })

    # ECOSTRESS validation rows (exactly six unique IDs)
    for eco in ecostress_granules:
        manifest_rows.append({
            "scene_id": eco["granule_id"],
            "source": "ecostress",
            "role": "validation",
            "platform": "ecostress",
            "year": eco["year"],
            "acquisition_datetime": _naive_to_utc(eco["datetime"]),
            "item_href": None,
            "aoi_clear_px": None,
            "aoi_total_px": None,
            "aoi_clear_frac": None,
            "cloud_cover": None,
            "solar_azimuth": None,
            "solar_elevation": None,
        })

    # ── Build pairings rows ────────────────────────────────────────────
    pairing_rows: list[dict] = []
    for pair in coupled:
        anchor = pair["anchor"]
        s2 = pair["s2"]
        dt_seconds = int(abs((anchor["datetime"] - s2["datetime"]).total_seconds()))
        l_clear = pair.get("landsat_clear_px") or anchor.get("aoi_clear_px") or 0
        j_clear = pair.get("joint_clear_px") or 0
        j_frac = pair.get("joint_clear_frac") or pair.get("clear_frac") or 0.0
        score = pair.get("score") or 0.0

        pairing_rows.append({
            "landsat_scene_id": anchor["scene_id"],
            "sentinel2_scene_id": s2["scene_id"],
            "dt_seconds": dt_seconds,
            "landsat_clear_px": l_clear,
            "joint_clear_px": j_clear,
            "joint_clear_frac": j_frac,
            "score": score,
        })

    # ── Build tables ───────────────────────────────────────────────────
    manifest_table = pa.Table.from_pylist(manifest_rows, schema=MANIFEST_SCHEMA)
    pairings_table = pa.Table.from_pylist(pairing_rows, schema=PAIRINGS_SCHEMA)

    # Compute initial hashes (before metadata — placeholder)
    manifest_hash = ""
    pairings_hash = ""

    # Attach metadata (needs hash placeholders)
    meta = bundle_metadata(
        p_hash, cutoff_utc,
        manifest_hash=manifest_hash,
        pairings_hash=pairings_hash,
    )
    manifest_table = manifest_table.replace_schema_metadata(table_metadata(meta))
    pairings_table = pairings_table.replace_schema_metadata(table_metadata(meta))

    # ── Validate before writing ────────────────────────────────────────
    from berlin_lst_downscaling.data.selection.validate import (
        validate_manifest_table,
        validate_pairings_table,
    )

    vr = validate_manifest_table(manifest_table, require_item_href=True)
    if not vr.ok:
        raise ValueError(
            "Manifest validation failed:\n" + "\n".join(vr.errors)
        )

    pr = validate_pairings_table(pairings_table, manifest_table)
    if not pr.ok:
        raise ValueError(
            "Pairings validation failed:\n" + "\n".join(pr.errors)
        )

    # ── Write bundle ───────────────────────────────────────────────────
    _ensure_dir(manifest_path)
    _ensure_dir(pairings_path)
    _ensure_dir(report_path)

    pq.write_table(manifest_table, manifest_path)
    pq.write_table(pairings_table, pairings_path)

    # Compute hashes from written files (after metadata attachment)
    manifest_hash = _file_hash(manifest_path)
    pairings_hash = _file_hash(pairings_path)

    # ── Build report ───────────────────────────────────────────────────
    n_coupled = len(coupled)
    n_dropped = len(dropped)
    n_anchors = n_coupled + n_dropped
    n_eco = len(ecostress_granules)

    report = {
        "bundle_id": f"bundle-{p_hash}-{now.strftime('%Y%m%dT%H%M%S')}",
        "generated_at": now.isoformat(),
        "cutoff_utc": cutoff_utc,
        "policy_hash": p_hash,
        "schema_version": 3,
        "manifest_hash": manifest_hash,
        "pairings_hash": pairings_hash,
        "counts": {
            "anchors_total": n_anchors,
            "coupled": n_coupled,
            "dropped": n_dropped,
            "ecostress": n_eco,
            "manifest_rows": len(manifest_rows),
            "pairing_rows": len(pairing_rows),
            "unique_landsat": len({
                r["scene_id"] for r in manifest_rows
                if r["source"] == "landsat-c2-l2"
            }),
            "unique_s2": len({
                r["scene_id"] for r in manifest_rows
                if r["source"] == "sentinel-2-l2a"
            }),
            "unique_ecostress": n_eco,
        },
        "dropped_reasons": _summarize_dropped(dropped),
        "warnings": pr.warnings + vr.warnings,
        "unresolved_errors": len(vr.errors) + len(pr.errors),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return BundleResult(
        manifest_path=manifest_path,
        pairings_path=pairings_path,
        report_path=report_path,
        manifest_hash=manifest_hash,
        pairings_hash=pairings_hash,
        n_anchors=n_anchors,
        n_coupled=n_coupled,
        n_dropped=n_dropped,
        n_ecostress=n_eco,
    )


# ── helpers ──────────────────────────────────────────────────────────


def _extract_platform(scene_id: str) -> str:
    """Derive platform from Landsat scene ID prefix."""
    prefix = scene_id[:4]
    if prefix == "LC08":
        return "landsat-8"
    if prefix == "LC09":
        return "landsat-9"
    return "unknown"


def _naive_to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC; if naive, assume UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _file_hash(path: str) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _summarize_dropped(dropped: list[dict]) -> dict[str, int]:
    """Count dropped anchors by reason."""
    reasons: dict[str, int] = {}
    for pair in dropped:
        r = pair.get("reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    return reasons


# ── result type ──────────────────────────────────────────────────────


class BundleResult:
    """Result of writing the manifest bundle."""

    def __init__(
        self,
        manifest_path: str,
        pairings_path: str,
        report_path: str,
        manifest_hash: str,
        pairings_hash: str,
        n_anchors: int,
        n_coupled: int,
        n_dropped: int,
        n_ecostress: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.pairings_path = pairings_path
        self.report_path = report_path
        self.manifest_hash = manifest_hash
        self.pairings_hash = pairings_hash
        self.n_anchors = n_anchors
        self.n_coupled = n_coupled
        self.n_dropped = n_dropped
        self.n_ecostress = n_ecostress


__all__ = [
    "write_bundle",
    "BundleResult",
]
