"""Write legacy v3 or exact-membership v4 manifest bundles.

The bundle consists of three artifacts:
  1. ``manifest.parquet`` — one unique executable scene per row.
  2. ``pairings.parquet`` — one Landsat→Sentinel-2 relation per anchor.
  3. ``manifest_report.json`` — publication gate with hashes, counts, policy.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.common.util import ensure_utc, sha256_bytes
from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import s2_mosaic_enabled
from berlin_lst_downscaling.data.io.storage import atomic_write
from berlin_lst_downscaling.data.selection.schema import (
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_V4,
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
    mosaic = s2_mosaic_enabled(cfg)
    manifest_version = 4 if mosaic else 3
    now = datetime.now(UTC)

    # ── Build manifest rows ────────────────────────────────────────────
    manifest_rows: list[dict] = []
    # Track referenced S2 IDs — each S2 scene appears only once in manifest
    seen_s2: set[str] = set()
    s2_contracts: dict[str, tuple] = {}

    for pair in coupled:
        anchor = pair["anchor"]
        s2 = pair["s2"]
        if mosaic:
            contract = (s2.get("s2_items"), s2.get("s2_mosaic_group"), s2.get("item_href"))
            if s2["scene_id"] in s2_contracts and s2_contracts[s2["scene_id"]] != contract:
                raise ValueError(f"Conflicting S2 membership for {s2['scene_id']!r}")
            s2_contracts[s2["scene_id"]] = contract

        # Anchor row (unique per Landsat scene)
        manifest_rows.append(
            {
                "scene_id": anchor["scene_id"],
                "source": "landsat-c2-l2",
                "role": "anchor",
                "platform": _extract_platform(anchor["scene_id"]),
                "year": anchor["year"],
                "acquisition_datetime": ensure_utc(anchor["datetime"]),
                "item_href": anchor.get("item_href"),
                "aoi_clear_px": anchor.get("aoi_clear_px"),
                "aoi_total_px": anchor.get("aoi_total_px"),
                "aoi_clear_frac": anchor.get("aoi_clear_frac"),
                "cloud_cover": anchor.get("cloud_cover"),
                "solar_azimuth": anchor.get("sun_azimuth"),
                "solar_elevation": anchor.get("sun_elevation"),
            }
        )

        # S2 predictor row (deduplicated — one row per unique S2 scene)
        if s2["scene_id"] not in seen_s2:
            seen_s2.add(s2["scene_id"])
            manifest_rows.append(
                {
                    "scene_id": s2["scene_id"],
                    "source": "sentinel-2-l2a",
                    "role": "predictor",
                    "platform": "sentinel-2",
                    "year": s2["year"],
                    "acquisition_datetime": ensure_utc(s2["datetime"]),
                    "item_href": s2.get("item_href"),
                    "aoi_clear_px": s2.get("aoi_clear_px"),
                    "aoi_total_px": s2.get("aoi_total_px"),
                    "aoi_clear_frac": s2.get("aoi_clear_frac"),
                    "cloud_cover": s2.get("cloud_cover"),
                    "solar_azimuth": None,
                    "solar_elevation": None,
                    **(
                        {
                            "s2_items": s2.get("s2_items"),
                            "s2_mosaic_group": s2.get("s2_mosaic_group"),
                        }
                        if mosaic
                        else {}
                    ),
                }
            )

    # ECOSTRESS validation rows (exactly six unique IDs)
    for eco in ecostress_granules:
        manifest_rows.append(
            {
                "scene_id": eco["granule_id"],
                "source": "ecostress",
                "role": "validation",
                "platform": "ecostress",
                "year": eco["year"],
                "acquisition_datetime": ensure_utc(eco["datetime"]),
                "item_href": None,
                "aoi_clear_px": None,
                "aoi_total_px": None,
                "aoi_clear_frac": None,
                "cloud_cover": None,
                "solar_azimuth": None,
                "solar_elevation": None,
            }
        )

    # ── Build pairings rows ────────────────────────────────────────────
    pairing_rows: list[dict] = []
    for pair in coupled:
        anchor = pair["anchor"]
        s2 = pair["s2"]
        dt_seconds = int(abs((anchor["datetime"] - s2["datetime"]).total_seconds()))
        pairing_rows.append(
            {
                "landsat_scene_id": anchor["scene_id"],
                "sentinel2_scene_id": s2["scene_id"],
                "dt_seconds": dt_seconds,
                "landsat_clear_px": int(pair["landsat_clear_px"]),
                "joint_clear_px": int(pair["joint_clear_px"]),
                "joint_clear_frac": float(pair["joint_clear_frac"]),
                "score": float(pair["score"]),
            }
        )

    # ── Build tables ───────────────────────────────────────────────────
    # Sort for deterministic output: anchors by scene_id, then predictors
    manifest_rows.sort(key=lambda r: (r["role"] != "anchor", r["scene_id"]))
    pairing_rows.sort(key=lambda r: r["landsat_scene_id"])

    manifest_table = pa.Table.from_pylist(
        manifest_rows, schema=MANIFEST_SCHEMA_V4 if mosaic else MANIFEST_SCHEMA
    )
    pairings_table = pa.Table.from_pylist(pairing_rows, schema=PAIRINGS_SCHEMA)

    # Attach metadata (no self-referential file hashes — those go in the report only)
    meta = bundle_metadata(p_hash, cutoff_utc, manifest_version=manifest_version)
    manifest_table = manifest_table.replace_schema_metadata(table_metadata(meta))
    pairings_table = pairings_table.replace_schema_metadata(table_metadata(meta))

    # ── Validate before writing ────────────────────────────────────────
    from berlin_lst_downscaling.data.selection.validate import (
        validate_manifest_table,
        validate_pairings_table,
    )

    vr = validate_manifest_table(manifest_table, require_item_href=True)
    if not vr.ok:
        raise ValueError("Manifest validation failed:\n" + "\n".join(vr.errors))

    pr = validate_pairings_table(pairings_table, manifest_table)
    if not pr.ok:
        raise ValueError("Pairings validation failed:\n" + "\n".join(pr.errors))

    # ── Write bundle ───────────────────────────────────────────────────
    # Serialize first and hash the upload bytes: re-reading the written file
    # via local open() fails for gs:// URIs, while hashing the uploaded bytes
    # guarantees the report matches the stored object on every backend.
    manifest_bytes = _table_bytes(manifest_table)
    pairings_bytes = _table_bytes(pairings_table)

    # Compute full SHA-256 hashes from the upload bytes (authoritative in report only)
    manifest_hash = sha256_bytes(manifest_bytes)
    pairings_hash = sha256_bytes(pairings_bytes)

    _ensure_dir(manifest_path)
    _ensure_dir(pairings_path)
    _ensure_dir(report_path)

    atomic_write(manifest_path, manifest_bytes)
    atomic_write(pairings_path, pairings_bytes)

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
        "schema_version": manifest_version,
        "manifest_hash": manifest_hash,
        "pairings_hash": pairings_hash,
        "counts": {
            "anchors_total": n_anchors,
            "coupled": n_coupled,
            "dropped": n_dropped,
            "ecostress": n_eco,
            "manifest_rows": len(manifest_rows),
            "pairing_rows": len(pairing_rows),
            "unique_landsat": len(
                {r["scene_id"] for r in manifest_rows if r["source"] == "landsat-c2-l2"}
            ),
            "unique_s2": len(
                {r["scene_id"] for r in manifest_rows if r["source"] == "sentinel-2-l2a"}
            ),
            "unique_ecostress": n_eco,
        },
        "dropped_reasons": _summarize_dropped(dropped),
        "warnings": pr.warnings + vr.warnings,
        "unresolved_errors": len(vr.errors) + len(pr.errors),
    }
    if mosaic or cfg.get("landsat", {}).get("anchor", {}).get("min_aoi_overlap_frac") is not None:
        report["dropped_anchors"] = [
            {
                "scene_id": entry["anchor"]["scene_id"],
                "reason": entry.get("reason", "unknown"),
                **{
                    key: entry["anchor"][key]
                    for key in ("aoi_total_px", "aoi_valid_px", "aoi_overlap_frac")
                    if key in entry["anchor"]
                },
            }
            for entry in sorted(dropped, key=lambda entry: entry["anchor"]["scene_id"])
        ]

    atomic_write(report_path, json.dumps(report, indent=2).encode("utf-8"))

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


def _table_bytes(table: pa.Table) -> bytes:
    """Serialize a table to parquet bytes for hashing and upload."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _ensure_dir(path: str) -> None:
    if path.startswith("gs://"):
        return  # object stores have no directories to create
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


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
