"""Write manifest Parquet per docs/ard-manifest-schema.md (v1 + v2 columns).

Manifest structure: one row per scene (Landsat anchor, S2 match, or ECOSTRESS granule).
Rows are tagged with ``status`` so the ARD pipeline can filter by source.

Status values (per schema spec):
  coupled   — Landsat anchor successfully paired with S2 (ECOSTRESS may follow)
  orphaned  — Landsat anchor with no S2 partner (below threshold)
  validated — Landsat anchor with both S2 and ECOSTRESS (for validation subset)
  dropped   — Landsat anchor entirely discarded (no row emitted, logged separately)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.selection import ManifestResult


def write_manifest(
    coupled: list[dict],
    dropped: list[dict],
    ecostress_by_anchor: dict[str, list[dict]],
    output_path: str,
) -> ManifestResult:
    """Write the ARD manifest Parquet and return a summary.

    Emits one row per scene with v1 + v2 columns per
    :ref:`ard-manifest-schema`.
    """
    rows: list[dict] = []

    # ── Coupled Landsat anchors (with S2 match) ─────────────────────────────
    for pair in coupled:
        anchor = pair["anchor"]
        s2 = pair["s2"]
        eco_list = ecostress_by_anchor.get(anchor["scene_id"], [])

        # Primary status: "validated" if ECOSTRESS also matched, else "coupled"
        status = "validated" if eco_list else "coupled"

        rows.append(
            {
                "scene_id": anchor["scene_id"],
                "source": "landsat-c2-l2",
                "year": anchor["year"],
                "status": status,
                "coupled_s2_id": s2["scene_id"],
                "ecostress_id": eco_list[0]["granule_id"] if eco_list else None,
                "paired_at": _naive_to_utc(s2["datetime"]),
                "clear_frac": pair.get("clear_frac"),
                "dt_days": s2["dt_days"],
                # v2 columns
                "date": anchor["date"],
                "item_href": anchor.get("item_href"),
                "acquisition_datetime": _naive_to_utc(anchor["datetime"]),
                "cloud_cover": anchor.get("cloud_cover"),
                "solar_azimuth": anchor.get("sun_azimuth"),
                "solar_elevation": anchor.get("sun_elevation"),
            }
        )

        # S2 partner row
        rows.append(
            {
                "scene_id": s2["scene_id"],
                "source": "sentinel-2-l2a",
                "year": s2["year"],
                "status": status,
                "coupled_s2_id": None,
                "ecostress_id": None,
                "paired_at": _naive_to_utc(s2["datetime"]),
                "clear_frac": pair.get("clear_frac"),
                "dt_days": s2["dt_days"],
                "date": s2["date"],
                "item_href": s2.get("item_href"),
                "acquisition_datetime": _naive_to_utc(s2["datetime"]),
                "cloud_cover": s2.get("cloud_cover"),
                "solar_azimuth": None,
                "solar_elevation": None,
            }
        )

        # ECOSTRESS rows (may be 0, 1, or more per anchor)
        for eco in eco_list:
            rows.append(
                {
                    "scene_id": eco["granule_id"],
                    "source": "ecostress",
                    "year": eco["year"],
                    "status": status,
                    "coupled_s2_id": None,
                    "ecostress_id": None,
                    "paired_at": _naive_to_utc(eco["datetime"]),
                    "clear_frac": eco.get("clear_frac"),
                    "dt_days": eco.get("dt_hours", 0.0) / 24.0,  # convert hours → days
                    "date": eco["date"],
                    "item_href": None,
                    "acquisition_datetime": _naive_to_utc(eco["datetime"]),
                    "cloud_cover": None,
                    "solar_azimuth": None,
                    "solar_elevation": None,
                }
            )

    # ── Orphaned Landsat anchors (no S2 above threshold) ────────────────────
    for pair in dropped:
        anchor = pair["anchor"]
        rows.append(
            {
                "scene_id": anchor["scene_id"],
                "source": "landsat-c2-l2",
                "year": anchor["year"],
                "status": "orphaned",
                "coupled_s2_id": None,
                "ecostress_id": None,
                "paired_at": None,
                "clear_frac": None,
                "dt_days": None,
                "date": anchor["date"],
                "item_href": anchor.get("item_href"),
                "acquisition_datetime": _naive_to_utc(anchor["datetime"]),
                "cloud_cover": anchor.get("cloud_cover"),
                "solar_azimuth": anchor.get("sun_azimuth"),
                "solar_elevation": anchor.get("sun_elevation"),
            }
        )

    # ── Write Parquet ────────────────────────────────────────────────────────
    table = pa.Table.from_pylist(rows, schema=_MANIFEST_SCHEMA)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pq.write_table(table, output_path)

    n_anchors = len(coupled) + len(dropped)
    return ManifestResult(
        n_anchors=n_anchors,
        n_coupled=len(coupled),
        n_dropped=len(dropped),
        n_ecostress=sum(len(v) for v in ecostress_by_anchor.values()),
        manifest_path=output_path,
    )


def _naive_to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC; if naive, assume UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


_MANIFEST_SCHEMA = pa.schema(
    [
        # v1 core
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("coupled_s2_id", pa.string(), nullable=True),
        pa.field("ecostress_id", pa.string(), nullable=True),
        pa.field("paired_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("clear_frac", pa.float32(), nullable=True),
        pa.field("dt_days", pa.float32(), nullable=True),
        # v2 optional
        pa.field("date", pa.string(), nullable=True),
        pa.field("item_href", pa.string(), nullable=True),
        pa.field("acquisition_datetime", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("cloud_cover", pa.float32(), nullable=True),
        pa.field("solar_azimuth", pa.float32(), nullable=True),
        pa.field("solar_elevation", pa.float32(), nullable=True),
    ]
)
