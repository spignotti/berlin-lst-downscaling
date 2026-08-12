"""Read-only Sentinel-2 snow/ice audit for the raw QA gate (WB2c-2, Stufe 1).

Quantifies the impact of the ARD flag-contract change that maps SCL
class 11 (snow/ice) to ``FLAG_SNOW_ICE``. For every unique paired S2
scene it compares the **source** SCL band — reloaded from the manifest
``item_href`` onto the canonical 10 m grid — against the **published**
S2 ARD flag COG and counts SCL=11 pixels that are currently unflagged.

Any CRS, shape, transform, or scene-identity mismatch fails that scene
row (``ok=False``) and makes the run exit non-zero: the audit is only
trustworthy if every scene was compared exactly.

Artifacts (overwrite-only, written under ``output_root``):
``scene_audit.parquet``, ``scene_audit.csv``, ``summary.json`` (last).
"""

from __future__ import annotations

import io
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rasterio.warp as rwarp

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.acquisition.pc_client import resolve_item_from_href
from berlin_lst_downscaling.data.acquisition.sentinel2 import load_s2_scene
from berlin_lst_downscaling.data.ard.ledger import Ledger
from berlin_lst_downscaling.data.io.storage import atomic_write
from berlin_lst_downscaling.data.selection.validate import load_bundle

_logger = logging.getLogger(__name__)

# ── artifact paths ──────────────────────────────────────────────────


def scene_audit_parquet_path(root: str) -> str:
    """URI of the per-scene audit Parquet."""
    return f"{root.rstrip('/')}/scene_audit.parquet"


def scene_audit_csv_path(root: str) -> str:
    """URI of the per-scene audit CSV."""
    return f"{root.rstrip('/')}/scene_audit.csv"


def audit_summary_path(root: str) -> str:
    """URI of the summary JSON (written last)."""
    return f"{root.rstrip('/')}/summary.json"


# ── per-scene row schema ────────────────────────────────────────────

_SCENE_SCHEMA = pa.schema(
    [
        pa.field("scene_id", pa.string()),
        pa.field("anchor_count", pa.int64()),
        pa.field("ok", pa.bool_()),
        pa.field("error", pa.string()),
        pa.field("aoi_px", pa.int64()),
        pa.field("scl11_px", pa.int64()),
        pa.field("scl11_invalid_px", pa.int64()),
        pa.field("scl11_unflagged_px", pa.int64()),
        pa.field("scl11_frac", pa.float64()),
        pa.field("unflagged_frac", pa.float64()),
        pa.field("unflagged_of_scl11_frac", pa.float64()),
    ]
)


@dataclass
class AuditResult:
    """Per-scene rows plus the aggregated summary."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ── main entry point ────────────────────────────────────────────────


def run_s2_snow_ice_audit(
    manifest_uri: str,
    ard_ledger_uri: str,
    aoi_mask_uri: str,
    output_root: str,
    run_id: str,
    max_scenes: int | None = None,
) -> AuditResult:
    """Audit SCL=11 vs published flags for every unique paired S2 scene.

    Parameters
    ----------
    manifest_uri :
        Canonical v3 bundle ``manifest.parquet`` (pairings derived from it).
    ard_ledger_uri :
        Published ARD ledger (provides per-scene ``path_flag``).
    aoi_mask_uri :
        Pre-rasterized Berlin mask (``aoi_10m.tif``, 1 = inside).
    output_root :
        Overwrite-only artifact root.
    run_id :
        Run identifier recorded in ``summary.json``.
    max_scenes :
        Bounded smoke support — audit only the first N paired S2 scenes.
    """
    bundle, bundle_result = load_bundle(manifest_uri, require_item_href=True)
    if not bundle_result.ok:
        raise RuntimeError(f"Manifest bundle invalid: {bundle_result.errors}")

    s2_anchor_counts = Counter(bundle.pairings_table.column("sentinel2_scene_id").to_pylist())
    manifest_lookup = {
        row["scene_id"]: row
        for row in bundle.manifest_table.to_pylist()
        if row["source"] == "sentinel-2-l2a"
    }

    ledger = Ledger.open(ard_ledger_uri)
    gbox = canon_grid_10m()
    aoi_mask = _load_aoi_mask(aoi_mask_uri, gbox)

    rows: list[dict[str, Any]] = []
    for scene_id, anchor_count in s2_anchor_counts.items():
        if max_scenes is not None and len(rows) >= max_scenes:
            break
        manifest_row = manifest_lookup.get(scene_id)
        if manifest_row is None:
            rows.append(
                _empty_row(
                    scene_id,
                    anchor_count,
                    error="paired S2 scene not in manifest sentinel-2-l2a rows",
                )
            )
            continue
        rows.append(_audit_one_scene(scene_id, anchor_count, manifest_row, ledger, gbox, aoi_mask))

    summary = _build_summary(rows, run_id, manifest_uri, ard_ledger_uri)
    _write_artifacts(rows, summary, output_root)
    return AuditResult(rows=rows, summary=summary)


# ── per-scene comparison ────────────────────────────────────────────


def _download_flag_bytes(flag_uri: str) -> bytes:
    """Download the flag COG bytes via GCS with bounded transient retry.

    Reads through the storage client (not /vsigs/) so transient range-read
    failures cannot corrupt the audit; the download is retried only on
    transient transport errors. 404s and 412 preconditions fail fast.
    """
    from google.api_core.exceptions import (
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    from google.cloud import storage
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    transient = (
        ConnectionError,
        TimeoutError,
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    client = storage.Client()
    bucket_name, key = flag_uri.removeprefix("gs://").split("/", 1)
    blob = client.bucket(bucket_name).blob(key)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(transient),
        reraise=True,
    )
    def _do() -> bytes:
        blob.reload()
        return blob.download_as_bytes(
            if_generation_match=blob.generation,
            checksum="auto",
        )

    return _do()


def _read_flag_band(flag_uri: str, gbox) -> tuple[np.ndarray | None, bool, str]:
    """Read band 1 of the flag COG in memory and check the canonical grid.

    Returns ``(flag, grid_ok, grid_detail)``; ``grid_detail`` describes
    the mismatch when ``grid_ok`` is False.
    """
    from rasterio.io import MemoryFile

    data = _download_flag_bytes(flag_uri)
    with MemoryFile(data) as mem:
        with mem.open() as src:
            crs_ok = str(src.crs).upper() == "EPSG:25833"
            shape_ok = (src.height, src.width) == (gbox.shape.y, gbox.shape.x)
            transform_ok = all(
                abs(g - e) < 0.01
                for g, e in zip(src.transform, gbox.transform, strict=True)
            )
            if not (crs_ok and shape_ok and transform_ok):
                detail = (
                    f"(crs={str(src.crs).upper()}, shape=({src.height},{src.width}), "
                    f"transform={src.transform})"
                )
                return None, False, detail
            return src.read(1), True, ""


def _empty_row(scene_id: str, anchor_count: int, error: str) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "anchor_count": anchor_count,
        "ok": False,
        "error": error,
        "aoi_px": None,
        "scl11_px": None,
        "scl11_invalid_px": None,
        "scl11_unflagged_px": None,
        "scl11_frac": None,
        "unflagged_frac": None,
        "unflagged_of_scl11_frac": None,
    }


def _audit_one_scene(
    scene_id: str,
    anchor_count: int,
    manifest_row: dict[str, Any],
    ledger: Ledger,
    gbox,
    aoi_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare source SCL against the published flag COG for one scene."""
    row = _empty_row(scene_id, anchor_count, error="")

    # ── reload source SCL from the exact manifest item ───────────────
    item_href = manifest_row.get("item_href")
    if not item_href:
        row["error"] = "manifest item_href missing"
        return row
    try:
        item = resolve_item_from_href(str(item_href), expected_id=scene_id)
        ds, _ = load_s2_scene(items=[item], bands=["SCL"], resolution=10)
        scl = np.round(ds["SCL"].values.squeeze()).astype(np.uint8)
    except Exception as exc:  # record and continue to next scene
        row["error"] = f"SCL reload failed: {exc}"
        return row

    # ── published ARD flag COG from the ledger ───────────────────────
    led = ledger.get(scene_id, "sentinel-2-l2a")
    if led is None or led.status != "done" or not led.path_flag:
        row["error"] = "ARD ledger row missing, not done, or flag path absent"
        return row
    try:
        flag, grid_ok, grid_detail = _read_flag_band(led.path_flag, gbox)
        if not grid_ok:
            row["error"] = f"flag COG grid mismatch vs canonical 10 m grid {grid_detail}"
            return row
        if flag is None:  # unreachable with grid_ok, kept for type narrowing
            raise RuntimeError("flag band read returned no data")
    except Exception as exc:  # record and continue to next scene
        row["error"] = f"flag COG read failed: {exc}"
        return row

    if scl.shape != flag.shape:
        row["error"] = f"SCL shape {scl.shape} != flag shape {flag.shape}"
        return row
    if scl.shape != aoi_mask.shape:
        row["error"] = f"SCL shape {scl.shape} != AOI mask shape {aoi_mask.shape}"
        return row

    # ── counts within the AOI ────────────────────────────────────────
    inside = aoi_mask
    scl11 = scl == 11
    invalid = flag != 0

    aoi_px = int(np.sum(inside))
    scl11_px = int(np.sum(scl11 & inside))
    scl11_invalid_px = int(np.sum(scl11 & invalid & inside))
    scl11_unflagged_px = int(np.sum(scl11 & ~invalid & inside))

    row.update(
        ok=True,
        error="",
        aoi_px=aoi_px,
        scl11_px=scl11_px,
        scl11_invalid_px=scl11_invalid_px,
        scl11_unflagged_px=scl11_unflagged_px,
        scl11_frac=(scl11_px / aoi_px) if aoi_px else None,
        unflagged_frac=(scl11_unflagged_px / aoi_px) if aoi_px else None,
        unflagged_of_scl11_frac=(scl11_unflagged_px / scl11_px) if scl11_px else None,
    )
    return row


# ── AOI mask ────────────────────────────────────────────────────────


def _load_aoi_mask(aoi_mask_uri: str, gbox) -> np.ndarray:
    """Reproject the AOI mask onto the canonical 10 m grid (nearest)."""
    with rasterio.open(aoi_mask_uri) as src:
        aoi_data = src.read(1)
        destination = np.empty((gbox.shape.y, gbox.shape.x), dtype=aoi_data.dtype)
        aoi_data, _ = rwarp.reproject(
            source=aoi_data,
            src_transform=src.transform,
            src_width=src.width,
            src_height=src.height,
            src_crs=src.crs,
            destination=destination,
            dst_crs="EPSG:25833",
            dst_transform=gbox.transform,
            resampling=rwarp.Resampling.nearest,
        )
    return aoi_data == 1


# ── summary + artifacts ─────────────────────────────────────────────


def _build_summary(
    rows: list[dict[str, Any]],
    run_id: str,
    manifest_uri: str,
    ard_ledger_uri: str,
) -> dict[str, Any]:
    ok_rows = [r for r in rows if r["ok"]]
    failed_rows = [r for r in rows if not r["ok"]]

    def _sum(key: str) -> int:
        return sum(int(r[key]) for r in ok_rows if r[key] is not None)

    totals = {
        "aoi_px": _sum("aoi_px"),
        "scl11_px": _sum("scl11_px"),
        "scl11_invalid_px": _sum("scl11_invalid_px"),
        "scl11_unflagged_px": _sum("scl11_unflagged_px"),
    }

    worst = sorted(
        (r for r in ok_rows if r["unflagged_of_scl11_frac"] is not None),
        key=lambda r: float(r["unflagged_of_scl11_frac"]),
        reverse=True,
    )[:10]

    return {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest_uri": manifest_uri,
        "ard_ledger_uri": ard_ledger_uri,
        "scenes_total": len(rows),
        "scenes_compared": len(ok_rows),
        "scenes_failed": len(failed_rows),
        "scenes_with_scl11": sum(1 for r in ok_rows if (r["scl11_px"] or 0) > 0),
        "scenes_with_unflagged_scl11": sum(
            1 for r in ok_rows if (r["scl11_unflagged_px"] or 0) > 0
        ),
        **totals,
        "overall_scl11_frac": (
            totals["scl11_px"] / totals["aoi_px"] if totals["aoi_px"] else None
        ),
        "overall_unflagged_frac": (
            totals["scl11_unflagged_px"] / totals["aoi_px"] if totals["aoi_px"] else None
        ),
        "overall_unflagged_of_scl11_frac": (
            totals["scl11_unflagged_px"] / totals["scl11_px"]
            if totals["scl11_px"]
            else None
        ),
        "worst_scenes": [
            {
                "scene_id": r["scene_id"],
                "anchor_count": r["anchor_count"],
                "scl11_px": r["scl11_px"],
                "scl11_unflagged_px": r["scl11_unflagged_px"],
                "unflagged_of_scl11_frac": r["unflagged_of_scl11_frac"],
            }
            for r in worst
        ],
    }


def _write_artifacts(rows: list[dict[str, Any]], summary: dict[str, Any], output_root: str) -> None:
    table = pa.Table.from_pylist(rows, schema=_SCENE_SCHEMA)

    parquet_buf = io.BytesIO()
    pq.write_table(table, parquet_buf)
    atomic_write(scene_audit_parquet_path(output_root), parquet_buf.getvalue(), overwrite=True)

    csv_buf = io.BytesIO()
    table.to_pandas().to_csv(csv_buf, index=False)
    atomic_write(scene_audit_csv_path(output_root), csv_buf.getvalue(), overwrite=True)

    atomic_write(
        audit_summary_path(output_root),
        json.dumps(summary, indent=2),
        overwrite=True,
    )


__all__ = [
    "AuditResult",
    "audit_summary_path",
    "run_s2_snow_ice_audit",
    "scene_audit_csv_path",
    "scene_audit_parquet_path",
]
