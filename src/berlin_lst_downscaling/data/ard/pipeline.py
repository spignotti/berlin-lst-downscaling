"""ARD pipeline orchestration — run, per-scene processing, STAC construction.

The outer :func:`run` dispatches by ``cfg.mode``:

* **smoke** — discover scene ID via lightweight STAC search, then process
  one scene per source (accepts double-load for simplicity).
* **full** — read a manifest of scenes, reconcile against ledger, process
  only scenes that need work (new, failed, interrupted, schema-changed).

Per-scene logic is in per-source runner functions registered in
:data:`_RUNNERS`: load → mask → write COG → write STAC → update ledger.

Supported sources: ``landsat-c2-l2``, ``sentinel-2-l2a``, ``ecostress``.
"""

from __future__ import annotations

import io
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import xarray as xr
from omegaconf import DictConfig

from berlin_lst_downscaling.data.acquisition.ecostress import load_ecostress_scene
from berlin_lst_downscaling.data.acquisition.landsat import load_landsat_scene
from berlin_lst_downscaling.data.acquisition.sentinel2 import load_s2_scene
from berlin_lst_downscaling.data.ard.aoi import compute_aoi_metrics
from berlin_lst_downscaling.data.ard.contract import Contract, contract_for_source
from berlin_lst_downscaling.data.ard.idempotency import reconcile
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow
from berlin_lst_downscaling.data.ard.masking import mask_ecostress, mask_landsat, mask_s2
from berlin_lst_downscaling.data.ard.paths import cog_path, flag_path, stac_path
from berlin_lst_downscaling.data.ard.reports import qa_report
from berlin_lst_downscaling.data.ard.solar_position import solar_position
from berlin_lst_downscaling.data.ard.writer import (
    write_cog_atomic,
    write_flag_cog_atomic,
    write_stac_atomic,
)

# ── per-source runner registry ───────────────────────────────────────


def _run_landsat_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
) -> xr.Dataset:
    """Load + mask a Landsat C2 L2 scene."""
    effective_date = scene_date or cfg.scene_date
    bbox = tuple(cfg.bbox)
    ds, loaded_ids = load_landsat_scene(
        date=effective_date,
        bbox=bbox,
        resolution=int(cfg.target_resolution_low),
        items=items,
    )
    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Scene {scene_id!r} not in loaded items {loaded_ids}. "
            f"Date {effective_date!r} or bbox may not cover the requested scene."
        )
    masked = mask_landsat(ds, cfg)
    return masked


def _run_sentinel2_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
) -> xr.Dataset:
    """Load + mask a Sentinel-2 L2A scene."""
    effective_date = scene_date or cfg.scene_date
    bbox = tuple(cfg.bbox)
    ds, loaded_ids = load_s2_scene(
        date=effective_date,
        bbox=bbox,
        resolution=int(cfg.target_resolution_high),
        bands=["B02", "B03", "B04", "B08", "SCL"],
        items=items,
    )
    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Scene {scene_id!r} not in loaded items {loaded_ids}. "
            f"Date {effective_date!r} or bbox may not cover the requested scene."
        )
    # Solar position for directional cloud-shadow projection
    # S2 uses NOAA computation (PC items lack view:sun_*)
    az, el = _solar_for_scene(ds, cfg)
    masked = mask_s2(ds, cfg, az, el)
    return masked


def _run_ecostress_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
) -> xr.Dataset:
    """Load + mask an ECOSTRESS L2T granule from local COGs.

    No STAC search — scene_id comes from the manifest (mode=full) or
    from listing the fixture directory (mode=smoke).  The raw_dir
    (``cfg.ecostress.raw_dir``) is expected to contain one sub-directory
    per granule, each with ECO_L2T_LSTE layer COGs.
    """
    bbox = tuple(cfg.bbox) if cfg.get("bbox") else None
    resolution = int(cfg.get("target_resolution_low", 70))
    raw_dir = str(cfg.ecostress.raw_dir)

    ds, loaded_ids = load_ecostress_scene(
        granule_id=scene_id,
        raw_dir=raw_dir,
        bbox=bbox,
        resolution=resolution,
    )

    # Assert the requested granule was loaded
    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Granule {scene_id!r} not in loaded IDs {loaded_ids}. "
            "Check that the granule exists in the raw_dir."
        )

    # No solar position needed — ECOSTRESS LST is atmospherically corrected
    masked = mask_ecostress(ds, cfg)
    return masked


# Registry maps source key → per-source runner function.
# Each runner returns a masked xr.Dataset ready for COG writing.
_RUNNERS: dict[str, Callable[..., Any]] = {
    "landsat-c2-l2": _run_landsat_scene,
    "sentinel-2-l2a": _run_sentinel2_scene,
    "ecostress": _run_ecostress_scene,
}

# ── main entry ───────────────────────────────────────────────────────


def run(cfg: DictConfig) -> int:
    """Execute the ARD pipeline — manifest-driven only (mode=full).

    Smoke mode was removed in favor of manifest-driven smoke using
    ``smoke_primary`` config which builds a 3-row manifest then runs
    ``mode=full``.

    Returns 0 on success, 1 if any scene failed.
    """
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)

    led = Ledger.open(f"{output_root}/ledger.parquet")
    _log(cfg, run_id, "start", {"mode": cfg.mode, "sources": list(cfg.sources)})

    sources = list(cfg.sources)

    for source in sources:
        contract = contract_for_source(source)
        _process_manifest(source, contract, cfg, led, run_id)

    # Final QA report
    report = qa_report(led, cfg, run_id)
    failed_count = sum(
        report.get("per_source", {}).get(s, {}).get("failed", 0) for s in sources
    )
    _log(cfg, run_id, "qa_report", report)

    # Note: ledger persists per-transition via upsert — no batch write needed
    return 0 if failed_count == 0 else 1


def _process_manifest(
    source: str,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
) -> None:
    """Process scenes listed in a manifest Parquet (mode=full)."""
    from berlin_lst_downscaling.data.io import exists

    manifest_uri = cfg.get("manifest_uri") or f"{cfg.output_root}/manifest.parquet"
    if not exists(manifest_uri):
        raise FileNotFoundError(
            f"mode=full requires a manifest at {manifest_uri}. "
            "Run Szenen-Selektion first."
        )

    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.ard.ledger import pc_equal
    from berlin_lst_downscaling.data.io import read_bytes

    tbl = pq.read_table(io.BytesIO(read_bytes(manifest_uri)))
    if tbl.num_rows == 0:
        return

    mask = pc_equal(tbl.column("source"), source)
    rows = tbl.filter(mask)
    if rows.num_rows == 0:
        return

    # Collect manifest rows per source — always carry scene_id + year
    scenes: list[tuple[str, str, int]] = []
    scene_dates: dict[str, str | None] = {}  # scene_id → date (optional)
    scene_hrefs: dict[str, str | None] = {}  # scene_id → item_href (optional)
    for i in range(rows.num_rows):
        r = rows.slice(i, 1).to_pydict()
        sid = str(r["scene_id"][0])
        scenes.append((sid, source, int(r["year"][0])))
        # Optional date column (forward-compatible schema extension)
        _dt = r.get("date", [None])[0]
        scene_dates[sid] = str(_dt) if _dt is not None else None
        # Optional item_href (bypasses STAC search)
        _href = r.get("item_href", [None])[0]
        scene_hrefs[sid] = str(_href) if _href else None

    todo = reconcile(scenes, ledger, contract)
    _log(cfg, run_id, "manifest_todo", {
        "source": source,
        "total": len(scenes),
        "n_todo": len(todo),
    })

    for scene_id, _source, year, _reason in todo:
        _run_scene(
            scene_id, source, year, contract, cfg, ledger, run_id,
            scene_date=scene_dates.get(scene_id),
            item_href=scene_hrefs.get(scene_id),
        )


# ── per-scene processing ─────────────────────────────────────────────


def _run_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
    item_href: str | None = None,
) -> None:
    """Process one scene: load → mask → write COG+STAC → update ledger.

    Parameters
    ----------
    items :
        Pre-fetched STAC items for the scene. Passed to acquisition
        loaders when provided (manifest-driven ``mode=full``).
        When ``None`` (smoke mode), the loader searches by date.
    scene_date :
        Overrides ``cfg.scene_date`` for this scene. Used by
        ``mode=full`` where each manifest row carries its own date.
    item_href :
        Direct STAC item URL from the manifest. When provided, the
        runner can bypass STAC search (future optimization; currently
        falls back to date-based search).
    """
    effective_date = scene_date or cfg.scene_date
    _log(cfg, run_id, "scene_start", {
        "scene_id": scene_id,
        "source": source,
        "scene_date": effective_date,
        "item_href": item_href,
    })
    t0 = time.perf_counter()

    # Mark as exporting (crash recovery entry)
    # Note: attempts is auto-managed by upsert (increments from existing)
    ledger.upsert(
        LedgerRow(
            scene_id=scene_id,
            source=source,
            year=year,
            status="exporting",
            run_id=run_id,
        )
    )

    try:
        # ── LOAD & MASK via per-source runner ──
        runner = _RUNNERS.get(source)
        if runner is None:
            raise ValueError(f"Unknown source: {source}")

        masked = runner(
            scene_id=scene_id,
            source=source,
            year=year,
            contract=contract,
            cfg=cfg,
            ledger=ledger,
            run_id=run_id,
            items=items,
            scene_date=scene_date,
        )

        # ── WRITE MAIN COG ──
        _log(cfg, run_id, "scene_writing", {"scene_id": scene_id, "source": source})
        root = str(cfg.output_root)
        cog_dst = cog_path(root, source, year, scene_id)

        # Split data bands from flag band
        flag_da: xr.DataArray | None = masked.get("flag")
        data_bands = [v for v in masked.data_vars if v != "flag"]
        if data_bands:
            ds_data = masked[data_bands]
        else:
            ds_data = masked

        write_cog_atomic(ds_data, cog_dst, contract, overwrite=True)

        # ── WRITE FLAG COG (separate uint8 file) ──
        flag_dst = flag_path(root, source, year, scene_id)
        if flag_da is not None and contract.flag_mode == "separate":
            write_flag_cog_atomic(flag_da, flag_dst, contract, overwrite=True)

        # ── COMPUTE AOI METRICS ──
        if flag_da is not None and contract.flag_mode == "separate":
            aoi_res = (
                int(cfg.target_resolution_low)
                if source == "landsat-c2-l2"
                else int(cfg.target_resolution_high)
            )
            aoi_base = cfg.get("aoi.mask_base", "data/boundaries")
            aoi_uri = f"{aoi_base}/aoi_{aoi_res}m.tif"
            try:
                _raw = compute_aoi_metrics(flag_dst, aoi_uri, contract)
                _v = _raw["aoi_clear_px"]
                aoi_clear_px = None if _v is None else int(_v)
                _v = _raw["aoi_cloudy_px"]
                aoi_cloudy_px = None if _v is None else int(_v)
                _v = _raw["aoi_shadow_px"]
                aoi_shadow_px = None if _v is None else int(_v)
                _v = _raw["aoi_cirrus_px"]
                aoi_cirrus_px = None if _v is None else int(_v)
                _v = _raw["aoi_saturated_px"]
                aoi_saturated_px = None if _v is None else int(_v)
                _v = _raw["aoi_fill_px"]
                aoi_fill_px = None if _v is None else int(_v)
                _v = _raw["aoi_total_px"]
                aoi_total_px = None if _v is None else int(_v)
                _v = _raw["aoi_overlap_px"]
                aoi_overlap_px = None if _v is None else int(_v)
                _v = _raw["aoi_clear_frac"]
                aoi_clear_frac = None if _v is None else float(_v)
            except Exception as _exc:
                # AOI metrics are best-effort; log and continue without them
                _log(cfg, run_id, "aoi_metrics_error", {
                    "scene_id": scene_id,
                    "aoi_uri": aoi_uri,
                    "error": str(_exc),
                })
                aoi_clear_px = aoi_cloudy_px = aoi_shadow_px = None
                aoi_cirrus_px = aoi_saturated_px = aoi_fill_px = None
                aoi_total_px = aoi_overlap_px = aoi_clear_frac = None
        else:
            aoi_clear_px = aoi_cloudy_px = aoi_shadow_px = None
            aoi_cirrus_px = aoi_saturated_px = aoi_fill_px = None
            aoi_total_px = aoi_overlap_px = aoi_clear_frac = None

        # Low-overlap warning: valid data covers a small fraction of the AOI intersection.
        # This catches off-target swaths where the COG covers the AOI bbox but LST is NaN.
        min_overlap = cfg.get("aoi", {}).get("min_overlap_px", None)
        if aoi_overlap_px is not None and min_overlap is not None and aoi_overlap_px < min_overlap:
            _log(cfg, run_id, "low_aoi_overlap", {
                "scene_id": scene_id,
                "aoi_overlap_px": aoi_overlap_px,
                "min_overlap_px": min_overlap,
            })

        # ── BUILD + WRITE STAC ──
        stac_dst = stac_path(root, source, year, scene_id)
        stac_item = _build_stac_item(
            scene_id, source, year, masked, contract, cog_dst, cfg,
            flag_dst=flag_dst if contract.flag_mode == "separate" else None,
        )
        write_stac_atomic(stac_item, stac_dst, overwrite=True)

        # ── UPDATE LEDGER ──
        elapsed = time.perf_counter() - t0
        ledger.upsert(
            LedgerRow(
                scene_id=scene_id,
                source=source,
                year=year,
                path_cog=cog_dst,
                path_stac=stac_dst,
                status="done",
                schema_hash=contract.schema_hash(),
                schema_version=contract.schema_version,
                run_id=run_id,
                updated_at=datetime.now(UTC),
                aoi_clear_px=aoi_clear_px,
                aoi_cloudy_px=aoi_cloudy_px,
                aoi_shadow_px=aoi_shadow_px,
                aoi_cirrus_px=aoi_cirrus_px,
                aoi_saturated_px=aoi_saturated_px,
                aoi_fill_px=aoi_fill_px,
                aoi_total_px=aoi_total_px,
                aoi_overlap_px=aoi_overlap_px,
                aoi_clear_frac=aoi_clear_frac,
            )
        )
        _attempts = row.attempts if (row := ledger.get(scene_id, source)) else 0
        _log(cfg, run_id, "scene_done", {
            "scene_id": scene_id,
            "source": source,
            "attempts": _attempts,
            "elapsed_s": round(elapsed, 2),
        })

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _attempts = row.attempts if (row := ledger.get(scene_id, source)) else 0
        _log(cfg, run_id, "scene_failed", {
            "scene_id": scene_id,
            "source": source,
            "attempts": _attempts,
            "error": str(exc),
            "elapsed_s": round(elapsed, 2),
        })
        # Note: attempts is auto-managed by upsert
        ledger.upsert(
            LedgerRow(
                scene_id=scene_id,
                source=source,
                year=year,
                status="failed",
                last_error=str(exc),
                run_id=run_id,
                updated_at=datetime.now(UTC),
            )
        )
        # do not re-raise — outer loop continues


# ── solar position ───────────────────────────────────────────────────


def _solar_for_scene(
    ds: xr.Dataset,
    cfg: DictConfig,
    stac_properties: dict | None = None,
) -> tuple[float, float]:
    """Return ``(azimuth_deg, elevation_deg)`` for the scene.

    When ``stac_properties`` are provided and contain
    ``view:sun_azimuth`` / ``view:sun_elevation``, those values
    are used directly (no NOAA computation).
    """
    from berlin_lst_downscaling.data.ard.solar_position import extract_solar_from_stac

    # Try STAC properties first (used by Landsat which has view:sun_*)
    if stac_properties:
        solar = extract_solar_from_stac(stac_properties)
        if solar is not None:
            return solar

    # Fall back to NOAA computation from acquisition time
    try:
        dt64 = ds.time.values[0]
        ts = dt64.astype("datetime64[us]").tolist()
        dt = datetime.fromtimestamp(ts.timestamp(), tz=UTC)
    except (IndexError, AttributeError, ValueError):
        date_str = cfg.scene_date
        dt = datetime.fromisoformat(date_str).replace(
            hour=10, minute=0, second=0, tzinfo=UTC,
        )
    return solar_position(dt)


# ── STAC item builder ────────────────────────────────────────────────


def _build_stac_item(
    scene_id: str,
    source: str,
    year: int,
    masked: xr.Dataset,
    contract: Contract,
    cog_path_rel: str,
    cfg: DictConfig,
    flag_dst: str | None = None,
) -> dict[str, Any]:
    """Build a minimal STAC item describing one ARD COG.

    Parameters
    ----------
    flag_dst :
        URI to the separate flag COG (``.flag.tif``). When provided and
        ``contract.flag_mode == "separate"``, a ``flag`` asset is added
        pointing to this file.
    """
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    crs = masked.rio.crs
    geo_transform = masked.rio.transform()

    first_band = list(masked.data_vars)[0]
    height, width = masked[first_band].shape[-2:]

    bounds = array_bounds(height, width, geo_transform)
    bbox_4326 = transform_bounds(crs, "EPSG:4326", *bounds)

    resolution = (
        cfg.target_resolution_low if source == "landsat-c2-l2"
        else cfg.target_resolution_high
    )

    assets: dict[str, Any] = {}
    # Data bands from contract.output_bands (flag is separate)
    for spec in contract.output_bands:
        # NaN nodata → None in JSON (STAC spec compatibility)
        nodata = None if spec.nodata is not None and _is_nan(spec.nodata) else spec.nodata
        assets[spec.name] = {
            "href": cog_path_rel,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": spec.description,
            "raster:bands": [
                {
                    "data_type": spec.dtype,
                    "nodata": nodata,
                    "spatial_resolution": resolution,
                }
            ],
        }

    # Flag band as separate asset
    if flag_dst is not None and contract.flag_mode == "separate":
        assets["flag"] = {
            "href": flag_dst,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": "Quality flag (bitmask: fill, cloudy, shadow, cirrus, saturated)",
            "raster:bands": [
                {
                    "data_type": "uint8",
                    "nodata": None,
                    "spatial_resolution": resolution,
                }
            ],
        }

    # Real acquisition datetime from dataset (T11)
    acq_dt = _acquisition_datetime(masked, cfg, year)

    item: dict[str, Any] = {
        "stac_version": "1.0.0",
        "stac_extensions": ["projection", "raster"],
        "type": "Feature",
        "id": scene_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox_4326[0], bbox_4326[1]],
                    [bbox_4326[2], bbox_4326[1]],
                    [bbox_4326[2], bbox_4326[3]],
                    [bbox_4326[0], bbox_4326[3]],
                    [bbox_4326[0], bbox_4326[1]],
                ]
            ],
        },
        "properties": {
            "datetime": (
                acq_dt.isoformat()
                if acq_dt
                else f"{cfg.get('scene_date', str(year))}T00:00:00Z"
            ),
            "crs": str(crs),
            "proj:epsg": crs.to_epsg(),
            "proj:shape": [height, width],
            "proj:transform": list(geo_transform),
            "ard:schema_hash": contract.schema_hash(),
            "ard:schema_version": contract.schema_version,
            "ard:source": source,
            "ard:scene_id": scene_id,
        },
        "assets": assets,
        "links": [],
    }

    return item


# ── STAC helpers ─────────────────────────────────────────────────────


def _acquisition_datetime(
    masked: xr.Dataset,
    cfg: DictConfig,
    year: int,
) -> datetime | None:
    """Extract the real acquisition datetime from the dataset.

    Returns ``None`` if the dataset has no ``time`` coordinate or it
    cannot be parsed (caller should fall back to config date).
    """
    try:
        dt64 = masked.time.values[0]
        ts = dt64.astype("datetime64[us]").tolist()
        return datetime.fromtimestamp(ts.timestamp(), tz=UTC)
    except (IndexError, AttributeError, ValueError):
        return None


def _is_nan(val: float) -> bool:
    """Check if a float is NaN without importing math."""
    return val != val


# ── logging ──────────────────────────────────────────────────────────


def _log(cfg: DictConfig, run_id: str, event: str, data: dict[str, Any]) -> None:
    """Emit a structured JSON log line to stderr and append to run log file."""
    entry = {
        "run_id": run_id,
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **data,
    }
    line = json.dumps(entry)

    print(line, file=sys.stderr)

    logging_root = cfg.get("logging_dir", cfg.output_root)
    # For GCS output_root, default logs to local ./logs/ directory
    if str(logging_root).startswith("gs://"):
        logging_root = "./logs"
    log_dir = Path(logging_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


__all__ = [
    "run",
]
