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
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import xarray as xr
from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
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
from berlin_lst_downscaling.data.ard.validate import validate_cog, validate_flag_cog
from berlin_lst_downscaling.data.ard.writer import (
    write_cog_atomic,
    write_flag_cog_atomic,
    write_stac_atomic,
)
from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

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
    ecostress_raw_dir: str | None = None,  # unused by Landsat
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
    ecostress_raw_dir: str | None = None,  # unused by S2
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
    ecostress_raw_dir: str | None = None,
) -> xr.Dataset:
    """Load + mask an ECOSTRESS L2T granule from local COGs.

    No STAC search — scene_id comes from the manifest.  The raw_dir
    (``cfg.ecostress.raw_dir`` or ``ecostress_raw_dir`` override) is
    expected to contain one sub-directory per granule, each with
    ECO_L2T_LSTE layer COGs.
    """
    bbox = tuple(cfg.bbox) if cfg.get("bbox") else None
    resolution = int(cfg.get("target_resolution_low", 70))
    raw_dir = ecostress_raw_dir or str(cfg.ecostress.raw_dir)

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


def run(cfg: DictConfig, run_id: str | None = None) -> int:
    """Execute the ARD pipeline — manifest-driven only (mode=full).

    Smoke mode was removed in favor of manifest-driven smoke using
    ``smoke_primary`` config which builds a 3-row manifest then runs
    ``mode=full``.

    Returns 0 on success, 1 if any scene failed.
    """
    if run_id is None:
        run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)

    led = Ledger.open(f"{output_root}/ledger.parquet")
    log_event(_logger, logging.INFO, "start", mode=cfg.mode, sources=list(cfg.sources))

    sources = list(cfg.sources)

    for source in sources:
        contract = contract_for_source(source)
        _process_manifest(source, contract, cfg, led, run_id)

    # Final QA report
    report = qa_report(led, cfg, run_id)
    failed_count = sum(
        report.get("per_source", {}).get(s, {}).get("failed", 0) for s in sources
    )
    log_event(_logger, logging.INFO, "qa_report", **report)

    # Note: ledger persists per-transition via upsert — no batch write needed
    return 0 if failed_count == 0 else 1


def _process_manifest(
    source: str,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
) -> None:
    """Process scenes listed in a manifest Parquet (mode=full).

    Reads the v3 manifest schema, validates the bundle, and resolves
    exact STAC items from item_href before passing them to loaders.
    """
    from berlin_lst_downscaling.data.io import exists

    manifest_uri = cfg.get("manifest_uri") or f"{cfg.output_root}/manifest.parquet"
    if not exists(manifest_uri):
        raise FileNotFoundError(
            f"mode=full requires a manifest at {manifest_uri}. "
            "Run Szenen-Selektion first."
        )

    import pyarrow.compute as pc  # type: ignore[attr-defined]
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.io import read_bytes

    tbl = pq.read_table(io.BytesIO(read_bytes(manifest_uri)))
    if tbl.num_rows == 0:
        return

    # Apply scene filter if configured (for cloud smoke)
    scene_filter = cfg.get("scene_filter")
    if scene_filter:
        filter_ids = scene_filter.get(f"{source.replace('-', '_')}_ids", [])
        if source == "landsat-c2-l2":
            filter_ids = scene_filter.get("landsat_ids", [])
        elif source == "sentinel-2-l2a":
            filter_ids = scene_filter.get("s2_ids", [])
        elif source == "ecostress":
            filter_ids = scene_filter.get("ecostress_ids", [])

        if filter_ids:
            mask = pc.is_in(tbl.column("scene_id"), filter_ids)  # type: ignore[attr-defined]
            tbl = tbl.filter(mask)
            if tbl.num_rows == 0:
                log_event(_logger, logging.INFO, "no_scenes_after_filter",
                    source=source, filter_ids=filter_ids)
                return

    mask = pc.equal(tbl.column("source"), source)  # type: ignore[attr-defined]
    rows = tbl.filter(mask)
    if rows.num_rows == 0:
        return

    # Collect manifest rows — carry scene_id, year, item_href, acq time
    scenes: list[tuple[str, str, int]] = []
    scene_meta: dict[str, dict] = {}
    for i in range(rows.num_rows):
        r = rows.slice(i, 1).to_pydict()
        sid = str(r["scene_id"][0])
        yr = int(r["year"][0])
        scenes.append((sid, source, yr))

        # Extract v3 fields
        href = r.get("item_href", [None])[0]
        acq_dt = r.get("acquisition_datetime", [None])[0]
        scene_meta[sid] = {
            "item_href": href,
            "acquisition_datetime": acq_dt,
            "year": yr,
        }

    max_attempts = cfg.get("max_scene_attempts", 3)
    todo = reconcile(scenes, ledger, contract, max_attempts=max_attempts)
    log_event(_logger, logging.INFO, "manifest_todo",
        source=source,
        total=len(scenes),
        n_todo=len(todo),
    )

    if source == "ecostress":
        _process_ecostress_todo(
            todo, source, contract, cfg, ledger, run_id, scene_meta,
        )
    else:
        for scene_id, _source, year, _reason in todo:
            meta = scene_meta.get(scene_id, {})
            # Resolve exact STAC item from manifest HREF
            items = _resolve_manifest_items(scene_id, source, meta)
            _run_scene(
                scene_id, source, year, contract, cfg, ledger, run_id,
                items=items,
                scene_date=meta.get("acquisition_datetime"),
            )


def _resolve_manifest_items(
    scene_id: str,
    source: str,
    meta: dict,
) -> list[Any] | None:
    """Resolve exact STAC items from manifest metadata.

    For PC STAC sources (Landsat/Sentinel-2), tries to resolve directly
    from item_href first (faster, avoids catalog search). Falls back to
    ID-based search if item_href is missing.
    Returns None for ECOSTRESS or when resolution fails.
    """
    if source == "ecostress":
        return None

    from berlin_lst_downscaling.data.acquisition.pc_client import (
        resolve_exact_item,
        resolve_item_from_href,
    )

    item_href = meta.get("item_href")

    # Try direct HREF resolution first (preferred path)
    if item_href:
        try:
            item = resolve_item_from_href(item_href, expected_id=scene_id)
            return [item]
        except Exception as exc:
            log_event(_logger, logging.WARNING, "href_resolve_failed",
                scene_id=scene_id, item_href=item_href, error=str(exc),
            )

    # Fallback: resolve by ID from catalog
    try:
        item = resolve_exact_item(collection=source, scene_id=scene_id)
        return [item]
    except Exception as exc:
        log_event(_logger, logging.WARNING, "exact_item_resolve_failed",
            scene_id=scene_id, error=str(exc),
        )
        return None


def _process_ecostress_todo(
    todo: list[tuple[str, str, int, str]],
    source: str,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    scene_meta: dict[str, dict],
) -> None:
    """Process ECOSTRESS scenes with pipeline-internal staging.

    For each scene: download granule from NASA Earthdata → stage →
    process → cleanup.  Stage lifecycle is managed by ``StageSession``.
    """
    from berlin_lst_downscaling.data.acquisition.ecostress import (
        download_and_stage_granule,
    )
    from berlin_lst_downscaling.data.io.staging import StageSession

    stage_base = cfg.get("ecostress.stage_base", "data/smoke/ecostress_stage")
    with StageSession(stage_base, run_id=run_id) as stage:
        for scene_id, _src, year, _reason in todo:
            try:
                download_and_stage_granule(scene_id, stage)
                granule_raw_dir = str(stage.uri.uri)
            except Exception as exc:
                log_event(_logger, logging.WARNING, "scene_failed",
                    scene_id=scene_id,
                    source=source,
                    error=f"Stage download failed: {exc}",
                )
                continue

            _run_scene(
                scene_id, source, year, contract, cfg, ledger, run_id,
                scene_date=scene_meta.get(scene_id, {}).get("acquisition_datetime"),
                ecostress_raw_dir=granule_raw_dir,
            )
        # Stage cleaned up automatically on StageSession.__exit__


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
    ecostress_raw_dir: str | None = None,
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
    ecostress_raw_dir :
        Per-scene override for ECOSTRESS raw_dir. Used by the
        manifest-driven ECOSTRESS pipeline where each granule is
        downloaded and staged on demand.
    """
    effective_date = scene_date or cfg.scene_date
    log_event(_logger, logging.INFO, "scene_start",
        scene_id=scene_id,
        source=source,
        scene_date=effective_date,
    )
    t0 = time.perf_counter()

    # Mark as exporting (crash recovery entry) with attempt tracking
    _attempts = ledger.begin_attempt(
        LedgerRow(
            scene_id=scene_id,
            source=source,
            year=year,
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
            ecostress_raw_dir=ecostress_raw_dir,
        )

        # ── WRITE MAIN COG ──
        log_event(_logger, logging.INFO, "scene_writing", scene_id=scene_id, source=source)
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

        # ── COG STRUCTURAL VALIDATION ──
        # Determine expected canonical grid from source resolution
        if source == "landsat-c2-l2":
            expected_grid = canon_grid_for_resolution(100)
        elif source == "sentinel-2-l2a":
            expected_grid = canon_grid_for_resolution(10)
        elif source == "ecostress":
            expected_grid = canon_grid_for_resolution(70)
        else:
            expected_grid = canon_grid_for_resolution(10)

        vig = validate_cog(cog_dst, contract, expected_grid)
        if vig.ok:
            log_event(_logger, logging.INFO, "cog_validated", scene_id=scene_id, source=source)
        else:
            raise RuntimeError(
                f"COG validation failed: {'; '.join(vig.errors)}"
            )

        if flag_da is not None and contract.flag_mode == "separate":
            vif = validate_flag_cog(flag_dst, expected_grid)
            if not vif.ok:
                raise RuntimeError(
                    f"Flag COG validation failed: {'; '.join(vif.errors)}"
                )

        # ── COMPUTE AOI METRICS ──
        aoi_clear_px: int | None = None
        aoi_cloudy_px: int | None = None
        aoi_shadow_px: int | None = None
        aoi_cirrus_px: int | None = None
        aoi_saturated_px: int | None = None
        aoi_fill_px: int | None = None
        aoi_total_px: int | None = None
        aoi_overlap_px: int | None = None
        aoi_clear_frac: float | None = None
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
                aoi_clear_px = _int_or_none(_raw.get("aoi_clear_px"))
                aoi_cloudy_px = _int_or_none(_raw.get("aoi_cloudy_px"))
                aoi_shadow_px = _int_or_none(_raw.get("aoi_shadow_px"))
                aoi_cirrus_px = _int_or_none(_raw.get("aoi_cirrus_px"))
                aoi_saturated_px = _int_or_none(_raw.get("aoi_saturated_px"))
                aoi_fill_px = _int_or_none(_raw.get("aoi_fill_px"))
                aoi_total_px = _int_or_none(_raw.get("aoi_total_px"))
                aoi_overlap_px = _int_or_none(_raw.get("aoi_overlap_px"))
                aoi_clear_frac = _float_or_none(_raw.get("aoi_clear_frac"))
            except Exception as _exc:
                # AOI metrics are best-effort; log and continue without them
                log_event(_logger, logging.WARNING, "aoi_metrics_error",
                    scene_id=scene_id,
                    aoi_uri=aoi_uri,
                    error=str(_exc),
                )
        else:
            aoi_clear_px = aoi_cloudy_px = aoi_shadow_px = None
            aoi_cirrus_px = aoi_saturated_px = aoi_fill_px = None
            aoi_total_px = aoi_overlap_px = aoi_clear_frac = None

        # Low-overlap warning: valid data covers a small fraction of the AOI intersection.
        # This catches off-target swaths where the COG covers the AOI bbox but LST is NaN.
        min_overlap = cfg.get("aoi", {}).get("min_overlap_px", None)
        if aoi_overlap_px is not None and min_overlap is not None and aoi_overlap_px < min_overlap:
            log_event(_logger, logging.WARNING, "low_aoi_overlap",
                scene_id=scene_id,
                aoi_overlap_px=aoi_overlap_px,
                min_overlap_px=min_overlap,
            )

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
                path_flag=flag_dst if contract.flag_mode == "separate" else None,
                path_stac=stac_dst,
                status="done",
                schema_hash=contract.schema_version_str(),
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
        log_event(_logger, logging.INFO, "scene_done",
            scene_id=scene_id,
            source=source,
            attempts=_attempts,
            elapsed_s=round(elapsed, 2),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _attempts = row.attempts if (row := ledger.get(scene_id, source)) else 0
        log_event(_logger, logging.ERROR, "scene_failed",
            scene_id=scene_id,
            source=source,
            attempts=_attempts,
            error=str(exc),
            elapsed_s=round(elapsed, 2),
        )
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
) -> tuple[float, float]:
    """Return ``(azimuth_deg, elevation_deg)`` for the scene.

    Computed from the dataset's acquisition time via NOAA.  Falls back to
    ``cfg.scene_date`` at 10:00 UTC when the dataset has no time coordinate.
    """
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
            "ard:schema_version": contract.schema_version_str(),
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


def _int_or_none(val) -> int | None:
    """Convert value to int, returning None for None or NaN."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN
        return None
    return int(val)


def _float_or_none(val) -> float | None:
    """Convert value to float, returning None for None or NaN."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN
        return None
    return float(val)


__all__ = [
    "run",
]
