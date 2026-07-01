"""ECOSTRESS download via AppEEARS — COG conversion and GCS upload.

The pipeline for each year:

1. Query CMR for available granules (for informational counts).
2. Submit one AppEEARS area task for the Berlin AOI and year.
3. Poll until the task completes.
4. List bundle files and download each GeoTIFF to a temp directory.
5. Convert each GeoTIFF to Cloud-Optimized GeoTIFF (COG) with
   consistent tiling, overviews, dtype and NoData.
6. Upload COGs to GCS at ``ard/validation/ecostress/{year}/``.
7. Clean up local temp files.

Usage::

    export_ecostress_by_year(cfg, year=2023, dry_run=True)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import rasterio
import rasterio.shutil
from omegaconf import DictConfig

from berlin_lst_downscaling.data.appeears_client import AppEEARSClient
from berlin_lst_downscaling.data.boundary import buffered_bbox_wgs84
from berlin_lst_downscaling.data.ecostress_scenes import (
    list_ecostress_granules,
    summarize_granules,
)
from berlin_lst_downscaling.data.gee_client import get_aoi_geojson_from_cfg

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_APPEARS_LAYER_MAP = {
    "LST": "LST",
    "cloud": "cloud",
    "QC": "QC",
}


def _build_appeears_task_name(year: int) -> str:
    """Human-readable task name for AppEEARS."""
    return f"ecostress-berlin-{year}"


def _build_appeears_dates(year: int, months: list[int] | None = None) -> list[dict[str, Any]]:
    """Build AppEEARS date specification for a single year.

    Each month gets its own entry with full start/end dates and
    recurring=True with the year range. AppEEARS expects MM-DD-YYYY
    when recurring=False, or MM-DD with recurring=True + yearRange.

    Args:
        year: Target year.
        months: List of months (1-12). If None, defaults to May–Sep.
    """
    if months is None:
        months = [5, 6, 7, 8, 9]

    def _last_day(m: int) -> str:
        """Return last day of month as DD string."""
        if m == 2:
            return "28"
        if m in (4, 6, 9, 11):
            return "30"
        return "31"

    return [
        {
            "startDate": f"{m:02d}-01",
            "endDate": f"{m:02d}-{_last_day(m)}",
            "recurring": True,
            "yearRange": [year, year],
        }
        for m in sorted(set(months))
    ]


def _build_appeears_layers(cfg: DictConfig) -> list[dict[str, str]]:
    """Build AppEEARS layer specifications from config."""
    product = f"{cfg.ecostress.product}.{cfg.ecostress.version}"
    return [
        {"product": product, "layer": layer}
        for layer in cfg.ecostress.appeears.layers
    ]


# ── COG conversion ────────────────────────────────────────────────────────────


def _convert_to_cog(
    src_path: Path,
    dst_path: Path,
    cfg: DictConfig,
) -> Path:
    """Convert a plain GeoTIFF to a Cloud-Optimized GeoTIFF.

    Uses ``rasterio.shutil.copy`` with the COG driver, which handles
    tiling, compression, and overview generation automatically.

    The source is expected to be float32 with NaN nodata (as delivered
    by AppEEARS).

    Args:
        src_path: Input GeoTIFF path.
        dst_path: Output COG path.
        cfg: Pipeline config (uses ``ecostress.cog`` section).

    Returns:
        Path to the generated COG.
    """
    cog_cfg = cfg.ecostress.cog
    tile_size = int(cog_cfg.tile_size)
    compress = str(cog_cfg.compress)

    # Build COG creation keyword arguments
    cog_kwargs: dict[str, Any] = {
        "driver": "COG",
        "compress": compress,
        "blocksize": tile_size,
    }

    ov_levels = cog_cfg.get("overview_levels")
    if ov_levels is not None:
        cog_kwargs["overview_levels"] = int(ov_levels)
    ov_resampling = cog_cfg.get("overview_resampling")
    if ov_resampling is not None:
        cog_kwargs["overview_resampling"] = str(ov_resampling)

    rasterio.shutil.copy(
        str(src_path),
        str(dst_path),
        **cog_kwargs,  # type: ignore[arg-type]
    )

    logger.info("COG written: %s", dst_path)
    return dst_path


# ── GCS upload ────────────────────────────────────────────────────────────────


def _upload_to_gcs(
    local_path: Path,
    gcs_blob_path: str,
    bucket_name: str,
    dry_run: bool = False,
) -> str | None:
    """Upload a local file to GCS.

    Args:
        local_path: Local file path.
        gcs_blob_path: GCS destination path (e.g. ``ard/validation/ecostress/2023/scene_LST.tif``).
        bucket_name: GCS bucket name.
        dry_run: If True, log the planned upload without executing.

    Returns:
        The GCS URI (``gs://bucket/path``) or ``None`` in dry-run mode.
    """
    uri = f"gs://{bucket_name}/{gcs_blob_path}"
    if dry_run:
        logger.info("[DRY-RUN] Would upload %s → %s", local_path, uri)
        return None

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)

    blob.upload_from_filename(str(local_path))
    logger.info("Uploaded: %s", uri)
    return uri


# ── Main export function ──────────────────────────────────────────────────────


def export_ecostress_by_year(
    cfg: DictConfig,
    year: int | None = None,
    dry_run: bool = True,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Download ECOSTRESS data for a given year via AppEEARS and produce COGs in GCS.

    The workflow is:
    1. CMR query (informational counts).
    2. Submit AppEEARS area task.
    3. Poll until complete.
    4. Download bundle to temp dir.
    5. Convert each GeoTIFF to COG.
    6. Upload COGs to GCS.
    7. Cleanup temp dir.

    Args:
        cfg: Pipeline config.
        year: Specific year to process. If ``None``, process all years in the
            config range (``cfg.ecostress.time.start_year`` to
            ``cfg.ecostress.time.end_year``).
        dry_run: If ``True``, print planned actions without submitting API calls.
        smoke: If ``True``, limit to 1 month and 1 file.

    Returns:
        List of result dicts with keys ``year``, ``gcs_paths``, and optionally
        ``error``.
    """
    years = _resolve_years(cfg, year)
    bucket = cfg.ard.output.bucket
    prefix = cfg.ecostress.export.prefix
    temp_base = Path(cfg.ecostress.export.temp_dir)
    wgs84_bbox = list(buffered_bbox_wgs84(cfg.ard.aoi.boundary_file))

    results: list[dict[str, Any]] = []

    for y in years:
        print(f"\n{'='*60}")
        print(f"ECOSTRESS {y}" + (" [DRY-RUN]" if dry_run else ""))
        print(f"{'='*60}")

        # Select months: in smoke mode, only the first month
        months = list(cfg.ecostress.time.months)
        if smoke:
            months = months[:1]
            print(f"  [SMOKE] limiting to month {months[0]:02d}")

        # ── Step 1: CMR query (informational) ──
        granules = list_ecostress_granules(
            wgs84_bbox=wgs84_bbox,
            start_year=y,
            end_year=y,
            months=months,
        )
        summary = summarize_granules(granules)
        print(f"  CMR granules: {summary['total']}")
        if not granules:
            print(f"  No ECOSTRESS granules found for {y}. Skipping.")
            continue

        # ── Step 2–3: Submit AppEEARS task ──
        if dry_run:
            print(f"  [DRY-RUN] Would submit AppEEARS task: {_build_appeears_task_name(y)}")
            print(f"  [DRY-RUN] Would download ~{summary['total']} files to {temp_base / str(y)}")
            results.append({
                "year": y,
                "status": "dry_run",
                "granules": summary["total"],
                "gcs_paths": [
                    f"gs://{bucket}/{prefix}/{y}/scene-{i}_LST.tif"
                    for i in range(min(3, summary["total"]))
                ],
            })
            continue

        # Real execution path
        try:
            result = _execute_year(
                cfg, y, granules, bucket, prefix, temp_base, wgs84_bbox, smoke=smoke,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("ECOSTRESS export failed for year %d", y)
            results.append({"year": y, "status": "error", "error": str(exc)})

    return results


def _execute_year(
    cfg: DictConfig,
    year: int,
    granules: list[dict[str, Any]],
    bucket: str,
    prefix: str,
    temp_base: Path,
    wgs84_bbox: list[float],
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute a full download + COG + upload cycle for one year.

    Called by ``export_ecostress_by_year`` when ``dry_run=False``.
    """

    client = AppEEARSClient()
    client._ensure_auth()

    temp_dir = temp_base / str(year)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Build and submit AppEEARS task
    geo_json = get_aoi_geojson_from_cfg(cfg)
    task_name = _build_appeears_task_name(year)
    layers = _build_appeears_layers(cfg)
    months = list(cfg.ecostress.time.months)
    if smoke:
        months = months[:1]
    dates = _build_appeears_dates(year, months=months)

    task_id = client.submit_area_task(
        name=task_name,
        geo_json=geo_json,
        layers=layers,
        dates=dates,
        output_format=cfg.ecostress.appeears.output_format,
        projection=cfg.ecostress.appeears.output_projection,
        filename_date=cfg.ecostress.appeears.filename_date,
    )
    print(f"  AppEEARS task submitted: {task_id}")

    # Poll until done
    print("  Waiting for task to complete...")
    client.wait_for_task(
        task_id,
        poll_interval_sec=int(cfg.ecostress.appeears.poll_interval_sec),
        timeout_hours=int(cfg.ecostress.appeears.timeout_hours),
    )
    print("  Task completed!")

    # List and download files
    bundle_files = client.list_bundle_files(task_id)
    print(f"  Bundle files: {len(bundle_files)}")

    # Separate GeoTIFF from metadata files
    def _is_tiff(f: dict[str, Any]) -> bool:
        return f["file_name"].lower().endswith((".tif", ".tiff"))
    tif_files = [f for f in bundle_files if _is_tiff(f)]
    meta_files = [f for f in bundle_files if not _is_tiff(f)]
    print(f"  GeoTIFF: {len(tif_files)}, metadata: {len(meta_files)}")

    # Optionally limit number of TIFFs (for testing)
    limit: int | None = cfg.ecostress.export.get("limit", None)
    if smoke:
        limit = 1  # smoke mode overrides: process only 1 file
    if limit is not None and limit < len(tif_files):
        tif_files = tif_files[:limit]
        print(f"  [LIMIT={limit}] processing {len(tif_files)} GeoTIFF(s)")

    gcs_paths: list[str] = []

    # ── Group TIFs by acquisition (datetime + aid + utm zone) ──
    # AppEEARS delivers LST/cloud/QC as separate files per acquisition.
    # We group by the acquisition suffix and combine LST+cloud into a
    # single 2-band COG; QC is uploaded as-is for QC analysis.
    acquisitions = _group_tifs_by_acquisition(tif_files)
    print(f"  Acquisitions: {len(acquisitions)}")
    for acq_id, layers in acquisitions.items():
        print(f"    {acq_id}: {sorted(layers.keys())}")

    # ── Download + process each acquisition ──
    for acq_idx, (acq_id, layers) in enumerate(acquisitions.items()):
        print(f"  [{acq_idx+1}/{len(acquisitions)}] Processing acquisition {acq_id}...")
        downloaded: dict[str, Path] = {}
        for layer_name, file_info in layers.items():
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            raw_path = temp_dir / f"{acq_id}_{layer_name}.tif"
            client.download_file(task_id, file_id, raw_path)
            if not raw_path.is_file() or raw_path.stat().st_size == 0:
                logger.error(
                    "Downloaded file missing/empty: %s (layer=%s)", raw_path, layer_name
                )
                continue
            downloaded[layer_name] = raw_path

        if "LST" not in downloaded:
            logger.warning("Acquisition %s has no LST band, skipping", acq_id)
            for p in downloaded.values():
                p.unlink(missing_ok=True)
            continue

        # ── Build combined 2-band COG (LST + cloud) ──
        lst_path = downloaded["LST"]
        cloud_path = downloaded.get("cloud")
        if cloud_path is not None:
            combined_path = temp_dir / f"{acq_id}_COG.tif"
            try:
                _combine_lst_cloud_to_cog(
                    lst_path, cloud_path, combined_path, cfg
                )
                upload_path = combined_path
                combined_name = f"{acq_id}_COG.tif"
            except Exception as exc:
                logger.warning(
                    "LST+cloud combine failed for %s, falling back to LST-only: %s",
                    acq_id, exc,
                )
                _convert_to_cog(lst_path, lst_path.with_name(f"{acq_id}_COG.tif"), cfg)
                upload_path = lst_path.with_name(f"{acq_id}_COG.tif")
                combined_name = f"{acq_id}_COG.tif"
        else:
            _convert_to_cog(lst_path, lst_path.with_name(f"{acq_id}_COG.tif"), cfg)
            upload_path = lst_path.with_name(f"{acq_id}_COG.tif")
            combined_name = f"{acq_id}_COG.tif"

        gcs_path = f"{prefix}/{year}/{combined_name}"
        uri = _upload_to_gcs(upload_path, gcs_path, bucket)
        if uri:
            gcs_paths.append(uri)

        # Upload standalone cloud + QC layers for QA / debug
        for layer_name, lp in downloaded.items():
            if layer_name == "LST" or layer_name == "cloud":
                continue  # already in combined COG
            gcs_layer_path = f"{prefix}/{year}/{acq_id}_{layer_name}.tif"
            _upload_to_gcs(lp, gcs_layer_path, bucket)

        # Cleanup
        for p in downloaded.values():
            p.unlink(missing_ok=True)
        upload_path.unlink(missing_ok=True)

        # In smoke mode, only process first acquisition
        if smoke:
            print("  [SMOKE] 1 acquisition processed — stopping.")
            break

    # ── Upload metadata files as-is ──
    for i, file_info in enumerate(meta_files):
        file_id = file_info["file_id"]
        file_name = file_info["file_name"]
        raw_path = temp_dir / file_name

        print(f"  [meta {i+1}/{len(meta_files)}] {file_name}...")
        client.download_file(task_id, file_id, raw_path)

        gcs_path = f"{prefix}/{year}/{file_name}"
        uri = _upload_to_gcs(raw_path, gcs_path, bucket)
        if uri:
            gcs_paths.append(uri)

        raw_path.unlink(missing_ok=True)

    # Cleanup temp dir for this year
    shutil.rmtree(temp_dir, ignore_errors=True)
    client.logout()

    return {
        "year": year,
        "status": "success",
        "task_id": task_id,
        "files_downloaded": len(bundle_files),
        "gcs_paths": gcs_paths,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _group_tifs_by_acquisition(
    tif_files: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Group TIF bundle files by acquisition id, keyed by layer name.

    AppEEARS file naming pattern::

        ECO_L2T_LSTE.002_{layer}_{datetime}_aid{N}_{zone}.tif

    The acquisition id is the suffix starting at the datetime, e.g.
    ``20230501T045756_aid0001_32N``. Files without the standard pattern
    are dropped with a warning.

    Returns:
        ``{acq_id: {layer_name: file_info}}``
    """
    import re

    pattern = re.compile(
        r"^ECO_L2T_LSTE\.\d+_(?P<layer>[A-Za-z]+)_(?P<acq>\d+T\d+_aid\d+_[A-Z0-9]+)\.tif$"
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for f in tif_files:
        m = pattern.match(f["file_name"])
        if not m:
            logger.warning("Skipping file with unexpected name: %s", f["file_name"])
            continue
        layer = m.group("layer")
        acq_id = m.group("acq")
        grouped.setdefault(acq_id, {})[layer] = f
    return grouped


def _combine_lst_cloud_to_cog(
    lst_path: Path,
    cloud_path: Path,
    dst_path: Path,
    cfg: DictConfig,
) -> Path:
    """Combine LST and cloud GeoTIFFs into a single 2-band COG.

    Both inputs are expected to share the same CRS, transform, and shape
    (which AppEEARS guarantees for layers of the same acquisition). The
    output has band descriptions ``["LST", "cloud_mask"]`` so the QA
    module and the smoke visualization can find the cloud band by
    description.

    Args:
        lst_path: Path to the LST GeoTIFF.
        cloud_path: Path to the cloud GeoTIFF.
        dst_path: Output 2-band COG path.
        cfg: Pipeline config (uses ``ecostress.cog`` section).

    Returns:
        Path to the written 2-band COG.
    """
    cog_cfg = cfg.ecostress.cog
    import numpy as np
    import rasterio
    import rasterio.shutil  # type: ignore[attr-defined]
    from rasterio.enums import Resampling

    with rasterio.open(lst_path) as lst_src, rasterio.open(cloud_path) as cld_src:
        if lst_src.shape != cld_src.shape:
            raise ValueError(
                f"LST and cloud shape mismatch: {lst_src.shape} vs {cld_src.shape}"
            )
        if str(lst_src.crs) != str(cld_src.crs):
            raise ValueError(
                f"LST and cloud CRS mismatch: {lst_src.crs} vs {cld_src.crs}"
            )

        profile = lst_src.profile.copy()
        profile.update(count=2, dtype="float32")
        if "blockxsize" in profile:
            profile.pop("blockxsize")
        if "blockysize" in profile:
            profile.pop("blockysize")
        if "tiled" in profile:
            profile.pop("tiled")

        nodata = float("nan")
        profile["nodata"] = nodata

        with rasterio.open(dst_path, "w", **profile) as dst:
            # Band 1: LST (passthrough)
            lst_data = lst_src.read(1).astype("float32")
            if lst_src.nodata is not None and not np.isnan(lst_src.nodata):
                lst_data = np.where(
                    np.isclose(lst_data, lst_src.nodata, rtol=1e-5), nodata, lst_data
                )
            dst.write(lst_data, 1)
            dst.set_band_description(1, "LST")

            # Band 2: cloud (1=clear, 0=cloud). AppEEARS encodes cloud as
            # values 0/1; pass through and apply NaN for nodata.
            cld_data = cld_src.read(1).astype("float32")
            if cld_src.nodata is not None and not np.isnan(cld_src.nodata):
                cld_data = np.where(
                    np.isclose(cld_data, cld_src.nodata, rtol=1e-5), nodata, cld_data
                )
            dst.write(cld_data, 2)
            dst.set_band_description(2, "cloud_mask")

            # Overviews
            ov_levels = cog_cfg.get("overview_levels")
            if ov_levels:
                factors = [2 ** (i + 1) for i in range(int(ov_levels))]
                resampling = getattr(
                    Resampling, str(cog_cfg.get("overview_resampling", "average")),
                    Resampling.average,
                )
                dst.build_overviews(factors, resampling)
                dst.update_tags(ns="rio_overview", resampling=str(resampling))

        # Re-write as proper COG with compression
        cog_tmp = dst_path.with_suffix(".cog.tmp")
        rasterio.shutil.copy(
            str(dst_path),
            str(cog_tmp),
            driver="COG",
            compress=str(cog_cfg.compress),
            blocksize=int(cog_cfg.tile_size),
        )
        cog_tmp.replace(dst_path)

    return dst_path


def _resolve_years(cfg: DictConfig, year: int | None) -> list[int]:
    """Return a list of years to process (config range or single year)."""
    if year is not None:
        return [year]
    return list(
        range(
            int(cfg.ecostress.time.start_year),
            int(cfg.ecostress.time.end_year) + 1,
        )
    )
