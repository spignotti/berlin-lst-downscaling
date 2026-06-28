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

    Returns:
        List of result dicts with keys ``year``, ``gcs_paths``, and optionally
        ``error``.
    """
    years = _resolve_years(cfg, year)
    bucket = cfg.ard.output.bucket
    prefix = cfg.ecostress.export.prefix
    temp_base = Path(cfg.ecostress.export.temp_dir)
    wgs84_bbox = list(cfg.ard.aoi.wgs84_bbox)

    results: list[dict[str, Any]] = []

    for y in years:
        print(f"\n{'='*60}")
        print(f"ECOSTRESS {y}" + (" [DRY-RUN]" if dry_run else ""))
        print(f"{'='*60}")

        # ── Step 1: CMR query (informational) ──
        granules = list_ecostress_granules(
            wgs84_bbox=wgs84_bbox,
            start_year=y,
            end_year=y,
            months=list(cfg.ecostress.time.months),
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
            result = _execute_year(cfg, y, granules, bucket, prefix, temp_base, wgs84_bbox)
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
    dates = _build_appeears_dates(year, months=list(cfg.ecostress.time.months))

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
    if limit is not None and limit < len(tif_files):
        tif_files = tif_files[:limit]
        print(f"  [LIMIT={limit}] processing {len(tif_files)} GeoTIFF(s)")

    gcs_paths: list[str] = []

    # ── Process GeoTIFF files → COG → upload ──
    for i, file_info in enumerate(tif_files):
        file_id = file_info["file_id"]
        file_name = file_info["file_name"]
        raw_path = temp_dir / file_name

        print(f"  [{i+1}/{len(tif_files)}] Downloading {file_name}...")
        client.download_file(task_id, file_id, raw_path)

        # Pre-flight check: file must exist and be non-empty
        if not raw_path.is_file():
            logger.error("Downloaded file missing: %s", raw_path)
            continue
        if raw_path.stat().st_size == 0:
            logger.error("Downloaded file empty: %s", raw_path)
            raw_path.unlink(missing_ok=True)
            continue

        # Convert to COG
        cog_name = file_name.replace(".tif", "_COG.tif").replace(".TIF", "_COG.tif")
        cog_path = temp_dir / cog_name
        cog_ok = False
        try:
            _convert_to_cog(raw_path, cog_path, cfg)
            cog_ok = True
            upload_path = cog_path
        except Exception as exc:
            logger.warning("COG conversion failed for %s: %s", file_name, exc)
            # Fallback: upload raw file with original name (no _COG suffix)
            upload_path = raw_path
            cog_name = file_name

        # Upload to GCS
        gcs_path = f"{prefix}/{year}/{cog_name}"
        uri = _upload_to_gcs(upload_path, gcs_path, bucket)
        if uri:
            gcs_paths.append(uri)

        # Cleanup
        if cog_ok:
            raw_path.unlink(missing_ok=True)
        upload_path.unlink(missing_ok=True)
        # Also remove the alternate path if it still exists
        alt = cog_path if upload_path == raw_path else raw_path
        alt.unlink(missing_ok=True)

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
