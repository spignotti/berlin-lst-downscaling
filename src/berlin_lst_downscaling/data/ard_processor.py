"""Reprojection, regridding, and GCS orchestration for the ARD pipeline.

Takes raw GEE/AppEEARS COG exports from GCS, reprojects/regrids them to
the canonical EPSG:25833 grid, and writes QA-validated ARD COGs back to GCS.

Source-to-grid mapping::

    Landsat     EPSG:25833  100m  →  Canonical 100m grid (regrid only)
    Sentinel-2  EPSG:25833  10m   →  Canonical 10m grid  (regrid only)
    ECOSTRESS   Native CRS  ~70m  →  Native CRS, ~70m    (passthrough, no reprojection)

Usage::

    from berlin_lst_downscaling.data.ard_processor import process_source
    results = process_source(cfg, "landsat", spec, year=2023, dry_run=False, smoke=True)
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
import rasterio.shutil
from affine import Affine
from omegaconf import DictConfig, OmegaConf
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.data.grid_spec import GridSpec

logger = logging.getLogger(__name__)

# shortcut: band-at-a-time memory optimization (replaces full multi-band array);
# ProcessPoolExecutor-based parallel processing with configurable max_workers.
# Windowed GCS reads deferred — not needed since GEE exports are already cropped.

# ── Public API ───────────────────────────────────────────────────────────


def list_scenes(cfg: DictConfig, source: str, year: int) -> list[str]:
    """List GCS blob URIs for a given source and year.

    Only returns files with valid raster extensions (``.tif``, ``.tiff``)
    to skip metadata files (CSVs, XMLs, etc.) that may coexist in the
    same prefix.

    Args:
        cfg: Pipeline config.
        source: ``"landsat"``, ``"sentinel2"``, or ``"ecostress"``.
        year: Year to list.

    Returns:
        Sorted list of ``gs://bucket/path`` URIs.
    """
    from google.cloud import storage

    bucket = cfg.ard.output.bucket
    prefix = f"{cfg.ard.process.sources[source].gcs_prefix}/{year}/"

    client = storage.Client()
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    raster_blobs = [b for b in blobs if b.name.lower().endswith((".tif", ".tiff"))]
    # Sort largest first — bigger files tend to have more valid data
    # (relevant for ECOSTRESS sparse swath tiles)
    raster_blobs.sort(key=lambda b: b.size or 0, reverse=True)
    uris = [f"gs://{bucket}/{b.name}" for b in raster_blobs]
    return uris


def process_source(
    cfg: DictConfig,
    source: str,
    spec: GridSpec,
    year: int | None = None,
    dry_run: bool = True,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Process all scenes for one source over the configured year range.

    Args:
        cfg: Pipeline config.
        source: Source name (``"landsat"``, ``"sentinel2"``, ``"ecostress"``).
        spec: Canonical grid specification.
        year: Single year, or ``None`` for all years in config range.
        dry_run: If ``True``, log planned actions without processing.
        smoke: If ``True``, process only 1 scene per year (stop for inspection).

    Returns:
        List of per-scene result dicts.
    """
    years = _resolve_years(cfg, year)
    total_results: list[dict[str, Any]] = []
    source_cfg = cfg.ard.process.sources[source]
    bucket = cfg.ard.output.bucket
    resume = cfg.ard.process.get("resume", False)

    for y in years:
        uris = list_scenes(cfg, source, y)
        if not uris:
            logger.info(
                "%s %s: no scenes found on GCS (prefix: %s/%s/)",
                source,
                y,
                source_cfg.gcs_prefix,
                y,
            )
            continue

        logger.info("%s %s: %s scene(s) available", source, y, len(uris))

        # Skip already-processed scenes if resume is enabled
        if resume and not dry_run:
            manifest = _read_manifest(bucket, source_cfg.output_prefix, y)
            completed_ids = set(manifest.get("completed", []))
            if completed_ids:
                uris_before = len(uris)
                uris = [u for u in uris if _parse_scene_id(u, source, cfg) not in completed_ids]
                skipped = uris_before - len(uris)
                if skipped:
                    logger.info("Resume: skipping %d already-processed scene(s)", skipped)

        if smoke:
            uris = uris[:1]
            logger.info("[SMOKE] limiting to 1 scene")

        # Decide execution mode
        parallel = not dry_run and cfg.ard.process.get("max_workers", 1) > 1 and len(uris) > 1

        if parallel:
            total_results.extend(
                _process_scenes_parallel(
                    uris,
                    source,
                    spec,
                    cfg,
                    y,
                    bucket,
                    source_cfg,
                    resume,
                    smoke,
                )
            )
            if smoke:
                # Smoke already stopped the parallel processing for this year
                return total_results
        else:
            for i, uri in enumerate(uris):
                result = process_scene(uri, source, spec, cfg, dry_run=dry_run)
                total_results.append(result)

                if dry_run:
                    continue

                scene_id = result.get("scene_id", "?")
                status = result.get("status", "?")
                logger.info("[%s/%s] %s → %s", i + 1, len(uris), scene_id, status)

                # Update completion manifest after each scene
                if resume and status == "success":
                    _update_manifest(
                        bucket, source_cfg.output_prefix, y, scene_id, status="completed"
                    )
                elif resume and status == "error":
                    err_msg = result.get("error", "unknown error")
                    _update_manifest(
                        bucket,
                        source_cfg.output_prefix,
                        y,
                        scene_id,
                        status="failed",
                        error=err_msg,
                    )

                if smoke and status == "success":
                    logger.info("SMOKE: Scene processed — stopping year loop.")
                    return total_results

        # Generate contact sheet from thumbnails uploaded this year
        if not dry_run:
            from berlin_lst_downscaling.data.quicklook import (
                generate_contact_sheet_from_gcs,
            )

            cs_output = Path(cfg.ard.process.temp_dir) / source / str(y) / "_contact_sheet.png"
            cs_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                generate_contact_sheet_from_gcs(
                    bucket=bucket,
                    prefix=source_cfg.output_prefix,
                    year=y,
                    output_path=cs_output,
                    cols=8,
                    thumb_width=256,
                )
                _upload_to_gcs(
                    cs_output,
                    f"{source_cfg.output_prefix}/{y}/_contact_sheet.png",
                    bucket,
                )
            except Exception as cs_exc:
                logger.warning("Contact sheet generation failed for %s %s: %s", source, y, cs_exc)

    return total_results


def process_scene(
    gcs_uri: str,
    source: str,
    spec: GridSpec,
    cfg: DictConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download, reproject/regrid, QA, and upload one scene.

    Args:
        gcs_uri: Input GCS blob URI (``gs://bucket/path.tif``).
        source: Source name.
        spec: Canonical grid specification.
        cfg: Pipeline config.
        dry_run: If ``True``, return planned paths without processing.

    Returns:
        Result dict with keys ``source``, ``scene_id``, ``output_path``,
        ``qa_report_path``, ``status``.
    """
    bucket_name = cfg.ard.output.bucket
    source_cfg = cfg.ard.process.sources[source]
    resampling_name = cfg.ard.process.resampling
    target_res = source_cfg.target_resolution
    temp_base = Path(cfg.ard.process.temp_dir)
    cog_cfg = cfg.ard.output.cog

    scene_id = _parse_scene_id(gcs_uri, source, cfg)
    year = _parse_year(gcs_uri)
    output_uri = f"gs://{bucket_name}/{source_cfg.output_prefix}/{year}/{scene_id}.tif"
    qa_uri = f"gs://{bucket_name}/{source_cfg.output_prefix}/{year}/{scene_id}_qa.json"

    result: dict[str, Any] = {
        "source": source,
        "scene_id": scene_id,
        "year": year,
        "input_uri": gcs_uri,
        "output_path": output_uri,
        "qa_report_path": qa_uri,
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    # ── Temp paths ──
    scene_temp = temp_base / source / str(year) / scene_id
    scene_temp.mkdir(parents=True, exist_ok=True)
    input_local = scene_temp / "input.tif"
    output_local = scene_temp / "output.tif"
    qa_local = scene_temp / "qa.json"

    try:
        # Step 1: Download
        _download_from_gcs(gcs_uri, input_local)

        # Step 2: Reproject / regrid
        _reproject_and_regrid(
            input_local,
            output_local,
            dst_crs=spec.crs,
            dst_resolution=target_res,
            dst_dtype=cog_cfg.dtype,
            resampling_name=resampling_name,
            spec=spec,
            cog_cfg=cog_cfg,
        )

        # Step 3: QA
        from berlin_lst_downscaling.data.ard_qa import generate_qa_report
        from berlin_lst_downscaling.data.boundary import landesgrenze_polygon_25833

        skip_grid = target_res is None  # ECOSTRESS: native CRS, skip grid check
        landesgrenze = landesgrenze_polygon_25833()
        qa_report = generate_qa_report(
            output_local,
            spec,
            target_resolution=target_res if target_res is not None else 0,
            cfg=cfg,
            scene_id=scene_id,
            skip_grid_check=skip_grid,
            landesgrenze_polygon=landesgrenze,
        )
        with open(qa_local, "w") as f:
            json.dump(qa_report, f, indent=2, default=str)

        result["qa_report"] = qa_report

        # Phase 2: Soft-warn instead of hard-fail. Coverage is now a warning
        # in qa_warnings; COG is uploaded regardless. strict_qa=true restores
        # the old hard-fail behaviour.
        if not qa_report.get("qa_passed", False):
            strict_qa = bool(cfg.ard.process.get("strict_qa", False))
            if strict_qa:
                raise ValueError(
                    f"QA failed for {scene_id} (strict_qa=true): "
                    f"aoi_coverage={qa_report.get('aoi_coverage_fraction', 0.0):.3f}"
                )
            logger.warning(
                "QA warnings for %s (strict_qa=false, uploading anyway): %s",
                scene_id,
                qa_report.get("qa_warnings", []),
            )

        # Step 4: Upload output COG
        _upload_to_gcs(
            output_local, f"{source_cfg.output_prefix}/{year}/{scene_id}.tif", bucket_name
        )

        # Step 5: Upload QA report
        _upload_to_gcs(
            qa_local,
            f"{source_cfg.output_prefix}/{year}/{scene_id}_qa.json",
            bucket_name,
        )

        # Step 6: Generate and upload STAC metadata item
        from berlin_lst_downscaling.data.stac_writer import write_stac_item

        # Derive GEE collection ID per source
        if source == "landsat":
            collections: list[str] = list(cfg.landsat.collections)
            collection_id = ", ".join(collections)
        elif source == "sentinel2":
            collection_id = str(cfg.sentinel2.collection)
        elif source == "ecostress":
            collection_id = f"{cfg.ecostress.product}/{cfg.ecostress.version}"
        else:
            collection_id = None

        stac_item = write_stac_item(
            cog_path=output_local,
            scene_id=scene_id,
            source=source,
            year=year,
            qa_report=qa_report,
            output_bucket=bucket_name,
            output_prefix=source_cfg.output_prefix,
            input_cog_path=input_local,
            config_dict=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
            collection_id=collection_id,
        )
        stac_local = scene_temp / "stac.json"
        with open(stac_local, "w") as f:
            json.dump(stac_item, f, indent=2, default=str)
        _upload_to_gcs(
            stac_local,
            f"{source_cfg.output_prefix}/{year}/{scene_id}_stac.json",
            bucket_name,
        )

        # Step 7: Generate and upload quicklook thumbnail
        from berlin_lst_downscaling.data.quicklook import generate_thumbnail

        thumbnail_local = scene_temp / "thumbnail.png"
        try:
            generate_thumbnail(output_local, thumbnail_local, width=512)
            _upload_to_gcs(
                thumbnail_local,
                f"{source_cfg.output_prefix}/{year}/thumbnails/{scene_id}.png",
                bucket_name,
            )
            result["thumbnail_path"] = str(thumbnail_local)
        except Exception as thumb_exc:
            logger.warning("Thumbnail generation failed for %s: %s", scene_id, thumb_exc)

        result["status"] = "success"
        result["stac_item"] = stac_item

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        # Cleanup
        shutil.rmtree(scene_temp, ignore_errors=True)

    return result


# ── Private helpers ──────────────────────────────────────────────────────


def _process_scenes_parallel(
    uris: list[str],
    source: str,
    spec: GridSpec,
    cfg: DictConfig,
    year: int,
    bucket: str,
    source_cfg: Any,
    resume: bool,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Process multiple scenes in parallel via ``ProcessPoolExecutor``.

    OmegaConf ``DictConfig`` is not reliably picklable, so the config is
    converted to a plain container dict before submitting to the pool
    and re-wrapped in the worker.
    """
    max_workers = int(cfg.ard.process.max_workers)
    total = len(uris)
    results: list[dict[str, Any]] = []
    completed_count = 0

    # Convert cfg to picklable container
    cfg_container = OmegaConf.to_container(cfg, resolve=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_scene_worker,
                uri,
                source,
                spec,
                cfg_container,
            ): uri
            for uri in uris
        }

        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed_count += 1

            scene_id = result.get("scene_id", "?")
            status = result.get("status", "?")
            logger.info("[%s/%s] %s → %s", completed_count, total, scene_id, status)

            # Update manifest from the main process (serial, no race)
            if resume and status == "success":
                _update_manifest(
                    bucket, source_cfg.output_prefix, year, scene_id, status="completed"
                )
            elif resume and status == "error":
                _update_manifest(
                    bucket,
                    source_cfg.output_prefix,
                    year,
                    scene_id,
                    status="failed",
                    error=result.get("error", "unknown error"),
                )

            if smoke and status == "success":
                logger.info("SMOKE: Scene processed — cancelling remaining futures.")
                # Cancel remaining futures and shutdown
                for f in future_map:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return results

    return results


def _process_scene_worker(
    gcs_uri: str,
    source: str,
    spec: GridSpec,
    cfg_container: Any,
) -> dict[str, Any]:
    """Wrapper around ``process_scene`` for use with ``ProcessPoolExecutor``.

    Re-wraps the container dict as a ``DictConfig`` and delegates.
    """
    cfg = OmegaConf.create(cfg_container)
    # Safe: cfg_container is always a dict for this pipeline config
    return process_scene(gcs_uri, source, spec, cfg, dry_run=False)  # type: ignore[arg-type]


def _download_from_gcs(gcs_uri: str, local_path: Path) -> Path:
    """Download a GCS blob to a local file."""
    from google.cloud import storage

    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    return local_path


def _upload_to_gcs(local_path: Path, gcs_path: str, bucket_name: str) -> str:
    """Upload a local file to GCS.

    Returns the GCS URI.
    """
    from google.cloud import storage

    uri = f"gs://{bucket_name}/{gcs_path}"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    return uri


def _reproject_and_regrid(
    src_path: Path,
    dst_path: Path,
    *,
    dst_crs: str,
    dst_resolution: float | None,
    dst_dtype: str,
    resampling_name: str,
    spec: GridSpec,
    cog_cfg: Any,  # DictConfig or dict-like
) -> Path:
    """Reproject and/or regrid a raster, writing a Cloud-Optimized GeoTIFF.

    Args:
        src_path: Input raster path.
        dst_path: Output COG path.
        dst_crs: Target CRS (e.g. ``"EPSG:25833"``).
        dst_resolution: Target pixel resolution in CRS units.
            ``None`` means keep native source resolution (ECOSTRESS).
        dst_dtype: Output data type (e.g. ``"float32"``).
        resampling_name: Resampling method name for
            ``rasterio.enums.Resampling``.
        spec: Canonical grid spec (used for origin alignment when
            regridding Landsat/S2).
        cog_cfg: Dict-like with keys ``compress``, ``tile_size``,
            ``nodata``.

    Returns:
        Path to the written COG.
    """
    # ECOSTRESS passthrough: preserve native CRS/resolution, apply COG profile
    if dst_resolution is None:
        return _copy_as_cog(src_path, dst_path, dst_dtype, cog_cfg)

    resampling = getattr(Resampling, resampling_name)

    with rasterio.open(src_path) as src:
        dst_transform, dst_width, dst_height = _compute_target_dims(
            src,
            spec,
            dst_crs,
            dst_resolution,
        )
        src_count = src.count

        # Build output profile
        profile = src.profile.copy()
        missing_val = _resolve_nodata(cog_cfg.nodata)

        # Strip keys inherited from the source that the COG driver
        # doesn't accept (COG handles tiling internally via blocksize).
        for _key in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(_key, None)

        cog_opts = dict(
            driver="COG",
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            dtype=dst_dtype,
            nodata=missing_val,
            compress=str(cog_cfg.compression),
            blocksize=int(cog_cfg.tile_size),
        )

        profile.update(**cog_opts)

        src_nodata = src.nodata

        # Nearest-neighbour for mask/flag/classification bands
        # (cloud_mask, lst_plausible, SCL) preserves categorical values.
        # Continuous bands use the configured resampling (bilinear).
        # If GEE exports don't set band descriptions, src.descriptions
        # returns empty strings and all bands use the configured resampling.
        mask_names = {"cloud_mask", "lst_plausible", "scl"}

        # Overview config (read before the write loop)
        ov_levels = cog_cfg.get("overview_levels")
        ov_resampling_str = cog_cfg.get("overview_resampling", "average")
        ov_resampling = getattr(Resampling, ov_resampling_str, Resampling.average)

        # Band-at-a-time processing to keep peak memory low.
        # For S2 (12 bands × 4980×4145 × float32), this drops peak
        # from ~990 MB (all bands) to ~165 MB (1 band per iteration).
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(src_count):
                band_desc = (src.descriptions[i] or "").lower()
                band_resamp = Resampling.nearest if band_desc in mask_names else resampling

                band_arr = np.zeros((dst_height, dst_width), dtype=dst_dtype)
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=band_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata=src_nodata,
                    dst_nodata=missing_val,
                    resampling=band_resamp,
                )
                # Safety net: catch any remaining pure nodata pixels
                if src_nodata is not None:
                    band_arr = np.where(
                        np.isclose(band_arr, src_nodata, rtol=1e-5),
                        missing_val,
                        band_arr,
                    )
                dst.write(band_arr, i + 1)
                dst.set_band_description(i + 1, src.descriptions[i] or f"band_{i + 1}")

            # Build overviews via GDAL internal overviews (avoids invalid
            # GDAL creation options for the COG driver on this platform).
            if ov_levels is not None:
                factors = [2 ** (i + 1) for i in range(int(ov_levels))]
                dst.build_overviews(factors, ov_resampling)
                dst.update_tags(ns="rio_overview", resampling=ov_resampling_str)

    return dst_path


def _copy_as_cog(src_path: Path, dst_path: Path, dst_dtype: str, cog_cfg: Any) -> Path:
    """Copy a raster as COG without reprojection, applying COG profile.

    Preserves source CRS, transform, and geometry. Applies compression,
    tiling, overviews, dtype, and nodata from ``cog_cfg``. Used for
    ECOSTRESS passthrough (keep native resolution/CRS).

    Args:
        src_path: Input raster path.
        dst_path: Output COG path.
        dst_dtype: Target data type.
        cog_cfg: Dict-like with keys ``compress``, ``tile_size``,
            ``nodata``.

    Returns:
        Path to the written COG.
    """
    missing_val = _resolve_nodata(cog_cfg.nodata)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

        # Strip keys the COG driver doesn't accept
        for _key in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(_key, None)

        profile.update(
            driver="COG",
            dtype=dst_dtype,
            nodata=missing_val,
            compress=str(cog_cfg.compression),
            blocksize=int(cog_cfg.tile_size),
        )

        ov_levels = cog_cfg.get("overview_levels")
        ov_resampling_str = cog_cfg.get("overview_resampling", "average")
        ov_resampling = getattr(Resampling, ov_resampling_str, Resampling.average)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(src.count):
                band = src.read(i + 1).astype(dst_dtype)
                band[~np.isfinite(band)] = missing_val
                dst.write(band, i + 1)

            # Build overviews via GDAL internal overviews
            if ov_levels is not None:
                factors = [2 ** (i + 1) for i in range(int(ov_levels))]
                dst.build_overviews(factors, ov_resampling)
                dst.update_tags(ns="rio_overview", resampling=ov_resampling_str)

    logger.info("COG passthrough (no reprojection): %s → %s", src_path, dst_path)
    return dst_path


def _compute_target_dims(
    src: rasterio.DatasetReader,
    spec: GridSpec,
    dst_crs: str,
    dst_resolution: float | None,
) -> tuple[Affine, int, int]:
    """Compute output transform and dimensions for reprojection.

    Args:
        src: Source raster dataset (must be open).
        spec: Canonical grid specification.
        dst_crs: Target CRS (e.g. ``"EPSG:25833"``).
        dst_resolution: Target pixel resolution in CRS units.

    For Landsat/S2 (``dst_resolution`` is set):
        The output is aligned to the canonical grid origin. The extent
        is the intersection of the source bounds + AOI bounds,
        snapped to the canonical grid.

    Note:
        ECOSTRESS passthrough (``dst_resolution=None``) is handled in
        ``_reproject_and_regrid`` and does not reach this function.
    """
    if dst_resolution is not None:
        # Source is already in EPSG:25833 — regrid to canonical origin
        # Intersect source bounds with AOI bounds (both same CRS)
        src_bounds = src.bounds
        xmin = max(src_bounds.left, spec.aoi_xmin)
        ymin = max(src_bounds.bottom, spec.aoi_ymin)
        xmax = min(src_bounds.right, spec.aoi_xmax)
        ymax = min(src_bounds.top, spec.aoi_ymax)

        # Align to canonical grid origin
        origin_x = spec.origin_x
        origin_y = spec.origin_y
        res = dst_resolution

        col_start = max(0, int((xmin - origin_x) / res))
        row_start = max(0, int((origin_y - ymax) / res))
        col_end = min(spec.width_10m, int((xmax - origin_x) / res + 0.5))
        row_end = min(spec.height_10m, int((origin_y - ymin) / res + 0.5))

        width = col_end - col_start
        height = row_end - row_start

        if width <= 0 or height <= 0:
            msg = f"Scene bounds ({src_bounds}) do not overlap AOI"
            raise ValueError(msg)

        transform = Affine(
            res,
            0,
            origin_x + col_start * res,
            0,
            -res,
            origin_y - row_start * res,
        )
        return transform, width, height

    # dst_resolution=None is handled in _reproject_and_regrid (ECOSTRESS passthrough)
    # and never reaches this function
    msg = "_compute_target_dims called with dst_resolution=None"
    raise ValueError(msg)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path`` into ``(bucket, path)``."""
    rest = uri.removeprefix("gs://")
    parts = rest.split("/", 1)
    return parts[0], parts[1]


def _parse_scene_id(gcs_uri: str, source: str, cfg: DictConfig) -> str:
    """Extract a stable scene identifier from a GCS blob path.

    Examples:
        ``gs://bucket/ard/dynamic/landsat/2023/LC08_XXXX_LST.tif``
        → ``LC08_XXXX``

        ``gs://bucket/ard/dynamic/sentinel2/2023/20230101T...tif``
        → ``20230101T...``

        ``gs://bucket/ard/validation/ecostress/2023/ECO_L2T_XXXX_COG.tif``
        → ``ECO_L2T_XXXX``
    """
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    stem = Path(blob_path).stem

    # Strip source-specific suffixes
    for suffix in ["_LST", "_COG"]:
        if stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
            break

    return stem


def _parse_year(gcs_uri: str) -> int:
    """Extract the 4-digit year from a GCS blob path."""
    import re

    m = re.search(r"/20\d{2}/", gcs_uri)
    if m:
        return int(m.group(0).strip("/"))
    return 0


def _resolve_nodata(nodata_val: Any) -> float:
    """Resolve nodata value to a float."""
    if nodata_val == "nan":
        return float("nan")
    return float(nodata_val)


def _resolve_years(cfg: DictConfig, year: int | None) -> list[int]:
    """Return a list of years to process."""
    if year is not None:
        return [year]
    return list(range(cfg.ard.time.start_year, cfg.ard.time.end_year + 1))


def _read_manifest(bucket: str, prefix: str, year: int) -> dict[str, Any]:
    """Read the completion manifest from GCS, or return empty if missing."""
    from google.cloud import storage

    client = storage.Client()
    manifest_path = f"{prefix}/{year}/_manifest.json"
    blob = client.bucket(bucket).blob(manifest_path)
    if not blob.exists():
        return {"completed": [], "failed": {}, "last_updated": None}
    content = blob.download_as_text()
    return json.loads(content)


def _update_manifest(
    bucket: str,
    prefix: str,
    year: int,
    scene_id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Append a scene result to the GCS completion manifest.

    Reads the existing manifest, appends the scene, and writes back.
    Safe for sequential processing; for parallel workers, a lock or
    per-scene manifest files should be used instead.
    """
    from datetime import datetime, timezone

    from google.cloud import storage

    client = storage.Client()
    manifest_path = f"{prefix}/{year}/_manifest.json"
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(manifest_path)

    if blob.exists():
        content = blob.download_as_text()
        manifest = json.loads(content)
    else:
        manifest = {"completed": [], "failed": {}, "last_updated": None}

    if status == "completed":
        if scene_id not in manifest["completed"]:
            manifest["completed"].append(scene_id)
        # Remove from failed if previously errored
        manifest["failed"].pop(scene_id, None)
    elif status == "failed":
        manifest["failed"][scene_id] = error or "unknown error"

    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    blob.upload_from_string(json.dumps(manifest, indent=2))
