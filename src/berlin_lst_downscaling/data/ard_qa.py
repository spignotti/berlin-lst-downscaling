"""Quality assurance for reprojected ARD COGs.

Provides grid-conformity checks and radiometric statistics for
pipeline outputs. No GCS or GEE dependencies — pure rasterio + numpy.

Typical usage::

    from berlin_lst_downscaling.data.ard_qa import generate_qa_report
    report = generate_qa_report("/path/to/cog.tif", spec, target_resolution=100, cfg=cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.io
import rasterio.windows
from omegaconf import DictConfig
from rasterio.mask import raster_geometry_mask

from berlin_lst_downscaling.data.grid_spec import GridSpec

# ── Public API ───────────────────────────────────────────────────────────


def compute_cloud_fraction(raster_path: Path) -> float:
    """Compute fraction of cloud pixels from the last band.

    The last band is assumed to be the cloud mask (1 = clear, 0 = cloud),
    matching the GEE export convention in this pipeline.

    Returns:
        Fraction of pixels flagged as cloud (0.0 to 1.0).
        Returns ``-1.0`` if the raster has fewer than 2 bands.
    """
    with rasterio.open(raster_path) as src:
        if src.count < 2:
            return -1.0

        mask_band = src.read(src.count, masked=True)  # last band is cloud mask
        valid = ~mask_band.mask
        if not valid.any():
            return -1.0

        cloud_pixels = np.sum((mask_band < 0.5) & valid)
        total_valid = np.sum(valid)
        return float(cloud_pixels / total_valid) if total_valid > 0 else 0.0


def compute_aoi_coverage_fraction(raster_path: Path, spec: GridSpec) -> float:
    """Compute the fraction of AOI pixels covered by valid raster data.

    AOI bounds are in ``spec.crs`` (EPSG:25833 by default). For native-CRS
    sources (ECOSTRESS, EPSG:32632), the AOI bounds are reprojected into
    the raster CRS before computing the window, so the fraction reflects
    true AOI overlap regardless of the raster's CRS.
    """
    with rasterio.open(raster_path) as src:
        geom = [spec.aoi_polygon_25833]
        if str(src.crs) != str(spec.crs):
            import geopandas as gpd  # noqa: PLC0415

            geom = [
                gpd.GeoSeries([spec.aoi_polygon_25833], crs=spec.crs)
                .to_crs(src.crs)
                .iloc[0]
            ]

        try:
            geom_mask, _, window = raster_geometry_mask(src, geom, crop=True)
        except ValueError:
            return 0.0

        total_pixels = int(np.count_nonzero(~geom_mask))
        if total_pixels == 0:
            return 0.0

        band = src.read(1, window=window, masked=True)
        valid_pixels = int(np.count_nonzero((~geom_mask) & (~np.ma.getmaskarray(band))))
        return float(valid_pixels / total_pixels)


def check_grid_conformity(
    raster_path: Path,
    spec: GridSpec,
    target_resolution: float,
    tolerance: float = 1e-2,
) -> dict[str, Any]:
    """Verify CRS, resolution, and origin alignment against the spec.

    Args:
        raster_path: Path to the raster to check.
        spec: Canonical grid specification.
        target_resolution: Expected pixel resolution in CRS units (e.g. 10 or 100).
        tolerance: Relative tolerance for origin alignment (in CRS units).

    Returns:
        Dict with keys ``crs_match``, ``resolution_match``, ``origin_ok``,
        ``bounds_ok``, and ``actual`` / ``expected`` sub-dicts.
    """
    with rasterio.open(raster_path) as src:
        expected_crs = spec.crs
        expected_res = float(target_resolution)
        actual_crs = str(src.crs)
        actual_res_x = abs(src.transform.a)
        actual_res_y = abs(src.transform.e)

        # CRS check
        crs_match = actual_crs == expected_crs

        # Resolution check
        res_x_ok = abs(actual_res_x - expected_res) / expected_res < 0.01
        res_y_ok = abs(actual_res_y - expected_res) / expected_res < 0.01
        resolution_match = res_x_ok and res_y_ok

        # Origin check — should be aligned to canonical grid
        origin_x = src.transform.c
        origin_y = src.transform.f
        origin_gap_x = (origin_x - spec.origin_x) % expected_res
        origin_gap_y = (spec.origin_y - origin_y) % expected_res
        origin_ok = (
            min(origin_gap_x, expected_res - origin_gap_x) < tolerance
            and min(origin_gap_y, expected_res - origin_gap_y) < tolerance
        )

        # Bounds check — should overlap AOI
        bounds = src.bounds
        bounds_ok = (
            bounds.left < spec.aoi_xmax
            and bounds.right > spec.aoi_xmin
            and bounds.bottom < spec.aoi_ymax
            and bounds.top > spec.aoi_ymin
        )

        return {
            "crs_match": crs_match,
            "resolution_match": resolution_match,
            "origin_ok": origin_ok,
            "bounds_ok": bounds_ok,
            "actual": {
                "crs": actual_crs,
                "resolution_x": round(actual_res_x, 4),
                "resolution_y": round(actual_res_y, 4),
                "origin_x": round(origin_x, 4),
                "origin_y": round(origin_y, 4),
                "bounds": {
                    "left": round(bounds.left, 4),
                    "bottom": round(bounds.bottom, 4),
                    "right": round(bounds.right, 4),
                    "top": round(bounds.top, 4),
                },
            },
            "expected": {
                "crs": expected_crs,
                "resolution": expected_res,
                "origin_x": spec.origin_x,
                "origin_y": spec.origin_y,
                "aoi_bounds": {
                    "xmin": spec.aoi_xmin,
                    "ymin": spec.aoi_ymin,
                    "xmax": spec.aoi_xmax,
                    "ymax": spec.aoi_ymax,
                },
            },
        }


def compute_radiometric_stats(
    raster_path: Path,
    nodata_threshold: float = 0.95,
) -> dict[str, dict]:
    """Compute per-band radiometric statistics.

    Each band is read with masked nodata. Bands with >``nodata_threshold``
    fraction of nodata pixels are reported but skipped from statistics.

    Args:
        raster_path: Path to the raster.
        nodata_threshold: Max fraction of nodata pixels allowed
            for meaningful statistics.

    Returns:
        Dict mapping band indices (1-based) or band descriptions to
        ``{"min":, "max":, "mean":, "std":, "nodata_pct":, "valid": bool}``.
    """
    with rasterio.open(raster_path) as src:
        stats: dict[str, dict] = {}

        for i in range(1, src.count + 1):
            data = src.read(i, masked=True)
            nodata_count = int(np.ma.count_masked(data))
            total_pixels = data.size
            nodata_pct = nodata_count / total_pixels if total_pixels > 0 else 1.0

            band_desc = src.descriptions[i - 1] or f"band_{i}"

            if nodata_pct > nodata_threshold or nodata_count == total_pixels:
                stats[band_desc] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "nodata_pct": round(nodata_pct, 4),
                    "valid": False,
                }
                continue

            valid = data[~data.mask]
            stats[band_desc] = {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=0)),
                "nodata_pct": round(nodata_pct, 4),
                "valid": True,
            }

        return stats


def generate_qa_report(
    raster_path: Path,
    spec: GridSpec,
    target_resolution: float,
    cfg: DictConfig,
    scene_id: str | None = None,
    skip_grid_check: bool = False,
    landesgrenze_polygon: object | None = None,
) -> dict:
    """Generate a full QA report for a processed ARD COG.

    Combines grid conformity check, radiometric statistics, and
    cloud fraction. Optionally includes ``scene_id`` for cohort
    outlier detection.

    The report contains both ``aoi_coverage_fraction`` (valid pixels
    inside the buffered AOI polygon) and ``city_coverage_fraction``
    (valid pixels inside the Berlin Landesgrenze, no buffer). When a
    ``landesgrenze_polygon`` is supplied, also reports ``clear_pixel_count``
    and ``city_total_pixels`` for ML/training use.

    Args:
        raster_path: Path to the raster.
        spec: Canonical grid specification.
        target_resolution: Expected pixel resolution.
        cfg: Pipeline config (used for ``qa.nodata_threshold``).
        scene_id: Scene identifier (optional, for cohort analysis).
        skip_grid_check: If ``True``, skip grid conformity check.
            Used for native-CRS sources (ECOSTRESS) where reprojection
            is not performed.
        landesgrenze_polygon: Optional Berlin-Landesgrenze polygon in the
            raster CRS. If provided, additional city-level metrics are
            computed.

    Returns:
        JSON-serializable QA report dict.
    """
    nodata_threshold = float(cfg.ard.process.qa.nodata_threshold)
    # Fail-safe default matches the documented value in
    # ``configs/ard/ard_process.yaml`` so that a missing/typo'd config key
    # cannot silently disable coverage gating.
    min_aoi_coverage = float(cfg.ard.process.qa.get("min_aoi_coverage", 0.80))

    stats = compute_radiometric_stats(raster_path, nodata_threshold=nodata_threshold)
    cloud_pct = compute_cloud_fraction(raster_path)
    aoi_coverage_fraction = compute_aoi_coverage_fraction(raster_path, spec)

    city_metrics = _compute_city_metrics(raster_path, landesgrenze_polygon)
    city_coverage_fraction = city_metrics["city_coverage_fraction"]
    city_total_pixels = city_metrics["city_total_pixels"]
    clear_pixel_count = city_metrics["clear_pixel_count"]

    if skip_grid_check:
        grid = {
            "checked": False,
            "reason": "native CRS source — grid conformity not applicable",
            "aoi_coverage_fraction": aoi_coverage_fraction,
        }
        # For sparse-swath sources (ECOSTRESS), any valid pixel is sufficient.
        # The standard nodata_threshold (0.95) is too strict for narrow swaths.
        qa_passed = any(b.get("nodata_pct", 1.0) < 1.0 for b in stats.values())
    else:
        grid = check_grid_conformity(raster_path, spec, target_resolution)
        grid["aoi_coverage_fraction"] = aoi_coverage_fraction
        qa_passed = (
            grid.get("crs_match", False)
            and grid.get("resolution_match", False)
            and aoi_coverage_fraction >= min_aoi_coverage
        )

    qa_warnings: list[str] = []
    if not skip_grid_check and aoi_coverage_fraction < min_aoi_coverage:
        qa_warnings.append(
            f"low_aoi_coverage: {aoi_coverage_fraction:.3f} < {min_aoi_coverage:.3f}"
        )
    if clear_pixel_count is not None and city_total_pixels:
        clear_fraction = clear_pixel_count / city_total_pixels
        if clear_fraction < min_aoi_coverage:
            qa_warnings.append(
                f"low_clear_fraction: {clear_fraction:.3f} < {min_aoi_coverage:.3f}"
            )

    report: dict = {
        "grid_conformity": grid,
        "radiometric_stats": stats,
        "cloud_fraction": cloud_pct,
        "aoi_coverage_fraction": aoi_coverage_fraction,
        "city_coverage_fraction": city_coverage_fraction,
        "city_total_pixels": city_total_pixels,
        "clear_pixel_count": clear_pixel_count,
        "qa_passed": qa_passed,
        "qa_warnings": qa_warnings,
    }

    if scene_id is not None:
        report["scene_id"] = scene_id

    return report


def _compute_city_metrics(
    raster_path: Path,
    landesgrenze_polygon: object | None,
) -> dict[str, float | int | None]:
    """Compute city-level coverage metrics inside the Landesgrenze polygon.

    Returns a dict with three keys:
        city_coverage_fraction: float — valid pixels inside the polygon
            divided by the polygon's pixel count in the raster grid.
        city_total_pixels: int — total number of raster pixels that fall
            inside the polygon (denominator).
        clear_pixel_count: int | None — valid AND not-cloudy pixels
            inside the polygon. ``None`` if the raster has no cloud_mask
            band.
    """
    if landesgrenze_polygon is None:
        return {
            "city_coverage_fraction": 0.0,
            "city_total_pixels": 0,
            "clear_pixel_count": None,
        }

    from rasterio.mask import raster_geometry_mask

    with rasterio.open(raster_path) as src:
        try:
            geom_mask, _, window = raster_geometry_mask(
                src, [landesgrenze_polygon], crop=True
            )
        except ValueError:
            return {
                "city_coverage_fraction": 0.0,
                "city_total_pixels": 0,
                "clear_pixel_count": None,
            }

        city_total_pixels = int(np.count_nonzero(~geom_mask))
        if city_total_pixels == 0:
            return {
                "city_coverage_fraction": 0.0,
                "city_total_pixels": 0,
                "clear_pixel_count": None,
            }

        band = src.read(1, window=window, masked=True)
        inside_mask = ~geom_mask
        valid_in_city = int(np.count_nonzero(inside_mask & (~np.ma.getmaskarray(band))))
        city_coverage_fraction = float(valid_in_city / city_total_pixels)

        # Cloud-mask band is identified by description "cloud_mask"
        cloud_mask = _try_read_cloud_mask(src, window)
        if cloud_mask is not None:
            not_cloud = cloud_mask >= 0.5
            clear_mask = inside_mask & (~np.ma.getmaskarray(band)) & not_cloud
            clear_pixel_count = int(np.count_nonzero(clear_mask))
        else:
            clear_pixel_count = None

    return {
        "city_coverage_fraction": city_coverage_fraction,
        "city_total_pixels": city_total_pixels,
        "clear_pixel_count": clear_pixel_count,
    }


def _try_read_cloud_mask(
    src: rasterio.io.DatasetReader,
    window: rasterio.windows.Window | None,
) -> np.ndarray | None:
    """Return the cloud_mask band as a 2D array, or None if absent.

    The cloud_mask band is identified by ``src.descriptions`` containing
    the literal ``"cloud_mask"``. Multi-band rasters that do not declare
    a cloud_mask band (e.g. single-band ECOSTRESS without Phase 3
    cloud band) return ``None``.
    """
    for i in range(1, src.count + 1):
        desc = (src.descriptions[i - 1] or "").lower()
        if desc == "cloud_mask":
            return src.read(i, window=window, masked=False)
    return None
