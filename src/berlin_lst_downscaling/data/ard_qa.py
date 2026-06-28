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
from omegaconf import DictConfig

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

        mask_band = src.read(src.count)  # last band is cloud mask
        valid = ~np.isnan(mask_band)
        if not valid.any():
            return -1.0

        cloud_pixels = np.sum((mask_band < 0.5) & valid)
        total_valid = np.sum(valid)
        return float(cloud_pixels / total_valid) if total_valid > 0 else 0.0


def detect_cohort_outliers(
    reports: list[dict],
    z_score_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Detect radiometric outliers across a cohort of QA reports.

    For each numeric statistic (min, max, mean, std) across all scenes,
    computes the z-score for each scene. Scenes with ``|z| > threshold``
    in any band/statistic are flagged.

    Args:
        reports: List of QA report dicts (each from ``generate_qa_report``).
        z_score_threshold: Z-score threshold for flagging.

    Returns:
        List of ``{"scene_id": str, "flags": list[str]}`` for outliers.
    """
    if not reports:
        return []

    from collections import defaultdict

    # Collect: {band: {stat: {scene_id: value}}}
    stats_map: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    scene_ids: list[str] = []
    for report in reports:
        sid: str = str(report.get("scene_id", "?"))
        scene_ids.append(sid)
        per_band = report.get("radiometric_stats", {})
        for band, s in per_band.items():
            if not isinstance(s, dict):
                continue
            for stat_key in ("min", "max", "mean", "std"):
                val = s.get(stat_key)
                if val is not None:
                    stats_map[band][stat_key][sid] = float(val)

    outliers: list[dict[str, Any]] = []
    for sid in scene_ids:
        flags: list[str] = []
        for band, stat_dict in stats_map.items():
            for stat_key, scene_vals in stat_dict.items():
                values = list(scene_vals.values())
                if len(values) < 3:
                    continue
                mean_v = float(np.mean(values))
                std_v = float(np.std(values, ddof=0))
                if std_v < 1e-12:
                    continue
                val = scene_vals.get(sid)
                if val is None:
                    continue
                z = abs(val - mean_v) / std_v
                if z > z_score_threshold:
                    flags.append(f"{band}/{stat_key}: z={z:.1f}")

        if flags:
            outliers.append({"scene_id": sid, "flags": flags})

    return outliers


def check_grid_conformity(
    raster_path: Path,
    spec: GridSpec,
    target_resolution: float,
    tolerance: float = 1e-6,
) -> dict[str, bool | dict]:
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
) -> dict:
    """Generate a full QA report for a processed ARD COG.

    Combines grid conformity check, radiometric statistics, and
    cloud fraction. Optionally includes ``scene_id`` for cohort
    outlier detection.

    Args:
        raster_path: Path to the raster.
        spec: Canonical grid specification.
        target_resolution: Expected pixel resolution.
        cfg: Pipeline config (used for ``qa.nodata_threshold``).
        scene_id: Scene identifier (optional, for cohort analysis).
        skip_grid_check: If ``True``, skip grid conformity check.
            Used for native-CRS sources (ECOSTRESS) where reprojection
            is not performed.

    Returns:
        JSON-serializable QA report dict.
    """
    nodata_threshold = float(cfg.ard.process.qa.nodata_threshold)

    stats = compute_radiometric_stats(raster_path, nodata_threshold=nodata_threshold)
    cloud_pct = compute_cloud_fraction(raster_path)

    if skip_grid_check:
        grid = {
            "checked": False,
            "reason": "native CRS source — grid conformity not applicable",
        }
        # For sparse-swath sources (ECOSTRESS), any valid pixel is sufficient.
        # The standard nodata_threshold (0.95) is too strict for narrow swaths.
        qa_passed = any(b.get("nodata_pct", 1.0) < 1.0 for b in stats.values())
    else:
        grid = check_grid_conformity(raster_path, spec, target_resolution)
        qa_passed = (
            grid.get("crs_match", False)
            and grid.get("resolution_match", False)
        )

    report: dict = {
        "grid_conformity": grid,
        "radiometric_stats": stats,
        "cloud_fraction": cloud_pct,
        "qa_passed": qa_passed,
    }

    if scene_id is not None:
        report["scene_id"] = scene_id

    return report
