"""Area-exact 10 m rasterization of building footprints.

Each building contributes its polygon intersection with a grid cell, not
a binary mark on cells whose center falls inside the footprint. A
building that misses every cell center still covers the cells it
overlaps, so garages and sheds are not dropped.

Per cell
--------
- ``building_coverage_ratio`` is the sum of intersection areas divided
  by the 100 m² cell area, clipped to ``[0, 1]``. Overlapping footprints
  are summed and then clipped. Berlin cadastral footprints rarely
  overlap; parent and part geometries are not both emitted (see
  :mod:`berlin_lst_downscaling.data.secondary.citygml`).
- ``building_height_mean`` is the mean of building heights weighted by
  those intersection areas.
- ``building_height_std`` is the area-weighted population standard
  deviation of the same heights. A cell touched by one building has
  standard deviation 0.
- ``building_height_max`` is the maximum height among buildings with a
  positive intersection. A missing height is never written as 0; the
  caller skips that building and counts it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio.transform
import rasterio.windows
from shapely import area as shapely_area
from shapely import box as shapely_box
from shapely import intersection as shapely_intersection
from shapely.geometry.base import BaseGeometry

CELL_AREA_M2: float = 100.0
_MIN_INTERSECTION_M2: float = 1e-6


def _empty_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty_i = np.array([], dtype=np.intp)
    empty_a = np.array([], dtype=np.float64)
    return empty_i, empty_i.copy(), empty_a


def cell_intersections(
    footprint: BaseGeometry,
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, cols, area_m2)`` for cells the footprint overlaps.

    *shape* is ``(height, width)`` of the destination array. The
    transform must be north-up (positive pixel width, negative pixel
    height), which is the canonical EPSG:25833 grid.
    """
    height, width = shape
    if transform.a <= 0 or transform.e >= 0:
        raise ValueError("building raster expects a north-up transform (positive a, negative e)")
    if footprint is None or footprint.is_empty:
        return _empty_indices()

    minx, miny, maxx, maxy = footprint.bounds
    if not all(np.isfinite(value) for value in (minx, miny, maxx, maxy)):
        return _empty_indices()
    try:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    except Exception:
        return _empty_indices()

    col0 = max(int(np.floor(window.col_off)), 0)
    row0 = max(int(np.floor(window.row_off)), 0)
    col1 = min(int(np.ceil(window.col_off + window.width)), width)
    row1 = min(int(np.ceil(window.row_off + window.height)), height)
    if col1 <= col0 or row1 <= row0:
        return _empty_indices()

    cols = np.arange(col0, col1)
    rows = np.arange(row0, row1)
    col_grid, row_grid = np.meshgrid(cols, rows)
    left = transform.c + col_grid * transform.a + row_grid * transform.b
    top = transform.f + col_grid * transform.d + row_grid * transform.e
    right = left + transform.a
    bottom = top + transform.e
    cells = shapely_box(left, bottom, right, top)
    try:
        overlaps = shapely_intersection(cells, footprint)
        areas = np.asarray(shapely_area(overlaps), dtype=np.float64)
    except Exception:
        return _empty_indices()

    keep = np.isfinite(areas) & (areas > _MIN_INTERSECTION_M2)
    if not np.any(keep):
        return _empty_indices()
    flat_keep = keep.ravel()
    return (
        row_grid.ravel()[flat_keep],
        col_grid.ravel()[flat_keep],
        areas.ravel()[flat_keep],
    )


def accumulate_building(
    footprint: BaseGeometry,
    height_m: float,
    *,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    sum_h: np.ndarray,
    sum_h2: np.ndarray,
    sum_area: np.ndarray,
    count: np.ndarray,
    max_h: np.ndarray,
) -> bool:
    """Add one building's area-weighted height and coverage.

    Returns True when at least one cell received a positive intersection.
    Non-finite or non-positive heights are ignored and add nothing: a
    missing height must not become a 0 m building.
    """
    if footprint is None or not np.isfinite(height_m) or height_m <= 0.0:
        return False
    rows, cols, areas = cell_intersections(footprint, transform, shape)
    if areas.size == 0:
        return False
    weighted = height_m * areas
    np.add.at(sum_h, (rows, cols), weighted)
    np.add.at(sum_h2, (rows, cols), height_m * weighted)
    np.add.at(sum_area, (rows, cols), areas)
    np.add.at(count, (rows, cols), 1)
    np.maximum.at(max_h, (rows, cols), height_m)
    return True


def morphology_bands(
    sum_h: np.ndarray,
    sum_h2: np.ndarray,
    sum_area: np.ndarray,
    count: np.ndarray,
    max_h: np.ndarray,
) -> dict[str, np.ndarray]:
    """Render accumulated statistics as the four morphology bands.

    Cells with no intersecting building are NaN in every band. The
    coverage ratio is clipped to ``[0, 1]`` after footprints are summed.
    """
    covered = count > 0
    safe_area = np.where(sum_area > 0.0, sum_area, np.nan)
    mean = sum_h / safe_area
    variance = sum_h2 / safe_area - mean * mean
    variance = np.where(count > 1, variance, 0.0)
    std = np.sqrt(np.maximum(variance, 0.0))
    coverage = np.clip(sum_area / CELL_AREA_M2, 0.0, 1.0)
    return {
        "building_height_mean": np.where(covered, mean, np.nan).astype(np.float32),
        "building_height_std": np.where(covered, std, np.nan).astype(np.float32),
        "building_coverage_ratio": np.where(covered, coverage, np.nan).astype(np.float32),
        "building_height_max": np.where(covered, max_h, np.nan).astype(np.float32),
    }


def new_accumulators(shape: tuple[int, int]) -> dict[str, Any]:
    """Allocate the statistics arrays for one grid."""
    return {
        "sum_h": np.zeros(shape, dtype=np.float64),
        "sum_h2": np.zeros(shape, dtype=np.float64),
        "sum_area": np.zeros(shape, dtype=np.float64),
        "count": np.zeros(shape, dtype=np.int32),
        "max_h": np.zeros(shape, dtype=np.float32),
    }


__all__ = [
    "CELL_AREA_M2",
    "accumulate_building",
    "cell_intersections",
    "morphology_bands",
    "new_accumulators",
]
