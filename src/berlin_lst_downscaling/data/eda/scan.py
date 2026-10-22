"""Per-scene reads of the published Landsat target and the V3 feature stack.

The training release stores cell identity, not LST or channel values. This
module reads those rasters and returns one row per eligible 100 m cell.
It does not write a training artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.common.grid import canon_grid_10m, canon_grid_100m
from berlin_lst_downscaling.data.eda.rules import GAP_CLASSES, KELVIN_OFFSET
from berlin_lst_downscaling.data.features.contracts import schema_matching_count
from berlin_lst_downscaling.data.qa.inventory import ResolvedScene
from berlin_lst_downscaling.data.qa.stage1_raw import _tile_windows, _window_offset
from berlin_lst_downscaling.data.stadtstruktur.sources import BLOCK_CELLS
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
    cell_id,
)
from berlin_lst_downscaling.data.training.eligibility import (
    EligibilityResult,
    compute_eligibility,
)

_TILE = 2560


@dataclass
class SceneScan:
    """Raster products for one scene."""

    observations: pd.DataFrame
    gap_counts: dict[str, int]
    gap_cells: dict[str, np.ndarray]
    provenance_target_valid: int
    provenance_eligible: int


def _origin_shift(grid: GeoBox) -> tuple[int, int]:
    row0 = round((CANON_GRID_ORIGIN_Y - grid.transform.yoff) / CELL_SIZE_M)
    col0 = round((grid.transform.xoff - CANON_GRID_ORIGIN_X) / CELL_SIZE_M)
    return int(row0), int(col0)


def _check_grid(provenance: dict[str, object], grid: GeoBox) -> None:
    recorded = provenance.get("grid")
    if not isinstance(recorded, dict):
        raise ValueError("provenance has no grid")
    shape = recorded.get("shape")
    origin = recorded.get("origin")
    expected = [grid.shape.y, grid.shape.x]
    if shape != expected:
        raise ValueError(f"provenance grid shape {shape} != canonical {expected}")
    if not isinstance(origin, list) or len(origin) != 2:
        raise ValueError("provenance grid origin is missing")
    if abs(float(origin[0]) - grid.transform.xoff) > 1e-6:
        raise ValueError("provenance grid x origin differs from the canonical grid")
    if abs(float(origin[1]) - grid.transform.yoff) > 1e-6:
        raise ValueError("provenance grid y origin differs from the canonical grid")


def _gap_masks(result: EligibilityResult) -> dict[str, np.ndarray]:
    target = result.target_valid
    support = result.support
    present = result.present
    full = present == 100
    return {
        "target_valid_support_100": target & (support == 100) & full,
        "target_valid_support_lt_100": target & (support < 100) & full,
        "target_invalid_support_100": (~target) & (support == 100) & full,
        "edge_lt_100": present < 100,
    }


def _cell_pairs(mask: np.ndarray, row0: int, col0: int) -> np.ndarray:
    local_row, local_col = np.where(mask)
    if local_row.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pairs = np.empty((local_row.size, 2), dtype=np.int32)
    pairs[:, 0] = local_row + row0
    pairs[:, 1] = local_col + col0
    return pairs


def _feature_means(
    uri: str,
    analysis_10: GeoBox,
    eligible: np.ndarray,
    local_row: np.ndarray,
    local_col: np.ndarray,
) -> np.ndarray:
    """Mean of the 100 valid 10 m pixels in each eligible cell."""
    with rasterio.open(uri) as src:
        schema = schema_matching_count(src.count)
    means = np.full((local_row.size, schema.n_channels), np.nan, dtype=np.float64)
    if local_row.size == 0:
        return means
    pairs = zip(local_row, local_col, strict=True)
    index = {(int(row), int(col)): position for position, (row, col) in enumerate(pairs)}
    with rasterio.open(uri) as src:
        if src.count != schema.n_channels:
            raise ValueError(f"{uri} has {src.count} bands, expected {schema.n_channels}")
        column_off, row_off = _window_offset(src, analysis_10, 10.0)
        for (r0, c0, r1, c1), _window in _tile_windows(analysis_10, _TILE):
            block_row0, block_col0 = r0 // 10, c0 // 10
            block_row1, block_col1 = r1 // 10, c1 // 10
            if block_row1 <= block_row0 or block_col1 <= block_col0:
                continue
            if not np.any(eligible[block_row0:block_row1, block_col0:block_col1]):
                continue
            height = (block_row1 - block_row0) * 10
            width = (block_col1 - block_col0) * 10
            window = Window(c0 + column_off, r0 + row_off, width, height)  # type: ignore[call-arg]
            data = src.read(window=window).astype(np.float64, copy=False)
            if data.shape[1] != height or data.shape[2] != width:
                raise ValueError(f"feature window {data.shape} != {(height, width)}")
            reduced = data.reshape(
                data.shape[0],
                block_row1 - block_row0,
                10,
                block_col1 - block_col0,
                10,
            ).mean(axis=(2, 4))
            hits = np.argwhere(eligible[block_row0:block_row1, block_col0:block_col1])
            for local_y, local_x in hits:
                position = index[(block_row0 + int(local_y), block_col0 + int(local_x))]
                means[position, :] = reduced[:, int(local_y), int(local_x)]
    return means


def _landsat_values(
    uri: str,
    analysis_100: GeoBox,
    local_row: np.ndarray,
    local_col: np.ndarray,
) -> np.ndarray:
    if local_row.size == 0:
        return np.zeros(0, dtype=np.float64)
    with rasterio.open(uri) as src:
        values = src.read(1).astype(np.float64, copy=False)
        column_off, row_off = _window_offset(src, analysis_100, 100.0)
    source_row = local_row + row_off
    source_col = local_col + column_off
    if source_row.min() < 0 or source_col.min() < 0:
        raise ValueError("Landsat window does not cover an eligible cell")
    if source_row.max() >= values.shape[0] or source_col.max() >= values.shape[1]:
        raise ValueError("Landsat window does not cover an eligible cell")
    return values[source_row, source_col]


def scan_scene(
    *,
    scene_id: str,
    year: int,
    split: str,
    month: int | None,
    season: str,
    s2_scene_id: str,
    landsat_cog: str,
    landsat_flag: str,
    feature_stack: str,
    feature_valid: str,
    provenance: dict[str, object],
    analysis_10: GeoBox | None = None,
) -> SceneScan:
    """Read one scene. Raises when the published grid or a raster does not match."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        return _read_scene(
            scene_id=scene_id,
            year=year,
            split=split,
            month=month,
            season=season,
            s2_scene_id=s2_scene_id,
            landsat_cog=landsat_cog,
            landsat_flag=landsat_flag,
            feature_stack=feature_stack,
            feature_valid=feature_valid,
            provenance=provenance,
            analysis_10=analysis_10,
        )


def _read_scene(
    *,
    scene_id: str,
    year: int,
    split: str,
    month: int | None,
    season: str,
    s2_scene_id: str,
    landsat_cog: str,
    landsat_flag: str,
    feature_stack: str,
    feature_valid: str,
    provenance: dict[str, object],
    analysis_10: GeoBox | None,
) -> SceneScan:
    grid_10 = canon_grid_10m() if analysis_10 is None else analysis_10
    grid_100 = canon_grid_100m() if analysis_10 is None else analysis_10.zoom_out(10)
    _check_grid(provenance, grid_100)
    counts = provenance.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"{scene_id}: provenance has no counts")
    resolved = ResolvedScene(
        scene_id=scene_id,
        year=year,
        s2_scene_id=s2_scene_id,
        geometry_id="",
        landsat_cog=landsat_cog,
        landsat_flag=landsat_flag,
        s2_cog="",
        s2_flag="",
        dynamic={},
        static_derived={},
        static_derived_meta={},
        static_sources={},
    )
    result = compute_eligibility(scene=resolved, analysis_10=grid_10, mask_uri=feature_valid)
    if result.errors:
        raise RuntimeError(f"{scene_id}: {result.errors[0]}")
    masks = _gap_masks(result)
    if not np.array_equal(masks["target_valid_support_100"], result.eligible):
        raise RuntimeError(f"{scene_id}: eligibility mask disagrees with the support cross-tab")
    row0, col0 = _origin_shift(grid_100)
    local_row, local_col = np.where(result.eligible)
    lst = _landsat_values(landsat_cog, grid_100, local_row, local_col)
    means = _feature_means(feature_stack, grid_10, result.eligible, local_row, local_col)
    finite = lst[np.isfinite(lst)]
    center = float(finite.mean()) if finite.size else math.nan
    global_row = local_row + row0
    global_col = local_col + col0
    frame = {
        "scene_id": scene_id,
        "year": year,
        "month": -1 if month is None else int(month),
        "season": season,
        "split": split,
        "cell_id": [
            cell_id(int(row), int(col)) for row, col in zip(global_row, global_col, strict=True)
        ],
        "row": global_row.astype(np.int32),
        "col": global_col.astype(np.int32),
        "block_row": (global_row // BLOCK_CELLS).astype(np.int32),
        "block_col": (global_col // BLOCK_CELLS).astype(np.int32),
        "lst_k": lst,
        "lst_c": lst - KELVIN_OFFSET,
        "lst_anomaly_k": lst - center,
    }
    for band, name in enumerate(schema_matching_count(means.shape[1]).names):
        frame[name] = means[:, band]
    observations = pd.DataFrame(frame)
    return SceneScan(
        observations=observations,
        gap_counts={name: int(np.count_nonzero(masks[name])) for name in GAP_CLASSES},
        gap_cells={name: _cell_pairs(masks[name], row0, col0) for name in GAP_CLASSES},
        provenance_target_valid=int(counts.get("target_valid_cells", -1)),
        provenance_eligible=int(counts.get("eligible_cells", -1)),
    )


__all__ = ["SceneScan", "scan_scene"]
