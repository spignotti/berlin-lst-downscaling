"""Canopy cover fraction from 1 m vegetation height.

The fraction is the share of 1 m pixels in a 10 m cell whose height is
strictly above a threshold. NoData (non-vegetated) counts in the
denominator and not in the numerator, so a single tall pixel does not
fill the cell. The default threshold is provisional
(:data:`CANOPY_COVER_THRESHOLD_M_DEFAULT`).

``vegetation_height_max`` stays on the source product for the vegetation
DSM. This band replaces it only in the features/v6 model roster.
"""

from __future__ import annotations

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.features.contracts import CANOPY_COVER_THRESHOLD_M_DEFAULT

# Source-window size for the 1 m raster. A 2048 px block is 16 MiB of
# float32 and is never the full Berlin raster.
_BLOCK_PX = 2048


def canopy_cover_fraction(
    height_m: np.ndarray,
    *,
    threshold_m: float = CANOPY_COVER_THRESHOLD_M_DEFAULT,
    factor: int = 10,
) -> np.ndarray:
    """Aggregate an aligned 1 m height array to canopy cover.

    ``height_m`` is metres, shape ``(factor * n_row, factor * n_col)``.
    Each output cell is the mean of a ``factor`` by ``factor`` block of
    the indicator ``height > threshold``. Non-finite pixels are not above
    the threshold and still occupy the denominator.

    Parameters
    ----------
    height_m :
        Vegetation height in metres. NoData is non-finite.
    threshold_m :
        Canopy threshold in metres. Pixels equal to the threshold are
        not canopy.
    factor :
        Linear aggregation factor. ``10`` maps 1 m pixels onto 10 m cells.

    Returns
    -------
    float32 array of shape ``(n_row, n_col)`` with values in ``[0, 1]``.
    """
    _validate_threshold(threshold_m)
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError(f"factor must be an int >= 1, got {factor!r}")
    height = np.asarray(height_m)
    if height.ndim != 2:
        raise ValueError(f"height_m must be 2-D, got shape {height.shape}")
    n_row, n_col = height.shape
    if n_row % factor or n_col % factor:
        raise ValueError(f"height shape {height.shape} is not divisible by factor {factor}")
    values = height.astype(np.float64, copy=False)
    blocks = values.reshape(n_row // factor, factor, n_col // factor, factor)
    above = np.isfinite(blocks) & (blocks > threshold_m)
    return above.mean(axis=(1, 3)).astype(np.float32)


def canopy_cover_fraction_on_grid(
    src: rasterio.DatasetReader,
    grid: GeoBox,
    *,
    threshold_m: float = CANOPY_COVER_THRESHOLD_M_DEFAULT,
    scale_to_m: float = 0.1,
    block_px: int = _BLOCK_PX,
) -> np.ndarray:
    """Windowed canopy cover on ``grid`` from an open 1 m height raster.

    Source pixels are read in blocks. A destination cell is the fraction
    of source-pixel centres that fall inside it and are strictly above
    ``threshold_m`` after ``scale_to_m`` (the Umweltatlas raster is
    stored in decimetres). Cells with no source centre stay NaN.

    The full 1 m raster is not loaded.
    """
    _validate_threshold(threshold_m)
    if scale_to_m <= 0:
        raise ValueError(f"scale_to_m must be positive, got {scale_to_m}")
    if block_px < 1:
        raise ValueError(f"block_px must be >= 1, got {block_px}")

    dest_sum = np.zeros((grid.shape.y, grid.shape.x), dtype=np.float64)
    dest_count = np.zeros((grid.shape.y, grid.shape.x), dtype=np.uint32)
    nodata = src.nodata
    for row_off in range(0, src.height, block_px):
        height = min(block_px, src.height - row_off)
        for col_off in range(0, src.width, block_px):
            width = min(block_px, src.width - col_off)
            window = rasterio.windows.Window(col_off, row_off, width, height)
            block = src.read(1, window=window)
            _accumulate_block(
                block,
                src_transform=src.window_transform(window),
                grid=grid,
                threshold_m=threshold_m,
                scale_to_m=scale_to_m,
                nodata=nodata,
                dest_sum=dest_sum,
                dest_count=dest_count,
            )

    cover = np.full(dest_sum.shape, np.nan, dtype=np.float32)
    covered = dest_count > 0
    cover[covered] = (dest_sum[covered] / dest_count[covered]).astype(np.float32)
    return cover


def _accumulate_block(
    block: np.ndarray,
    *,
    src_transform,
    grid: GeoBox,
    threshold_m: float,
    scale_to_m: float,
    nodata: float | None,
    dest_sum: np.ndarray,
    dest_count: np.ndarray,
) -> None:
    """Add one source window into the destination sum and count."""
    height, width = block.shape
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    xs = src_transform.c + (cols + 0.5) * src_transform.a
    ys = src_transform.f + (rows + 0.5) * src_transform.e
    dest_col = np.floor((xs - grid.transform.c) / grid.transform.a).astype(np.int32)
    dest_row = np.floor((ys - grid.transform.f) / grid.transform.e).astype(np.int32)
    dest_col_grid, dest_row_grid = np.meshgrid(dest_col, dest_row)
    inside = (
        (dest_row_grid >= 0)
        & (dest_row_grid < dest_sum.shape[0])
        & (dest_col_grid >= 0)
        & (dest_col_grid < dest_sum.shape[1])
    )
    if not np.any(inside):
        return
    values = block.astype(np.float64, copy=False)
    if nodata is not None and np.isfinite(nodata):
        values = np.where(values == nodata, np.nan, values)
    values = values * scale_to_m
    above = np.isfinite(values) & (values > threshold_m)
    rows_in = dest_row_grid[inside]
    cols_in = dest_col_grid[inside]
    np.add.at(dest_count, (rows_in, cols_in), 1)
    np.add.at(dest_sum, (rows_in, cols_in), above[inside].astype(np.float64))


def _validate_threshold(threshold_m: float) -> None:
    if isinstance(threshold_m, bool) or not isinstance(threshold_m, (int, float)):
        raise ValueError(f"threshold_m must be a real number, got {threshold_m!r}")
    if not np.isfinite(threshold_m) or float(threshold_m) < 0:
        raise ValueError(f"threshold_m must be finite and >= 0, got {threshold_m}")


__all__ = [
    "canopy_cover_fraction",
    "canopy_cover_fraction_on_grid",
]
