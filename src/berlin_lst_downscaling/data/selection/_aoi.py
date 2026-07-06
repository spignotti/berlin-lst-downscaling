"""Shared AOI mask loading and reprojection for selection module."""

from __future__ import annotations

import numpy as np
import rasterio
import rasterio.warp as rwarp
import rioxarray  # noqa: F401  — registers rio accessor on xr.Dataset


def load_aoi_mask(
    aoi_path: str,
    target_ds,
) -> np.ndarray:
    """Load and reproject the Berlin AOI mask to match the target dataset grid.

    Parameters
    ----------
    aoi_path :
        Path to uint8 AOI COG (1 = inside Berlin).
    target_ds :
        xarray Dataset with ``crs``, ``rio`` accessor (from rioxarray).

    Returns
    -------
    np.ndarray
        2D boolean array (y, x), True = inside Berlin.
    """
    with rasterio.open(aoi_path) as aoi_src:
        aoi_data = aoi_src.read(1).astype(bool)
        aoi_crs = aoi_src.crs
        aoi_transform = aoi_src.transform
        aoi_width = aoi_src.width
        aoi_height = aoi_src.height

    target_transform = target_ds.rio.transform()
    target_crs = target_ds.rio.crs
    target_height, target_width = target_ds.dims["y"], target_ds.dims["x"]

    # Use uint8 for GDAL/rasterio compatibility (bool causes TypeError)
    destination = np.empty((target_height, target_width), dtype=np.uint8)
    rwarp.reproject(
        source=aoi_data.astype(np.uint8),
        src_crs=aoi_crs,
        src_transform=aoi_transform,
        src_width=aoi_width,
        src_height=aoi_height,
        destination=destination,
        dst_crs=target_crs,
        dst_transform=target_transform,
        resampling=rwarp.Resampling.nearest,
    )
    return destination.astype(bool)
