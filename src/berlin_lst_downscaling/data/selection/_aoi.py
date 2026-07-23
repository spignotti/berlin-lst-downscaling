"""Shared AOI mask loading and reprojection for selection module."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp as rwarp
import rioxarray  # noqa: F401  — registers rio accessor on xr.Dataset
import xarray as xr


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

def select_time_slice(
    ds: xr.Dataset,
    target_dt: datetime | pd.Timestamp,
) -> xr.Dataset:
    """Select the time-slice whose UTC date matches target_dt's date.

    When a dataset contains multiple solar-day slices (e.g. from a ±1-day
    search), ``values[0]`` would pick the first chronological slice — which
    may not correspond to the anchor's date.  This function finds the slice
    whose UTC date matches ``target_dt`` and returns that slice.

    Parameters
    ----------
    ds :
        xarray Dataset with a ``time`` dimension.
    target_dt :
        Datetime of the anchor (UTC). The date part is used for matching.

    Returns
    -------
    xr.Dataset
        Dataset sliced to the matching time index. If the dataset has no
        ``time`` dimension (single slice), returns it unchanged.
    """
    if "time" not in ds.dims:
        return ds  # scalar — already a single slice

    time_vals = pd.to_datetime(ds["time"].values)
    target_ts = pd.Timestamp(target_dt)
    # Convert to days-since-epoch via numpy datetime64 (pyright-safe)
    time_days = time_vals.to_numpy().astype("datetime64[D]").astype(np.int64)
    target_day = target_ts.to_numpy().astype("datetime64[D]").astype(np.int64)
    idx = int(np.argmin(np.abs(time_days - target_day)))
    return ds.isel(time=idx)