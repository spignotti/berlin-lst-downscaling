"""COG writer + STAC item emission with atomic write.

``write_cog_atomic`` and ``write_stac_atomic`` each write to a
temporary path first, then ``os.replace`` to the final location.

The COG writer uses a 2-pass procedure:
1. Write all bands with final compression (deflate) to a temp file.
2. Build overviews in-place.
3. ``os.replace`` to the destination.

This avoids the data-loss window of writing directly to the destination
and saves the expensive recompress pass (pass 3 eliminated).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling

from berlin_lst_downscaling.data.ard.contract import Contract

# ── COG write (main band file, float32) ──────────────────────────────


def write_cog_atomic(
    ds: xr.Dataset,
    dst: Path,
    contract: Contract,
    overwrite: bool = False,
    bands_order: list[str] | None = None,
) -> Path:
    """Write a multi-band COG atomically.

    The COG is assembled from bands in ``ds`` (or ``bands_order`` if
    given), written to a ``.tmp`` sibling, then ``os.replace``-ed to
    final path.

    Parameters
    ----------
    ds :
        Dataset whose variables are bands to write.  Each variable must
        be a 2D (or 3D with singleton ``time``) ``float32`` DataArray
        carrying CRS and transform via ``rio``.
    dst :
        Final output path (e.g. ``…/<scene_id>.tif``).
    contract :
        Contract describing tiling, compression, and expected nodata.
    overwrite :
        If ``False`` and ``dst`` exists, a :class:`FileExistsError`
        is raised.
    bands_order :
        Band variable names in order they should appear in the COG.
        Defaults to ``list(ds.data_vars)``.

    Returns
    -------
    Path
        The final ``dst`` path on success.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))

    bands = bands_order or [str(k) for k in ds.data_vars]
    arrays: list[tuple[str, np.ndarray]] = []
    h = w = 0
    crs = None
    geo_transform = None

    for name in bands:
        arr = ds[name].values.squeeze()
        arr_2d: np.ndarray = arr if arr.ndim == 2 else arr[0]  # type: ignore[assignment]
        if len(arrays) == 0:
            h, w = arr_2d.shape
            crs = ds[name].rio.crs
            geo_transform = ds[name].rio.transform()
        arrays.append((name, arr_2d))

    dtypes = [ds[name].values.squeeze().dtype for name in bands]
    common_dtype = _common_dtype(dtypes)

    # Set nodata for float types
    nodata = float("nan") if "float" in common_dtype else None

    profile = _build_profile(
        dst=dst,
        common_dtype=common_dtype,
        n_bands=len(bands),
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
        nodata=nodata,
    )

    # Clear stale temp files
    tmp_dir = dst.parent / ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst_tmp = tmp_dir / f"_{dst.name}.cog"

    try:
        # --- pass 1: write with final compression + nodata ---
        with rasterio.open(dst_tmp, "w", **profile) as tmp:
            for i, (name, arr) in enumerate(arrays, 1):
                out_arr = arr.astype(common_dtype, copy=False)
                tmp.write(out_arr, i)
                tmp.set_band_description(i, name)

        # --- pass 2: build overviews in-place ---
        ov_levels = list(contract.tiling.overviews)
        if ov_levels:
            with rasterio.open(dst_tmp, "r+") as tmp:
                tmp.build_overviews(ov_levels, Resampling.average)

        # Atomic replace
        os.replace(str(dst_tmp), str(dst))

    except BaseException:
        # Clean up temp files on any error
        dst_tmp.unlink(missing_ok=True)
        raise

    return dst


# ── COG write (flag band, uint8) ─────────────────────────────────────


def write_flag_cog_atomic(
    flag_da: xr.DataArray,
    dst: Path,
    contract: Contract,
    overwrite: bool = False,
) -> Path:
    """Write a single-band uint8 flag COG atomically.

    The flag band stores a bitmask (fill, cloudy, shadow, cirrus,
    saturated).  It is written as a separate COG to avoid promoting
    uint8 to float32 in the multi-band COG.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))

    arr = flag_da.values.squeeze()
    arr_2d: np.ndarray = arr if arr.ndim == 2 else arr[0]  # type: ignore[assignment]
    h, w = arr_2d.shape
    crs = flag_da.rio.crs
    geo_transform = flag_da.rio.transform()

    profile = _build_profile(
        dst=dst,
        common_dtype="uint8",
        n_bands=1,
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
    )
    profile["compress"] = "zstd"  # fast for uint8 bitmask data
    profile["predictor"] = 1  # no prediction for integer data

    tmp_dir = dst.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst_tmp = tmp_dir / f"_{dst.name}"

    try:
        with rasterio.open(dst_tmp, "w", **profile) as tmp:
            tmp.write(arr_2d, 1)
            tmp.set_band_description(1, "flag")

        os.replace(str(dst_tmp), str(dst))
    except BaseException:
        dst_tmp.unlink(missing_ok=True)
        raise

    return dst


# ── STAC item ────────────────────────────────────────────────────────


def write_stac_atomic(
    stac_item: dict[str, Any],
    dst: Path,
    overwrite: bool = False,
) -> Path:
    """Write a STAC item as JSON atomically.

    Parameters
    ----------
    stac_item :
        The STAC item dictionary.  Must include ``stac_version``,
        ``id``, ``type``, ``geometry``, ``properties``, ``assets``,
        and ``links`` (at minimum).
    dst :
        Final output path (e.g. ``…/<scene_id>.stac.json``).
    overwrite :
        If ``False`` and ``dst`` exists, a :class:`FileExistsError`
        is raised.

    Returns
    -------
    Path
        The final ``dst`` path on success.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_tmp = dst.parent / ".tmp" / f"_{dst.name}"

    dst_tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_tmp, "w", encoding="utf-8") as f:
        json.dump(stac_item, f, indent=2)

    os.replace(str(dst_tmp), str(dst))
    return dst


# ── helpers ──────────────────────────────────────────────────────────


def _build_profile(
    dst: Path,
    common_dtype: str,
    n_bands: int,
    h: int,
    w: int,
    crs: Any,
    transform: Any,
    contract: Contract,
    nodata: float | None = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": common_dtype,
        "count": n_bands,
        "width": w,
        "height": h,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": contract.tiling.blocksize,
        "blockysize": contract.tiling.blocksize,
        "compress": contract.tiling.compress,
        "predictor": contract.tiling.predictor,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    return profile


def _common_dtype(dtypes: Sequence[np.dtype]) -> str:
    """Return a single dtype string that all bands can be cast to."""
    dt_set = set(str(d) for d in dtypes)
    if len(dt_set) == 1:
        return dt_set.pop()
    # For mixed float+uint: promote to float32
    if any("float" in d for d in dt_set):
        return "float32"
    # Mixed uints → smallest common that fits all
    sizes = [int(d[-2:]) for d in dt_set if d[-2:].isdigit()]
    max_bits = max(sizes) if sizes else 8
    return f"uint{max_bits}"


__all__ = [
    "write_cog_atomic",
    "write_flag_cog_atomic",
    "write_stac_atomic",
]
