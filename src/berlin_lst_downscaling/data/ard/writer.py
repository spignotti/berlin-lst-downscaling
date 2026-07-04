"""COG writer + STAC item emission backed by ``atomic_write``.

``write_cog_atomic`` and ``write_stac_atomic`` accept ``str`` destination
URIs (local path, ``gs://`` bucket, or ``~/.mnt/`` mount) and write
atomically via the storage module.

The COG writer uses a 2-pass procedure on a local temp file:
1. Write all bands with final compression (deflate) to a temp file.
2. Build overviews in-place.
3. Read the temp file into bytes, call ``atomic_write``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.io.storage import atomic_write, exists

# ── COG write (main band file, float32) ──────────────────────────────


def write_cog_atomic(
    ds: xr.Dataset,
    dst: str,
    contract: Contract,
    overwrite: bool = False,
    bands_order: list[str] | None = None,
) -> str:
    """Write a multi-band COG atomically.

    The COG is assembled from bands in ``ds`` (or ``bands_order`` if
    given), written to a local temp file, then pushed via
    ``atomic_write`` to *dst* (local path or GCS URI).

    Parameters
    ----------
    ds :
        Dataset whose variables are bands to write.  Each variable must
        be a 2D (or 3D with singleton ``time``) ``float32`` DataArray
        carrying CRS and transform via ``rio``.
    dst :
        Final output URI (e.g. ``data/ard/…/<scene_id>.tif`` or
        ``gs://berlin-lst-data/…/<scene_id>.tif``).
    contract :
        Contract describing tiling, compression, and expected nodata.
    overwrite :
        If ``False`` and *dst* exists, a :class:`FileExistsError`
        is raised.
    bands_order :
        Band variable names in order they should appear in the COG.
        Defaults to ``list(ds.data_vars)``.

    Returns
    -------
    str
        The final *dst* URI on success.
    """
    if exists(dst) and not overwrite:
        raise FileExistsError(dst)

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

    nodata = float("nan") if "float" in common_dtype else None

    profile = _build_profile(
        common_dtype=common_dtype,
        n_bands=len(bands),
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
        nodata=nodata,
    )

    # Write to local temp file (2-pass: write + overviews)
    tmp_dir = Path(".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst_tmp = tmp_dir / f"_{Path(dst).name}.cog"

    try:
        with rasterio.open(dst_tmp, "w", **profile) as tmp:
            for i, (name, arr) in enumerate(arrays, 1):
                out_arr = arr.astype(common_dtype, copy=False)
                tmp.write(out_arr, i)
                tmp.set_band_description(i, name)

        ov_levels = list(contract.tiling.overviews)
        if ov_levels:
            with rasterio.open(dst_tmp, "r+") as tmp:
                tmp.build_overviews(ov_levels, Resampling.average)

        # Read bytes and push via atomic_write
        cog_bytes = dst_tmp.read_bytes()
        atomic_write(dst, cog_bytes, overwrite=overwrite)

    except BaseException:
        dst_tmp.unlink(missing_ok=True)
        raise

    return dst


# ── COG write (flag band, uint8) ─────────────────────────────────────


def write_flag_cog_atomic(
    flag_da: xr.DataArray,
    dst: str,
    contract: Contract,
    overwrite: bool = False,
) -> str:
    """Write a single-band uint8 flag COG atomically.

    The flag band stores a bitmask (fill, cloudy, shadow, cirrus,
    saturated).  It is written as a separate COG to avoid promoting
    uint8 to float32 in the multi-band COG.
    """
    if exists(dst) and not overwrite:
        raise FileExistsError(dst)

    arr = flag_da.values.squeeze()
    arr_2d: np.ndarray = arr if arr.ndim == 2 else arr[0]  # type: ignore[assignment]
    h, w = arr_2d.shape
    crs = flag_da.rio.crs
    geo_transform = flag_da.rio.transform()

    profile = _build_profile(
        common_dtype="uint8",
        n_bands=1,
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
    )
    profile["compress"] = "zstd"
    profile["predictor"] = 1

    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst_tmp = tmp_dir / f"_{Path(dst).name}"

    try:
        with rasterio.open(dst_tmp, "w", **profile) as tmp:
            tmp.write(arr_2d, 1)
            tmp.set_band_description(1, "flag")

        cog_bytes = dst_tmp.read_bytes()
        atomic_write(dst, cog_bytes, overwrite=overwrite)

    except BaseException:
        dst_tmp.unlink(missing_ok=True)
        raise

    return dst


# ── STAC item ────────────────────────────────────────────────────────


def write_stac_atomic(
    stac_item: dict[str, Any],
    dst: str,
    overwrite: bool = False,
) -> str:
    """Write a STAC item as JSON atomically.

    Parameters
    ----------
    stac_item :
        The STAC item dictionary.
    dst :
        Final output URI (e.g. ``…/<scene_id>.stac.json``).
    overwrite :
        If ``False`` and *dst* exists, a :class:`FileExistsError`
        is raised.

    Returns
    -------
    str
        The final *dst* URI on success.
    """
    if exists(dst) and not overwrite:
        raise FileExistsError(dst)

    json_bytes = json.dumps(stac_item, indent=2).encode("utf-8")
    atomic_write(dst, json_bytes, overwrite=overwrite)

    return dst


# ── helpers ──────────────────────────────────────────────────────────


def _build_profile(
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
    if any("float" in d for d in dt_set):
        return "float32"
    sizes = [int(d[-2:]) for d in dt_set if d[-2:].isdigit()]
    max_bits = max(sizes) if sizes else 8
    return f"uint{max_bits}"


__all__ = [
    "write_cog_atomic",
    "write_flag_cog_atomic",
    "write_stac_atomic",
]
