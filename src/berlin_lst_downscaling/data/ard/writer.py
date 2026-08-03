"""COG writer + STAC item emission backed by ``atomic_write``.

``write_cog_atomic`` and ``write_stac_atomic`` accept ``str`` destination
URIs (local path, ``gs://`` bucket, or ``~/.mnt/`` mount) and write
atomically via the storage module.

The COG writer uses a 2-pass procedure on a local temp file:
1. Write all bands to a staging temp file (no overviews).
2. Copy to final COG via GDAL COG driver (CreateCopy).
3. Strict-validate the temp COG.
4. Upload the validated COG.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil
import xarray as xr

from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.io.storage import atomic_upload, atomic_write, exists

# Overview resampling: average for numeric data, nearest for flags
_NUMERIC_OV_RESAMPLING = "AVERAGE"
_FLAG_OV_RESAMPLING = "NEAREST"

# ── COG write (main band file, float32) ──────────────────────────────

def write_cog_atomic(
    ds: xr.Dataset,
    dst: str,
    contract: Contract,
    overwrite: bool = False,
) -> str:
    """Write a multi-band COG atomically.

    Bands are written in ``ds.data_vars`` order. The file is written to a
    local staging temp, then copied to a COG temp via the GDAL COG driver,
    validated, and uploaded via ``atomic_upload`` to *dst*.

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

    Returns
    -------
    str
        The final *dst* URI on success.
    """
    if exists(dst) and not overwrite:
        raise FileExistsError(dst)

    bands = [str(k) for k in ds.data_vars]
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

    # Derive nodata: float → NaN, integer → contract's first band nodata
    if "float" in common_dtype:
        nodata = float("nan")
    elif contract.output_bands:
        nodata = contract.output_bands[0].nodata
    else:
        nodata = None

    staging_profile = _build_staging_profile(
        common_dtype=common_dtype,
        n_bands=len(bands),
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
        nodata=nodata,
    )

    # 2-pass: staging → COG copy → validate → upload
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        staging_path = tmp_dir / f"_staging_{Path(dst).name}.tif"
        cog_path = tmp_dir / f"_cog_{Path(dst).name}.tif"

        # 1. Write staging GTiff (no overviews)
        with rasterio.open(staging_path, "w", **staging_profile) as tmp:
            for i, (name, arr) in enumerate(arrays, 1):
                out_arr = arr.astype(common_dtype, copy=False)
                tmp.write(out_arr, i)
                tmp.set_band_description(i, name)

        # 2. Copy to COG via GDAL COG driver
        cog_options: dict[str, str] = {
            "BLOCKSIZE": str(contract.tiling.blocksize),
            "COMPRESS": contract.tiling.compress.upper(),
            "PREDICTOR": str(contract.tiling.predictor),
            "BIGTIFF": "IF_SAFER",
            "OVERVIEW_RESAMPLING": _NUMERIC_OV_RESAMPLING,
        }

        rasterio.shutil.copy(
            str(staging_path),
            str(cog_path),
            driver="COG",
            strict=True,
            **cog_options,
        )

        # 3. Strict COG validation — warnings are failures
        strict_result = validate_strict_cog(str(cog_path))
        if not strict_result.valid:
            raise ValueError(
                f"COG strict validation failed for {dst}: "
                + "; ".join(strict_result.errors + strict_result.warnings)
            )

        # 4. Upload via streaming (no full-COG-in-RAM for large multi-band files)
        atomic_upload(cog_path, dst, overwrite=overwrite)

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

    staging_profile = _build_staging_profile(
        common_dtype="uint8",
        n_bands=1,
        h=h,
        w=w,
        crs=crs,
        transform=geo_transform,
        contract=contract,
    )
    # Override compression for flags
    staging_profile["compress"] = "zstd"
    staging_profile["predictor"] = 1

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        staging_path = tmp_dir / f"_staging_{Path(dst).name}.tif"
        cog_path = tmp_dir / f"_cog_{Path(dst).name}.tif"

        # 1. Write staging GTiff
        with rasterio.open(staging_path, "w", **staging_profile) as tmp:
            tmp.write(arr_2d, 1)
            tmp.set_band_description(1, "flag")

        # 2. Copy to COG via GDAL COG driver (nearest for flags)
        cog_options: dict[str, str] = {
            "BLOCKSIZE": str(contract.tiling.blocksize),
            "COMPRESS": "ZSTD",
            "PREDICTOR": "1",
            "BIGTIFF": "IF_SAFER",
            "OVERVIEW_RESAMPLING": _FLAG_OV_RESAMPLING,
        }

        rasterio.shutil.copy(
            str(staging_path),
            str(cog_path),
            driver="COG",
            strict=True,
            **cog_options,
        )

        # 3. Strict COG validation — warnings are failures
        strict_result = validate_strict_cog(str(cog_path))
        if not strict_result.valid:
            raise ValueError(
                f"COG strict validation failed for flag {dst}: "
                + "; ".join(strict_result.errors + strict_result.warnings)
            )

        # 4. Upload
        atomic_upload(cog_path, dst, overwrite=overwrite)

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

def _build_staging_profile(
    common_dtype: str,
    n_bands: int,
    h: int,
    w: int,
    crs: Any,
    transform: Any,
    contract: Contract,
    nodata: float | None = None,
) -> dict[str, Any]:
    """Build a staging GTiff profile (no overviews, no COG driver)."""
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
