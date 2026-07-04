"""COG writer + STAC item emission with atomic write.

``write_cog_atomic`` and ``write_stac_atomic`` each write to a
temporary path first, then ``os.replace`` to the final location.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy

from berlin_lst_downscaling.data.ard.contract import Contract

# ── COG write ────────────────────────────────────────────────────────


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
        be a 2D (or 3D with singleton ``time``) ``float32`` or
        ``uint8`` DataArray carrying CRS and transform via ``rio``.
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
    transform = None

    for name in bands:
        arr = ds[name].values.squeeze()
        arr_2d: np.ndarray = arr if arr.ndim == 2 else arr[0]  # type: ignore[assignment]
        if len(arrays) == 0:
            h, w = arr_2d.shape
            crs = ds[name].rio.crs
            transform = ds[name].rio.transform()
        arrays.append((name, arr_2d))

    # resolve per-band dtypes
    dtypes = [ds[name].values.squeeze().dtype for name in bands]

    # Unique dtype — COG is single-dtype per file.  Use common if
    # mixed; plan normally keeps landsat st (float32) + flag (uint8)
    # in one file → cast uint8 up to float32.
    common_dtype = _common_dtype(dtypes)

    profile = {
        "driver": "GTiff",
        "dtype": common_dtype,
        "count": len(bands),
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

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_tmp = dst.parent / ".tmp" / f"_{dst.name}.cog"

    # --- pass 1: write to temp without overviews, weak compression ---
    dst_tmp.parent.mkdir(parents=True, exist_ok=True)
    pass1_profile = {**profile, "compress": "none"}
    with rasterio.open(dst_tmp, "w", **pass1_profile) as tmp:
        for i, (name, arr) in enumerate(arrays, 1):
            out_arr = arr.astype(common_dtype, copy=False)
            tmp.write(out_arr, i)
            tmp.set_band_description(i, name)

    # --- pass 2: add overviews ---
    ov_levels = list(contract.tiling.overviews)
    if ov_levels:
        with rasterio.open(dst_tmp, "r+") as tmp:
            tmp.build_overviews(ov_levels, Resampling.average)

    # --- pass 3: copy with final compression / COG layout ---
    rio_opts: dict[str, object] = {
        k: profile[k]
        for k in (
            "tiled",
            "blockxsize",
            "blockysize",
            "compress",
            "predictor",
        )
    }
    rio_copy(dst_tmp, dst, copy_src_overviews=True, **rio_opts)

    # clean up temp
    dst_tmp.unlink(missing_ok=True)

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
    "write_stac_atomic",
]
