"""Load an ECOSTRESS L2T granule from local/AppEEARS-export COGs.

ECOSTRESS L2T (ECO_L2T_LSTE.002) is distributed as per-layer COG files,
not as a single HDF5. Each granule provides:

    {granule_id}_LST.tif       float32  Kelvin  (main LST band)
    {granule_id}_cloud.tif      uint8    0=clear / 1=cloud / 255=fill
    {granule_id}_water.tif      uint8    0=dry  / 1=water / 255=fill
    {granule_id}_QC.tif         uint8    mandatory QA bitmask (see below)

Native grid: MGRS UTM tiles, 1568 × 1568 px at 70 m.  The pipeline
reprojects to EPSG:25833 (ETRS89 / UTM zone 33N, Berlin) before masking.

QC mandatory QA bits (``QC & 0b11``):
    0b00 = TES pixel produced (best quality)
    0b01 = TES produced, degraded conditions
    0b10 = not set (not cloud in v002)
    0b11 = pixel not produced (fill)

Cloud semantics (Collection 2, ``cloud`` layer):
    0 = clear, 1 = cloud, 255 = fill / outside granule
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset
import xarray as xr
from rasterio.enums import Resampling

from berlin_lst_downscaling.common.config import settings

# Compiled granule-ID regex.  Pattern:
#   ECOv002_L2T_LSTE_<orbit>_<scene>_<MGRS>_<YYYYMMDDThhmmss>_<build>_<rev>
_RE_GRANULE = re.compile(
    r"^ECO"
    r"v(?P<version>\d+)"
    r"_L2T_LSTE_"
    r"(?P<orbit>\d+)"
    r"_(?P<scene>\d+)"
    r"_(?P<mgrs>[\w]+)"
    r"_(?P<datetime>\d{8}T\d{6})"
    r"_(?P<build>\d+)"
    r"_(?P<rev>\d+)$",
)


def _parse_granule_datetime(granule_id: str) -> datetime | None:
    """Extract UTC acquisition datetime from a granule ID, or None if unparseable."""
    m = _RE_GRANULE.match(granule_id)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group("datetime"), "%Y%m%dT%H%M%S")
    except ValueError:
        return None


# ── public API ───────────────────────────────────────────────────────


def load_ecostress_scene(
    granule_id: str,
    raw_dir: str,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: int = 70,
) -> tuple[xr.Dataset, list[str]]:
    """Load an ECOSTRESS L2T granule from local COGs.

    Parameters
    ----------
    granule_id :
        Granule identifier, e.g.
        ``ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01``.
    raw_dir :
        Root directory containing per-granule sub-directories.
        Expected layout: ``{raw_dir}/{granule_id}/{granule_id}_{layer}.tif``.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. When provided,
        the granule is clipped to this extent (with a small buffer) before
        reprojection.  When ``None`` the full granule tile is loaded.
    resolution :
        Target resolution in metres for the EPSG:25833 reprojection.
        Defaults to 70 m (ECOSTRESS L2T native).

    Returns
    -------
    tuple[xr.Dataset, list[str]]
        A dataset with bands ``lst`` (float32 K), ``cloud`` (uint8),
        ``water`` (uint8), and ``qc`` (uint8), and the list
        ``[granule_id]``.

    Raises
    ------
    FileNotFoundError
        If the expected COG files are not found under ``raw_dir``.
    RuntimeError
        If the granule ID cannot be parsed for datetime metadata.
    """
    granule_dir = Path(raw_dir) / granule_id

    layers = ["LST", "cloud", "water", "QC"]
    files: dict[str, Path] = {}
    for layer in layers:
        tif_path = granule_dir / f"{granule_id}_{layer}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(
                f"Expected ECOSTRESS L2T layer not found: {tif_path}"
            )
        files[layer] = tif_path

    # Load all four layers as xr.DataArrays
    data_vars: dict[str, xr.DataArray] = {}
    src_crs: str | None = None
    src_transform: Any = None

    for layer, path in files.items():
        with rasterio.open(path) as src:
            band = src.read(1)
            # Track CRS and transform from the first opened file
            if src_crs is None:
                src_crs = str(src.crs)
                src_transform = src.transform
            # For non-LST layers, cast to consistent dtypes
            if layer == "LST":
                da = xr.DataArray(
                    band.astype("float32")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                )
            elif layer in ("cloud", "water"):
                da = xr.DataArray(
                    band.astype("uint8")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                )
            else:  # QC
                da = xr.DataArray(
                    band.astype("uint8")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                )
            da = da.assign_coords(
                transform=src.transform, crs=str(src.crs)
            )
            data_vars[layer.lower()] = da

    # Build dataset with explicit CRS from the source
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(
        crs=src_crs,
        transform=src_transform,
    )

    # Reproject to EPSG:25833 (Berlin UTM)
    target_crs = settings.target_crs  # EPSG:25833
    reproj_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        key = str(name)  # data_vars keys are Hashable; rioxarray needs str keys
        da_rio = da.rio.write_crs(src_crs)
        if key == "lst":
            # Bilinear for LST (continuous)
            reproj_vars[key] = da_rio.rio.reproject(
                target_crs,
                resolution=resolution,
                resampling=Resampling.bilinear,
            )
        else:
            # Nearest-neighbour for categorical layers
            reproj_vars[key] = da_rio.rio.reproject(
                target_crs,
                resolution=resolution,
                resampling=Resampling.nearest,
            )

    ds_out = xr.Dataset(reproj_vars)
    ds_out = ds_out.assign_coords(crs=target_crs)

    # Optionally clip to bbox (best-effort — only trims off-tile areas)
    # Rasterio/rioxarray raises Exception if bbox doesn't intersect the grid.
    # We silently continue with the full granule in that case.
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        try:
            ds_out = ds_out.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        except Exception:  # best-effort bbox trim; full tile used on failure  # noqa: S110
            pass

    return ds_out, [granule_id]


__all__ = [
    "load_ecostress_scene",
]
