"""Load an ECOSTRESS L2T granule from local or GCS COGs.

ECOSTRESS L2T (ECO_L2T_LSTE.002) is distributed as per-layer COG files,
not as a single HDF5. Each granule provides:

    {granule_id}_LST.tif       float32  Kelvin  (main LST band)
    {granule_id}_cloud.tif      uint8    0=clear / 1=cloud / 255=fill
    {granule_id}_water.tif      uint8    0=dry  / 1=water / 255=fill
    {granule_id}_QC.tif         uint8    mandatory QA bitmask (see below)

``raw_dir`` accepts any URI scheme supported by :mod:`rasterio`:

==============  ===========================================================
Scheme          Notes
==============  ===========================================================
Local POSIX     ``/path/to/stage/`` or ``data/stage/``
GCS             ``gs://bucket/path`` — rasterio opens via /vsigs/
Mounted FUSE    ``~/.mnt/bucket/path`` — same as local POSIX
==============  ===========================================================

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
    """Load an ECOSTRESS L2T granule from local or GCS COGs.

    Parameters
    ----------
    granule_id :
        Granule identifier, e.g.
        ``ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01``.
    raw_dir :
        Root URI containing per-granule sub-directories.
        Accepted schemes: local POSIX path, ``gs://bucket/path`` (opened
        via rasterio /vsigs/), or ``~/.mnt/bucket/path`` (FUSE mount).
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
    layers = ["LST", "cloud", "water", "QC"]
    tif_paths: dict[str, str] = {}
    for layer in layers:
        tif_uri = _resolve_granule_uri(raw_dir, granule_id, layer)
        _assert_granule_layer_exists(tif_uri)
        tif_paths[layer] = tif_uri

    # Load all four layers as xr.DataArrays
    data_vars: dict[str, xr.DataArray] = {}
    src_crs: str | None = None

    for layer, uri in tif_paths.items():
        with rasterio.open(uri) as src:
            band = src.read(1)
            # Track CRS, transform and dimensions from the first opened file
            if src_crs is None:
                src_crs = str(src.crs)
            # Build x/y coordinate arrays from the Affine transform (pixel center)
            # transform = Affine(a, b, c, d, e, f) where a=dx, e=dy, c/f = origin
            x_coords = src.transform.xoff + src.transform.a * (0.5 + np.arange(src.width))
            y_coords = src.transform.yoff + src.transform.e * (0.5 + np.arange(src.height))
            if layer == "LST":
                da = xr.DataArray(
                    band.astype("float32")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                    coords={"band": [0], "y": y_coords, "x": x_coords},
                )
            elif layer in ("cloud", "water"):
                da = xr.DataArray(
                    band.astype("uint8")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                    coords={"band": [0], "y": y_coords, "x": x_coords},
                )
            else:  # QC
                da = xr.DataArray(
                    band.astype("uint8")[np.newaxis, ...],
                    dims=("band", "y", "x"),
                    coords={"band": [0], "y": y_coords, "x": x_coords},
                )
            da = da.assign_coords(crs=str(src.crs))
            data_vars[layer.lower()] = da

    # Build dataset with explicit CRS from the source
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(crs=src_crs)

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

    # Clip to bbox (WGS84) after reprojection to target CRS.
    # transform_bounds converts WGS84 → target CRS so clip_box gets valid metres.
    if bbox is not None:
        from rasterio.warp import transform_bounds

        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", target_crs, *bbox)
        ds_out = ds_out.rio.clip_box(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        )

    return ds_out, [granule_id]


# ── internal helpers ──────────────────────────────────────────────────


def _resolve_granule_uri(raw_dir: str, granule_id: str, layer: str) -> str:
    """Return the URI string for a granule layer TIF.

    Handles local POSIX paths, ``gs://`` (converted to /vsigs/ for rasterio),
    and FUSE mount paths (``~/.mnt/...``).
    """
    import os

    granule_leaf = f"{granule_id}_{layer}.tif"
    raw_str = str(raw_dir).rstrip("/")

    if raw_str.startswith("gs://"):
        # Convert gs://bucket/key → /vsigs/bucket/key for rasterio
        path = raw_str.removeprefix("gs://")
        bucket, key = path.split("/", 1)
        return f"/vsigs/{bucket}/{key}/{granule_id}/{granule_leaf}"
    elif raw_str.startswith("~/"):
        # Expand FUSE mount home
        return f"{os.path.expanduser(raw_str)}/{granule_id}/{granule_leaf}"
    else:
        # Local POSIX
        return f"{raw_str}/{granule_id}/{granule_leaf}"


def _assert_granule_layer_exists(uri: str) -> None:
    """Raise FileNotFoundError if the URI cannot be opened by rasterio."""
    try:
        with rasterio.open(uri):
            pass
    except Exception as exc:  # rasterio raises RasterioIOError on 404 / ENOENT on missing
        raise FileNotFoundError(
            f"ECOSTRESS L2T layer not found or not readable: {uri}"
        ) from exc


__all__ = [
    "load_ecostress_scene",
]

