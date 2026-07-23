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
import tempfile
from datetime import datetime
from pathlib import Path

import earthaccess
import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset
import xarray as xr
from earthaccess.store import Store
from rasterio.enums import Resampling

from berlin_lst_downscaling.common.grid import canon_grid_70m
from berlin_lst_downscaling.data.io.staging import StageManager

# Compiled granule-ID regex.  Pattern:
#   ECOv002_L2T_LSTE_<orbit>_<scene>_<MGRS>_<YYYYMMDDThhmmss>_<build>_<rev>
RE_GRANULE = re.compile(
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


def parse_granule_datetime(granule_id: str) -> datetime | None:
    """Extract UTC acquisition datetime from a granule ID, or None if unparseable."""
    m = RE_GRANULE.match(granule_id)
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
            dtype = "float32" if layer == "LST" else "uint8"
            da = xr.DataArray(
                band.astype(dtype)[np.newaxis, ...],
                dims=("band", "y", "x"),
                coords={"band": [0], "y": y_coords, "x": x_coords},
            )
            da = da.assign_coords(crs=str(src.crs))
            data_vars[layer.lower()] = da

    # Build dataset with explicit CRS from the source
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(crs=src_crs)

    # Reproject to EPSG:25833 (Berlin UTM) on the canonical 70m grid
    gbox = canon_grid_70m()
    reproj_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        key = str(name)  # data_vars keys are Hashable; rioxarray needs str keys
        da_rio = da.rio.write_crs(src_crs)
        kwargs = dict(
            dst_crs=gbox.crs,
            shape=gbox.shape,
            transform=gbox.transform,
        )
        if key == "lst":
            # Bilinear for LST (continuous)
            reproj_vars[key] = da_rio.rio.reproject(
                **kwargs,
                resampling=Resampling.bilinear,
            )
        else:
            # Nearest-neighbour for categorical layers
            reproj_vars[key] = da_rio.rio.reproject(
                **kwargs,
                resampling=Resampling.nearest,
            )

    ds_out = xr.Dataset(reproj_vars)
    ds_out = ds_out.assign_coords(crs=gbox.crs)

    # Clip to bbox (WGS84) after reprojection to target CRS.
    # transform_bounds converts WGS84 → target CRS so clip_box gets valid metres.
    if bbox is not None:
        from rasterio.warp import transform_bounds

        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", str(gbox.crs), *bbox)
        ds_out = ds_out.rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
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
        raise FileNotFoundError(f"ECOSTRESS L2T layer not found or not readable: {uri}") from exc


def download_and_stage_granule(
    granule_id: str,
    stage: StageManager,
) -> str:
    """Download one ECOSTRESS L2T granule from NASA Earthdata → stage.

    Searches CMR by granule ID, downloads the 4 layer COGs (LST, cloud,
    water, QC) to a local temp directory, then uploads them into *stage*
    via :meth:`StageManager.put` (scheme-aware — works for local POSIX,
    FUSE mounts, and ``gs://``).

    Parameters
    ----------
    granule_id :
        Full granule ID, e.g.
        ``ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01``.
    stage :
        A :class:`StageManager` (or :class:`StageSession`) pointing at the
        stage root for the run. Files are uploaded under
        ``{stage.uri}/{granule_id}/``.

    Returns
    -------
    str
        The ``raw_dir`` URI the granule's COGs were uploaded into
        (``str(stage.uri.uri)``). Pass this to
        :func:`load_ecostress_scene` as ``raw_dir``.
    """
    dt = parse_granule_datetime(granule_id)
    if dt is None:
        raise ValueError(f"Cannot parse datetime from granule ID: {granule_id}")
    date_compact = dt.strftime("%Y%m%d")
    mgrs = parse_granule_mgrs(granule_id)
    if mgrs is None:
        raise ValueError(f"Cannot parse MGRS tile from granule ID: {granule_id}")

    # ── Login to Earthdata ──────────────────────────────────────────────
    auth = earthaccess.login()

    # ── CMR search by granule_name pattern ──────────────────────────────
    results = earthaccess.search_data(
        short_name="ECO_L2T_LSTE",
        version="002",
        granule_name=f"ECOv002_L2T_LSTE_*_{mgrs}_{date_compact}T*",
        count=5,
    )
    granule = next(
        (g for g in results if g["meta"]["native-id"] == granule_id),
        None,
    )
    if granule is None:
        raise FileNotFoundError(
            f"ECOSTRESS granule {granule_id!r} not found in CMR. "
            f"Searched for pattern: ECOv002_L2T_LSTE_*_{mgrs}_{date_compact}T*"
        )

    # ── Download to local temporary directory ───────────────────────────
    # TemporaryDirectory is auto-cleaned on exit, even on exception.
    with tempfile.TemporaryDirectory(prefix=f"eco_dl_{granule_id[-12:]}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        downloaded = _download_to_tmp(granule, tmp_dir, auth)

        # Upload into stage (scheme-aware via StageManager.put).
        # Downloaded files are inside tmp_dir which is cleaned up on exit.
        for local_path in downloaded:
            stage.put(local_path, f"{granule_id}/{local_path.name}")

    return str(stage.uri.uri)


def _download_to_tmp(
    granule: dict,
    tmp_dir: Path,
    auth: earthaccess.auth.Auth,
) -> list[Path]:
    """Download one granule's 4 layer COGs to a local temp directory.

    Uses ``earthaccess.Store.get()`` which always downloads locally.
    Retries up to 3 times with exponential backoff.
    """
    store = Store(auth=auth)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            downloaded = store.get([granule], local_path=str(tmp_dir), threads=4)  # type: ignore[arg-type]
            return [Path(p) for p in downloaded if p]
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                import time as _time

                _time.sleep(2**attempt)
    raise RuntimeError(
        f"Download failed after 3 attempts for {granule['meta']['native-id']}: {last_exc}"
    ) from last_exc


def parse_granule_mgrs(granule_id: str) -> str | None:
    """Extract MGRS tile from a granule ID (e.g. 33UUU)."""
    parts = granule_id.split("_")
    if len(parts) >= 6:
        return parts[5]
    return None


__all__ = [
    "load_ecostress_scene",
    "download_and_stage_granule",
    "parse_granule_datetime",
    "parse_granule_mgrs",
]
