"""ASTER GED v3 (AG100.003) static emissivity for the emissivity diagnostic.

AG100.003 is the ASTER Global Emissivity Dataset v3 (Hulley et al. 2015):
a 100 m, 1 x 1 degree tiled HDF5 product with the 2000-2008 mean
emissivity of the five ASTER TIR bands (10-14) in ``Emissivity/Mean``
(integer DN, ``emissivity * 1000``, fill <= 0) and pixel-centre
``Geolocation/Latitude`` / ``Geolocation/Longitude`` arrays. Tiles are
discovered through NASA CMR and downloaded from LP DAAC with an Earthdata
bearer token.

Spectral adjustment to Landsat 8 TIRS Band 10 uses the regression of the
USGS Collection 2 ST algorithm::

    eps_B10 = 0.6820 * eps_13 + 0.2578 * eps_14 + 0.0584

Malakar, N. K., Hulley, G. C., Hook, S. J., Laraby, K., Cook, M., Schott,
J. R. (2018), IEEE TGRS 56(10):5717-5735, doi:10.1109/TGRS.2018.2824828.
The same coefficients appear in Ermida et al. (2020), Remote Sens.
12(9):1471 (SMW ``compute_emissivity``). No NDVI/vegetation/water/snow
adjustment is applied here: this is the *static* GED emissivity.

Georeferencing rule: the HDF5 file carries no usable geotransform. The
affine transform is rebuilt from the pixel-centre geolocation arrays
(spacing 1/999 deg, since 1000 samples include both tile ends) with a
half-pixel shift to the pixel edge, and the arrays are checked to be a
regular north-up grid (non-regular or rotated grids raise).

Token hygiene: the token is read only from the ``EARTHDATA_TOKEN``
environment variable, is sent only in the ``Authorization`` header, and is
scrubbed from every error message. It is never logged or printed.

Cache: downloaded tiles live in ``.scratch/aster-ged-v3/`` (gitignored);
a valid cached tile skips the download (no token needed for the download
step itself); CMR discovery still runs per call.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import numpy as np
import rasterio
import requests
from rasterio import Affine
from rasterio.errors import NotGeoreferencedWarning

AG100_SHORT_NAME = "AG100"
AG100_VERSION = "003"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
EARTHDATA_TOKEN_ENV = "EARTHDATA_TOKEN"  # noqa: S105 — env var name, not a secret
DEFAULT_GED_CACHE_DIR = ".scratch/aster-ged-v3"
GED_EMIS_SCALE = 0.001
GED_BAND13_INDEX = 3  # Emissivity/Mean band order = ASTER 10, 11, 12, 13, 14
GED_BAND14_INDEX = 4
GED_CRS = "EPSG:4326"
GEOLOCATION_TOL_DEG = 1e-5
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

GED_NOTE = (
    "ASTER GED v3 (AG100.003) Emissivity/Mean bands 13/14 x0.001, Malakar 2018 L8 TIRS B10 "
    "spectral adjustment on the native 100 m grid; no NDVI/vegetation/water/snow adjustment"
)


class GedCoefficients(NamedTuple):
    """ASTER GED band 13/14 -> TIRS B10 regression coefficients."""

    c13: float
    c14: float
    c: float


# Malakar et al. 2018, IEEE TGRS 56(10):5717-5735, doi:10.1109/TGRS.2018.2824828
# (spectral adjustment step of the USGS C2 ST algorithm; same coefficients in
# Ermida et al. 2020, Remote Sens. 12(9):1471, SMW compute_emissivity).
MALAKAR2018_L8_B10 = GedCoefficients(c13=0.6820, c14=0.2578, c=0.0584)


class AsterGedError(RuntimeError):
    """ASTER GED discovery, download, or decoding failed."""


class MissingEarthdataTokenError(AsterGedError):
    """The Earthdata bearer token is not available in the environment."""


def ged_coefficients_label(platform: str) -> str:
    """Human-readable label of the spectral-adjustment coefficients per platform."""
    if platform == "landsat-8":
        return "L8 B10 (Malakar 2018)"
    if platform == "landsat-9":
        # Malakar 2018 predates Landsat 9; the TIRS-2 B10 RSR is near-identical
        # to TIRS B10, so the L8 coefficients are used as an approximation.
        return "L8 B10 (Malakar 2018) applied to L9 — approximation"
    raise ValueError(f"unknown platform {platform!r}")


# ── pure physics / georeferencing ────────────────────────────────────────


def ged_dn_to_emissivity(dn: np.ndarray) -> np.ndarray:
    """GED integer DN -> unitless emissivity (float64); DN <= 0 (fill) -> NaN."""
    dn = np.asarray(dn)
    out = dn.astype(np.float64) * GED_EMIS_SCALE
    out[dn <= 0] = np.nan
    return out


def ged_b10_from_bands(
    e13: np.ndarray, e14: np.ndarray, coeffs: GedCoefficients = MALAKAR2018_L8_B10
) -> np.ndarray:
    """Spectrally adjust ASTER bands 13/14 to TIRS B10. NaN propagates; no clamping."""
    return (
        coeffs.c13 * np.asarray(e13, dtype=np.float64)
        + coeffs.c14 * np.asarray(e14, dtype=np.float64)
        + coeffs.c
    )


def geotransform_from_geolocation(
    lat: np.ndarray, lon: np.ndarray, tol: float = GEOLOCATION_TOL_DEG
) -> Affine:
    """Affine transform (pixel edges) from pixel-CENTRE latitude/longitude arrays.

    Accepts 2D arrays of equal shape (rows, cols) or 1D ``lat`` (rows) with
    1D ``lon`` (cols). Row 0 must be north, column 0 west. The full arrays
    must match a regular grid to within ``tol`` degrees, else ValueError.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        nrows, ncols = lat.size, lon.size
        lat = np.broadcast_to(lat[:, None], (nrows, ncols))
        lon = np.broadcast_to(lon[None, :], (nrows, ncols))
    elif lat.ndim == 2 and lat.shape == lon.shape:
        nrows, ncols = lat.shape
    else:
        raise ValueError(f"lat/lon shapes {lat.shape} and {lon.shape} are not compatible")
    if nrows < 2 or ncols < 2:
        raise ValueError("geolocation needs at least 2 rows and 2 columns")
    lon0 = float(lon[0, 0])
    lat0 = float(lat[0, 0])
    dx = (float(lon[0, -1]) - lon0) / (ncols - 1)
    dy = (float(lat[-1, 0]) - lat0) / (nrows - 1)
    if not dx > 0:
        raise ValueError(f"longitude must increase eastwards along columns (dx={dx})")
    if not dy < 0:
        raise ValueError(f"latitude must decrease southwards along rows (dy={dy})")
    ideal_lon = lon0 + np.arange(ncols, dtype=np.float64) * dx
    ideal_lat = lat0 + np.arange(nrows, dtype=np.float64) * dy
    dev_lon = float(np.max(np.abs(lon - ideal_lon[None, :])))
    dev_lat = float(np.max(np.abs(lat - ideal_lat[:, None])))
    if not (dev_lon < tol and dev_lat < tol):
        raise ValueError(
            f"geolocation is not a regular north-up grid (max deviation lon {dev_lon:.3g}, "
            f"lat {dev_lat:.3g} deg >= tol {tol:g})"
        )
    return Affine(dx, 0.0, lon0 - dx / 2.0, 0.0, dy, lat0 - dy / 2.0)


def _scrub(text: str, token: str) -> str:
    """Replace the token in ``text`` with ``***`` (no-op for an empty token)."""
    if not token:
        return text
    return text.replace(token, "***")


# ── auth / discovery / download ──────────────────────────────────────────


def require_earthdata_token() -> str:
    """Return the Earthdata bearer token from the environment or raise."""
    token = os.environ.get(EARTHDATA_TOKEN_ENV)
    if token is None or not token.strip():
        raise MissingEarthdataTokenError(
            "EARTHDATA_TOKEN is not set; ASTER GED download needs a NASA Earthdata bearer token"
        )
    return token.strip()


def cmr_granule_urls(
    bbox: tuple[float, float, float, float],
    *,
    session: requests.Session | None = None,
    timeout: float = 60,
) -> list[str]:
    """Discover AG100.003 granule download URLs (https ``.h5``) covering ``bbox``."""
    west, south, east, north = bbox
    params = {
        "short_name": AG100_SHORT_NAME,
        "version": AG100_VERSION,
        "bounding_box": f"{west},{south},{east},{north}",
        "page_size": 200,
    }
    if session is None:
        session = requests.Session()
    failure: str | None = None
    resp: Any = None
    try:
        resp = session.get(CMR_GRANULES_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        failure = f"CMR request failed: {type(exc).__name__}: {exc}"
    if failure is not None:
        raise AsterGedError(failure)
    if resp.status_code != 200:
        raise AsterGedError(f"CMR granule search returned HTTP {resp.status_code}")
    urls: list[str] = []
    try:
        entries = resp.json().get("feed", {}).get("entry", [])
    except ValueError:
        raise AsterGedError("CMR granule search returned non-JSON content") from None
    for entry in entries:
        for link in entry.get("links", []):
            href = str(link.get("href", ""))
            if href.startswith("https://") and href.endswith(".h5") and href not in urls:
                urls.append(href)
                break  # one URL per granule
    if not urls:
        raise AsterGedError(
            f"CMR returned no {AG100_SHORT_NAME}.{AG100_VERSION} granules for bbox {bbox}"
        )
    return urls


def _read_subdataset(path: str | Path, name: str) -> np.ndarray:
    """Read all bands of one HDF5 subdataset as (bands, rows, cols). Test seam."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(f'HDF5:"{path}"://{name}') as src:
            return src.read()


def _is_valid_ged_file(path: str | Path) -> bool:
    """True if ``path`` is an HDF5 GED tile with a 5-band ``Emissivity/Mean``."""
    try:
        p = Path(path)
        if not p.is_file():
            return False
        with open(p, "rb") as f:
            if f.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
                return False
        arr = _read_subdataset(p, "Emissivity/Mean")
        return arr.ndim == 3 and arr.shape[0] == 5
    except Exception:  # noqa: BLE001 — any read failure means "not a valid tile"
        return False


def download_granule(
    url: str,
    cache_dir: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (30, 600),
) -> Path:
    """Download one GED tile into ``cache_dir`` (atomic); reuse a valid cached tile."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / os.path.basename(urlparse(url).path)
    if _is_valid_ged_file(target):
        return target
    token = require_earthdata_token()
    if session is None:
        session = requests.Session()
    # The error is raised outside the ``except`` block so the original
    # exception (whose text may embed the token) is not chained as __context__.
    failure: str | None = None
    resp: Any = None
    try:
        resp = session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            allow_redirects=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        failure = _scrub(f"GED download failed for {url}: {type(exc).__name__}: {exc}", token)
    if failure is not None:
        raise AsterGedError(failure)
    tmp_name: str | None = None
    try:
        if resp.status_code != 200:
            raise AsterGedError(
                _scrub(f"GED download returned HTTP {resp.status_code} for {url}", token)
            )
        if "text/html" in str(resp.headers.get("Content-Type", "")).lower():
            raise AsterGedError(
                _scrub(
                    f"GED download for {url} got an HTML page (Earthdata login?) instead of HDF5",
                    token,
                )
            )
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as tmp:
            tmp_name = tmp.name
            try:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    tmp.write(chunk)
            except requests.RequestException as exc:
                failure = _scrub(
                    f"GED download interrupted for {url}: {type(exc).__name__}: {exc}", token
                )
        if failure is not None:
            raise AsterGedError(failure)
        with open(tmp_name, "rb") as f:
            head = f.read(len(HDF5_SIGNATURE))
        if head != HDF5_SIGNATURE:
            hint = " (looks like HTML; Earthdata login?)" if head.startswith(b"<") else ""
            raise AsterGedError(f"downloaded file for {url} is not HDF5{hint}")
        if not _is_valid_ged_file(tmp_name):
            raise AsterGedError(f"downloaded file for {url} is not a valid GED tile")
        os.replace(tmp_name, target)
        tmp_name = None
        return target
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        close = getattr(resp, "close", None)
        if callable(close):
            close()


# ── reading / mosaicking ─────────────────────────────────────────────────


def _squeeze_geo(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def read_ged_b10(path: str | Path) -> dict:
    """Read one tile: B10 emissivity on the native 100 m grid plus its transform."""
    emis = _read_subdataset(path, "Emissivity/Mean")
    if emis.ndim != 3 or emis.shape[0] != 5:
        raise AsterGedError(f"Emissivity/Mean in {path} has shape {emis.shape}; expected (5, R, C)")
    nrows, ncols = emis.shape[1:]
    lat = _squeeze_geo(_read_subdataset(path, "Geolocation/Latitude"))
    lon = _squeeze_geo(_read_subdataset(path, "Geolocation/Longitude"))
    if lat.ndim == 2 and lon.ndim == 2:
        ok = lat.shape == (nrows, ncols) and lon.shape == (nrows, ncols)
    else:
        ok = lat.shape == (nrows,) and lon.shape == (ncols,)
    if not ok:
        raise AsterGedError(
            f"geolocation shapes {lat.shape}/{lon.shape} do not match emissivity grid "
            f"({nrows}, {ncols}) in {path}"
        )
    e13 = ged_dn_to_emissivity(emis[GED_BAND13_INDEX])
    e14 = ged_dn_to_emissivity(emis[GED_BAND14_INDEX])
    return {
        "array": ged_b10_from_bands(e13, e14),
        "transform": geotransform_from_geolocation(lat, lon),
        "crs": GED_CRS,
    }


def _near_int(value: float, what: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-3:
        raise ValueError(f"GED tile {what} offset {value} is not an integer pixel count")
    return int(rounded)


def mosaic_ged_tiles(tiles: list[dict]) -> tuple[np.ndarray, Affine]:
    """Place same-resolution tiles on their union grid (no resampling).

    Adjacent GED tiles share their edge row/column; in overlapping cells
    the first finite value wins.
    """
    if not tiles:
        raise ValueError("no tiles to mosaic")
    if len(tiles) == 1:
        return tiles[0]["array"], tiles[0]["transform"]
    dx = tiles[0]["transform"].a
    dy = tiles[0]["transform"].e
    for t in tiles[1:]:
        if abs(t["transform"].a - dx) > 1e-6 * abs(dx) or abs(t["transform"].e - dy) > 1e-6 * abs(
            dy
        ):
            raise ValueError("GED tiles have different resolutions; refusing to mosaic")
    left = min(t["transform"].c for t in tiles)
    top = max(t["transform"].f for t in tiles)
    right = max(t["transform"].c + t["array"].shape[1] * dx for t in tiles)
    bottom = min(t["transform"].f + t["array"].shape[0] * dy for t in tiles)
    ncols = _near_int((right - left) / dx, "union width")
    nrows = _near_int((bottom - top) / dy, "union height")
    dest = np.full((nrows, ncols), np.nan, dtype=np.float64)
    for t in tiles:
        c0 = _near_int((t["transform"].c - left) / dx, "column")
        r0 = _near_int((t["transform"].f - top) / dy, "row")
        h, w = t["array"].shape
        window = dest[r0 : r0 + h, c0 : c0 + w]
        fill = ~np.isfinite(window)
        window[fill] = t["array"][fill]
    return dest, Affine(dx, 0.0, left, 0.0, dy, top)


def load_ged_b10_for_bbox(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path = DEFAULT_GED_CACHE_DIR,
    *,
    session: requests.Session | None = None,
) -> dict:
    """Discover, download (cached), read, and mosaic GED B10 tiles for ``bbox``."""
    require_earthdata_token()  # fail fast, before any network access
    urls = cmr_granule_urls(bbox, session=session)
    paths = [download_granule(u, cache_dir, session=session) for u in urls]
    tiles = [read_ged_b10(p) for p in paths]
    array, transform = mosaic_ged_tiles(tiles)
    return {
        "array": array,
        "transform": transform,
        "crs": GED_CRS,
        "granules": [p.name for p in paths],
        "note": GED_NOTE,
    }


__all__ = [
    "AG100_SHORT_NAME",
    "AG100_VERSION",
    "CMR_GRANULES_URL",
    "DEFAULT_GED_CACHE_DIR",
    "EARTHDATA_TOKEN_ENV",
    "GED_BAND13_INDEX",
    "GED_BAND14_INDEX",
    "GED_CRS",
    "GED_EMIS_SCALE",
    "GED_NOTE",
    "GEOLOCATION_TOL_DEG",
    "HDF5_SIGNATURE",
    "MALAKAR2018_L8_B10",
    "AsterGedError",
    "GedCoefficients",
    "MissingEarthdataTokenError",
    "cmr_granule_urls",
    "download_granule",
    "ged_b10_from_bands",
    "ged_coefficients_label",
    "ged_dn_to_emissivity",
    "geotransform_from_geolocation",
    "load_ged_b10_for_bbox",
    "mosaic_ged_tiles",
    "read_ged_b10",
    "require_earthdata_token",
]
