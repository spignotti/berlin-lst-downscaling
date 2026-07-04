"""Cloud-masking per sensor + S2 directional cloud-shadow projection.

Functions are pure — they take an ``xr.Dataset`` (loaded by the
acquisition layer) and return a new ``xr.Dataset`` whose bands match
the :class:`Contract` for that source.
"""

from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from omegaconf import DictConfig
from scipy.ndimage import binary_dilation

from berlin_lst_downscaling.data.ard.contract import Contract

# ── Landsat ─────────────────────────────────────────────────────────##


_LS_ST_SCALE = 0.00341802  # USGS Collection 2 Level-2 ST scale
_LS_ST_OFFSET = 149.0  # K


def mask_landsat(ds: xr.Dataset, cfg: DictConfig) -> xr.Dataset:
    """Apply Landsat ARD masking: ST (Kelvin) + flag band.

    Parameters
    ----------
    ds :
        Dataset from :func:`~berlin_lst_downscaling.data.acquisition.load_landsat_scene`
        containing ``lwir11`` and ``qa_pixel``.
    cfg :
        Hydra config (uses ``cloud_dilation_px``).

    Returns
    -------
    xr.Dataset with bands ``st`` (float32, Kelvin) and ``flag`` (uint8).
    """
    contract = _contract("landsat-c2-l2")

    # --- derive flag from qa_pixel ---
    qa = ds["qa_pixel"].values.squeeze().astype(np.uint16)
    flag = np.zeros(qa.shape, dtype=np.uint8)

    # bit 0: fill
    flag[(qa & 0b1) != 0] |= contract.FLAG_FILL

    # cloud: bit 3 with confidence ≥ medium (bits 8-9 ≥ 2)
    cloud_raw = (qa >> 3) & 1
    cloud_conf = (qa >> 8) & 0b11
    cloudy = (cloud_raw != 0) & (cloud_conf >= 2)
    flag[cloudy] |= contract.FLAG_CLOUDY

    # apply additional dilation to the cloud mask
    dilate_px = cfg.get("cloud_dilation_px", 2)
    if dilate_px > 0 and cloudy.any():
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
        buffered = binary_dilation(cloudy, structure=struct)
        # dilated buffer → fill bit
        flag[buffered & ~cloudy] |= contract.FLAG_FILL

    # cloud shadow: bit 4
    flag[((qa >> 4) & 1) != 0] |= contract.FLAG_SHADOW

    # cirrus: bit 2 (UA cirrus flag in Collection 2)
    flag[((qa >> 2) & 1) != 0] |= contract.FLAG_CIRRUS

    # --- ST band ---
    raw = ds["lwir11"].values.squeeze().astype(np.float32)
    st_kelvin = raw * _LS_ST_SCALE + _LS_ST_OFFSET
    st_kelvin[(flag & contract.FLAG_FILL) != 0] = float("nan")

    # --- build output Dataset ---
    coords = dict(ds.coords)
    dims = ds.dims

    out = xr.Dataset(
        {
            "st": xr.DataArray(
                st_kelvin[np.newaxis, ...] if "time" in dims else st_kelvin,
                dims=ds["lwir11"].dims,
                attrs={"long_name": "Surface Temperature", "units": "K"},
            ),
            "flag": xr.DataArray(
                flag[np.newaxis, ...] if "time" in dims else flag,
                dims=ds["lwir11"].dims,
                attrs={"long_name": "Quality flag", "flags": _FLAG_DOC},
            ),
        },
        coords=coords,
    )
    # propagate CRS via rioxarray
    for var in out.data_vars:
        out[var].rio.write_crs(ds.rio.crs, inplace=True)
        out[var].rio.write_transform(ds.rio.transform(), inplace=True)

    return out


# ── Sentinel-2 ───────────────────────────────────────────────────────##


_S2_DN_SCALE = 1.0 / 10000.0  # Baseline 04.00 scaled reflectance


def mask_s2(
    ds: xr.Dataset,
    cfg: DictConfig,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> xr.Dataset:
    """Apply S2 ARD masking: scaled reflectance + flag band.

    Parameters
    ----------
    ds :
        Dataset from :func:`~berlin_lst_downscaling.data.acquisition.load_s2_scene`
        containing ``B02, B03, B04, B08`` (float32 raw DN) and
        ``SCL`` (float32).
    cfg :
        Hydra config (uses ``cloud_base_height_m``).
    sun_azimuth_deg, sun_elevation_deg :
        Solar position for cloud-shadow projection.

    Returns
    -------
    xr.Dataset with bands ``B02, B03, B04, B08`` (float32 0-1) and
    ``flag`` (uint8).
    """
    contract = _contract("sentinel-2-l2a")

    # --- flag from SCL (Scene Classification Layer) ---
    # SCL comes as float32 from odc.stac.load; round to int for class values
    scl_raw = ds["SCL"].values.squeeze()
    scl = np.round(scl_raw).astype(np.uint8)
    flag = np.zeros(scl.shape, dtype=np.uint8)

    # fill
    flag[scl == 0] |= contract.FLAG_FILL

    # cloudy: medium (8) and high (9) probability
    flag[(scl == 8) | (scl == 9)] |= contract.FLAG_CLOUDY

    # cloud shadow (disjunctive SCL detector — lower bound)
    flag[scl == 3] |= contract.FLAG_SHADOW

    # cirrus
    flag[scl == 10] |= contract.FLAG_CIRRUS

    # saturated
    flag[scl == 1] |= contract.FLAG_SATURATED

    # --- directional cloud-shadow projection ---
    cloud_mask = (scl == 8) | (scl == 9)
    if cloud_mask.any() and sun_elevation_deg > 0.5:
        transform = ds.rio.transform()
        proj_mask = _project_cloud_shadow(
            cloud_mask,
            sun_azimuth_deg,
            sun_elevation_deg,
            cfg.get("cloud_base_height_m", 1000),
            transform,
        )
        flag[proj_mask] |= contract.FLAG_SHADOW

    # --- scale reflectance bands ---
    bands_10m = ["B02", "B03", "B04", "B08"]
    scaled = {}
    for b in bands_10m:
        arr = ds[b].values.squeeze().astype(np.float32)
        arr = arr * _S2_DN_SCALE
        arr = np.clip(arr, 0.0, 1.0)
        # propagate fill NaN
        arr[(flag & contract.FLAG_FILL) != 0] = float("nan")
        scaled[b] = arr

    # --- build output Dataset ---
    coords = dict(ds.coords)
    dims = ds.dims
    out_vars = {}
    for b in bands_10m:
        out_vars[b] = xr.DataArray(
            scaled[b][np.newaxis, ...] if "time" in dims else scaled[b],
            dims=ds[b].dims,
            attrs={"long_name": f"S2 {b}", "units": "1"},
        )
    out_vars["flag"] = xr.DataArray(
        flag[np.newaxis, ...] if "time" in dims else flag,
        dims=ds[bands_10m[0]].dims,
        attrs={"long_name": "Quality flag", "flags": _FLAG_DOC},
    )

    out = xr.Dataset(out_vars, coords=coords)
    for var in out.data_vars:
        out[var].rio.write_crs(ds.rio.crs, inplace=True)
        out[var].rio.write_transform(ds.rio.transform(), inplace=True)

    return out


# ── cloud-shadow projection (directional offset, not ray-cast) ───────##


def _project_cloud_shadow(
    cloud_mask: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    cloud_height_m: float,
    transform: rasterio.Affine,
) -> np.ndarray:
    """Directional-offset cloud-shadow projection.

    For each cloud pixel, shift the mask in the solar direction by
    ``cloud_height × tan(zenith)`` meters.  Returns a boolean mask
    of the same shape where ``True`` = shadow.

    .. note::
       This is **not** ray-cast — shadows behind tall DSM features are
       not caught.  Full ray-cast is Stage 3 (Sekundärdaten-Pipeline).
    """
    if sun_elevation_deg <= 0.5:
        return np.zeros_like(cloud_mask, dtype=bool)

    zenith_rad = np.deg2rad(90.0 - sun_elevation_deg)
    azimuth_rad = np.deg2rad(sun_azimuth_deg)

    # horizontal offset magnitude
    horiz_m = cloud_height_m * np.tan(zenith_rad)
    if horiz_m < 1.0:
        return np.zeros_like(cloud_mask, dtype=bool)

    # ground displacement in CRS (easting, northing)
    # shadow is cast opposite the sun: -x (east), -y (north)
    dx_m = -horiz_m * np.sin(azimuth_rad)
    dy_m = -horiz_m * np.cos(azimuth_rad)

    # convert to pixel shifts
    # transform.a + transform.e = pixel resolution in CRS units (m)
    # transform.e is typically negative (north-up)
    dx_px = dx_m / abs(transform.a)
    dy_px = -dy_m / abs(transform.e)  # negative because y-axis is inverted in CRS

    # nearest-neighbour shift of the boolean cloud mask
    from scipy.ndimage import shift as _ndshift

    shifted = _ndshift(
        cloud_mask.astype(np.float32),
        shift=(dy_px, dx_px),
        order=0,
        mode="constant",
        cval=0.0,
    )
    return shifted > 0.5


# ── contract helper ──────────────────────────────────────────────────##


def _contract(source: str) -> Contract:
    from berlin_lst_downscaling.data.ard.contract import contract_for_source as _cf

    return _cf(source)


_FLAG_DOC = "bit0=fill, bit1=cloudy, bit2=cloud_shadow, bit3=cirrus, bit4=saturated"

__all__ = [
    "mask_landsat",
    "mask_s2",
]
