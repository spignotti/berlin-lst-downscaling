"""Cloud-masking per sensor + S2 directional cloud-shadow projection.

Functions are pure — they take an ``xr.Dataset`` (loaded by the
acquisition layer) and return a new ``xr.Dataset`` whose bands match
the :class:`Contract` for that source.
"""

from __future__ import annotations

import numpy as np
import rasterio
import rioxarray  # noqa: F401  — registers rio accessor with xr
import xarray as xr
from omegaconf import DictConfig
from rasterio.warp import Resampling
from scipy.ndimage import binary_dilation

from berlin_lst_downscaling.data.ard.contract import contract_for_source

# ── Landsat ─────────────────────────────────────────────────────────##

_LS_ST_SCALE = 0.00341802  # USGS Collection 2 Level-2 ST scale
_LS_ST_OFFSET = 149.0  # K

def landsat_qa_to_clear_bits(qa: np.ndarray) -> np.ndarray:
    """Return boolean array — True = pixel is clear of cloud/shadow/cirrus/fill.

    Landsat C2 L2 QA_PIXEL bits used:
      bit 0 (1)  — Fill (designated nodata)
      bit 2 (4)  — Cirrus (high confidence)
      bit 3 (8)  — Cloud (raw flag)
      bit 4 (16) — Cloud shadow
    Cloud with confidence ≥ medium (bits 8–9 ≥ 2) is the canonical
    "cloud" interpretation; lower-confidence hints are ignored.
    Bit 1 (dilated cloud) is not included — it is handled separately
    as an ARD-only dilation buffer.

    Snow (bit 6, value 64) and water (clear-water QA=192) are *clear*
    by this function — water bodies are common in the AOI and must not
    be excluded from coupling.
    """
    cloud_raw = (qa >> 3) & 1
    cloud_conf = (qa >> 8) & 0b11
    cloudy = (cloud_raw != 0) & (cloud_conf >= 2)
    cirrus = (qa >> 2) & 1
    shadow = (qa >> 4) & 1
    fill = qa & 1
    return ~(fill.astype(bool) | cloudy | shadow.astype(bool) | cirrus.astype(bool))

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
    contract = contract_for_source("landsat-c2-l2")

    qa = ds["qa_pixel"].values.squeeze().astype(np.uint16)
    flag = np.zeros(qa.shape, dtype=np.uint8)

    cloud_raw = (qa >> 3) & 1
    cloud_conf = (qa >> 8) & 0b11
    cloudy = (cloud_raw != 0) & (cloud_conf >= 2)
    shadow = (qa >> 4) & 1
    cirrus = (qa >> 2) & 1

    flag[(qa & 1) != 0] |= contract.FLAG_FILL
    flag[cloudy] |= contract.FLAG_CLOUDY
    flag[shadow.astype(bool)] |= contract.FLAG_SHADOW
    flag[cirrus.astype(bool)] |= contract.FLAG_CIRRUS

    # apply additional dilation to the cloud mask (ARD-only buffer)
    dilate_px = cfg.get("cloud_dilation_px", 2)
    if dilate_px > 0 and cloudy.any():
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
        buffered = binary_dilation(cloudy, structure=struct)
        # dilated buffer → fill bit
        flag[buffered & ~cloudy] |= contract.FLAG_FILL

    raw = ds["lwir11"].values.squeeze().astype(np.float32)
    # DN=0 is fill in USGS C2 L2 ST; not always caught by QA_PIXEL bit 0
    flag[raw == 0] |= contract.FLAG_FILL
    st_kelvin = raw * _LS_ST_SCALE + _LS_ST_OFFSET
    st_kelvin[(flag & contract.FLAG_FILL) != 0] = float("nan")

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

# Native 20m SWIR bands, masked at 20m before bilinear 20→10m resampling.
_S2_SWIR_BAND_NAMES = ("B11", "B12")

def _s2_flag(
    scl: np.ndarray,
    contract,
    *,
    cloud_mask: np.ndarray | None = None,
    transform: rasterio.Affine | None = None,
    cfg: DictConfig | None = None,
    sun_azimuth_deg: float = 0.0,
    sun_elevation_deg: float = 0.0,
) -> np.ndarray:
    """Derive the ARD flag bitmask from SCL classes.

    ``scl`` is the rounded uint8 SCL array.  When ``cloud_mask`` and
    ``transform`` are given, the directional cloud-shadow projection is
    OR'd onto the shadow bit (used at 10m, matching the pre-SWIR
    behaviour).  Without them, only the direct SCL classes are used
    (used at 20m to mask SWIR before resampling).
    """
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

    # snow / ice — SCL class 11 (excluded from clear sky like the
    # selection layer, which treats SCL 11 as not-clear)
    flag[scl == 11] |= contract.FLAG_SNOW_ICE

    if (
        cloud_mask is not None
        and cloud_mask.any()
        and transform is not None
        and cfg is not None
        and sun_elevation_deg > 0.5
    ):
        proj_mask = _project_cloud_shadow(
            cloud_mask,
            sun_azimuth_deg,
            sun_elevation_deg,
            cfg.get("cloud_base_height_m", 1000),
            transform,
        )
        flag[proj_mask] |= contract.FLAG_SHADOW

    return flag


def mask_s2(
    ds_10m: xr.Dataset,
    ds_20m: xr.Dataset,
    cfg: DictConfig,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> xr.Dataset:
    """Apply S2 ARD masking: six scaled-reflectance bands + flag band.

    Parameters
    ----------
    ds_10m :
        Dataset from :func:`~berlin_lst_downscaling.data.acquisition.load_s2_scene`
        on the canonical 10m grid, containing ``B02, B03, B04, B08``
        (float32 raw DN) and ``SCL``.
    ds_20m :
        Same source, loaded on the canonical 20m grid with
        ``B11, B12`` (float32 raw DN) and ``SCL``.
    cfg :
        Hydra config (uses ``cloud_base_height_m``).
    sun_azimuth_deg, sun_elevation_deg :
        Solar position for cloud-shadow projection.

    Returns
    -------
    xr.Dataset with bands ``B02, B03, B04, B08, B11, B12`` (float32 0-1)
    and ``flag`` (uint8, 10m).

    SWIR handling: B11/B12 are masked at their native 20m resolution
    (any non-zero SCL flag → NaN) **before** the bilinear 20→10m
    resampling, so cloudy/fill SWIR samples never contribute to the
    10m product. NaN is declared as nodata on the source, so it
    propagates through the bilinear kernel (verified empirically: a
    NaN 20m sample yields NaN in its 2×2 10m footprint, valid
    neighbours interpolate cleanly). After upsampling the 10m fill
    bit is applied like the other four bands.
    """
    contract = contract_for_source("sentinel-2-l2a")

    # ── 10m flag (existing behaviour, incl. shadow projection) ──────
    scl_10m = np.round(ds_10m["SCL"].values.squeeze()).astype(np.uint8)
    cloud_mask = (scl_10m == 8) | (scl_10m == 9)
    flag = _s2_flag(
        scl_10m,
        contract,
        cloud_mask=cloud_mask,
        transform=ds_10m.rio.transform(),
        cfg=cfg,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
    )

    # ── 20m SWIR: mask at native resolution before resampling ───────
    scl_20m = np.round(ds_20m["SCL"].values.squeeze()).astype(np.uint8)
    flag_20m = _s2_flag(scl_20m, contract)

    swir_scaled: dict[str, np.ndarray] = {}
    for b in _S2_SWIR_BAND_NAMES:
        arr = ds_20m[b].values.squeeze().astype(np.float32)
        arr = arr * _S2_DN_SCALE
        arr = np.clip(arr, 0.0, 1.0)
        # any non-zero flag → invalid SWIR sample, excluded from the
        # bilinear kernel (NaN propagates; see docstring)
        arr[flag_20m != 0] = float("nan")
        swir_scaled[b] = arr

    # ── assemble output on the 10m grid ─────────────────────────────
    bands_10m = ["B02", "B03", "B04", "B08"]
    scaled: dict[str, np.ndarray] = {}
    for b in bands_10m:
        arr = ds_10m[b].values.squeeze().astype(np.float32)
        arr = arr * _S2_DN_SCALE
        arr = np.clip(arr, 0.0, 1.0)
        # propagate fill NaN
        arr[(flag & contract.FLAG_FILL) != 0] = float("nan")
        scaled[b] = arr

    coords = dict(ds_10m.coords)
    dims = ds_10m.dims
    out_vars: dict[str, xr.DataArray] = {}
    for b in bands_10m:
        out_vars[b] = xr.DataArray(
            scaled[b][np.newaxis, ...] if "time" in dims else scaled[b],
            dims=ds_10m[b].dims,
            attrs={"long_name": f"S2 {b}", "units": "1"},
        )

    # bilinear 20m → 10m for the masked SWIR bands

    # 2D 10m match target (odc loads may carry a size-1 ``time`` dim;
    # the reproject is purely spatial).
    target_10m = ds_10m["B08"] if "time" not in dims else ds_10m["B08"].isel(time=0)

    for b in _S2_SWIR_BAND_NAMES:
        swir_da = xr.DataArray(
            swir_scaled[b],
            dims=("y", "x"),
            attrs={"long_name": f"S2 {b}", "units": "1"},
        ).rio.write_crs(ds_20m.rio.crs).rio.write_transform(ds_20m.rio.transform())
        # Declare NaN as nodata so GDAL excludes invalid SWIR samples
        # from the bilinear kernel instead of smearing them (see docstring).
        swir_da = swir_da.rio.write_nodata(float("nan"), encoded=False)
        # Match the loaded 10m grid (rioxarray requires rio metadata on
        # the target; ds_10m bands carry it from the odc load).
        swir_10m = swir_da.rio.reproject_match(
            target_10m, resampling=Resampling.bilinear
        )
        arr_10m = np.asarray(swir_10m.values.squeeze(), dtype=np.float32)
        arr_10m[(flag & contract.FLAG_FILL) != 0] = float("nan")
        out_vars[b] = xr.DataArray(
            arr_10m[np.newaxis, ...] if "time" in dims else arr_10m,
            dims=ds_10m["B08"].dims,
            attrs={"long_name": f"S2 {b}", "units": "1"},
        )

    out_vars["flag"] = xr.DataArray(
        flag[np.newaxis, ...] if "time" in dims else flag,
        dims=ds_10m[bands_10m[0]].dims,
        attrs={"long_name": "Quality flag", "flags": _FLAG_DOC},
    )

    out = xr.Dataset(out_vars, coords=coords)
    for var in out.data_vars:
        out[var].rio.write_crs(ds_10m.rio.crs, inplace=True)
        out[var].rio.write_transform(ds_10m.rio.transform(), inplace=True)

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

# ── ECOSTRESS ───────────────────────────────────────────────────────##

def mask_ecostress(ds: xr.Dataset, cfg: DictConfig) -> xr.Dataset:
    """Apply ECOSTRESS L2T masking: LST + flag band from cloud/water/QC layers.

    Dataset ``ds`` from
    :func:`~berlin_lst_downscaling.data.acquisition.ecostress.load_ecostress_scene`
    contains bands ``lst`` (float32 K), ``cloud`` (uint8), ``water`` (uint8),
    and ``qc`` (uint8).

    Flag derivation (Collection 2 L2T semantics):

    | Source value | Flag bit | Meaning |
    |---|---|---|
    | ``cloud == 255`` | ``FLAG_FILL`` | fill / outside granule |
    | ``cloud == 1`` | ``FLAG_CLOUDY`` | high-confidence cloud |
    | ``(qc & 0b11) == 3`` | ``FLAG_FILL`` | pixel not produced |
    | ``(qc & 0b11) == 1`` | ``FLAG_CLOUDY`` | TES degraded (conservative) |
    | ``water == 1`` | ``FLAG_FILL`` | water bodies excluded from urban LST |

    There is **no cloud-shadow projection** for ECOSTRESS — the L2T product
    does not carry the per-pixel geometry needed for directional shadow
    casting (ray-cast shadows deferred to Sekundärdaten-Pipeline Stage 3).

    Returns
    -------
    xr.Dataset with bands ``lst`` (float32 Kelvin) and ``flag`` (uint8).
    """
    contract = contract_for_source("ecostress")

    lst_arr = ds["lst"].values.astype(np.float32)
    cloud_arr = ds["cloud"].values.astype(np.uint8)
    water_arr = ds["water"].values.astype(np.uint8)
    qc_arr = ds["qc"].values.astype(np.uint8)

    flag = np.zeros(lst_arr.shape, dtype=np.uint8)

    # cloud (0=clear, 1=cloud, 255=fill)
    flag[cloud_arr == 255] |= contract.FLAG_FILL
    flag[cloud_arr == 1] |= contract.FLAG_CLOUDY

    # water (0=dry, 1=water, 255=fill)
    flag[water_arr == 255] |= contract.FLAG_FILL
    flag[water_arr == 1] |= contract.FLAG_FILL

    # QC mandatory QA (bits 0-1): 0b00=best, 0b01=degraded, 0b10=not-set, 0b11=not-produced
    qc_low2 = qc_arr & 0b11
    flag[qc_low2 == 3] |= contract.FLAG_FILL  # pixel not produced
    flag[qc_low2 == 1] |= contract.FLAG_CLOUDY  # TES degraded (conservative)

    lst_arr[(flag & contract.FLAG_FILL) != 0] = float("nan")

    # Close the NaN⟺fill invariant: bilinear LST reprojection spreads NaN
    # beyond the nearest-reprojected fill/water/QC regions, so any NaN the
    # mask left unclassified is treated as fill.
    flag[np.isnan(lst_arr)] |= contract.FLAG_FILL

    coords = dict(ds.coords)
    dims = ds.dims

    out = xr.Dataset(
        {
            "lst": xr.DataArray(
                lst_arr[np.newaxis, ...] if "time" in dims else lst_arr,
                dims=ds["lst"].dims,
                attrs={"long_name": "Land Surface Temperature", "units": "K"},
            ),
            "flag": xr.DataArray(
                flag[np.newaxis, ...] if "time" in dims else flag,
                dims=ds["lst"].dims,
                attrs={"long_name": "Quality flag", "flags": _FLAG_DOC},
            ),
        },
        coords=coords,
    )
    for var in out.data_vars:
        out[var].rio.write_crs(ds.rio.crs, inplace=True)
        out[var].rio.write_transform(ds.rio.transform(), inplace=True)

    return out

_FLAG_DOC = "bit0=fill, bit1=cloudy, bit2=cloud_shadow, bit3=cirrus, bit4=saturated, bit5=snow_ice"

__all__ = [
    "landsat_qa_to_clear_bits",
    "mask_landsat",
    "mask_s2",
    "mask_ecostress",
]