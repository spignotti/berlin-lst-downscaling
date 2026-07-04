"""Pixel-wise clear_frac computation on the canonical 10 m EPSG:25833 grid.

clear_frac = sum(aoi & l8_clear & s2_clear) / sum(aoi & l8_clear)
             = fraction of clear S2 pixels among the Landsat-clear AOI pixels

The denominator uses Landsat clear as the reference baseline, per the
Szenen-Selektion spec: "clear_frac immer relativ zur Schnittmenge mit
klaren Landsat-Pixeln, nicht zur ganzen Szene."

This function is the expensive part of the coupling (pixel loads).  It is
called only for the top-N S2 candidates per anchor (N is typically small,
e.g. 3–7 scenes in a ±3-day window).  The volume-scan mode does NOT call
this function.
"""

from __future__ import annotations

import numpy as np
import odc.stac
import rasterio.warp as rwarp
import xarray as xr


def compute_clear_frac(
    l8_items: list,
    s2_items: list,
    anchor_bbox: tuple[float, float, float, float],
    aoi_mask_path: str = "data/boundaries/aoi_10m.tif",
    resolution: int = 10,
) -> float:
    """Compute clear_frac for a (Landsat, S2) pair on the 10 m canonical grid.

    Parameters
    ----------
    l8_items :
        STAC items for the Landsat scene (pystac Item list).
    s2_items :
        STAC items for the Sentinel-2 candidate (pystac Item list).
    anchor_bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)`` of the coupling region.
    aoi_mask_path :
        Path to the pre-baked Berlin AOI mask (uint8, 1=inside).
    resolution :
        Target resolution in metres (default 10 m).

    Returns
    -------
    float
        clear_frac in [0.0, 1.0].  Returns NaN if the AOI intersection
        contains zero Landsat-clear pixels.
    """
    if not l8_items or not s2_items:
        return float("nan")

    # ── load both sources onto the same 10-m EPSG:25833 grid ─────────────────
    l8_ds = odc.stac.load(
        items=l8_items,
        bands=["qa_pixel"],
        crs="EPSG:25833",
        resolution=resolution,
        bbox=anchor_bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )

    s2_ds = odc.stac.load(
        items=s2_items,
        bands=["SCL"],
        crs="EPSG:25833",
        resolution=resolution,
        bbox=anchor_bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )

    # ── build boolean clear masks ───────────────────────────────────────────
    l8_clear = _landsat_is_clear(l8_ds)
    s2_clear = _s2_is_clear(s2_ds)

    # ── AOI mask (reproject to the scene grid) ──────────────────────────────
    aoi = _load_aoi_mask(aoi_mask_path, l8_ds)

    # ── compute clear_frac ───────────────────────────────────────────────────
    # denominator: AOI ∩ Landsat-clear pixels
    denom = int(np.sum(aoi & l8_clear))
    if denom == 0:
        return float("nan")

    # numerator: AOI ∩ Landsat-clear ∩ S2-clear pixels
    numer = int(np.sum(aoi & l8_clear & s2_clear))
    return numer / denom


def _landsat_is_clear(ds: xr.Dataset) -> np.ndarray:
    """Return boolean array where True = clear according to QA_PIXEL.

    QA_PIXEL bits (Collection 2):
      bit 1  (1):  fill
      bit 2  (2):  dilated cloud
      bit 3  (4):  cloud
      bit 4  (8):  cloud shadow
      bit 5  (16): water
      bit 6  (32): snow / ice (kept as clear)
      bit 7  (64): cirrus
      bit 8 (128): drop?  (radiometric saturation flags in bits 9-11)
    Clear = none of {fill, cloud, cloud shadow, cirrus} set.
    """
    qa = ds["qa_pixel"].values[0]  # (y, x)

    fill     = (qa & 1)   != 0
    cloud    = (qa & 4)   != 0
    shadow   = (qa & 8)   != 0
    cirrus   = (qa & 64)  != 0

    clear = ~fill & ~cloud & ~shadow & ~cirrus
    return clear


def _s2_is_clear(ds: xr.Dataset) -> np.ndarray:
    """Return boolean array where True = clear according to SCL.

    SCL classes (only vegetation, bare, water kept as clear):
      0: no_data / fill
      1: saturated / defective
      2: dark area pixels
      3: cloud shadows
      4: vegetation
      5: bare (not vegetated)
      6: water
      7: unclassified
      8: cloud (medium probability)
      9: cloud (high probability)
      10: thin cirrus
      11: snow / ice
    Clear = classes 4, 5, 6 (vegetation, bare, water).
    """
    scl = ds["SCL"].values[0]  # (y, x)
    clear_classes = {4, 5, 6}
    return np.isin(scl, list(clear_classes))


def _load_aoi_mask(
    aoi_path: str,
    target_ds: xr.Dataset,
) -> np.ndarray:
    """Load and reproject the Berlin AOI mask to match the target dataset grid.

    Parameters
    ----------
    aoi_path :
        Path to uint8 AOI COG (1 = inside Berlin).
    target_ds :
        xarray Dataset with ``crs``, ``rio`` accessor (from rioxarray).

    Returns
    -------
    np.ndarray
        2D boolean array (y, x), True = inside Berlin.
    """
    import rasterio

    with rasterio.open(aoi_path) as aoi_src:
        aoi_data = aoi_src.read(1).astype(bool)
        aoi_crs = aoi_src.crs
        aoi_transform = aoi_src.transform
        aoi_width = aoi_src.width
        aoi_height = aoi_src.height

    target_transform = target_ds.rio.transform()
    target_crs = target_ds.rio.crs
    target_height, target_width = target_ds.dims["y"], target_ds.dims["x"]

    destination = np.empty((target_height, target_width), dtype=aoi_data.dtype)
    rwarp.reproject(
        source=aoi_data.astype(np.uint8),
        src_crs=aoi_crs,
        src_transform=aoi_transform,
        src_width=aoi_width,
        src_height=aoi_height,
        destination=destination,
        dst_crs=target_crs,
        dst_transform=target_transform,
        resampling=rwarp.Resampling.nearest,
    )
    return destination.astype(bool)
