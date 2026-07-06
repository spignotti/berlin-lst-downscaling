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

from datetime import datetime

import numpy as np
import odc.stac
import xarray as xr

from berlin_lst_downscaling.data.ard.masking import landsat_qa_to_clear_bits
from berlin_lst_downscaling.data.selection._aoi import load_aoi_mask, select_time_slice


def compute_clear_frac(
    l8_items: list,
    s2_items: list,
    anchor_bbox: tuple[float, float, float, float],
    aoi_mask_path: str = "data/boundaries/aoi_10m.tif",
    resolution: int = 10,
    anchor_dt: datetime | None = None,
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
    l8_clear = _landsat_is_clear(l8_ds, anchor_dt)
    s2_clear = _s2_is_clear(s2_ds, anchor_dt)

    # ── AOI mask (reproject to the scene grid) ──────────────────────────────
    aoi = load_aoi_mask(aoi_mask_path, l8_ds)

    # ── compute clear_frac ───────────────────────────────────────────────────
    denom = int(np.sum(aoi & l8_clear))
    if denom == 0:
        return float("nan")

    numer = int(np.sum(aoi & l8_clear & s2_clear))
    return numer / denom


def compute_clear_frac_with_counts(
    l8_items: list,
    s2_items: list,
    anchor_bbox: tuple[float, float, float, float],
    aoi_mask_path: str = "data/boundaries/aoi_10m.tif",
    resolution: int = 10,
    anchor_dt: datetime | None = None,
) -> tuple[float, dict]:
    """Same as compute_clear_frac but also returns intermediate pixel counts.

    Returns (clear_frac, counts_dict).
    counts_dict keys: ``aoi_px``, ``l8_clear_px``, ``s2_clear_px``,
    ``intersect_px``.
    """
    if not l8_items or not s2_items:
        return float("nan"), _empty_counts()

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
    l8_clear = _landsat_is_clear(l8_ds, anchor_dt)
    s2_clear = _s2_is_clear(s2_ds, anchor_dt)

    # ── AOI mask (reproject to the scene grid) ──────────────────────────────
    aoi = load_aoi_mask(aoi_mask_path, l8_ds)

    # ── compute clear_frac + counts ─────────────────────────────────────────
    aoi_px = int(np.sum(aoi))
    l8_clear_px = int(np.sum(aoi & l8_clear))
    s2_clear_px = int(np.sum(aoi & s2_clear))
    intersect_px = int(np.sum(aoi & l8_clear & s2_clear))

    if l8_clear_px == 0:
        cf = float("nan")
        return cf, _counts_dict(cf, aoi_px, l8_clear_px, s2_clear_px, intersect_px)

    cf = intersect_px / l8_clear_px
    return cf, _counts_dict(cf, aoi_px, l8_clear_px, s2_clear_px, intersect_px)


def _counts_dict(
    clear_frac: float,
    aoi_px: int,
    l8_clear_px: int,
    s2_clear_px: int,
    intersect_px: int,
) -> dict:
    return {
        "clear_frac": clear_frac,
        "aoi_px": aoi_px,
        "l8_clear_px": l8_clear_px,
        "s2_clear_px": s2_clear_px,
        "intersect_px": intersect_px,
    }


def _empty_counts() -> dict:
    return {
        "clear_frac": float("nan"),
        "aoi_px": 0,
        "l8_clear_px": 0,
        "s2_clear_px": 0,
        "intersect_px": 0,
    }


def _landsat_is_clear(ds: xr.Dataset, anchor_dt: datetime | None = None) -> np.ndarray:
    """Return boolean array where True = clear according to QA_PIXEL.

    Uses the production-tested ``landsat_qa_to_clear_bits`` from the ARD
    masking module (bits 0 fill, 2 cirrus, 3 cloud w/ conf≥2, 4 shadow).
    No dilation — dilation is ARD-only.

    When ``anchor_dt`` is provided, selects the solar-day slice matching the
    anchor's date rather than ``values[0]`` (first chronological slice).
    """
    if anchor_dt is not None:
        ds = select_time_slice(ds, anchor_dt)
    qa = ds["qa_pixel"].values[0].astype(np.uint16)
    return landsat_qa_to_clear_bits(qa)


_S2_CLOUD_CLASSES = {0, 1, 8, 9, 10, 11}  # fill, saturated, cloud, cirrus, snow


def _s2_is_clear(ds: xr.Dataset, anchor_dt: datetime | None = None) -> np.ndarray:
    """Return boolean array where True = clear according to SCL.

    Inverts the SCL cloud classification: any class NOT in the cloudy set
    is considered clear sky. This includes class 7 (unclassified), which
    covers urban impervious surfaces over Berlin — these should not be
    excluded from coupling even though Sen2Cor does not classify them
    as vegetation/bare/water.

    When ``anchor_dt`` is provided, selects the solar-day slice matching the
    anchor's date rather than ``values[0]`` (first chronological slice).
    """
    if anchor_dt is not None:
        ds = select_time_slice(ds, anchor_dt)
    scl = ds["SCL"].values[0].astype(np.uint8)
    return ~np.isin(scl, list(_S2_CLOUD_CLASSES))

