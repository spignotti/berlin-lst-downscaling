"""Canonical raster grid — shared origin across all satellite sources.

All three sources (Landsat 100m, Sentinel-2 10m, ECOSTRESS 70m) write COGs
on the same canonical grid to ensure pixel-level alignment for downstream
stacking.  The 10m grid is the base; 70m and 100m are derived by
``zoom_out``, which guarantees all origins match exactly.

Grid origin (EPSG:25833, Berlin bbox):
    (369190, 5838410)

Example
-------
>>> from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
>>> from odc.stac import load as odc_load
>>> ds = odc_load(items=items, bands=bands, geobox=canon_grid_for_resolution(10))
"""

from __future__ import annotations

from functools import lru_cache

from odc.geo.geobox import GeoBox
from rasterio.warp import transform_bounds

from berlin_lst_downscaling.common.config import BERLIN_BBOX, TARGET_CRS, TARGET_RESOLUTION


@lru_cache(maxsize=1)
def canon_grid_10m() -> GeoBox:
    """Return the canonical 10m EPSG:25833 GeoBox for Berlin bbox."""
    bbox_25833 = transform_bounds("EPSG:4326", TARGET_CRS, *BERLIN_BBOX)
    return GeoBox.from_bbox(bbox_25833, crs=TARGET_CRS, resolution=10)


@lru_cache(maxsize=1)
def canon_grid_70m() -> GeoBox:
    """Return the canonical 70m EPSG:25833 GeoBox (7× nested from 10m)."""
    return canon_grid_10m().zoom_out(7)


@lru_cache(maxsize=1)
def canon_grid_100m() -> GeoBox:
    """Return the canonical 100m EPSG:25833 GeoBox (10× nested from 10m)."""
    return canon_grid_10m().zoom_out(10)


def canon_grid_for_resolution(res: int) -> GeoBox:
    """Return the canonical GeoBox for a given resolution.

    Parameters
    ----------
    res : int
        Target resolution in metres.  One of ``10``, ``70``, ``100``.

    Returns
    -------
    GeoBox
        The canonical grid at *res*, guaranteed to share the same origin
        as all other resolutions.
    """
    return {10: canon_grid_10m, 70: canon_grid_70m, 100: canon_grid_100m}[res]()


def smoke_grid(bbox_wgs84: tuple[float, float, float, float]) -> GeoBox:
    """Return a 10 m canonical-aligned subset grid for a WGS84 bbox.

    The result is the largest 10 m grid aligned to the canonical origin
    that fits inside *bbox_wgs84*.  Used for local real-data smoke tests
    where the full Berlin grid is too large for a single DGM/LoD2 tile.

    Parameters
    ----------
    bbox_wgs84 :
        (west, south, east, north) in WGS84.
    """
    bbox_native = transform_bounds("EPSG:4326", TARGET_CRS, *bbox_wgs84)
    return GeoBox.from_bbox(bbox_native, crs=TARGET_CRS, resolution=TARGET_RESOLUTION)


def grid_from_cog(uri: str) -> GeoBox:
    """Infer the GeoBox from a rasterio-readable COG.

    Parameters
    ----------
    uri :
        Local path or ``gs://…`` URI of the COG.
    """
    import rasterio

    with rasterio.open(uri) as src:
        from odc.geo.geobox import GeoBox as _GB

        return _GB.from_rio(src)


__all__ = [
    "canon_grid_10m",
    "canon_grid_70m",
    "canon_grid_100m",
    "canon_grid_for_resolution",
    "smoke_grid",
    "grid_from_cog",
]
