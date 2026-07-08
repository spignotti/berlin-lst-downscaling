"""Load a Landsat Collection 2 Level-2 scene via Planetary Computer STAC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pystac
import xarray as xr

from berlin_lst_downscaling.common.config import settings
from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog, stac_load

_LANDSAT_COLLECTION = "landsat-c2-l2"
_LANDSAT_BANDS = [
    "red",
    "green",
    "blue",
    "nir08",
    "swir16",
    "swir22",
    "lwir11",
    "qa_pixel",
]


def load_landsat_scene(
    date: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    bands: Sequence[str] | None = None,
    max_items: int | None = None,
    resolution: int | None = None,
    items: list[pystac.Item] | None = None,
    **odc_kw: Any,
) -> tuple[xr.Dataset, list[str]]:
    """Search Planetary Computer for Landsat scenes and load them.

    Loads matching scenes (default: all intersecting; typically 2 WRS-2
    rows for Berlin at path 193) and merges onto the shared ``bbox``-aligned
    GeoBox via ``odc.stac.load``
    (``groupby="solar_day"`` fuses overlapping pixels with last-item-wins).

    Parameters
    ----------
    date :
        ISO date string (e.g. ``"2024-07-15"``). Defaults to
        ``settings.default_date``. Ignored when ``items`` is provided.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to
        ``settings.berlin_bbox``.
    bands :
        Band asset keys to load. Defaults to ``_LANDSAT_BANDS``.
    resolution :
        Target resolution in meters.  Defaults to
        ``settings.target_resolution``.  The ARD pipeline typically
        passes 100 m (native LST, anti-leakage).
    items :
        Pre-fetched STAC items.  When provided, ``date`` and
        ``max_items`` are ignored and no STAC search is performed.
        Use this for manifest-driven ``mode=full`` to avoid
        re-querying PC by date.
    **odc_kw :
        Additional keyword arguments forwarded to ``odc.stac.load``
        (e.g. ``chunks``, ``resampling``).

    Returns
    -------
    tuple[xr.Dataset, list[str]]
        Loaded scene on ``settings.target_crs`` at the chosen
        resolution, and the STAC item IDs of **every** matching scene.

    Raises
    ------
    RuntimeError
        If no matching scene is found.
    """
    if items is not None:
        # Skip STAC search — caller provides pre-fetched items
        item_ids = [it.id for it in items]
    else:
        date = date or settings.default_date
        bbox = bbox or settings.berlin_bbox
        catalog = get_catalog()
        search = catalog.search(
            collections=[_LANDSAT_COLLECTION],
            bbox=bbox,
            datetime=date,
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=max_items,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"No Landsat scene found for date={date} bbox={bbox}")
        item_ids = [it.id for it in items]

    bands = list(bands) if bands is not None else _LANDSAT_BANDS
    res = resolution if resolution is not None else settings.target_resolution

    ds = stac_load(
        items=items,
        bands=bands,
        crs=settings.target_crs,
        resolution=res,
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        **odc_kw,
    )
    return ds, item_ids
