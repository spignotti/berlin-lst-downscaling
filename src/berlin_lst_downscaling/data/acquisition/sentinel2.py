"""Load a Sentinel-2 Level-2A scene via Planetary Computer STAC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pystac
import xarray as xr

from berlin_lst_downscaling.common.config import (
    BERLIN_BBOX,
    DEFAULT_DATE,
    TARGET_RESOLUTION,
)
from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog, stac_load

_S2_COLLECTION = "sentinel-2-l2a"
_S2_BANDS = ["B02", "B03", "B04", "B08", "SCL"]

def load_s2_scene(
    date: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    bands: Sequence[str] | None = None,
    max_items: int | None = 3,
    resolution: int | None = None,
    items: list[pystac.Item] | None = None,
    **odc_kw: Any,
) -> tuple[xr.Dataset, list[str]]:
    """Search Planetary Computer for Sentinel-2 L2A scenes and load them.

    Loads matching tiles (default: 3 MGRS tiles covering Berlin) and
    mosaics them onto the shared ``bbox``-aligned GeoBox via
    ``odc.stac.load(groupby="solar_day")`` (last-item-wins for overlapping
    pixels).

    Parameters
    ----------
    date :
        ISO date string (e.g. ``"2024-07-15"``). Defaults to
        ``DEFAULT_DATE``.  Ignored when ``items`` is provided.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to
        ``BERLIN_BBOX``.
    bands :
        Band asset keys to load. Defaults to ``_S2_BANDS``
        (native S2 names: ``B02``, ``B03``, ``B04``, ``B08``, ``SCL``).
    resolution :
        Target resolution in meters.  Defaults to
        ``TARGET_RESOLUTION``.  The ARD pipeline typically
        passes 10 m.
    items :
        Pre-fetched STAC items.  When provided, ``date`` and
        ``max_items`` are ignored and no STAC search is performed.
        Use this for manifest-driven ``mode=full``.
    **odc_kw :
        Additional keyword arguments forwarded to ``odc.stac.load``.

    Returns
    -------
    tuple[xr.Dataset, list[str]]
        Loaded scene on the configured target CRS at the chosen
        resolution, and the STAC item IDs.

    Raises
    ------
    RuntimeError
        If no matching scene is found.
    """
    if items is not None:
        item_ids = [it.id for it in items]
    else:
        date = date or DEFAULT_DATE
        bbox = bbox or BERLIN_BBOX
        catalog = get_catalog()
        search = catalog.search(
            collections=[_S2_COLLECTION],
            bbox=bbox,
            datetime=date,
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=max_items,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"No Sentinel-2 scene found for date={date} bbox={bbox}")
        item_ids = [it.id for it in items]

    bands = list(bands) if bands is not None else _S2_BANDS
    res = resolution if resolution is not None else TARGET_RESOLUTION

    ds = stac_load(
        items=items,
        bands=bands,
        geobox=canon_grid_for_resolution(res),
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        **odc_kw,
    )
    return ds, item_ids