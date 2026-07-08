"""Load a Sentinel-2 Level-2A scene via Planetary Computer STAC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pystac
import xarray as xr

from berlin_lst_downscaling.common.config import settings
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

    Loads matching tiles (default: 3 — enough for full Berlin coverage:
    east+west MGRS tiles plus central strip) and merges onto the shared
    ``bbox``-aligned GeoBox via ``odc.stac.load``.

    .. note::

       The production pipeline will select the best tile per date using
       the ``clear_frac - λ·Δt/3`` score rather than loading all
       intersecting tiles.

    Parameters
    ----------
    date :
        ISO date string (e.g. ``"2024-07-15"``). Defaults to
        ``settings.default_date``.  Ignored when ``items`` is provided.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to
        ``settings.berlin_bbox``.
    bands :
        Band asset keys to load. Defaults to ``_S2_BANDS``
        (native S2 names: ``B02``, ``B03``, ``B04``, ``B08``, ``SCL``).
    resolution :
        Target resolution in meters.  Defaults to
        ``settings.target_resolution``.  The ARD pipeline typically
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
        Loaded scene on ``settings.target_crs`` at the chosen
        resolution, and the STAC item IDs.

    Raises
    ------
    RuntimeError
        If no matching scene is found.
    """
    if items is not None:
        item_ids = [it.id for it in items]
    else:
        date = date or settings.default_date
        bbox = bbox or settings.berlin_bbox
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
