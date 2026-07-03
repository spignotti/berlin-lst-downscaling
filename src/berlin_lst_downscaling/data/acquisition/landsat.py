"""Load a Landsat Collection 2 Level-2 scene via Planetary Computer STAC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import odc.stac
import xarray as xr

from berlin_lst_downscaling.common.config import settings
from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog

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
    **odc_kw: Any,
) -> tuple[xr.Dataset, str]:
    """Search Planetary Computer for one Landsat scene and load it.

    Parameters
    ----------
    date :
        ISO date string (e.g. ``"2024-07-15"``). Defaults to
        ``settings.default_date``.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to
        ``settings.berlin_bbox``.
    bands :
        Band asset keys to load. Defaults to ``_LANDSAT_BANDS``.
    **odc_kw :
        Additional keyword arguments forwarded to ``odc.stac.load``
        (e.g. ``chunks``, ``resampling``).

    Returns
    -------
    tuple[xr.Dataset, str]
        Loaded scene on ``settings.target_crs`` at
        ``settings.target_resolution``, and the STAC item ID.

    Raises
    ------
    RuntimeError
        If no matching scene is found.
    """
    date = date or settings.default_date
    bbox = bbox or settings.berlin_bbox
    bands = list(bands) if bands is not None else _LANDSAT_BANDS

    catalog = get_catalog()
    search = catalog.search(
        collections=[_LANDSAT_COLLECTION],
        bbox=bbox,
        datetime=date,
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Landsat scene found for date={date} bbox={bbox}"
        )

    item = items[0]

    ds = odc.stac.load(
        items=[item],
        bands=bands,
        crs=settings.target_crs,
        resolution=settings.target_resolution,
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        **odc_kw,
    )
    return ds, item.id
