"""Load a Sentinel-2 Level-2A scene via Planetary Computer STAC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import odc.stac
import xarray as xr

from berlin_lst_downscaling.common.config import settings
from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog

_S2_COLLECTION = "sentinel-2-l2a"
_S2_BANDS = ["B02", "B03", "B04", "B08", "SCL"]


def load_s2_scene(
    date: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    bands: Sequence[str] | None = None,
    **odc_kw: Any,
) -> tuple[xr.Dataset, str]:
    """Search Planetary Computer for one Sentinel-2 L2A scene and load it.

    Parameters
    ----------
    date :
        ISO date string (e.g. ``"2024-07-15"``). Defaults to
        ``settings.default_date``.
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to
        ``settings.berlin_bbox``.
    bands :
        Band asset keys to load. Defaults to ``_S2_BANDS``
        (native S2 names: ``B02``, ``B03``, ``B04``, ``B08``, ``SCL``).
    **odc_kw :
        Additional keyword arguments forwarded to ``odc.stac.load``.

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
    bands = list(bands) if bands is not None else _S2_BANDS

    catalog = get_catalog()
    search = catalog.search(
        collections=[_S2_COLLECTION],
        bbox=bbox,
        datetime=date,
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scene found for date={date} bbox={bbox}"
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
