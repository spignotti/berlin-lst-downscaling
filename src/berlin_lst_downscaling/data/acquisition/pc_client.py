"""Planetary Computer STAC catalog factory with asset signing + retry."""

from __future__ import annotations

import logging

import planetary_computer
import pystac
import pystac_client
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_logger = logging.getLogger(__name__)


def get_catalog() -> pystac_client.Client:
    """Return a PC STAC Client with automatic asset signing wired in.

    Uses ``planetary_computer.sign_inplace`` as the modifier so every
    STAC item returned by ``search()`` has its asset URLs signed for
    CloudFront access. No network IO at import time — the Client lazily
    fetches the root catalog on first use.
    """
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def stac_load(**kwargs):
    """odc.stac.load with PC URL re-signing + retry on network failure.

    Usage
    -----
    .. code-block:: python

        from berlin_lst_downscaling.data.acquisition.pc_client import stac_load

        ds = stac_load(
            items=items,
            bands=bands,
            geobox=gbox,
            chunks={"x": 2048, "y": 2048},
            groupby="solar_day",
        )

    All keyword arguments are forwarded to ``odc.stac.load``.
    ``patch_url`` defaults to ``planetary_computer.sign_url`` to re-sign
    SAS-expired URLs at read time.
    """
    import odc.stac  # lazy import — odc-stac may not be imported at module level

    kwargs.setdefault("patch_url", planetary_computer.sign_url)
    return odc.stac.load(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def pc_search(
    collections: list[str],
    bbox: tuple[float, float, float, float] | None = None,
    datetime: str | None = None,
    ids: list[str] | None = None,
    query: dict | None = None,
    max_items: int | None = None,
) -> list:
    """Search PC STAC with bounded retry on transient network failures.

    Retries only on ConnectionError/TimeoutError; other exceptions
    (4xx, invalid parameters) propagate immediately.

    Returns a list of pystac.Item objects.
    """
    cat = get_catalog()
    kwargs: dict = {"collections": collections}
    if bbox is not None:
        kwargs["bbox"] = bbox
    if datetime is not None:
        kwargs["datetime"] = datetime
    if ids is not None:
        kwargs["ids"] = ids
    if query is not None:
        kwargs["query"] = query
    if max_items is not None:
        kwargs["max_items"] = max_items

    search = cat.search(**kwargs)
    items = list(search.items())
    return items


def resolve_exact_item(
    collection: str,
    scene_id: str,
) -> pystac.Item:
    """Resolve a single exact STAC item by collection and ID.

    Raises RuntimeError if the item is not found or the returned ID
    does not match the requested one.
    """
    items = pc_search(
        collections=[collection],
        ids=[scene_id],
        max_items=1,
    )
    if not items:
        raise RuntimeError(
            f"STAC item {scene_id!r} not found in collection {collection!r}"
        )
    item = items[0]
    if item.id != scene_id:
        raise RuntimeError(
            f"STAC returned item {item.id!r} but expected {scene_id!r}"
        )
    return item


__all__ = [
    "get_catalog",
    "stac_load",
    "pc_search",
    "resolve_exact_item",
]
