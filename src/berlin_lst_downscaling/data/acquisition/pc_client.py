"""Planetary Computer STAC catalog factory with asset signing + retry."""

from __future__ import annotations

import planetary_computer
import pystac_client
from tenacity import retry, stop_after_attempt, wait_exponential


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
