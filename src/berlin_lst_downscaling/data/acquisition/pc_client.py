"""Planetary Computer STAC catalog factory with asset signing."""

from __future__ import annotations

import planetary_computer
import pystac_client


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
