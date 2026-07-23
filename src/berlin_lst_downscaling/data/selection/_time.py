"""Shared datetime parsing helpers for STAC-driven selection stages.

Both the Landsat-anchor loader and the S2-candidate loader rely on
these utilities; centralising them prevents drift between the two
call sites.
"""
from __future__ import annotations

from datetime import datetime


def parse_cutoff(cutoff_str: str) -> datetime:
    """Parse a cutoff timestamp as UTC datetime.

    Accepts the ``Z`` suffix as well as explicit ``+00:00`` offsets.
    """
    try:
        return datetime.fromisoformat(cutoff_str.replace("Z", "+00:00"))
    except ValueError as err:
        raise ValueError(
            f"Invalid cutoff_utc format: {cutoff_str!r}. "
            "Expected ISO format, e.g. '2026-07-17T23:59:59Z'."
        ) from err


def parse_item_datetime(item) -> datetime | None:
    """Extract UTC datetime from a STAC item's datetime property.

    Returns ``None`` if the property is missing or unparseable.
    """
    dt_str = item.properties.get("datetime")
    if dt_str is None:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None
