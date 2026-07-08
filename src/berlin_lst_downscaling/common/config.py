"""Shared constants for acquisition modules.

Replaces pydantic-settings with simple module-level defaults.
Values can be overridden via ``BERLIN_LST_*`` env vars at runtime
(optional — not wired below; add ``os.environ.get()`` if needed).
"""

from __future__ import annotations

# Berlin bounding box (WGS84)
BERLIN_BBOX: tuple[float, float, float, float] = (
    13.08,
    52.34,
    13.76,
    52.68,
)

# Target CRS (ETRS89 / UTM zone 33N, Berlin)
TARGET_CRS: str = "EPSG:25833"

# Target resolution in metres
TARGET_RESOLUTION: int = 10

# Default scene date for smoke tests
DEFAULT_DATE: str = "2024-06-29"


# Backward-compatible access (old import: `from ...config import settings`)
class _Settings:
    """Simple settings bag matching the old pydantic-settings API.

    Allows existing code to access ``settings.berlin_bbox`` etc.
    without changes.
    """

    berlin_bbox = BERLIN_BBOX
    target_crs = TARGET_CRS
    target_resolution = TARGET_RESOLUTION
    default_date = DEFAULT_DATE


settings = _Settings()
