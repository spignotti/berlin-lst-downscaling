"""Shared constants for acquisition modules."""

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