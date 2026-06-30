"""Authoritative target grid specification for the Berlin LST downscaling project.

The grid is a nested 10m / 100m system with a shared origin in EPSG:25833.
The 100m grid is an exact 10Ã—10 aggregate of the 10m grid (same origin,
aligned boundaries). This module is the single source of truth for ALL
spatial operations in the pipeline.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from affine import Affine
from shapely.geometry import box

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "boundaries"
AOI_FILE = DATA_DIR / "berlin_landesgrenze_2km_buffer.geojson"


@dataclass(frozen=True)
class GridSpec:
    """Immutable description of the nested 10m / 100m target grid."""

    # Grid origin (top-left corner) in EPSG:25833
    origin_x: float
    origin_y: float

    # 10m grid
    width_10m: int
    height_10m: int

    # 100m grid â€” exact 10Ã—10 aggregate of 10m grid
    width_100m: int
    height_100m: int

    # AOI bounding box in EPSG:25833 (from fetched boundary + 2 km buffer)
    aoi_xmin: float
    aoi_ymin: float
    aoi_xmax: float
    aoi_ymax: float

    # WGS84 bbox for GEE queries
    wgs84_bbox: tuple[float, float, float, float]

    # Buffered AOI polygon in EPSG:25833 (single source of truth for QA masking)
    aoi_polygon_25833: Any

    # Optional overrides with defaults
    crs: str = "EPSG:25833"
    resolution_10m: float = 10.0
    resolution_100m: float = 100.0

    @property
    def transform_10m(self) -> Affine:
        return Affine(self.resolution_10m, 0, self.origin_x,
                      0, -self.resolution_10m, self.origin_y)

    @property
    def transform_100m(self) -> Affine:
        return Affine(self.resolution_100m, 0, self.origin_x,
                      0, -self.resolution_100m, self.origin_y)

    @property
    def xmax_10m(self) -> float:
        return self.origin_x + self.width_10m * self.resolution_10m

    @property
    def ymin_10m(self) -> float:
        return self.origin_y - self.height_10m * self.resolution_10m

    def to_dict(self) -> dict:
        return {
            "crs": self.crs,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "resolution_10m": self.resolution_10m,
            "width_10m": self.width_10m,
            "height_10m": self.height_10m,
            "resolution_100m": self.resolution_100m,
            "width_100m": self.width_100m,
            "height_100m": self.height_100m,
            "aoi_25833": [self.aoi_xmin, self.aoi_ymin, self.aoi_xmax, self.aoi_ymax],
        }


def make_grid_spec(
    *,
    origin_x: float,
    origin_y: float,
    aoi_25833: tuple[float, float, float, float],
    wgs84_bbox: tuple[float, float, float, float],
    aoi_polygon_25833: Any | None = None,
    res_10m: float = 10.0,
    res_100m: float = 100.0,
) -> GridSpec:
    """Build a GridSpec ensuring 10m dimensions are multiples of 10 for nested alignment."""
    axmin, aymin, axmax, aymax = aoi_25833

    # Compute pixels needed to cover the AOI
    w10 = int(ceil((axmax - origin_x) / res_10m))
    h10 = int(ceil((origin_y - aymin) / res_10m))

    # Align width/height to next multiple of 10 (for exact 10Ã—10 nesting)
    w10_aligned = ((w10 + 9) // 10) * 10
    h10_aligned = ((h10 + 9) // 10) * 10

    w100 = w10_aligned // 10
    h100 = h10_aligned // 10

    return GridSpec(
        origin_x=origin_x,
        origin_y=origin_y,
        width_10m=w10_aligned,
        height_10m=h10_aligned,
        width_100m=w100,
        height_100m=h100,
        aoi_xmin=axmin,
        aoi_ymin=aymin,
        aoi_xmax=axmax,
        aoi_ymax=aymax,
        wgs84_bbox=wgs84_bbox,
        aoi_polygon_25833=aoi_polygon_25833 or box(axmin, aymin, axmax, aymax),
    )


@functools.cache
def get_spec() -> GridSpec:
    """Load the AOI GeoJSON and return a cached GridSpec singleton.

    The GeoJSON must have been created by ``scripts/fetch_berlin_boundary.py``.
    Raises ``FileNotFoundError`` with a helpful message if the file doesn't exist.
    """
    import json

    import geopandas as gpd  # type: ignore[import-untyped]

    if not AOI_FILE.exists():
        raise FileNotFoundError(
            f"AOI file not found at {AOI_FILE}. "
            "Run `python scripts/fetch_berlin_boundary.py` first."
        )

    with open(AOI_FILE) as f:
        data = json.load(f)

    # Parse the buffered AOI polygon (first feature) and derive its bbox
    coords = data["features"][0]["geometry"]["coordinates"][0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    aoi_25833 = (min(xs), min(ys), max(xs), max(ys))

    # Hardcoded origin aligned to 10m (from ard config)
    origin_x = 368000.0
    origin_y = 5839000.0

    # For WGS84 bbox, transform back
    gdf = gpd.read_file(AOI_FILE)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    wgs = gdf_wgs84.total_bounds
    wgs84_bbox = (float(wgs[0]), float(wgs[1]), float(wgs[2]), float(wgs[3]))

    return make_grid_spec(
        origin_x=origin_x,
        origin_y=origin_y,
        aoi_25833=aoi_25833,
        wgs84_bbox=wgs84_bbox,
        aoi_polygon_25833=gdf.geometry.iloc[0],
    )
