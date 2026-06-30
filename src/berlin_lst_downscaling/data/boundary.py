"""Berlin boundary loading and polygon mask generation.

Single source of truth: ``data/boundaries/berlin_landesgrenze_2km_buffer.geojson``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "boundaries"
LANDESGRENZE_FILE = DATA_DIR / "berlin_landesgrenze.geojson"
BUFFER_FILE = DATA_DIR / "berlin_landesgrenze_2km_buffer.geojson"


@cache
def load_landesgrenze() -> gpd.GeoDataFrame:
    """Return the Berlin administrative boundary as a GeoDataFrame."""
    return gpd.read_file(LANDESGRENZE_FILE)


@cache
def load_buffered_polygon(boundary_file: str | Path | None = None) -> gpd.GeoDataFrame:
    """Return the Berlin boundary buffered by 2 km as a GeoDataFrame."""
    return gpd.read_file(Path(boundary_file) if boundary_file is not None else BUFFER_FILE)


def buffered_bbox_wgs84(
    boundary_file: str | Path | None = None,
) -> tuple[float, float, float, float]:
    """Return the buffered AOI bounding box in WGS84."""
    bounds = load_buffered_polygon(boundary_file).to_crs("EPSG:4326").total_bounds
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def buffered_bbox_25833(
    boundary_file: str | Path | None = None,
) -> tuple[float, float, float, float]:
    """Return the buffered AOI bounding box in EPSG:25833."""
    bounds = load_buffered_polygon(boundary_file).to_crs("EPSG:25833").total_bounds
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def buffered_geojson_wgs84(boundary_file: str | Path | None = None) -> dict:
    """Return the buffered AOI polygon as GeoJSON FeatureCollection in WGS84."""
    gdf = load_buffered_polygon(boundary_file).to_crs("EPSG:4326")
    geom = gdf.geometry.iloc[0]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {},
            }
        ],
    }


def polygon_mask(
    transform,
    shape: tuple[int, int],
    *,
    crs: str = "EPSG:25833",
    buffered: bool = True,
    boundary_file: str | Path | None = None,
) -> np.ndarray:
    """Return a boolean AOI mask (``True`` inside the polygon)."""
    gdf = load_buffered_polygon(boundary_file) if buffered else load_landesgrenze()
    if str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)
    return ~geometry_mask(gdf.geometry, transform=transform, out_shape=shape, invert=False)
