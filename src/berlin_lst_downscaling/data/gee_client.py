"""GEE client initialization and AOI geometry helpers."""

from collections.abc import Sequence
from typing import Any

import ee
from omegaconf import DictConfig


def initialize(cfg: DictConfig) -> None:
    """Initialize Earth Engine with the configured project."""
    project = cfg.ard.gee.project
    ee.Initialize(project=project)


def get_aoi_geometry(
    bbox_wgs84: Sequence[float],
) -> Any:
    """Return the AOI as an ``ee.Geometry.Rectangle`` in WGS84.

    ``bbox_wgs84`` should be ``(west, south, east, north)``.
    This is the standard GEE region for export and filtering.
    """
    return ee.Geometry.Rectangle(list(bbox_wgs84))


def get_aoi_from_cfg(cfg: DictConfig) -> Any:
    """Convenience: read the WGS84 bbox from the Hydra config."""
    return get_aoi_geometry(cfg.ard.aoi.wgs84_bbox)


def get_aoi_geojson_from_cfg(cfg: DictConfig) -> dict:
    """Return the AOI as a GeoJSON FeatureCollection dict (EPSG:4326).

    Used for AppEEARS area task submission (not GEE).
    """
    west, south, east, north = cfg.ard.aoi.wgs84_bbox
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }
