"""GEE client initialization and AOI geometry helpers."""

from collections.abc import Sequence
from typing import Any

import ee
from omegaconf import DictConfig

from berlin_lst_downscaling.data.boundary import buffered_bbox_wgs84, buffered_geojson_wgs84


def initialize(cfg: DictConfig) -> None:
    """Initialize Earth Engine with the configured project via service account."""
    project = cfg.ard.gee.project
    credentials = ee.ServiceAccountCredentials(
        cfg.ard.gee.service_account_email,
        cfg.ard.gee.service_account_key_path,
    )
    ee.Initialize(credentials, project=project)


def get_aoi_geometry(
    bbox_wgs84: Sequence[float],
) -> Any:
    """Return the AOI as an ``ee.Geometry.Rectangle`` in WGS84.

    ``bbox_wgs84`` should be ``(west, south, east, north)``.
    This is the standard GEE region for export and filtering.
    """
    return ee.Geometry.Rectangle(list(bbox_wgs84))


def get_aoi_from_cfg(cfg: DictConfig) -> Any:
    """Convenience: derive the WGS84 bbox from the buffered AOI polygon."""
    return get_aoi_geometry(buffered_bbox_wgs84(cfg.ard.aoi.boundary_file))


def get_aoi_geojson_from_cfg(cfg: DictConfig) -> dict:
    """Return the AOI as a GeoJSON FeatureCollection dict (EPSG:4326).

    Used for AppEEARS area task submission (not GEE).
    """
    return buffered_geojson_wgs84(cfg.ard.aoi.boundary_file)
