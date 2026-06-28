"""STAC metadata generation for ARD COGs.

Writes minimal STAC 1.1.0 ``Feature`` items as JSON sidecars alongside
each processed COG. No ``pystac`` dependency — raw JSON with Geographic
coordinates in EPSG:25833.

Usage::

    from berlin_lst_downscaling.data.stac_writer import write_stac_item

    item_uri = write_stac_item(
        cog_path=Path("/tmp/output.tif"),
        scene_id="LC08_XXXXX",
        source="landsat",
        year=2023,
        qa_report=qa_report,
        output_bucket="berlin-lst-data",
        output_prefix="ard/processed/landsat",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio

logger = logging.getLogger(__name__)

_STAC_VERSION = "1.1.0"
_UTC = timezone.utc  # noqa: UP017  — avoids UP017 on every usage

_SOURCE_META: dict[str, dict[str, Any]] = {
    "landsat": {
        "constellation": "landsat",
        "instruments": ["tirs"],
        "gsd": 100.0,
        "description": "Landsat Collection 2 Level-2 surface temperature",
    },
    "sentinel2": {
        "constellation": "sentinel-2",
        "instruments": ["msi"],
        "gsd": 10.0,
        "description": "Sentinel-2 Level-2A surface reflectance (Harmonized)",
    },
    "ecostress": {
        "constellation": "ecostress",
        "instruments": ["ecostress"],
        "gsd": 70.0,
        "description": "ECOSTRESS Level-2 land surface temperature and emissivity",
    },
}


# ── Public API ───────────────────────────────────────────────────────────


def write_stac_item(
    cog_path: Path,
    scene_id: str,
    source: str,
    year: int,
    qa_report: dict[str, Any] | None,
    output_bucket: str,
    output_prefix: str,
    input_cog_path: Path | None = None,
    config_dict: dict[str, Any] | None = None,
    collection_id: str | None = None,
) -> dict[str, Any]:
    """Build and return a STAC 1.1.0 ``Feature`` dict for a processed COG.

    Args:
        cog_path: Local path to the output COG.
        scene_id: Scene identifier (parsed from GCS URI).
        source: Source name (``"landsat"``, ``"sentinel2"``, ``"ecostress"``).
        year: Processing year.
        qa_report: QA report dict (optional, for cloud_fraction).
        output_bucket: GCS bucket name.
        output_prefix: GCS output prefix (e.g. ``"ard/processed/landsat"``).
        input_cog_path: Local path to the source (pre-reproject) COG. Used to
            read geographic lat/lon for solar geometry.
        config_dict: Pipeline config as a plain dict. Used for config hash.
        collection_id: GEE / AppEEARS collection identifier.

    Returns:
        STAC item as a JSON-serializable dict.
    """
    cog_uri = f"gs://{output_bucket}/{output_prefix}/{year}/{scene_id}.tif"
    qa_uri = f"gs://{output_bucket}/{output_prefix}/{year}/{scene_id}_qa.json"

    # Read geographic metadata from the COG
    with rasterio.open(cog_path) as src:
        b = src.bounds
        bbox = tuple(b)  # type: ignore[arg-type]
        geometry = _bbox_to_polygon(bbox)
        crs = str(src.crs)
        transform = list(src.transform)[:6]  # [a, b, c, d, e, f]

    # Determine acquisition datetime (with time if possible)
    dt = _parse_overflight_datetime(scene_id, source, year)

    # Build properties
    meta = _SOURCE_META.get(source, {})
    properties: dict[str, Any] = {
        "datetime": dt.isoformat() if dt else None,
        "created": datetime.now(_UTC).isoformat(),  # noqa: UP017
        "constellation": meta.get("constellation", source),
        "instruments": meta.get("instruments", []),
        "gsd": meta.get("gsd", 10.0),
        "description": meta.get("description", ""),
        "processing:level": "ARD",
        "processing:version": "0.1.0",
    }

    # Refined overflight datetime (covers cases with full time precision)
    if dt:
        properties["overflight_datetime"] = dt.isoformat()
        properties["start_datetime"] = dt.isoformat()
        properties["end_datetime"] = dt.isoformat()

    # Solar angles from scene center + datetime
    if dt and input_cog_path and input_cog_path.exists():
        _add_solar_angles(properties, input_cog_path, dt)

    # Cloud fraction from QA report
    if qa_report and isinstance(qa_report, dict):
        cloud_pct = qa_report.get("cloud_fraction") or qa_report.get(
            "cloud_pixel_fraction"
        )
        if cloud_pct is not None:
            properties["cloud_fraction"] = cloud_pct

    # Config hash
    if config_dict:
        properties["processing:config_hash"] = _config_hash(config_dict)

    # Collection ID
    if collection_id:
        properties["processing:collection_id"] = collection_id

    # Geo transform
    properties["proj:transform"] = transform

    assets = {
        "cog": {
            "href": cog_uri,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": f"{scene_id} COG",
            "roles": ["data", "reflectance"],
            "description": meta.get("description", ""),
        },
        "qa-json": {
            "href": qa_uri,
            "type": "application/json",
            "title": f"{scene_id} QA report",
            "roles": ["metadata", "quality"],
        },
    }

    item: dict[str, Any] = {
        "stac_version": _STAC_VERSION,
        "stac_extensions": [],
        "type": "Feature",
        "id": scene_id,
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "assets": assets,
        "links": [],
    }

    if crs:
        item["crs"] = crs

    return item


# ── Solar geometry ──────────────────────────────────────────────────────


def _add_solar_angles(
    properties: dict[str, Any],
    cog_path: Path,
    dt: datetime,
) -> None:
    """Compute and set ``sun_azimuth`` / ``sun_elevation`` from scene center.

    Handles CRS reprojection automatically: if the COG CRS is not
    WGS84 (EPSG:4326), the center coordinate is reprojected before
    computing solar position.
    """
    try:
        with rasterio.open(cog_path) as src:
            left, bottom, right, top = src.bounds
            center_x = (left + right) / 2.0
            center_y = (bottom + top) / 2.0
            src_crs = src.crs
    except Exception:
        logger.warning("Failed to read bounds from %s for solar position", cog_path)
        return

    # Convert to WGS84 lat/lon if needed
    try:
        if src_crs and str(src_crs).upper() not in ("EPSG:4326", "WGS84", "GCS_WGS_1984"):
            from rasterio.warp import transform  # noqa: PLC0415

            # transform expects (src_crs, dst_crs, xs, ys)
            xy = transform(
                src_crs, "EPSG:4326", [center_x], [center_y]
            )
            center_lat = float(xy[1][0])
            center_lon = float(xy[0][0])
        else:
            center_lat, center_lon = center_y, center_x
    except Exception:
        logger.warning(
            "Failed to reproject center for solar position; using native coords"
        )
        center_lat, center_lon = center_y, center_x

    try:
        az, elev = _solar_position(center_lat, center_lon, dt)
        properties["sun_azimuth"] = round(az, 1)
        properties["sun_elevation"] = round(elev, 1)
    except Exception:
        logger.warning("Solar position computation failed", exc_info=True)


def _solar_position(
    lat: float, lon: float, dt: datetime
) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) via simplified SPA.

    Implements the NOAA Solar Position Algorithm in a compact form using
    only ``math`` (no external dependencies). Accuracy ~±0.5°.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)

    # Julian date
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    y, m = dt.year, dt.month
    d = dt.day + frac
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5

    # Days since J2000.0
    n = jd - 2451545.0

    # Solar mean longitude, mean anomaly
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)

    # Ecliptic longitude (with equation of centre correction)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(
        2 * g
    )
    eps = math.radians(23.439 - 0.0000004 * n)

    # Right ascension and declination
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))

    # Hour angle
    gmst_base = 6.6974243242 + 0.0657098283 * n
    gmst = (gmst_base + dt.hour + dt.minute / 60.0 + dt.second / 3600.0) % 24
    ha = math.radians(gmst * 15.0 + lon - math.degrees(ra))

    # Elevation
    lat_r = math.radians(lat)
    sin_elev = (
        math.sin(lat_r) * math.sin(dec)
        + math.cos(lat_r) * math.cos(dec) * math.cos(ha)
    )
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    # Azimuth
    elev_r = math.radians(elev)
    cos_az = (math.sin(dec) - math.sin(lat_r) * math.sin(elev_r)) / (
        math.cos(lat_r) * math.cos(elev_r)
    )
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if ha > 0:
        az = 360.0 - az

    return az, elev


# ── Config hash ─────────────────────────────────────────────────────────


def _config_hash(cfg_dict: dict[str, Any]) -> str:
    """Return a deterministic short SHA256 hex digest of the config.

    Only the ``ard`` section is hashed (excludes CLI args, system env).
    """
    ard_section = cfg_dict.get("ard", cfg_dict)
    raw = json.dumps(ard_section, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Overflight datetime ─────────────────────────────────────────────────


def _parse_overflight_datetime(
    scene_id: str,
    source: str,
    year: int,
) -> datetime | None:
    """Parse acquisition datetime with as much precision as scene ID allows.

    Sensor-specific patterns tried in priority order:

    * **Landsat:** ``YYYYMMDD`` prefix in scene_id → date at 10:00 UTC
      (approximate sun-synchronous descending-node overpass for Berlin).
    * **Sentinel-2:** ``YYYYMMDDTHHMMSS`` prefix → full datetime.
    * **ECOSTRESS:** ``YYYYMMDDTHHMMSS`` infix → full datetime.
    * **Fallback:** July 1 of the year (mid-season for May–Sep window).
    """
    if source == "sentinel2":
        m = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", scene_id)
        if m:
            parts = [int(g) for g in m.groups()]
            return datetime(*parts, tzinfo=_UTC)  # noqa: UP017

    if source == "ecostress":
        # ECO_L2T_LSTE_12345_001_YYYYMMDDTHHMMSS_...
        m = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", scene_id)
        if m:
            parts = [int(g) for g in m.groups()]
            return datetime(*parts, tzinfo=_UTC)  # noqa: UP017

    if source == "landsat":
        # Landsat scene_ids often end with YYYYMMDD or start with
        # YYYYDDD (year + Julian day). Try ``YYYYMMDD`` first.
        m = re.search(r"(\d{4})(\d{2})(\d{2})", scene_id)
        if m:
            parts = [int(g) for g in m.groups()]
            if 1 <= parts[1] <= 12 and 1 <= parts[2] <= 31:
                return datetime(
                    parts[0], parts[1], parts[2],
                    10, 0, 0,  # 10:00 UTC approximate overpass
                    tzinfo=_UTC,
                )

    # Fallback: July 1 (mid-season guess for May–Sep window)
    return datetime(year, 7, 1, tzinfo=_UTC)  # noqa: UP017


# ── Geometry helpers ────────────────────────────────────────────────────


def _bbox_to_polygon(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Convert a bounding box to a GeoJSON-like Polygon dict."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]],
    }
