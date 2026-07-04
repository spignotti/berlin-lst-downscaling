"""Landsat anchor search — PC STAC query filtered to Berlin AOI and season."""

from __future__ import annotations

from datetime import datetime

from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog


def build_anchors(cfg) -> list:
    """Return Landsat C2 L2 scenes as coupling anchors.

    Queries PC STAC for all scenes intersecting the configured bbox within
    the year range, then filters to May–September (configurable months).
    """
    cat = get_catalog()

    min_year = min(cfg.years)
    max_year = max(cfg.years)
    # Build a season-spanning datetime for the STAC query
    # Use a single range covering all requested years + seasons
    query_start = f"{min_year}-05-01"
    query_end = f"{max_year}-09-30"

    search = cat.search(
        collections=[cfg.landsat.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{query_start}/{query_end}",
        query={"eo:cloud_cover": {"lt": cfg.landsat.cloud_cover_max}},
    )

    anchors: list = []
    for item in search.items():
        dt_utc = _parse_item_datetime(item)
        if dt_utc is None:
            continue

        # Filter by configured months (1-indexed)
        if dt_utc.month not in tuple(cfg.months):
            continue
        # Filter by configured years
        if dt_utc.year not in tuple(cfg.years):
            continue

        # Extract solar angles from STAC properties
        sun_az = item.properties.get("view:sun_azimuth")
        sun_el = item.properties.get("view:sun_elevation")
        cloud_cover = item.properties.get("eo:cloud_cover")

        # item.get_self_href() returns the absolute self link (may be None for some catalogs)
        # For mode=full the pipeline re-resolves via date; we store item.id only.
        item_href = item.get_self_href() if hasattr(item, "get_self_href") else None

        anchors.append({
            "scene_id": item.id,
            "source": "landsat-c2-l2",
            "year": dt_utc.year,
            "datetime": dt_utc,
            "date": dt_utc.strftime("%Y-%m-%d"),
            "cloud_cover": cloud_cover,
            "sun_azimuth": sun_az,
            "sun_elevation": sun_el,
            "item_href": item_href,
        })

    # Sort chronologically
    anchors.sort(key=lambda a: a["datetime"])
    return anchors


def _parse_item_datetime(item) -> datetime | None:
    """Extract UTC datetime from a STAC item's datetime property."""
    dt_str = item.properties.get("datetime")
    if dt_str is None:
        return None
    # e.g. "2024-06-29T10:15:00Z"
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None
