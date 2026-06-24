"""ECOSTRESS scene listing via NASA CMR (Common Metadata Repository).

Uses ``earthaccess`` for lightweight, read-only queries against the CMR
to discover available granules without submitting AppEEARS tasks.

Usage::

    from berlin_lst_downscaling.data.ecostress_scenes import list_ecostress_granules

    granules = list_ecostress_granules(
        wgs84_bbox=[13.0, 52.3, 13.8, 52.7],
        start_year=2018,
        end_year=2025,
        months=[5, 6, 7, 8, 9],
    )
    for g in granules[:5]:
        print(g["acquisition_date"], g.get("GranuleUR", ""))
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any

import earthaccess

logger = logging.getLogger(__name__)

# Short name for ECOSTRESS L2T LSTE Collection 2
_PRODUCT_SHORT_NAME = "ECO_L2T_LSTE"
_PRODUCT_VERSION = "002"


def _ensure_auth() -> None:
    """Authenticate with Earthdata Login via earthaccess.

    Uses ``EARTHDATA_TOKEN`` or ``.netrc``; already configured in the
    project's ``.env``. No-op if already authenticated.
    """
    earthaccess.login(strategy="environment", persist=False)


def _extract_date_time(granule: dict[str, Any]) -> datetime | None:
    """Extract acquisition datetime from a CMR granule dict.

    ``DataGranule.get("umm")`` returns a dict with ``TemporalExtent`` →
    ``RangeDateTime`` → ``BeginningDateTime``.

    Args:
        granule: A ``DataGranule`` (dict-like) object.

    Returns:
        ``datetime`` object or ``None`` if the date cannot be parsed.
    """
    umm = granule.get("umm", {})
    extent = umm.get("TemporalExtent", {})
    date_str = extent.get("RangeDateTime", {}).get("BeginningDateTime", "")
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def list_ecostress_granules(
    wgs84_bbox: list[float],
    start_year: int = 2018,
    end_year: int = 2025,
    months: list[int] | None = None,
    max_results: int = -1,
) -> list[dict[str, Any]]:
    """Query CMR for ECOSTRESS L2T LSTE granules over a given AOI and time range.

    Lightweight metadata query — no actual data is downloaded.

    Args:
        wgs84_bbox: ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
        start_year: First year to query.
        end_year: Last year to query (inclusive).
        months: If set, only include granules whose acquisition month is in
            this list.
        max_results: Maximum granules to return. ``-1`` means unlimited.

    Returns:
        List of granule metadata dicts with at least ``"acquisition_date"``
        (``datetime``) and the raw ``"umm"`` and ``"meta"`` dicts.

    Raises:
        RuntimeError: If CMR query fails.
    """
    _ensure_auth()

    # Build temporal range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year + 1}-01-01"  # Exclusive upper bound

    # Unpack bounding box — earthaccess expects individual args
    bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max = wgs84_bbox

    results = earthaccess.search_data(
        short_name=_PRODUCT_SHORT_NAME,
        version=_PRODUCT_VERSION,
        bounding_box=(bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max),
        temporal=(start_date, end_date),
        count=max_results if max_results > 0 else -1,
    )

    granules: list[dict[str, Any]] = []
    for r in results:
        dt = _extract_date_time(r)
        if dt is None:
            continue
        if months is not None and dt.month not in months:
            continue
        granules.append({
            "acquisition_date": dt,
            "meta": r.get("meta", {}),
            "umm": r.get("umm", {}),
        })

    logger.info(
        "CMR query: ECO_L2T_LSTE.002 over bbox=%s, %s-%s → %d granules",
        wgs84_bbox,
        start_year,
        end_year,
        len(granules),
    )
    return granules


def summarize_granules(
    granules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics from a list of CMR granules.

    Returns per-year counts, month counts, and Landsat-adjacent window counts.
    """
    years: Counter[int] = Counter()
    month_counts: Counter[int] = Counter()
    landsat_window: Counter[int] = Counter()  # UTC hour 6-11
    total = 0

    for g in granules:
        dt = g.get("acquisition_date")
        if dt is None:
            continue
        total += 1
        years[dt.year] += 1
        month_counts[dt.month] += 1
        if 6 <= dt.hour <= 11:
            landsat_window[dt.year] += 1

    return {
        "total": total,
        "by_year": dict(sorted(years.items())),
        "by_month": dict(sorted(month_counts.items())),
        "landsat_window_by_year": dict(sorted(landsat_window.items())),
    }
