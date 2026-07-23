"""ECOSTRESS granule search via NASA CMR (earthaccess).

Queries NASA CMR (LP DAAC) for ECO_L2T_LSTE.002 granules covering the
configured bbox within a date range.
"""

from __future__ import annotations

import earthaccess

from berlin_lst_downscaling.data.acquisition.ecostress import (
    parse_granule_datetime,
    parse_granule_mgrs,
)

# Berlin bbox (WGS84) for overlap computation
_BERLIN_BBOX = (13.08, 52.34, 13.76, 52.68)


def search_ecostress(
    start: str,
    end: str,
    bbox: tuple[float, float, float, float] | None = None,
    version: str = "002",
) -> list[dict]:
    """Query CMR for ECO_L2T_LSTE.002 granules covering bbox in [start, end].

    Parameters
    ----------
    start :
        Start date string "YYYY-MM-DD".
    end :
        End date string "YYYY-MM-DD" (inclusive).
    bbox :
        WGS84 bounding box ``(minx, miny, maxx, maxy)``. Defaults to the
        internal Berlin bbox.
    version :
        Collection version (default: "002").

    Returns
    -------
    list[ECOSTRESSMatch]
        Granules sorted by datetime.
    """
    if bbox is None:
        bbox = _BERLIN_BBOX

    _ensure_earthdata_login()

    minx, miny, maxx, maxy = bbox
    try:
        results = earthaccess.search_data(
            short_name="ECO_L2T_LSTE",
            version=version,
            bounding_box=(minx, miny, maxx, maxy),
            temporal=(start, end),
            count=500,
        )
    except Exception as exc:
        raise RuntimeError(f"CMR query failed: {exc}") from exc

    matches: list[dict] = []
    for granule in results:
        granule_id: str = granule["meta"]["native-id"]

        dt = parse_granule_datetime(granule_id)
        if dt is None:
            continue

        mgrs = parse_granule_mgrs(granule_id)
        overlap = _footprint_overlap(granule, bbox)
        if overlap < 0.10:
            continue

        matches.append(
            {
                "granule_id": granule_id,
                "source": "ecostress",
                "year": dt.year,
                "datetime": dt,
                "date": dt.strftime("%Y-%m-%d"),
                "dt_hours": 0.0,  # caller must fill relative to anchor
                "mgrs_tile": mgrs,
                "overlap_frac": overlap,
                "clear_frac": None,  # computed later in ecostress_subset
            }
        )

    matches.sort(key=lambda m: m["datetime"])
    return matches


def _footprint_overlap(
    granule: dict,
    bbox: tuple[float, float, float, float],
) -> float:
    """Return fraction [0,1] of bbox overlapped by the granule's CMR footprint.

    Falls back to 1.0 (permissive) if metadata is missing.
    """
    berlin_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    try:
        rects = (
            granule["umm"]
            .get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
            .get("BoundingRectangles", [])
        )
        if not rects:
            return 1.0
        r = rects[0]
        iw = max(r["WestBoundingCoordinate"], bbox[0])
        ie = min(r["EastBoundingCoordinate"], bbox[2])
        is_ = max(r["SouthBoundingCoordinate"], bbox[1])
        in_ = min(r["NorthBoundingCoordinate"], bbox[3])
        if iw >= ie or is_ >= in_:
            return 0.0
        return (ie - iw) * (in_ - is_) / berlin_area
    except Exception:
        return 1.0


def _ensure_earthdata_login() -> None:
    """Authenticate with NASA Earthdata, raising RuntimeError on failure."""
    try:
        earthaccess.login()
    except Exception as exc:
        raise RuntimeError(
            f"NASA Earthdata login failed. Run `python -c 'import earthaccess; "
            f"earthaccess.login()'` interactively once to cache credentials. "
            f"Detail: {exc}"
        ) from exc


__all__ = ["search_ecostress", "parse_granule_datetime", "parse_granule_mgrs"]
