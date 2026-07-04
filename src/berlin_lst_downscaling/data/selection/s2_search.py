"""Sentinel-2 candidate search — ±window_days around each Landsat anchor.

This module provides two functions:

1. ``match_s2_candidates`` — lightweight STAC search returning candidate
   metadata (scene_id, datetime, dt_days, cloud_cover).  Used by
   ``run_scan`` where pixel loads are not needed.

2. ``match_s2_candidates_with_clear_frac`` — same search but also
   computes pixel-wise ``clear_frac`` for each candidate on the
   canonical 10-m EPSG:25833 grid.  Used by the coupling step.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog
from berlin_lst_downscaling.data.selection import Anchor, S2Candidate
from berlin_lst_downscaling.data.selection.clear_frac import compute_clear_frac


def match_s2_candidates(anchor: Anchor, cfg) -> list[S2Candidate]:
    """Return S2 L2A candidates within ±window_days of anchor's acquisition.

    Lightweight metadata-only search — no pixel loads.
    Used by ``run_scan`` for volume estimation.
    """
    cat = get_catalog()
    window_days: int = cfg.sentinel2.window_days

    start_dt = anchor.datetime - timedelta(days=window_days)
    end_dt = anchor.datetime + timedelta(days=window_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    search = cat.search(
        collections=[cfg.sentinel2.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{start_str}/{end_str}",
        query={"eo:cloud_cover": {"lt": cfg.sentinel2.cloud_cover_max}},
    )

    candidates: list[S2Candidate] = []
    for item in search.items():
        dt_utc = _parse_item_datetime(item)
        if dt_utc is None:
            continue

        dt_days = abs((dt_utc - anchor.datetime).total_seconds()) / 86400.0
        if dt_days > window_days + 1e-6:
            continue

        cloud_cover = item.properties.get("eo:cloud_cover")
        item_href = item.get_self_href() if hasattr(item, "get_self_href") else None

        candidates.append(
            S2Candidate(
                scene_id=item.id,
                source="sentinel-2-l2a",
                year=dt_utc.year,
                datetime=dt_utc,
                date=dt_utc.strftime("%Y-%m-%d"),
                dt_days=dt_days,
                cloud_cover=cloud_cover,
                item_href=item_href,
            )
        )

    candidates.sort(key=lambda c: c.dt_days)
    return candidates


def match_s2_candidates_with_clear_frac(
    anchor: Anchor,
    l8_items: list,
    cfg,
) -> list[S2Candidate]:
    """Return S2 candidates with pixel-wise clear_frac pre-computed.

    Loads Landsat + S2 via odc.stac and computes clear_frac per candidate.
    Reuses the same l8_items for all candidates (the anchor scene).
    """
    candidates = match_s2_candidates(anchor, cfg)
    if not candidates:
        return candidates

    # Resolve S2 items for the candidates (search by date ± 1 day tolerance)
    s2_items_map = _resolve_s2_items([c.datetime for c in candidates], cfg)

    for c in candidates:
        s2_items = s2_items_map.get(c.datetime)
        if s2_items is None:
            c.clear_frac = None
            continue
        try:
            c.clear_frac = compute_clear_frac(
                l8_items=l8_items,
                s2_items=s2_items,
                anchor_bbox=tuple(cfg.bbox),
                aoi_mask_path=f"{cfg.aoi.mask_base}/aoi_10m.tif",
                resolution=10,
            )
        except Exception:
            c.clear_frac = None

    return candidates


def _resolve_s2_items(
    datetimes: list,
    cfg,
) -> dict:
    """Resolve S2 STAC items by datetime.

    Returns dict mapping UTC datetime → list of pystac Item.
    Since odc.stac.load uses groupby="solar_day", we search per date.
    """

    cat = get_catalog()
    result: dict = {}

    for dt in datetimes:
        day_start = dt.replace(hour=0, minute=0, second=0)
        day_end = dt.replace(hour=23, minute=59, second=59)

        search = cat.search(
            collections=[cfg.sentinel2.collection],
            bbox=tuple(cfg.bbox),
            datetime=f"{day_start.strftime('%Y-%m-%d')}/{day_end.strftime('%Y-%m-%d')}",
            query={"eo:cloud_cover": {"lt": cfg.sentinel2.cloud_cover_max}},
        )
        items = list(search.items())
        if items:
            result[dt] = items

    return result


def _parse_item_datetime(item) -> datetime | None:
    """Extract UTC datetime from a STAC item."""
    dt_str = item.properties.get("datetime")
    if dt_str is None:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None
