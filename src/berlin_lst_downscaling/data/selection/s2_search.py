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


def match_s2_candidates(anchor: dict, cfg) -> list[dict]:
    """Return S2 L2A candidates within ±window_days of anchor's acquisition.

    Lightweight metadata-only search — no pixel loads.
    Returns list of candidate dicts with keys: scene_id, source, year,
    datetime, date, dt_days, cloud_cover, clear_frac, item_href.
    """
    cat = get_catalog()
    window_days: int = cfg.sentinel2.window_days

    anchor_dt = anchor["datetime"]
    start_dt = anchor_dt - timedelta(days=window_days)
    end_dt = anchor_dt + timedelta(days=window_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    search = cat.search(
        collections=[cfg.sentinel2.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{start_str}/{end_str}",
        query={"eo:cloud_cover": {"lt": cfg.sentinel2.cloud_cover_max}},
    )

    candidates: list[dict] = []
    for item in search.items():
        dt_utc = _parse_item_datetime(item)
        if dt_utc is None:
            continue

        dt_days = abs((dt_utc - anchor_dt).total_seconds()) / 86400.0
        if dt_days > window_days + 1e-6:
            continue

        cloud_cover = item.properties.get("eo:cloud_cover")
        item_href = item.get_self_href() if hasattr(item, "get_self_href") else None

        candidates.append(
            {
                "scene_id": item.id,
                "source": "sentinel-2-l2a",
                "year": dt_utc.year,
                "datetime": dt_utc,
                "date": dt_utc.strftime("%Y-%m-%d"),
                "dt_days": dt_days,
                "cloud_cover": cloud_cover,
                "clear_frac": None,  # filled by _with_clear_frac variant
                "item_href": item_href,
            }
        )

    candidates.sort(key=lambda c: c["dt_days"])
    return candidates


def match_s2_candidates_with_clear_frac(
    anchor: dict,
    l8_items: list,
    cfg,
) -> list[dict]:
    """Return S2 candidates with pixel-wise clear_frac pre-computed.

    Loads Landsat + S2 via odc.stac and computes clear_frac per candidate.
    Reuses the same l8_items for all candidates (the anchor scene).
    """
    import json
    import sys

    candidates = match_s2_candidates(anchor, cfg)
    if not candidates:
        return candidates

    # Resolve S2 items for the candidates (search by date ± 1 day tolerance)
    s2_items_map = _resolve_s2_items([c["datetime"] for c in candidates], cfg)

    candidate_diagnostics = []
    for c in candidates:
        s2_items = s2_items_map.get(c["datetime"])
        if s2_items is None:
            c["clear_frac"] = None
            candidate_diagnostics.append(_cf_diagnostic_entry(c, None, None))
            continue
        try:
            from berlin_lst_downscaling.data.selection.clear_frac import (
                compute_clear_frac_with_counts,
            )

            cf, counts = compute_clear_frac_with_counts(
                l8_items=l8_items,
                s2_items=s2_items,
                anchor_bbox=tuple(cfg.bbox),
                aoi_mask_path=f"{cfg.aoi.mask_base}/aoi_10m.tif",
                anchor_dt=anchor["datetime"],
            )
            c["clear_frac"] = cf
            candidate_diagnostics.append(_cf_diagnostic_entry(c, cf, counts))
        except Exception as exc:
            c["clear_frac"] = None
            import traceback

            print(f"  [clear_frac error] {c['scene_id']}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            candidate_diagnostics.append(_cf_diagnostic_entry(c, None, None))

    # Log structured diagnostic event for this anchor
    event = {
        "event": "clear_frac_diagnostic",
        "anchor_id": anchor["scene_id"],
        "anchor_date": anchor["date"],
        "n_candidates": len(candidate_diagnostics),
        "candidates": candidate_diagnostics,
    }
    print(json.dumps(event), file=sys.stderr)

    return candidates


def _cf_diagnostic_entry(candidate: dict, clear_frac: float | None, counts: dict | None) -> dict:
    entry = {
        "s2_id": candidate["scene_id"],
        "dt_days": candidate.get("dt_days"),
        "cloud_cover": candidate.get("cloud_cover"),
        "clear_frac": clear_frac,
    }
    if counts is not None:
        entry.update(
            {
                "aoi_px": counts["aoi_px"],
                "l8_clear_px": counts["l8_clear_px"],
                "s2_clear_px": counts["s2_clear_px"],
                "intersect_px": counts["intersect_px"],
            }
        )
    return entry


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
        # Use date-only range to avoid PC STAC datetime-parsing issues.
        target_date = dt.date()
        start_date = target_date - timedelta(days=1)
        end_date = target_date + timedelta(days=1)

        search = cat.search(
            collections=[cfg.sentinel2.collection],
            bbox=tuple(cfg.bbox),
            datetime=f"{start_date}/{end_date}",
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
