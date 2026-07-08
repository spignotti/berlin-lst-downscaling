"""Landsat anchor search — PC STAC query filtered to Berlin AOI and season."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset
from odc.geo.geobox import GeoBox
from rasterio.warp import transform_bounds

from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog, stac_load
from berlin_lst_downscaling.data.ard.masking import landsat_qa_to_clear_bits
from berlin_lst_downscaling.data.selection._aoi import load_aoi_mask, select_time_slice


def build_anchors(cfg) -> tuple[list, dict]:
    """Return Landsat C2 L2 scenes as coupling anchors.

    Queries PC STAC for all scenes intersecting the configured bbox within
    the year range, then filters to May–September (configurable months).

    Returns a tuple ``(anchors, stats)`` where ``anchors`` is a list of dicts
    and ``stats`` is a dict with keys ``n_total``, ``n_kept``, ``n_dropped``
    (pixel-filter stats; ``n_dropped`` is 0 when ``min_clear_frac`` is 0).
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
        # scene-level cloud_cover filter removed per audit decision:
        # all Landsat overpasses over Berlin loaded; fitness decided
        # pixel-wise via QA_PIXEL ∩ AOI below.
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

        anchors.append(
            {
                "scene_id": item.id,
                "source": "landsat-c2-l2",
                "year": dt_utc.year,
                "datetime": dt_utc,
                "date": dt_utc.strftime("%Y-%m-%d"),
                "cloud_cover": cloud_cover,
                "sun_azimuth": sun_az,
                "sun_elevation": sun_el,
                "item_href": item_href,
            }
        )

    # ── Pixel-wise anchor fitness gate ─────────────────────────────────────
    min_cf = getattr(cfg.landsat.anchor, "min_clear_frac", 0.0)
    n_before_filter = len(anchors)
    if min_cf > 0:
        kept, dropped = _filter_by_pixel_clear_frac(anchors, cfg, min_cf)
        anchors = kept
    else:
        dropped = []

    # Sort chronologically
    anchors.sort(key=lambda a: a["datetime"])

    stats = {
        "n_total": n_before_filter,
        "n_kept": len(anchors),
        "n_dropped": len(dropped),
    }
    return anchors, stats


def _filter_by_pixel_clear_frac(
    anchors: list[dict],
    cfg,
    min_clear_frac: float,
) -> tuple[list[dict], list[dict]]:
    """Drop anchors whose AOI clear fraction is below min_clear_frac.

    Loads QA_PIXEL via odc.stac for each anchor over the Berlin bbox and
    computes ``AOI ∩ clear / AOI``.  Anchors below the threshold are
    dropped from the list and logged to stderr.

    Returns ``(kept, dropped)``.
    """
    kept, dropped = [], []
    for anchor in anchors:
        cf = compute_anchor_clear_frac(anchor, cfg)
        if cf is not None and cf >= min_clear_frac:
            kept.append(anchor)
        else:
            dropped.append(anchor)
            cf_str = f"{cf:.3f}" if cf is not None else "N/A"
            import sys

            print(
                f"[anchor_filter] dropped {anchor['scene_id']} "
                f"(clear_frac={cf_str} < {min_clear_frac})",
                file=sys.stderr,
            )
    import sys

    print(
        f"[anchor_filter] kept {len(kept)}/{len(anchors)} anchors "
        f"(min_clear_frac={min_clear_frac})",
        file=sys.stderr,
    )
    return kept, dropped


def compute_anchor_clear_frac(
    anchor: dict,
    cfg,
) -> float | None:
    """Compute pixel-wise clear fraction for a Landsat anchor over the Berlin AOI.

    Loads QA_PIXEL via odc.stac over the configured bbox, intersects with
    the Berlin AOI mask, and returns the fraction of clear pixels.

    Returns None on load failure.
    """
    from datetime import timedelta

    anchor_dt = anchor["datetime"]
    day_start = anchor_dt - timedelta(days=1)
    day_end = anchor_dt + timedelta(days=1)

    cat = get_catalog()
    search = cat.search(
        collections=[cfg.landsat.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{day_start.strftime('%Y-%m-%d')}/{day_end.strftime('%Y-%m-%d')}",
    )
    items = list(search.items())
    if not items:
        return None

    try:
        # Use explicit GeoBox — odc.stac has a bug where (crs, resolution, bbox)
        # with EPSG:25833 coordinates causes OverflowError on Landsat items.
        bbox_wgs84 = tuple(cfg.bbox)
        bbox_25833 = transform_bounds("EPSG:4326", "EPSG:25833", *bbox_wgs84)
        gbox = GeoBox.from_bbox(bbox_25833, crs="EPSG:25833", resolution=10)
        ds = stac_load(
            items=items,
            bands=["qa_pixel"],
            geobox=gbox,
            chunks={"x": 2048, "y": 2048},
            groupby="solar_day",
        )
    except Exception:
        return None

    qa_ds = select_time_slice(ds, anchor["datetime"])
    qa = qa_ds["qa_pixel"].values[0].astype(np.uint16)  # (y, x)

    # Use the production-tested shared helper (no dilation — coupling only)
    l8_clear = landsat_qa_to_clear_bits(qa)

    # AOI mask
    aoi = load_aoi_mask(
        f"{cfg.aoi.mask_base}/aoi_10m.tif",
        ds,
    )

    denom = int(np.sum(aoi))
    if denom == 0:
        return None
    numer = int(np.sum(aoi & l8_clear))
    return numer / denom


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
