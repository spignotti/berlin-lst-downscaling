"""Landsat anchor search — PC STAC query filtered to Berlin AOI and season."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset

from berlin_lst_downscaling.common.grid import canon_grid_10m
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

    Uses ThreadPoolExecutor for parallel I/O (2 workers) and writes
    a checkpoint every 50 anchors so the filter can resume if killed.

    Loads QA_PIXEL via stac_load for each anchor over the Berlin bbox and
    computes ``AOI ∩ clear / AOI``.  Anchors below the threshold are
    dropped from the list and logged to stderr.

    Returns ``(kept, dropped)``.
    """
    import pickle
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    ckpt_path = "data/ard/anchor_filter_checkpoint.pkl"

    # Load checkpoint — cached clear_frac by scene_id
    cf_cache: dict[str, float | None] = {}
    if Path(ckpt_path).exists():
        try:
            with open(ckpt_path, "rb") as f:
                cf_cache = pickle.load(f)  # noqa: S301 — internal checkpoint
            print(
                f"  [anchor_filter] Resumed checkpoint: {len(cf_cache)} anchors cached",
                file=sys.stderr,
            )
        except Exception:
            print("  [anchor_filter] Checkpoint load failed — starting fresh", file=sys.stderr)

    n_total = len(anchors)
    todo = [a for a in anchors if a["scene_id"] not in cf_cache]

    if todo:
        done = len(cf_cache)
        print(
            f"  [anchor_filter] processing {len(todo)}/{n_total} anchors "
            f"(min_clear_frac={min_clear_frac})",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(compute_anchor_clear_frac, a, cfg): a for a in todo}
            for future in as_completed(futures):
                anchor = futures[future]
                cf = future.result()
                cf_cache[anchor["scene_id"]] = cf
                done += 1
                if done % 20 == 0 or done == n_total:
                    print(
                        f"  [anchor_filter] progress: {done}/{n_total} anchors "
                        f"(last: {anchor['scene_id']})",
                        file=sys.stderr,
                    )
                if done % 50 == 0:
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(cf_cache, f)

    # Apply filter using cached results
    kept, dropped = [], []
    for anchor in anchors:
        cf = cf_cache.get(anchor["scene_id"])
        if cf is not None and cf >= min_clear_frac:
            kept.append(anchor)
        else:
            dropped.append(anchor)
            cf_str = f"{cf:.3f}" if cf is not None else "N/A"
            print(
                f"[anchor_filter] dropped {anchor['scene_id']} "
                f"(clear_frac={cf_str} < {min_clear_frac})",
                file=sys.stderr,
            )

    print(
        f"[anchor_filter] kept {len(kept)}/{len(anchors)} anchors "
        f"(min_clear_frac={min_clear_frac})",
        file=sys.stderr,
    )

    # Delete checkpoint on successful completion
    Path(ckpt_path).unlink(missing_ok=True)
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
        # Load QA_PIXEL on the canonical 10m grid
        gbox = canon_grid_10m()
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
