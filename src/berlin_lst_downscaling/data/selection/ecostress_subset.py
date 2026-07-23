"""ECOSTRESS subset: only granules on Landsat anchor days, ±window_hours local.

Per the Szenen-Selektion spec:
  "ECOSTRESS-Subset für Validierung: nur an Landsat-Anker-Tagen,
   nur Granules mit Überflugzeit im Fenster Landsat-Zeit ±2 h
   (etwa 08:00–12:00 lokal), die Berlin überlappen und klare Pixel haben."
"""

from __future__ import annotations

import logging
from datetime import UTC
from zoneinfo import ZoneInfo

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.selection.ecostress import (
    search_ecostress,
)

_logger = logging.getLogger(__name__)


def build_ecostress_subset(
    pairs: list[dict],
    cfg,
) -> dict[str, list[dict]]:
    """Find ECOSTRESS granules for each coupled pair.

    For each anchor, searches CMR for ECOSTRESS granules on the anchor's
    acquisition date within ±window_hours of local Berlin time, then
    filters by Berlin footprint overlap and (optionally) clear-pixel fraction.

    Performance: queries CMR *once* for the full date range of all coupled
    pairs, then filters in-memory per anchor (avoids 500+ individual CMR
    queries).

    Parameters
    ----------
    pairs :
        List of successfully coupled pairs (with S2) as dicts.
    cfg :
        Hydra config with ``ecostress.*`` and ``bbox``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping ``anchor.scene_id`` → list of matched ECOSTRESS granule dicts.
        Empty list if no granules pass the filters.
    """
    if not pairs:
        return {}

    tz = ZoneInfo(cfg.ecostress.local_tz)
    window_hours: int = cfg.ecostress.window_hours
    clear_frac_min: float = cfg.ecostress.clear_frac_min

    # ── Query CMR once for full date range of all coupled anchors ────────
    min_year = min(p["anchor"]["year"] for p in pairs)
    max_year = max(p["anchor"]["year"] for p in pairs)
    all_granules = search_ecostress(
        start=f"{min_year}-01-01",
        end=f"{max_year}-12-31",
        bbox=tuple(cfg.bbox),
        version=cfg.ecostress.version,
    )
    log_event(
        _logger,
        logging.INFO,
        "cmr_ecostress_query",
        n_granules=len(all_granules),
        min_year=min_year,
        max_year=max_year,
    )

    result: dict[str, list[dict]] = {}

    for pair in pairs:
        anchor = pair["anchor"]

        # ── Convert anchor UTC → Berlin local → ±window_hours ─────────────────
        anchor_utc = anchor["datetime"]
        anchor_local = anchor_utc.astimezone(tz)

        # ── Filter cached granules by time window ─────────────────────────
        matches: list[dict] = []
        for g in all_granules:
            # dt_hours relative to anchor in local Berlin time
            if g["datetime"].tzinfo is None:
                g_dt_utc = g["datetime"].replace(tzinfo=UTC)
            else:
                g_dt_utc = g["datetime"]
            g_local = g_dt_utc.astimezone(tz)
            dt_hours = abs((g_local - anchor_local).total_seconds()) / 3600.0

            # Enforce ±window_hours filter
            if dt_hours > window_hours:
                continue

            # footprint_overlap already ≥ 0.10 from search_ecostress
            if g["overlap_frac"] < cfg.ecostress.overlap_min:
                continue

            # clear_frac: if the granule has a local raw_dir with cloud.tif,
            # compute it; otherwise leave as None (scan mode)
            if clear_frac_min > 0 and cfg.ecostress.get("raw_dir"):
                try:
                    g["clear_frac"] = _compute_ecostress_clear_frac(
                        g["granule_id"],
                        cfg.ecostress.raw_dir,
                        tuple(cfg.bbox),
                    )
                    if g["clear_frac"] is not None and g["clear_frac"] < clear_frac_min:
                        continue
                except Exception as exc:
                    import warnings

                    warnings.warn(f"clear_frac failed for {g['granule_id']}: {exc}", stacklevel=2)

            g["dt_hours"] = dt_hours
            matches.append(g)

        result[anchor["scene_id"]] = matches

    return result


def _compute_ecostress_clear_frac(
    granule_id: str,
    raw_dir: str,
    bbox: tuple[float, float, float, float],
) -> float | None:
    """Compute fraction of cloud==0 pixels inside Berlin for one ECOSTRESS granule.

    Loads the cloud layer from the staged granule directory (raw_dir/granule_id/).
    Returns None on any error (file not found, rasterio error, etc.)
    """
    from pathlib import Path

    import numpy as np
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    layer_path = Path(str(raw_dir).rstrip("/")) / granule_id / f"{granule_id}_cloud.tif"

    if not layer_path.exists():
        return None

    try:
        # Reproject Berlin bbox to source CRS
        target_crs = "EPSG:25833"
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", target_crs, *bbox)

        geom = [box(minx, miny, maxx, maxy).__geo_interface__]
        with rasterio.open(layer_path) as src:
            clipped_cloud, _ = rio_mask(src, geom, crop=True)

        cloud_clipped = clipped_cloud[0]
        total = cloud_clipped.size
        if total == 0:
            return None
        clear_px = np.sum(cloud_clipped == 0)
        fill_px = np.sum(cloud_clipped == 255)
        valid = total - fill_px
        if valid == 0:
            return 0.0
        return clear_px / valid
    except Exception:
        return None
