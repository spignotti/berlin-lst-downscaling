"""ECOSTRESS subset: only granules on Landsat anchor days, ±window_hours local.

Per the Szenen-Selektion spec:
  "ECOSTRESS-Subset für Validierung: nur an Landsat-Anker-Tagen,
   nur Granules mit Überflugzeit im Fenster Landsat-Zeit ±2 h
   (etwa 08:00–12:00 lokal), die Berlin überlappen und klare Pixel haben."
"""

from __future__ import annotations

from datetime import timedelta

import pytz

from berlin_lst_downscaling.data.selection.ecostress import (
    search_ecostress,
)


def build_ecostress_subset(
    pairs: list[dict],
    cfg,
) -> dict[str, list[dict]]:
    """Find ECOSTRESS granules for each coupled pair.

    For each anchor, searches CMR for ECOSTRESS granules on the anchor's
    acquisition date within ±window_hours of local Berlin time, then
    filters by Berlin footprint overlap and (optionally) clear-pixel fraction.

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
    tz = pytz.timezone(cfg.ecostress.local_tz)
    window_hours: int = cfg.ecostress.window_hours
    clear_frac_min: float = cfg.ecostress.clear_frac_min

    result: dict[str, list[dict]] = {}

    for pair in pairs:
        anchor = pair["anchor"]

        # ── Convert anchor UTC → Berlin local → ±window_hours ─────────────────
        anchor_utc = anchor["datetime"]
        anchor_local = anchor_utc.astimezone(tz)

        window_start_local = anchor_local - timedelta(hours=window_hours)
        window_end_local = anchor_local + timedelta(hours=window_hours)

        # Convert bounds back to UTC for CMR query
        window_start_utc = window_start_local.astimezone(pytz.UTC)
        window_end_utc = window_end_local.astimezone(pytz.UTC)

        start_str = window_start_utc.strftime("%Y-%m-%d")
        end_str = window_end_utc.strftime("%Y-%m-%d")

        # ── Search CMR for granules in the narrow time window ───────────────
        try:
            granules = search_ecostress(
                start=start_str,
                end=end_str,
                bbox=tuple(cfg.bbox),
                version=cfg.ecostress.version,
            )
        except RuntimeError:
            result[anchor["scene_id"]] = []
            continue

        # ── Convert ECOSTRESSMatch objects to dicts ────────────────────────────
        granules = [_eco_match_to_dict(g) for g in granules]

        # ── Filter and enrich with dt_hours + clear_frac ─────────────────────
        matches: list[dict] = []
        for g in granules:
            # dt_hours relative to anchor in local Berlin time
            g_local = g["datetime"].astimezone(tz)
            dt_hours = abs((g_local - anchor_local).total_seconds()) / 3600.0

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


def _eco_match_to_dict(g) -> dict:
    """Convert an ECOSTRESSMatch dataclass to a plain dict."""
    return {
        "granule_id": g.granule_id,
        "source": g.source,
        "year": g.year,
        "datetime": g.datetime,
        "date": g.date,
        "dt_hours": g.dt_hours,
        "mgrs_tile": g.mgrs_tile,
        "overlap_frac": g.overlap_frac,
        "clear_frac": g.clear_frac,
    }


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
