"""AOI-based QA metrics — pixel counts within the Berlin boundary.

``compute_aoi_metrics`` reads the flag COG and the pre-rasterized AOI
mask, intersects them, and returns per-category pixel counts for the
scene's AOI area.

AOI masks (``aoi_10m.tif``, ``aoi_100m.tif``) are pre-baked by
``scripts/build_aoi.py`` from ``berlin_landesgrenze.geojson`` (EPSG:25833).
They are uint8, 1 = inside Berlin, 0 = outside.
"""

from __future__ import annotations

import numpy as np
import rasterio

from berlin_lst_downscaling.data.ard.contract import Contract

# ── flag bit definitions (must match contract.py) ───────────────────

_FLAG_FILL = 1 << 0  # 1
_FLAG_CLOUDY = 1 << 1  # 2
_FLAG_SHADOW = 1 << 2  # 4
_FLAG_CIRRUS = 1 << 3  # 8
_FLAG_SATURATED = 1 << 4  # 16


def load_aoi_mask(aoi_uri: str) -> tuple[np.ndarray, dict]:
    """Load an AOI mask COG (uint8, 1=inside, 0=outside).

    Returns ``(mask, profile)`` where *mask* is a 2D uint8 numpy array
    and *profile* is the rasterio profile dict.

    Raises ``FileNotFoundError`` if the AOI COG does not exist.
    """
    with rasterio.open(aoi_uri) as src:
        data = src.read(1)  # single band
        profile = src.profile.copy()
    return data, profile


def compute_aoi_metrics(
    flag_uri: str,
    aoi_uri: str,
    contract: Contract,
) -> dict[str, int | float]:
    """Compute AOI intersection metrics from the flag COG and AOI mask.

    Parameters
    ----------
    flag_uri :
        URI to the flag COG (uint8 bitmask).
    aoi_uri :
        URI to the pre-rasterized AOI mask COG (uint8, 1=inside Berlin).
    contract :
        Contract carrying flag bit definitions.

    Returns
    -------
    dict
        ``aoi_clear_px``, ``aoi_cloudy_px``, ``aoi_shadow_px``,
        ``aoi_cirrus_px``, ``aoi_saturated_px``, ``aoi_fill_px``,
        ``aoi_total_px`` (non-fill, non-no-data pixels inside AOI),
        ``aoi_overlap_px`` (all pixels inside COG∩AOI, including fill),
        ``aoi_clear_frac`` (clear / total inside AOI, NaN if total=0).
    """
    # ── load flag COG ────────────────────────────────────────────────
    with rasterio.open(flag_uri) as src:
        flag_data = src.read(1)
        flag_profile = src.profile.copy()
        flag_crs = src.crs
        flag_transform = src.transform

    # ── load AOI mask ────────────────────────────────────────────────
    # Reproject AOI mask to match the flag COG's grid (resolution,
    # extent, and CRS) — needed even when CRS is identical because
    # the pre-baked AOI covers the full Berlin bounding box while the
    # scene covers only its tile.
    import rasterio.warp as rwarp

    with rasterio.open(aoi_uri) as src:
        aoi_data = src.read(1)
        aoi_crs = src.crs
        aoi_transform = src.transform
        aoi_width = src.width
        aoi_height = src.height

    # Always reproject — handles both CRS mismatch and extent/resolution
    # mismatch. The AOI mask is pre-baked at a fixed resolution covering
    # the full Berlin bounding box; reproject it to match the flag COG's
    # grid (same CRS, same transform, same pixel grid).
    # Note: provide destination array to avoid conflict between dst_transform
    # and dst_width/dst_height in rasterio's reproject API.
    destination = np.empty(
        (flag_profile["height"], flag_profile["width"]), dtype=aoi_data.dtype
    )
    aoi_data, _ = rwarp.reproject(
        source=aoi_data,
        src_transform=aoi_transform,
        src_width=aoi_width,
        src_height=aoi_height,
        src_crs=aoi_crs,
        destination=destination,
        dst_crs=flag_crs,
        dst_transform=flag_transform,
        resampling=rwarp.Resampling.nearest,
    )

    # Cast to bool for masking
    inside = aoi_data == 1

    fill_mask = (flag_data & _FLAG_FILL) != 0
    cloudy_mask = (flag_data & _FLAG_CLOUDY) != 0
    shadow_mask = (flag_data & _FLAG_SHADOW) != 0
    cirrus_mask = (flag_data & _FLAG_CIRRUS) != 0
    saturated_mask = (flag_data & _FLAG_SATURATED) != 0

    # Clear = not fill, not cloudy, not shadow, not cirrus, not saturated
    clear_mask = ~fill_mask & ~cloudy_mask & ~shadow_mask & ~cirrus_mask & ~saturated_mask

    aoi_fill_px = int(np.sum(inside & fill_mask))
    aoi_cloudy_px = int(np.sum(inside & cloudy_mask))
    aoi_shadow_px = int(np.sum(inside & shadow_mask))
    aoi_cirrus_px = int(np.sum(inside & cirrus_mask))
    aoi_saturated_px = int(np.sum(inside & saturated_mask))
    aoi_clear_px = int(np.sum(inside & clear_mask))

    # Total AOI pixels that are not fill (usable area)
    aoi_total_px = aoi_clear_px + aoi_cloudy_px + aoi_shadow_px + aoi_cirrus_px + aoi_saturated_px

    # All pixels in the COG∩AOI intersection (including fill) — used to detect
    # scenes whose valid data only covers a tiny fraction of the overlap area
    # (e.g. off-target swaths where the COG covers AOI but all LST pixels are NaN).
    aoi_overlap_px = int(np.sum(inside))

    aoi_clear_frac = float(aoi_clear_px) / float(aoi_total_px) if aoi_total_px > 0 else float("nan")

    return {
        "aoi_clear_px": aoi_clear_px,
        "aoi_cloudy_px": aoi_cloudy_px,
        "aoi_shadow_px": aoi_shadow_px,
        "aoi_cirrus_px": aoi_cirrus_px,
        "aoi_saturated_px": aoi_saturated_px,
        "aoi_fill_px": aoi_fill_px,
        "aoi_total_px": aoi_total_px,
        "aoi_overlap_px": aoi_overlap_px,
        "aoi_clear_frac": aoi_clear_frac,
    }


__all__ = [
    "load_aoi_mask",
    "compute_aoi_metrics",
]
