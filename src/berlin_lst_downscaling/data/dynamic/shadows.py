"""Shadow product adapter — derive building/vegetation shadow masks.

For each Landsat anchor scene, produces one binary shadow mask COG per
geometry component (building, vegetation) by comparing the pre-computed
horizon cube with the scene's solar geometry.

Algorithm
---------
1. Read the 36-band horizon cube (10° azimuth steps, centidegrees).
2. Interpolate the horizon elevation angle at the solar azimuth between
   the two neighboring 10° azimuth bands.
3. Set ``shadow = 1`` where ``horizon_elevation > solar_elevation``,
   ``0`` elsewhere.
4. Mark nodata where the horizon cube has nodata pixels.

Output: uint8 COG with ``0`` = lit, ``1`` = shadowed, ``255`` = nodata.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from hashlib import sha256

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

_N_DIRECTIONS = 36
_AZIMUTHS_DEG = np.linspace(0, 350, _N_DIRECTIONS, dtype=np.float64)
_CENTIDEGREE_SCALE = 100.0
_NODATA_UINT16 = 65535

_SHADOW_NODATA = 255


# ── contract ───────────────────────────────────────────────────────────


def contract_for_shadow(component: str) -> Contract:
    """Return the output Contract for shadow COGs."""
    return Contract(
        source=f"shadow_{component}",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name=f"shadow_{component}",
                dtype="uint8",
                nodata=_SHADOW_NODATA,
                description=(
                    f"Binary shadow mask ({component} geometry). 0=lit, 1=shadowed, 255=nodata."
                ),
                unit="",
                valid_range=(0.0, 1.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


# ── core shadow computation ────────────────────────────────────────────


def shadow_from_horizon(
    horizon_uri: str,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    grid=None,
) -> np.ndarray:
    """Compute a binary shadow mask from a pre-computed horizon cube.

    Parameters
    ----------
    horizon_uri :
        URI to the 36-band horizon COG (building or vegetation).
    azimuth_deg :
        Solar azimuth (0–360, clockwise from N).
    elevation_deg :
        Solar elevation above horizon (positive = daytime).
    grid :
        Output GeoBox (for shape).  Defaults to full canonical grid.

    Returns
    -------
    np.ndarray
        uint8 array: 0=lit, 1=shadowed, 255=nodata.
    """
    import rasterio

    grid = grid or canon_grid_10m()
    h, w = grid.shape.y, grid.shape.x

    # Nighttime: no shadow computation
    if elevation_deg <= 0:
        return np.full((h, w), _SHADOW_NODATA, dtype=np.uint8)

    # Read horizon cube
    with rasterio.open(horizon_uri) as src:
        horizon = src.read()  # shape: (36, h, w)

    # Find neighboring azimuth bands for interpolation
    az = azimuth_deg % 360.0
    az_idx_f = az / 10.0  # fractional index into 36 bands
    idx_lo = int(az_idx_f) % _N_DIRECTIONS
    idx_hi = (idx_lo + 1) % _N_DIRECTIONS

    # Fractional interpolation weight
    frac = az_idx_f - math.floor(az_idx_f)

    # Extract the two neighboring horizon bands
    h_lo = horizon[idx_lo].astype(np.float64) / _CENTIDEGREE_SCALE
    h_hi = horizon[idx_hi].astype(np.float64) / _CENTIDEGREE_SCALE

    # Interpolate
    horizon_at_az = h_lo * (1.0 - frac) + h_hi * frac

    # Build valid mask (where neither band is nodata)
    valid = (horizon[idx_lo] != _NODATA_UINT16) & (horizon[idx_hi] != _NODATA_UINT16)

    # Shadow = horizon elevation > solar elevation
    shadow = np.where(
        valid,
        np.where(horizon_at_az > elevation_deg, 1, 0),
        _SHADOW_NODATA,
    ).astype(np.uint8)

    return shadow


# ── prepare ───────────────────────────────────────────────────────────


def prepare_shadow(
    component: str,
    horizon_uri: str,
    azimuth_deg: float,
    elevation_deg: float,
    scene_id: str,
    output_root: str,
    run_id: str,
    *,
    grid=None,
    geometry_id: str = "",
    geometry_hash: str = "",
    acquisition_datetime: datetime | None = None,
    day_of_year: int | None = None,
    scene_year: int | None = None,
) -> PreparedSecondaryProduct:
    """Prepare a binary shadow mask for a single component and scene.

    Parameters
    ----------
    component :
        ``"building"`` or ``"vegetation"``.
    horizon_uri :
        URI to the component's 36-band horizon COG.
    azimuth_deg, elevation_deg :
        Solar geometry at scene acquisition time.
    scene_id :
        Landsat scene ID.
    output_root, run_id :
        Pipeline context.
    acquisition_datetime :
        Scene acquisition time (UTC). Required for correct provenance.
    day_of_year :
        Day of year (1–366). Derived from acquisition_datetime if not given.
    scene_year :
        Scene year. Derived from acquisition_datetime if not given.
    """
    grid = grid or canon_grid_10m()
    c_hash = sha256(f"shadow_{component}:{scene_id}".encode()).hexdigest()[:12]

    # Derive temporal fields from acquisition_datetime
    if scene_year is None and acquisition_datetime is not None:
        scene_year = acquisition_datetime.year
    if day_of_year is None and acquisition_datetime is not None:
        day_of_year = acquisition_datetime.timetuple().tm_yday
    if scene_year is None:
        scene_year = datetime.now(UTC).year

    log_event(
        _logger,
        logging.INFO,
        "shadow_computing",
        component=component,
        scene_id=scene_id,
        azimuth=round(azimuth_deg, 1),
        elevation=round(elevation_deg, 1),
    )

    shadow_arr = shadow_from_horizon(
        horizon_uri,
        azimuth_deg,
        elevation_deg,
        grid=grid,
    )

    # Wrap as xr.Dataset
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {f"shadow_{component}": (("y", "x"), shadow_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    # QA stats
    total_px = shadow_arr.size
    shadow_px = int(np.sum(shadow_arr == 1))
    lit_px = int(np.sum(shadow_arr == 0))
    nodata_px = int(np.sum(shadow_arr == _SHADOW_NODATA))

    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source=f"shadow_{component}",
        item_key=scene_id,
        category="dynamic",
        dataset=ds,
        contract=contract_for_shadow(component),
        nominal_interval=vintage_interval(scene_year),
        source_metadata={
            "horizon_uri": horizon_uri,
            "solar_azimuth_deg": round(azimuth_deg, 3),
            "solar_elevation_deg": round(elevation_deg, 3),
            "geometry_id": geometry_id,
            "geometry_temporal_mode": "retrospective_static",
            "component": component,
            "scene_year": scene_year,
            "day_of_year": day_of_year,
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "shadow_frac": round(shadow_px / max(total_px - nodata_px, 1), 4),
            "shadow_pixels": shadow_px,
            "lit_pixels": lit_px,
            "nodata_pixels": nodata_px,
            "total_pixels": total_px,
            "solar_azimuth": round(azimuth_deg, 1),
            "solar_elevation": round(elevation_deg, 1),
            "scene_year": scene_year,
            "day_of_year": day_of_year,
        },
        config_hash=c_hash,
        acquisition_datetime=acquisition_datetime,
        stac_properties={
            "shadow:component": component,
            "shadow:solar_azimuth": round(azimuth_deg, 3),
            "shadow:solar_elevation": round(elevation_deg, 3),
            "shadow:geometry_temporal_mode": "retrospective_static",
            "shadow:encoding": "uint8 (0=lit, 1=shadow, 255=nodata)",
            "acquisition:datetime": (
                acquisition_datetime.isoformat() if acquisition_datetime else None
            ),
            "acquisition:doy": day_of_year,
            "acquisition:year": scene_year,
        },
    )


__all__ = [
    "contract_for_shadow",
    "shadow_from_horizon",
    "prepare_shadow",
]
