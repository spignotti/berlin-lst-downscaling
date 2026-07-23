"""Horizon angle computation — pre-computed per-azimuth horizon cubes.

Computes the maximum elevation angle visible from each cell along
36 evenly-spaced azimuth directions (0°–350°, 10° steps).  The result
is stored as a 36-band ``uint16`` COG encoding centidegrees (×100)
with nodata=65535.

This is a one-time pre-computation; per-scene shadow lookup is then
a fast O(pixels × 1) lookup against the stored horizon cube.

Processing
----------
1. Read the component DSM from the canonical product path.
2. For each azimuth direction, cast rays outward from each cell and
   find the maximum elevation angle along the ray.
3. Encode angles as ``uint16`` centidegrees (0–9000 range).
4. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts.

The kernel is Numba-accelerated for full-AOI performance.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from hashlib import sha256

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
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
_MAX_RADIUS_M = 200.0  # 200 m search radius
_CELL_SIZE_M = 10.0

# ── contract ───────────────────────────────────────────────────────────


def contract_for_horizon(building_or_vegetation: str) -> Contract:
    """Return the output Contract for horizon COGs."""
    return Contract(
        source=f"horizon_{building_or_vegetation}",
        target_crs="EPSG:25833",
        output_bands=tuple(
            BandSpec(
                name=f"az_{int(az):03d}",
                dtype="uint16",
                nodata=_NODATA_UINT16,
                description=f"Horizon angle at {int(az)}° azimuth (centidegrees)",
                unit="°×100",
                valid_range=(0.0, 9000.0),
            )
            for az in np.arange(0, 360, 10)
        ),
        tiling=TilingSpec(blocksize=256),  # smaller blocks for many bands
        schema_version=1,
        flag_mode="none",
    )


def config_hash_for_horizon(
    component: str,
    max_radius_m: float,
    upstream_hash: str,
) -> str:
    """Return a stable config hash for a horizon cube."""
    raw = f"horizon:{component}:r={max_radius_m}:u={upstream_hash}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── Numba horizon kernel ──────────────────────────────────────────────


def _compute_horizon_cube(
    dsm: np.ndarray,
    cell_size: float,
    max_radius_m: float,
    n_azimuths: int = 36,
) -> np.ndarray:
    """Compute horizon angles for all cells and azimuth directions.

    Parameters
    ----------
    dsm :
        2D float32 array of DSM heights (metres).
    cell_size :
        Cell size in metres.
    max_radius_m :
        Maximum ray-cast distance in metres.
    n_azimuths :
        Number of azimuth directions (default 36 = 10° steps).

    Returns
    -------
    np.ndarray
        Shape ``(n_azimuths, height, width)`` of uint16 centidegrees.
    """
    from numba import njit, prange  # noqa: F811

    h, w = dsm.shape
    max_radius_cells = int(math.ceil(max_radius_m / cell_size))

    # Precompute azimuth directions as (dx, dy) unit vectors
    azimuths_rad = np.linspace(0, 2 * math.pi, n_azimuths, endpoint=False)
    cos_az = np.cos(azimuths_rad).astype(np.float64)
    sin_az = np.sin(azimuths_rad).astype(np.float64)

    # Precompute ray offsets for each azimuth and radius
    # For each (az_idx, r), store the (dy, dx) offset in cells
    offsets_y = np.zeros((n_azimuths, max_radius_cells), dtype=np.float64)
    offsets_x = np.zeros((n_azimuths, max_radius_cells), dtype=np.float64)
    for az_idx in range(n_azimuths):
        for r in range(1, max_radius_cells + 1):
            offsets_y[az_idx, r - 1] = -sin_az[az_idx] * r  # y points down
            offsets_x[az_idx, r - 1] = cos_az[az_idx] * r

    @njit(parallel=True, cache=True)  # type: ignore[misc]
    def _kernel(
        dsm: np.ndarray,
        result: np.ndarray,
        offsets_y: np.ndarray,
        offsets_x: np.ndarray,
        cell_size: float,
        max_radius_cells: int,
        n_azimuths: int,
        h: int,
        w: int,
    ) -> None:
        nodata = 65535
        for y in prange(h):
            for x in range(w):
                observer_h = dsm[y, x]
                if np.isnan(observer_h):
                    for az_idx in range(n_azimuths):
                        result[az_idx, y, x] = nodata
                    continue

                for az_idx in range(n_azimuths):
                    max_angle_cd = 0
                    for r_idx in range(max_radius_cells):
                        ty = int(round(y + offsets_y[az_idx, r_idx]))
                        tx = int(round(x + offsets_x[az_idx, r_idx]))

                        if ty < 0 or ty >= h or tx < 0 or tx >= w:
                            continue

                        target_h = dsm[ty, tx]
                        if np.isnan(target_h):
                            continue

                        dist_m = (r_idx + 1) * cell_size
                        elev_diff = target_h - observer_h
                        angle_rad = math.atan2(elev_diff, dist_m)
                        angle_deg = angle_rad * (180.0 / math.pi)
                        if angle_deg < 0.0:
                            angle_deg = 0.0
                        angle_cd = int(round(angle_deg * 100.0))

                        if angle_cd > max_angle_cd:
                            max_angle_cd = angle_cd

                    result[az_idx, y, x] = max_angle_cd

    result = np.full((n_azimuths, h, w), _NODATA_UINT16, dtype=np.uint16)
    _kernel(dsm, result, offsets_y, offsets_x, cell_size, max_radius_cells, n_azimuths, h, w)
    return result


# ── prepare ───────────────────────────────────────────────────────────


def prepare_horizon(
    dsm_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    component: str,
    upstream_hash: str,
    max_radius_m: float = _MAX_RADIUS_M,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Compute horizon angles from a component DSM.

    Parameters
    ----------
    dsm_uri :
        URI of the component DSM COG (building_dsm or vegetation_dsm).
    component :
        ``"building"`` or ``"vegetation"``.
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.
    """
    import rasterio

    grid = grid or canon_grid_10m()
    c_hash = config_hash_for_horizon(component, max_radius_m, upstream_hash)

    # Read the component DSM
    with rasterio.open(dsm_uri) as src:
        dsm_data = src.read(1).astype(np.float32)

    # Compute horizon cube
    n_azimuths = 36
    log_event(
        _logger,
        logging.INFO,
        "horizon_computing",
        component=component,
        n_azimuths=n_azimuths,
        max_radius_m=max_radius_m,
    )
    horizon_cube = _compute_horizon_cube(
        dsm_data,
        _CELL_SIZE_M,
        max_radius_m,
        n_azimuths,
    )

    # Build canonical xr.Dataset with one variable per azimuth band
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    band_vars = {}
    for i, az in enumerate(np.arange(0, 360, 10)):
        band_vars[f"az_{int(az):03d}"] = (("y", "x"), horizon_cube[i])

    ds = xr.Dataset(band_vars, coords={"x": xs, "y": ys})
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    # QA stats from band 0 (north)
    b0 = horizon_cube[0]
    valid = b0[b0 != _NODATA_UINT16]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source=f"horizon_{component}",
        item_key=item_key,
        category="morphology",
        dataset=ds,
        contract=contract_for_horizon(component),
        nominal_interval=vintage_interval(int(item_key) if item_key.isdigit() else 2021),
        source_metadata={
            "dsm_uri": dsm_uri,
            "upstream_hash": upstream_hash,
            "component": component,
            "n_azimuths": n_azimuths,
            "azimuth_step_deg": 10.0,
            "max_radius_m": max_radius_m,
            "encoding": "uint16 centidegrees",
            "nodata": _NODATA_UINT16,
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "valid_frac": (round(float(len(valid)) / b0.size, 4) if b0.size > 0 else 0.0),
            "min_angle_cd": int(valid.min()) if len(valid) > 0 else None,
            "max_angle_cd": int(valid.max()) if len(valid) > 0 else None,
            "shape": list(horizon_cube.shape),
            "n_bands": n_azimuths,
        },
        config_hash=c_hash,
    )


__all__ = [
    "config_hash_for_horizon",
    "contract_for_horizon",
    "prepare_horizon",
]
