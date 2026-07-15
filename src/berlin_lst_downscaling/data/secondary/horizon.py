"""Horizon angle computation — pre-computed per-azimuth horizon cubes.

Computes the maximum elevation angle visible from each cell along
36 evenly-spaced azimuth directions (0°–350°, 10° steps).  The result
is stored as a 36-band ``uint16`` COG encoding centidegrees (×100)
with nodata=65535.

This is a one-time pre-computation; per-scene shadow lookup is then
a fast O(pixels × 1) lookup against the stored horizon cube.

Processing
----------
1. Read the combined DSM from the canonical product path.
2. For each azimuth direction, cast rays outward from each cell and
   find the maximum elevation angle along the ray.
3. Encode angles as ``uint16`` centidegrees (0–36000 range).
4. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts.

The kernel is implemented in pure NumPy (no Numba) for portability.
A Numba-accelerated version can be added later if profiling shows
the pure-NumPy version is too slow.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from hashlib import sha256

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

# ── constants ──────────────────────────────────────────────────────────

_N_DIRECTIONS = 36
_AZIMUTHS_DEG = np.linspace(0, 350, _N_DIRECTIONS, dtype=np.float64)
_CENTIDEGREE_SCALE = 100.0
_NODATA_UINT16 = 65535
_MAX_RADIUS_M = 200.0  # 200 m search radius
_CELL_SIZE_M = 10.0

# ── contract ───────────────────────────────────────────────────────────


def contract_for_horizon(building_or_vegetation: str) -> Contract:
    """Return the output Contract for horizon COGs.

    Parameters
    ----------
    building_or_vegetation :
        Either ``"building"`` or ``"vegetation"``.
    """
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
                valid_range=(0.0, 36000.0),
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


# ── horizon computation ───────────────────────────────────────────────


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
    h, w = dsm.shape
    max_radius_cells = int(math.ceil(max_radius_m / cell_size))
    azimuths = np.linspace(0, 2 * math.pi, n_azimuths, endpoint=False)

    result = np.full((n_azimuths, h, w), _NODATA_UINT16, dtype=np.uint16)

    for az_idx, az in enumerate(azimuths):
        dx = math.cos(az)
        dy = -math.sin(az)  # negative because y-axis points down

        for r in range(1, max_radius_cells + 1):
            # Offset in pixels
            step_x = dx * r
            step_y = dy * r

            # For each cell, sample the DSM at the ray endpoint
            for y in range(h):
                for x in range(w):
                    # Target cell
                    tx = int(round(x + step_x))
                    ty = int(round(y + step_y))

                    if tx < 0 or tx >= w or ty < 0 or ty >= h:
                        continue

                    # Elevation angle from observer to target
                    dist_m = r * cell_size
                    elev_diff = dsm[ty, tx] - dsm[y, x]
                    angle_rad = math.atan2(elev_diff, dist_m)
                    angle_deg = max(0.0, math.degrees(angle_rad))
                    angle_cd = int(round(angle_deg * _CENTIDEGREE))

                    # Update if this angle is larger than current
                    current = result[az_idx, y, x]
                    if current == _NODATA_UINT16 or angle_cd > current:
                        result[az_idx, y, x] = angle_cd

    return result


_CENTIDEGREE = 100.0  # multiply degrees by this to get centidegrees


# ── prepare ───────────────────────────────────────────────────────────


def prepare_horizon(
    combined_dsm_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    component: str,
    upstream_hash: str,
    max_radius_m: float = _MAX_RADIUS_M,
) -> PreparedSecondaryProduct:
    """Compute horizon angles from the combined DSM.

    Parameters
    ----------
    combined_dsm_uri :
        URI of the finalized combined_dsm COG.
    output_root :
        Root URI for all outputs.
    run_id :
        Unique run identifier.
    item_key :
        Vintage key for the input DSM.
    component :
        ``"building"`` or ``"vegetation"`` (for naming only — both
        use the same combined DSM as input).
    upstream_hash :
        Config hash of the upstream combined_dsm.
    max_radius_m :
        Maximum ray-cast distance in metres.

    Returns
    -------
    PreparedSecondaryProduct
        36-band horizon cube on the canonical 10 m grid.
    """
    import rasterio

    grid = canon_grid_10m()
    c_hash = config_hash_for_horizon(component, max_radius_m, upstream_hash)

    # Read the combined DSM
    with rasterio.open(combined_dsm_uri) as src:
        dsm_data = src.read(1).astype(np.float32)

    # Compute horizon cube
    n_azimuths = 36
    print(
        f"  Horizon ({component}): computing {n_azimuths} directions, "
        f"radius={max_radius_m}m..."
    )
    horizon_cube = _compute_horizon_cube(
        dsm_data, _CELL_SIZE_M, max_radius_m, n_azimuths,
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
            "combined_dsm_uri": combined_dsm_uri,
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
            "valid_frac": (
                round(float(len(valid)) / b0.size, 4)
                if b0.size > 0 else 0.0
            ),
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
