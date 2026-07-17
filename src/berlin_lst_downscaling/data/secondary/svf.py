"""Sky View Factor (SVF) computation — wrapper around ``xarray-spatial``.

Computes SVF from a combined DSM using the Zakek 2011 hemispherical
view analysis algorithm.  The computation is numba-backed and takes
~21s for the full Berlin AOI (12M pixels, max_radius=3, n_directions=16).

The SVF measures the fraction of the sky hemisphere visible from each
cell on a scale from 0 (fully obstructed) to 1 (flat open terrain).

Processing
----------
1. Read the combined DSM from the canonical product path.
2. Cast to xr.DataArray with explicit cell sizes (10 m).
3. Compute SVF via ``xrspatial.sky_view_factor()``.
4. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from hashlib import sha256

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from xrspatial import sky_view_factor

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── contract ───────────────────────────────────────────────────────────


def contract_for_svf() -> Contract:
    """Return the output Contract for SVF COGs."""
    return Contract(
        source="svf",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="svf",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Sky View Factor [0, 1]: fraction of sky hemisphere "
                    "visible from each cell (Zakek 2011)"
                ),
                unit="",
                valid_range=(-0.01, 1.01),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def config_hash_for_svf(
    max_radius: int,
    n_directions: int,
    upstream_hash: str,
) -> str:
    """Return a stable config hash for SVF with given parameters."""
    raw = f"svf:r={max_radius}:d={n_directions}:u={upstream_hash}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── prepare ───────────────────────────────────────────────────────────


def prepare_svf(
    combined_dsm_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    upstream_hash: str,
    max_radius: int = 3,
    n_directions: int = 16,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Compute SVF from the combined DSM.

    Parameters
    ----------
    combined_dsm_uri :
        URI of the finalized combined_dsm COG.
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.
    """
    import rasterio

    grid = grid or canon_grid_10m()
    c_hash = config_hash_for_svf(max_radius, n_directions, upstream_hash)

    # Read the combined DSM
    with rasterio.open(combined_dsm_uri) as src:
        dsm_data = src.read(1).astype(np.float32)

    # Build xr.DataArray for xarray-spatial
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    dsm_da = xr.DataArray(
        dsm_data,
        dims=["y", "x"],
        coords={"x": xs, "y": ys},
        attrs={"res": (10.0, 10.0)},
    )

    # Compute SVF
    log_event(
        _logger, logging.INFO, "svf_computing",
        max_radius=max_radius, n_directions=n_directions,
    )
    svf_data = sky_view_factor(
        dsm_da,
        max_radius=max_radius,
        n_directions=n_directions,
        cellsize_x=10.0,
        cellsize_y=10.0,
        name="svf",
    )

    svf_arr = svf_data.values.astype(np.float32)

    # Build canonical xr.Dataset
    ds = xr.Dataset(
        {"svf": (("y", "x"), svf_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    valid = svf_arr[~np.isnan(svf_arr)]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source="svf",
        item_key=item_key,
        category="morphology",
        dataset=ds,
        contract=contract_for_svf(),
        nominal_interval=vintage_interval(int(item_key) if item_key.isdigit() else 2021),
        source_metadata={
            "combined_dsm_uri": combined_dsm_uri,
            "upstream_hash": upstream_hash,
            "max_radius": max_radius,
            "n_directions": n_directions,
            "algorithm": "Zakek 2011 via xarray-spatial",
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "valid_frac": (
                round(float(len(valid)) / svf_arr.size, 4)
                if svf_arr.size > 0 else 0.0
            ),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "mean": float(np.nanmean(valid)) if len(valid) > 0 else None,
            "shape": list(svf_arr.shape),
        },
        config_hash=c_hash,
    )


__all__ = [
    "config_hash_for_svf",
    "contract_for_svf",
    "prepare_svf",
]
