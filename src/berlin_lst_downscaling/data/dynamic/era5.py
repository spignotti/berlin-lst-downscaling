"""ERA5-Land meteorology adapter — download, cache, and extract scene channels.

Produces one three-band COG per Landsat anchor scene containing:
- ``t2m_scene``: 2m air temperature (K) at acquisition time
- ``ssrd_scene``: surface solar radiation downwards (W/m²) at acquisition time
- ``ssrd_antecedent_72h_mean``: 72-hour rolling mean of SSRD (W/m²) before acquisition

ERA5-Land variables:
- ``t2m``: instantaneous 2m temperature (K) — direct read
- ``ssrd``: cumulative surface solar radiation (J/m²) — accumulates 00:00→23:59,
  resets daily at 00:00.  Convert to hourly W/m² via first-difference / 3600.

Processing
----------
1. Cache monthly GRIB files under ``_raw/dynamic/era5_land/YYYY-MM/``.
   Fetch the preceding month too when the scene month's first 72h window
   spills into the previous month.
2. Decode with ``cfgrib`` engine.
3. For each scene: extract t2m at the acquisition hour; derive hourly ssrd
   at the acquisition hour; compute 72-hour antecedent mean of hourly ssrd.
4. Expand the nearest ERA5 grid cell to the canonical 10m grid via
   nearest-neighbour fill (constant per ERA5 cell ≈ 9 km).
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import xarray as xr

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.dynamic.paths import era5_cache_path
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.io.storage import atomic_write, exists
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

# Berlin AOI bbox (WGS84): S, W, N, E
_BERLIN_BBOX = (52.34, 13.08, 52.68, 13.76)

# CDS variable short names
_ERA5_VARIABLES = ["2m_temperature", "surface_solar_radiation_downwards"]

# Hours in an antecedent window
_ANTECEDENT_HOURS = 72

# ERA5-Land accumulation: ssrd resets daily at 00:00 UTC
# ssrd at HH:00 = sum from 00:00 to HH:00 on same day
# ssrd at 00:00 = full previous day's 24h sum


# ── contract ───────────────────────────────────────────────────────────


def contract_for_era5_scene() -> Contract:
    """Return the output Contract for ERA5 scene COGs (3 bands)."""
    return Contract(
        source="era5_land",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="t2m_scene",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "ERA5-Land 2m air temperature at Landsat acquisition time. "
                    "Instantaneous value, nearest-hourly."
                ),
                unit="K",
                valid_range=(200.0, 350.0),
            ),
            BandSpec(
                name="ssrd_scene",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "ERA5-Land surface solar radiation downwards at acquisition hour. "
                    "Derived from daily accumulation: delta(ssrd) / 3600."
                ),
                unit="W/m²",
                valid_range=(-1.0, 1500.0),
            ),
            BandSpec(
                name="ssrd_antecedent_72h_mean",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "72-hour rolling mean of hourly SSRD (W/m²) ending at "
                    "the acquisition hour. Antecedent solar context."
                ),
                unit="W/m²",
                valid_range=(-1.0, 1500.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


# ── ERA5 cache management ─────────────────────────────────────────────


def _cache_grib_path(output_root: str, year: int, month: int) -> str:
    """Return local cache path for ERA5 GRIB file."""
    return era5_cache_path(output_root, year, month)


def _ensure_month_cached(
    output_root: str,
    year: int,
    month: int,
    run_id: str,
) -> str | None:
    """Ensure a monthly ERA5-Land GRIB file is cached locally.

    Returns the local path on success, None on failure.
    """
    cache_path = _cache_grib_path(output_root, year, month)

    if exists(cache_path):
        log_event(_logger, logging.DEBUG, "era5_cache_hit", year=year, month=month)
        return cache_path

    # Download to local temp then atomic-write to cache
    local_tmp = Path(tempfile.mkdtemp()) / f"era5_land_{year:04d}{month:02d}.grib"

    log_event(_logger, logging.INFO, "era5_download", year=year, month=month)
    t0 = time.perf_counter()

    try:
        _download_era5_month(year, month, local_tmp)
        elapsed = time.perf_counter() - t0
        log_event(
            _logger,
            logging.INFO,
            "era5_downloaded",
            year=year,
            month=month,
            elapsed_s=round(elapsed, 1),
            size_mb=round(local_tmp.stat().st_size / 1024 / 1024, 1),
        )

        # Atomic write to cache location
        atomic_write(cache_path, local_tmp.read_bytes(), overwrite=False)
        return cache_path
    except Exception as exc:
        log_event(
            _logger,
            logging.ERROR,
            "era5_download_failed",
            year=year,
            month=month,
            error=str(exc),
        )
        return None


def _download_era5_month(year: int, month: int, target: Path) -> None:
    """Retrieve a single month of ERA5-Land for Berlin AOI via CDS API."""
    import cdsapi

    client = cdsapi.Client()

    # Number of days in this month
    import calendar

    n_days = calendar.monthrange(year, month)[1]

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": _ERA5_VARIABLES,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [_BERLIN_BBOX[2], _BERLIN_BBOX[0], _BERLIN_BBOX[1], _BERLIN_BBOX[3]],
            # N, W, S, E (CDS order)
            "format": "grib",
        },
        str(target),
    )


# ── ERA5 decode and processing ────────────────────────────────────────


def _decode_monthly_grib(grib_path: str) -> xr.Dataset:
    """Decode a monthly ERA5 GRIB file with cfgrib.

    Returns an xarray Dataset with variables ``t2m`` (K) and ``ssrd`` (J/m²).
    """
    return xr.open_dataset(grib_path, engine="cfgrib")


def _ssrd_to_hourly(ssrd: xr.DataArray) -> xr.DataArray:
    """Convert cumulative SSRD (J/m²) to hourly irradiance (W/m²).

    SSRD accumulates within each day from 00:00:
    - At HH:00, ssrd = sum of hourly values from 00:00 to HH:00
    - At 00:00 of next day, ssrd = full previous day's 24h total

    Strategy: group by date, compute diff within each day, divide by 3600.
    """
    # Convert to pandas for easy grouping
    time_coords = ssrd.time.values

    # Compute hourly increments by day
    hourly = ssrd.copy()

    for date in np.unique(time_coords.astype("datetime64[D]")):
        day_mask = time_coords.astype("datetime64[D]") == date
        day_data = ssrd.isel(time=day_mask)

        if len(day_data) < 2:
            continue

        # First hour of day: value is previous day's total → set to 0
        # (the 24h accumulated total is not an "hourly" value)
        # Actually, at 00:00, ssrd is the previous day's 24h accumulation.
        # For the current day's first hour, the accumulation starts fresh.
        # So: diff at hour 0 = ssrd[0] - 0 = ssrd[0] (but this is prev day total)
        # We need to handle this carefully.

        # Better approach: within each day, the increment is diff of cumulative.
        # The first entry of each day (00:00) = previous day's 24h sum.
        # So we set the first entry to 0 (or NaN) and diff from there.

        indices = np.where(day_mask)[0]
        if len(indices) < 2:
            continue

        # Set the 00:00 value to 0 (it's the previous day's accumulated total)
        hourly.data[indices[0]] = 0.0

        # Now diff gives the hourly increment for hours 1..23
        for j in range(1, len(indices)):
            hourly.data[indices[j]] = float(ssrd.data[indices[j]]) - float(
                ssrd.data[indices[j - 1]]
            )

    # Convert J/m² to W/m²
    return hourly / 3600.0


def _extract_scene_values(
    hourly_t2m: xr.DataArray,
    hourly_ssrd: xr.DataArray,
    acquisition_dt: datetime,
) -> dict[str, float]:
    """Extract ERA5 values at the scene acquisition time.

    Uses nearest-hourly match.  Returns dict with t2m_scene and ssrd_scene.
    """
    # Find nearest hour
    time_da = hourly_t2m.time
    acq_np = np.datetime64(acquisition_dt.replace(tzinfo=None))
    diffs = np.abs(time_da.values - acq_np)
    nearest_idx = int(diffs.argmin())

    t2m_val = float(hourly_t2m.isel(time=nearest_idx).mean().values)
    ssrd_val = float(hourly_ssrd.isel(time=nearest_idx).mean().values)

    return {"t2m_scene": t2m_val, "ssrd_scene": max(ssrd_val, 0.0)}


def _compute_antecedent_mean(
    hourly_ssrd: xr.DataArray,
    acquisition_dt: datetime,
    hours: int = _ANTECEDENT_HOURS,
) -> float:
    """Compute the rolling mean of hourly SSRD over the preceding N hours."""
    acq_np = np.datetime64(acquisition_dt.replace(tzinfo=None))
    time_vals = hourly_ssrd.time.values

    # Select all hours in [acquisition - hours, acquisition]
    window_start = acq_np - np.timedelta64(hours, "h")
    mask = (time_vals >= window_start) & (time_vals <= acq_np)

    if not np.any(mask):
        return 0.0

    values = hourly_ssrd.values[mask]
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return 0.0

    return float(np.mean(valid))


def _expand_to_canonical_grid(
    scalar_value: float,
    grid,
) -> xr.Dataset:
    """Expand a single ERA5 grid-cell value to the canonical 10m grid.

    Nearest-neighbour fill: every pixel in the output gets the same value
    (constant per ERA5 cell ≈ 9 km).
    """
    shape = (grid.shape.y, grid.shape.x)
    arr = np.full(shape, scalar_value, dtype=np.float32)

    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    ds = xr.Dataset(
        {
            "t2m_scene": (("y", "x"), arr),
            "ssrd_scene": (("y", "x"), arr),
            "ssrd_antecedent_72h_mean": (("y", "x"), arr),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)
    return ds


# ── public API ─────────────────────────────────────────────────────────


def prepare_era5_scene(
    scene_id: str,
    acquisition_dt: datetime,
    output_root: str,
    run_id: str,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Prepare ERA5-Land scene channels for a Landsat anchor.

    Downloads and caches the relevant monthly GRIB files, decodes,
    computes the three channels, and returns a canonical-grid dataset.

    Parameters
    ----------
    scene_id :
        Landsat scene ID (used as item_key).
    acquisition_dt :
        Scene acquisition datetime (UTC).
    output_root :
        Root URI for ERA5 cache and final products.
    run_id :
        Current run identifier.
    grid :
        Output GeoBox.  Defaults to full canonical 10m grid.
    """
    grid = grid or canon_grid_10m()
    c_hash = sha256(f"era5_land:{scene_id}".encode()).hexdigest()[:12]

    # ── 1. ensure relevant months are cached ──────────────────────────
    acq_year = acquisition_dt.year
    acq_month = acquisition_dt.month

    # Determine which months we need (might need preceding month for antecedent)
    months_needed = [(acq_year, acq_month)]
    if acq_month == 1:
        months_needed.append((acq_year - 1, 12))
    else:
        months_needed.append((acq_year, acq_month - 1))

    grib_paths: dict[tuple[int, int], str] = {}
    for year, month in months_needed:
        path = _ensure_month_cached(output_root, year, month, run_id)
        if path is not None:
            grib_paths[(year, month)] = path

    if (acq_year, acq_month) not in grib_paths:
        raise ValueError(
            f"Cannot process {scene_id}: ERA5 cache missing for {acq_year}-{acq_month:02d}"
        )

    # ── 2. decode and process ────────────────────────────────────────
    log_event(_logger, logging.INFO, "era5_processing", scene_id=scene_id)

    # Open the acquisition month
    primary_ds = _decode_monthly_grib(grib_paths[(acq_year, acq_month)])

    # Find variables
    t2m_var = _find_var(primary_ds, ["t2m"])
    ssrd_var = _find_var(primary_ds, ["ssrd", "ssrd"])

    if t2m_var is None or ssrd_var is None:
        raise ValueError(
            f"Cannot find t2m/ssrd in ERA5 GRIB for {scene_id}: vars={list(primary_ds.data_vars)}"
        )

    # Convert SSRD to hourly W/m²
    hourly_ssrd = _ssrd_to_hourly(primary_ds[ssrd_var])
    hourly_t2m = primary_ds[t2m_var]

    # ── 3. extract scene values ──────────────────────────────────────
    scene_vals = _extract_scene_values(hourly_t2m, hourly_ssrd, acquisition_dt)
    antecedent = _compute_antecedent_mean(hourly_ssrd, acquisition_dt)

    log_event(
        _logger,
        logging.DEBUG,
        "era5_scene_values",
        scene_id=scene_id,
        t2m=round(scene_vals["t2m_scene"], 2),
        ssrd=round(scene_vals["ssrd_scene"], 2),
        ssrd_antecedent=round(antecedent, 2),
    )

    # ── 4. expand to canonical grid ──────────────────────────────────
    # For now: nearest-neighbour fill (constant per ERA5 cell)
    # Future: could interpolate if ERA5 grid is finer than canonical grid
    shape = (grid.shape.y, grid.shape.x)
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    t2m_ds = xr.Dataset(
        {
            "t2m_scene": (("y", "x"), np.full(shape, scene_vals["t2m_scene"], dtype=np.float32)),
            "ssrd_scene": (("y", "x"), np.full(shape, scene_vals["ssrd_scene"], dtype=np.float32)),
            "ssrd_antecedent_72h_mean": (("y", "x"), np.full(shape, antecedent, dtype=np.float32)),
        },
        coords={"x": xs, "y": ys},
    )
    t2m_ds = t2m_ds.rio.write_crs(str(grid.crs))
    t2m_ds = t2m_ds.rio.write_transform(grid.transform)

    primary_ds.close()

    retrieved_at = datetime.now(UTC).isoformat()

    # ── 5. build prepared product ────────────────────────────────────
    return PreparedSecondaryProduct(
        source="era5_land",
        item_key=scene_id,
        category="dynamic",
        dataset=t2m_ds,
        contract=contract_for_era5_scene(),
        nominal_interval=vintage_interval(acquisition_dt.year),
        source_metadata={
            "era5_variables": _ERA5_VARIABLES,
            "era5_months_used": [f"{y:04d}-{m:02d}" for y, m in grib_paths],
            "acquisition_time_utc": acquisition_dt.isoformat(),
            "grid_expansion": "nearest_neighbour",
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "t2m_scene": round(scene_vals["t2m_scene"], 2),
            "ssrd_scene": round(scene_vals["ssrd_scene"], 2),
            "ssrd_antecedent_72h_mean": round(antecedent, 2),
            "shape": [grid.shape.y, grid.shape.x],
        },
        config_hash=c_hash,
        acquisition_datetime=acquisition_dt,
        stac_properties={
            "era5:temporal_mode": "scene_timestamp",
            "era5:t2m_unit": "K",
            "era5:ssrd_unit": "W/m²",
            "era5:antecedent_hours": _ANTECEDENT_HOURS,
        },
    )


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    """Find a variable in the dataset by trying candidate names."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


__all__ = [
    "contract_for_era5_scene",
    "prepare_era5_scene",
]
