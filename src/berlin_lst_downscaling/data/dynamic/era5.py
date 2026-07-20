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
1. Cache monthly NetCDF files under ``_raw/dynamic/era5_land/YYYY-MM/``.
   Fetch the preceding month when the scene month's first 72h window
   spills into the previous month.
2. Decode with xarray (NetCDF), concatenate months.
3. For each scene: extract t2m at the acquisition hour; derive hourly ssrd
   at the acquisition hour; compute 72-hour antecedent mean of hourly ssrd.
4. Expand each ERA5 grid cell to the canonical 10m grid via nearest-neighbour.
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

# ERA5-Land grid resolution
_ERA5_GRID_DEG = 0.0625  # ~6.9 km at 52°N


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
    return era5_cache_path(output_root, year, month)


def _ensure_month_cached(
    output_root: str, year: int, month: int, run_id: str,
) -> str | None:
    """Ensure a monthly ERA5-Land GRIB file is cached locally."""
    cache_path = _cache_grib_path(output_root, year, month)
    if exists(cache_path):
        return cache_path

    local_tmp = Path(tempfile.mkdtemp()) / f"era5_land_{year:04d}{month:02d}.nc"
    log_event(_logger, logging.INFO, "era5_download", year=year, month=month)
    t0 = time.perf_counter()

    try:
        _download_era5_month(year, month, local_tmp)
        elapsed = time.perf_counter() - t0
        log_event(_logger, logging.INFO, "era5_downloaded",
                  year=year, month=month, elapsed_s=round(elapsed, 1),
                  size_mb=round(local_tmp.stat().st_size / 1024 / 1024, 1))
        atomic_write(cache_path, local_tmp.read_bytes(), overwrite=False)
        return cache_path
    except Exception as exc:
        log_event(_logger, logging.ERROR, "era5_download_failed",
                  year=year, month=month, error=str(exc))
        return None


def _download_era5_month(year: int, month: int, target: Path) -> None:
    """Retrieve a single month of ERA5-Land for Berlin AOI via CDS API.

    CDS returns NetCDF files wrapped in a ZIP archive. This function
    downloads the ZIP, extracts the NetCDF, and writes it to ``target``.
    """
    import calendar
    import zipfile

    import cdsapi

    client = cdsapi.Client()
    n_days = calendar.monthrange(year, month)[1]

    # retrieve() downloads the ZIP to a temp path; we get the path back
    zip_path = Path(target).with_suffix(".zip")
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": _ERA5_VARIABLES,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            # CDS order: N, W, S, E
            "area": [_BERLIN_BBOX[2], _BERLIN_BBOX[0], _BERLIN_BBOX[1], _BERLIN_BBOX[3]],
            "format": "netcdf",
        },
        str(zip_path),
    )

    # Extract NetCDF from ZIP
    with zipfile.ZipFile(zip_path) as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise ValueError(f"No .nc file in CDS ZIP for {year}-{month:02d}")
        # Write the first (usually only) NetCDF entry to target
        target.write_bytes(zf.read(nc_names[0]))
    zip_path.unlink(missing_ok=True)


# ── ERA5 decode and processing ────────────────────────────────────────


def _decode_monthly_grib(
    grib_path: str,
    time_slice: tuple[str, str] | None = None,
) -> xr.Dataset:
    """Decode a monthly ERA5 file (NetCDF or GRIB).

    netCDF4 cannot read GCS URIs directly, so remote files are
    copied to a local temp path first.

    Parameters
    ----------
    time_slice : (start, end) ISO datetime strings, optional
        If given, only load data within this time window to reduce memory.
    """
    if grib_path.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes

        local_tmp = Path(tempfile.mkdtemp()) / Path(grib_path).name
        local_tmp.write_bytes(read_bytes(grib_path))
        grib_path = str(local_tmp)

    ds = xr.open_dataset(grib_path)

    if time_slice is not None:
        # Find the time coordinate name (ERA5 uses 'valid_time' or 'time')
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})

    return ds


def _ssrd_to_hourly(ssrd: xr.DataArray) -> xr.DataArray:
    """Convert cumulative SSRD (J/m²) to hourly irradiance (W/m²).

    SSRD accumulates within each day from 00:00:
    - At HH:00, ssrd = sum of hourly values from 00:00 to HH:00
    - At 00:00 of next day, ssrd = full previous day's 24h sum

    Strategy: group by date, first-difference within each day, divide by 3600.
    Handles multi-dimensional arrays (time × lat × lon).
    """
    time_vals = ssrd.time.values
    hourly = ssrd.copy()

    for date in np.unique(time_vals.astype("datetime64[D]")):
        day_mask = time_vals.astype("datetime64[D]") == date
        indices = np.where(day_mask)[0]

        if len(indices) < 2:
            continue

        # Set 00:00 to 0 (it's previous day's accumulated total)
        hourly.data[indices[0]] = 0.0

        # Vectorised diff over all spatial dims simultaneously
        for j in range(1, len(indices)):
            hourly.data[indices[j]] = (
                ssrd.data[indices[j]].astype(np.float64)
                - ssrd.data[indices[j - 1]].astype(np.float64)
            )

    return (hourly / 3600.0).astype(np.float32)


def _extract_era5_at_scene(
    t2m: xr.DataArray,
    ssrd_hourly: xr.DataArray,
    acquisition_dt: datetime,
    berlin_lat: float = 52.52,
    berlin_lon: float = 13.42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract t2m, ssrd, and 72h-antecedent for all ERA5 cells over Berlin.

    Returns
    -------
    t2m_vals : np.ndarray shape (n_lat, n_lon)
        Temperature in K at the acquisition hour.
    ssrd_vals : np.ndarray shape (n_lat, n_lon)
        Hourly SSRD in W/m² at the acquisition hour.
    antecedent_vals : np.ndarray shape (n_lat, n_lon)
        72h rolling mean of hourly SSRD in W/m².
    """
    acq_np = np.datetime64(acquisition_dt.replace(tzinfo=None))

    # Nearest hour
    diffs = np.abs(t2m.time.values - acq_np)
    nearest_idx = int(diffs.argmin())

    t2m_vals = t2m.isel(time=nearest_idx).values.astype(np.float32)
    ssrd_vals = ssrd_hourly.isel(time=nearest_idx).values.astype(np.float32)
    ssrd_vals = np.clip(ssrd_vals, 0.0, None)

    # 72h antecedent mean
    window_start = acq_np - np.timedelta64(_ANTECEDENT_HOURS, "h")
    time_vals = ssrd_hourly.time.values
    mask = (time_vals >= window_start) & (time_vals <= acq_np)
    window_data = ssrd_hourly.values[mask]  # shape: (n_hours, n_lat, n_lon)
    with np.errstate(invalid="ignore"):
        antecedent_vals = np.nanmean(window_data, axis=0).astype(np.float32)

    return t2m_vals, ssrd_vals, antecedent_vals


def _expand_to_canonical_grid(
    era5_2d: np.ndarray,
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
    grid,
    berlin_lat: float = 52.52,
    berlin_lon: float = 13.42,
) -> np.ndarray:
    """Expand an ERA5 2D field (lat × lon) to the canonical 10m grid.

    Strategy: find the nearest ERA5 grid cell to Berlin center,
    fill entire canonical grid with that value.

    Returns a 2D float32 array of shape (grid_y, grid_x).
    """
    h, w = grid.shape.y, grid.shape.x

    # Find nearest ERA5 cell to Berlin center
    lat_idx = int(np.abs(era5_lat - berlin_lat).argmin())
    lon_idx = int(np.abs(era5_lon - berlin_lon).argmin())

    val = float(era5_2d[lat_idx, lon_idx])
    return np.full((h, w), val, dtype=np.float32)


# ── public API ─────────────────────────────────────────────────────────


def prepare_era5_scene(
    scene_id: str,
    acquisition_dt: datetime,
    output_root: str,
    run_id: str,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Prepare ERA5-Land scene channels for a Landsat anchor."""
    grid = grid or canon_grid_10m()
    c_hash = sha256(f"era5_land:{scene_id}".encode()).hexdigest()[:12]

    # ── 1. ensure relevant months are cached ──────────────────────────
    acq_year = acquisition_dt.year
    acq_month = acquisition_dt.month

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
            f"Cannot process {scene_id}: ERA5 cache missing for "
            f"{acq_year}-{acq_month:02d}"
        )

    # ── 2. decode and concatenate months ──────────────────────────────
    log_event(_logger, logging.INFO, "era5_processing", scene_id=scene_id)

    # Compute time window: 72h before acquisition to acquisition time
    window_start = acquisition_dt - __import__("datetime").timedelta(hours=_ANTECEDENT_HOURS)
    time_slice = (str(window_start), str(acquisition_dt))

    primary_ds = _decode_monthly_grib(
        grib_paths[(acq_year, acq_month)], time_slice=time_slice,
    )

    # If we need previous month for antecedent, decode and concatenate
    prev_month_key = months_needed[1] if len(months_needed) > 1 else None
    if prev_month_key and prev_month_key in grib_paths:
        prev_ds = _decode_monthly_grib(
            grib_paths[prev_month_key], time_slice=time_slice,
        )
        # Concatenate along time dimension
        primary_ds = xr.concat([prev_ds, primary_ds], dim="time")
        primary_ds = primary_ds.sortby("time")

    # Find variables
    t2m_var = _find_var(primary_ds, ["t2m"])
    ssrd_var = _find_var(primary_ds, ["ssrd"])

    if t2m_var is None or ssrd_var is None:
        raise ValueError(
            f"Cannot find t2m/ssrd in ERA5 GRIB for {scene_id}: "
            f"vars={list(primary_ds.data_vars)}"
        )

    # ── 3. convert SSRD to hourly W/m² ───────────────────────────────
    hourly_ssrd = _ssrd_to_hourly(primary_ds[ssrd_var])
    hourly_t2m = primary_ds[t2m_var]

    # ── 4. extract scene values (per-cell) ────────────────────────────
    t2m_2d, ssrd_2d, antecedent_2d = _extract_era5_at_scene(
        hourly_t2m, hourly_ssrd, acquisition_dt,
    )

    # ── 5. expand to canonical grid (nearest ERA5 cell) ───────────────
    lat_vals = primary_ds.latitude.values if "latitude" in primary_ds.coords else np.array([52.5])
    lon_vals = primary_ds.longitude.values if "longitude" in primary_ds.coords else np.array([13.4])

    t2m_grid = _expand_to_canonical_grid(t2m_2d, lat_vals, lon_vals, grid)
    ssrd_grid = _expand_to_canonical_grid(ssrd_2d, lat_vals, lon_vals, grid)
    ant_grid = _expand_to_canonical_grid(antecedent_2d, lat_vals, lon_vals, grid)

    # Build xr.Dataset
    shape = (grid.shape.y, grid.shape.x)
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    t2m_ds = xr.Dataset(
        {
            "t2m_scene": (("y", "x"), t2m_grid),
            "ssrd_scene": (("y", "x"), ssrd_grid),
            "ssrd_antecedent_72h_mean": (("y", "x"), ant_grid),
        },
        coords={"x": xs, "y": ys},
    )
    t2m_ds = t2m_ds.rio.write_crs(str(grid.crs))
    t2m_ds = t2m_ds.rio.write_transform(grid.transform)

    primary_ds.close()

    # Scene channel values for logging/QA
    t2m_val = float(t2m_grid[shape[0] // 2, shape[1] // 2])
    ssrd_val = float(ssrd_grid[shape[0] // 2, shape[1] // 2])
    ant_val = float(ant_grid[shape[0] // 2, shape[1] // 2])

    log_event(_logger, logging.DEBUG, "era5_scene_values",
              scene_id=scene_id, t2m=round(t2m_val, 2),
              ssrd=round(ssrd_val, 2), ssrd_antecedent=round(ant_val, 2))

    retrieved_at = datetime.now(UTC).isoformat()
    doy = acquisition_dt.timetuple().tm_yday

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
            "scene_year": acquisition_dt.year,
            "day_of_year": doy,
            "grid_expansion": "nearest_era5_cell",
            "era5_grid_resolution_deg": _ERA5_GRID_DEG,
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "t2m_scene": round(t2m_val, 2),
            "ssrd_scene": round(ssrd_val, 2),
            "ssrd_antecedent_72h_mean": round(ant_val, 2),
            "shape": list(shape),
        },
        config_hash=c_hash,
        acquisition_datetime=acquisition_dt,
        stac_properties={
            "era5:temporal_mode": "scene_timestamp",
            "era5:t2m_unit": "K",
            "era5:ssrd_unit": "W/m²",
            "era5:antecedent_hours": _ANTECEDENT_HOURS,
            "acquisition:datetime": acquisition_dt.isoformat(),
            "acquisition:doy": doy,
            "acquisition:year": acquisition_dt.year,
        },
    )


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


__all__ = [
    "contract_for_era5_scene",
    "prepare_era5_scene",
]
