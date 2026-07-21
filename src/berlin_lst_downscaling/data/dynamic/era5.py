"""ERA5-Land meteorology adapter — download, cache, and extract scene channels.

Produces one three-band COG per Landsat anchor scene containing:
- ``t2m_scene``: 2m air temperature (K) at acquisition time
- ``ssrd_scene``: surface solar radiation downwards (W/m²) at acquisition time
- ``ssrd_antecedent_72h_mean``: 72-hour rolling mean of SSRD (W/m²) before acquisition

ERA5-Land variables:
- ``t2m``: instantaneous 2m temperature (K) — direct read
- ``ssrd``: cumulative surface solar radiation (J/m²) — accumulates 00:00→23:59,
  resets at 01:00 next day.  Convert to hourly W/m² via ECMWF conversion rule:
  - 01 UTC: ssrd / 3600
  - Otherwise: (ssrd[t] - ssrd[t-1]) / 3600

Processing
----------
1. Cache monthly NetCDF files under ``_raw/dynamic/era5_land/YYYY-MM/``.
   Fetch the preceding month when the scene month's first 72h window
   spills into the previous month.
2. Decode with xarray (NetCDF), concatenate months.
3. For each scene: normalize acquisition to nearest UTC hour; extract t2m;
   derive hourly ssrd via ECMWF differencing; compute 72h antecedent mean.
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
from berlin_lst_downscaling.data.io.storage import exists
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

# ERA5-Land grid resolution (official CDS default: 0.1° × 0.1°)
_ERA5_GRID_DEG = 0.1

# Berlin center for nearest-cell selection
_BERLIN_LAT = 52.52
_BERLIN_LON = 13.42


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
                    "Derived via ECMWF conversion: ssrd/3600 at 01 UTC, "
                    "delta(ssrd)/3600 otherwise."
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
    return era5_cache_path(output_root, year, month, cache_version="v2")


def _ensure_month_cached(
    output_root: str,
    year: int,
    month: int,
    run_id: str,
    *,
    local_dir: Path | None = None,
) -> Path | None:
    """Ensure a monthly ERA5-Land NetCDF file is available locally.

    Returns a local file path for decoding. Downloads from GCS using
    streaming (no full-file RAM load). Files are written to ``local_dir``
    if given, otherwise to a new temp directory.

    Parameters
    ----------
    local_dir : directory to write the .nc file into. Caller is responsible
        for cleanup. If None, a new temp dir is created (legacy behaviour).
    """
    cache_path = _cache_grib_path(output_root, year, month)
    fname = f"era5_land_{year:04d}{month:02d}.nc"

    if local_dir is not None:
        target = local_dir / fname
    else:
        target = Path(tempfile.mkdtemp()) / fname

    if target.exists() and target.stat().st_size > 0:
        return target

    # If already cached on GCS, stream-download to local
    if exists(cache_path):
        _download_gcs_to_local(cache_path, target)
        return target

    # Download from CDS, then upload to GCS cache
    log_event(_logger, logging.INFO, "era5_download", year=year, month=month)
    t0 = time.perf_counter()

    try:
        _download_era5_month(year, month, target)
        elapsed = time.perf_counter() - t0
        log_event(_logger, logging.INFO, "era5_downloaded",
                  year=year, month=month, elapsed_s=round(elapsed, 1),
                  size_mb=round(target.stat().st_size / 1024 / 1024, 1))
        # Upload to GCS cache via streaming (no full-file RAM load)
        from berlin_lst_downscaling.data.io.storage import atomic_upload
        atomic_upload(target, cache_path, overwrite=False)
        return target
    except Exception as exc:
        log_event(_logger, logging.ERROR, "era5_download_failed",
                  year=year, month=month, error=str(exc))
        return None


def _download_gcs_to_local(gcs_uri: str, local_path: Path) -> None:
    """Stream-download a GCS object to a local path (no full-file RAM load)."""
    from google.cloud import storage

    bucket_name, key = gcs_uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)
    blob.download_to_filename(str(local_path))


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
            # CDS area order: N, W, S, E
            "area": [_BERLIN_BBOX[2], _BERLIN_BBOX[1], _BERLIN_BBOX[0], _BERLIN_BBOX[3]],
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


def _decode_monthly_era5(
    nc_path: str | Path,
    time_slice: tuple[str, str] | None = None,
) -> xr.Dataset:
    """Decode a monthly ERA5-Land NetCDF file.

    netCDF4 cannot read GCS URIs directly, so remote files are
    copied to a local temp path first.

    Parameters
    ----------
    nc_path : local or GCS path to .nc file
    time_slice : (start, end) ISO datetime strings, optional
        If given, only load data within this time window.
    """
    nc_str = str(nc_path)
    if nc_str.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes

        local_tmp = Path(tempfile.mkdtemp()) / Path(nc_str).name
        local_tmp.write_bytes(read_bytes(nc_str))
        nc_str = str(local_tmp)

    ds = xr.open_dataset(nc_str)

    if time_slice is not None:
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})

    return ds


def _ssrd_to_hourly(ssrd: xr.DataArray) -> xr.DataArray:
    """Convert cumulative SSRD (J/m²) to hourly irradiance (W/m²).

    ECMWF ERA5-Land convention (CDS documentation):
      - SSRD accumulates from 00 UTC to the hour ending at the forecast step.
      - At 01 UTC, ssrd = accumulation for 00:00–01:00 (1 hour).
      - At 02+ UTC, ssrd = accumulation for 00:00–HH:00.
      - At 00 UTC (next day), ssrd = full 24h accumulation of previous day.

    Conversion:
      - 01 UTC:  hourly = ssrd / 3600
      - Otherwise: hourly = (ssrd[t] - ssrd[t-1]) / 3600

    At 00 UTC this yields the 24th hour's value (previous day's last hour).
    The resulting array has shape (time, lat, lon) with a per-element linear
    loop over timesteps.  For ~744 steps × ~396 × 3214 cells the cost is
    dominated by I/O, not this loop.
    """
    time_dim = "valid_time" if "valid_time" in ssrd.dims else "time"
    time_vals = ssrd[time_dim].values
    hourly = np.empty_like(ssrd.data, dtype=np.float32)

    for t in range(len(time_vals)):
        h = int(time_vals[t].astype("datetime64[h]").astype(int) % 24)
        if h == 1:
            # 01 UTC: ssrd = accumulation for 00:00–01:00
            hourly[t] = ssrd.data[t].astype(np.float32) / 3600.0
        else:
            # All other hours: (ssrd[t] - ssrd[t-1]) / 3600
            hourly[t] = (
                (ssrd.data[t].astype(np.float64) - ssrd.data[t - 1].astype(np.float64))
                / 3600.0
            ).astype(np.float32)

    return xr.DataArray(
        hourly, coords=ssrd.coords, dims=ssrd.dims, attrs=ssrd.attrs,
    )


def _extract_era5_at_scene(
    t2m: xr.DataArray,
    ssrd_hourly: xr.DataArray,
    acquisition_dt: datetime,
    berlin_lat: float = _BERLIN_LAT,
    berlin_lon: float = _BERLIN_LON,
) -> tuple[float, float, float]:
    """Extract t2m, ssrd, and 72h-antecedent at the nearest ERA5 cell to Berlin.

    Returns
    -------
    t2m_val : float
        Temperature in K at the acquisition hour.
    ssrd_val : float
        Hourly SSRD in W/m² at the acquisition hour.
    antecedent_val : float
        72h rolling mean of hourly SSRD in W/m².
    """
    acq_np = np.datetime64(acquisition_dt.replace(tzinfo=None))

    # Find the time dimension name (ERA5 uses 'valid_time' or 'time')
    time_dim = "valid_time" if "valid_time" in t2m.dims else "time"

    # Select nearest Berlin cell before any heavy computation
    lat_vals = t2m.latitude.values if "latitude" in t2m.coords else np.array([berlin_lat])
    lon_vals = t2m.longitude.values if "longitude" in t2m.coords else np.array([berlin_lon])
    lat_idx = int(np.abs(lat_vals - berlin_lat).argmin())
    lon_idx = int(np.abs(lon_vals - berlin_lon).argmin())

    t2m_cell = t2m.isel(latitude=lat_idx, longitude=lon_idx) if "latitude" in t2m.dims else t2m
    ssrd_dims = "latitude" in ssrd_hourly.dims
    ssrd_cell = ssrd_hourly.isel(latitude=lat_idx, longitude=lon_idx) if ssrd_dims else ssrd_hourly

    # Nearest hour
    diffs = np.abs(t2m_cell[time_dim].values - acq_np)
    nearest_idx = int(diffs.argmin())

    t2m_val = float(t2m_cell.isel({time_dim: nearest_idx}).values)
    ssrd_val = float(ssrd_cell.isel({time_dim: nearest_idx}).values)
    ssrd_val = max(ssrd_val, 0.0)

    # 72h antecedent mean
    window_start = acq_np - np.timedelta64(_ANTECEDENT_HOURS, "h")
    time_vals = ssrd_cell[time_dim].values
    mask = (time_vals >= window_start) & (time_vals <= acq_np)
    window_data = ssrd_cell.values[mask]

    if window_data.size == 0:
        raise ValueError(
            f"Empty 72h antecedent window for acquisition {acq_np}: "
            f"no ERA5 timesteps in [{window_start}, {acq_np}]"
        )

    antecedent_val = float(np.nanmean(window_data))
    if not np.isfinite(antecedent_val):
        raise ValueError(
            f"Antecedent value is non-finite for acquisition {acq_np}"
        )

    return t2m_val, ssrd_val, antecedent_val


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
    local_dir: Path | None = None,
) -> PreparedSecondaryProduct:
    """Prepare ERA5-Land scene channels for a Landsat anchor.

    Parameters
    ----------
    local_dir : directory for ERA5 monthly cache files. Caller manages cleanup.
    """
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

    grib_paths: dict[tuple[int, int], Path] = {}
    for year, month in months_needed:
        path = _ensure_month_cached(output_root, year, month, run_id, local_dir=local_dir)
        if path is not None:
            grib_paths[(year, month)] = path

    if (acq_year, acq_month) not in grib_paths:
        raise ValueError(
            f"Cannot process {scene_id}: ERA5 cache missing for "
            f"{acq_year}-{acq_month:02d}"
        )

    # ── 2. decode and concatenate months ──────────────────────────────
    log_event(_logger, logging.INFO, "era5_processing", scene_id=scene_id)

    # Normalize acquisition time to nearest UTC hour (round, not truncate)
    acq_naive = acquisition_dt.replace(tzinfo=None)
    if acq_naive.minute >= 30:
        acq_hour = (acq_naive + __import__("datetime").timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0,
        )
    else:
        acq_hour = acq_naive.replace(minute=0, second=0, microsecond=0)

    # Time window: 72h + 1h padding before acquisition (for diff)
    window_start = acq_hour - __import__("datetime").timedelta(hours=_ANTECEDENT_HOURS + 1)
    time_slice = (str(window_start), str(acq_hour))

    primary_ds = _decode_monthly_era5(
        grib_paths[(acq_year, acq_month)], time_slice=time_slice,
    )

    # If we need previous month for antecedent, decode and concatenate
    prev_month_key = months_needed[1] if len(months_needed) > 1 else None
    if prev_month_key and prev_month_key in grib_paths:
        prev_ds = _decode_monthly_era5(
            grib_paths[prev_month_key], time_slice=time_slice,
        )
        # Find time dimension name for concat
        time_dim = "valid_time" if "valid_time" in primary_ds.dims else "time"
        # Concatenate along time dimension
        primary_ds = xr.concat([prev_ds, primary_ds], dim=time_dim)
        primary_ds = primary_ds.sortby(time_dim)

    # ── 2b. preflight validation ────────────────────────────────────
    t2m_var = _find_var(primary_ds, ["t2m"])
    ssrd_var = _find_var(primary_ds, ["ssrd"])

    if t2m_var is None or ssrd_var is None:
        raise ValueError(
            f"Cannot find t2m/ssrd in ERA5 NetCDF for {scene_id}: "
            f"vars={list(primary_ds.data_vars)}"
        )

    # Validate spatial coverage: coordinates must include Berlin area
    if "latitude" in primary_ds.coords:
        lat_range = float(primary_ds.latitude.min()), float(primary_ds.latitude.max())
        lon_range = float(primary_ds.longitude.min()), float(primary_ds.longitude.max())
        if not (lat_range[0] <= _BERLIN_LAT <= lat_range[1]):
            raise ValueError(
                f"ERA5 latitude range {lat_range} does not cover Berlin ({_BERLIN_LAT})"
            )
        if not (lon_range[0] <= _BERLIN_LON <= lon_range[1]):
            raise ValueError(
                f"ERA5 longitude range {lon_range} does not cover Berlin ({_BERLIN_LON})"
            )

    # Validate time coverage
    time_dim = "valid_time" if "valid_time" in primary_ds.dims else "time"
    time_vals = primary_ds[time_dim].values
    acq_np = np.datetime64(acq_hour)
    if not np.any(np.abs(time_vals - acq_np) < np.timedelta64(2, "h")):
        raise ValueError(
            f"ERA5 time range does not cover acquisition {acq_hour}"
        )

    # ── 3. convert SSRD to hourly W/m² ───────────────────────────────
    hourly_ssrd = _ssrd_to_hourly(primary_ds[ssrd_var])
    hourly_t2m = primary_ds[t2m_var]

    # ── 4. extract scene values (nearest Berlin cell only) ───────────
    t2m_val, ssrd_val, antecedent_val = _extract_era5_at_scene(
        hourly_t2m, hourly_ssrd, acq_hour,
    )

    # ── 5. fill canonical grid with scalar values ────────────────────
    shape = (grid.shape.y, grid.shape.x)
    t2m_grid = np.full(shape, t2m_val, dtype=np.float32)
    ssrd_grid = np.full(shape, ssrd_val, dtype=np.float32)
    ant_grid = np.full(shape, antecedent_val, dtype=np.float32)

    # Build xr.Dataset
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
    t2m_val_log = t2m_val
    ssrd_val_log = ssrd_val
    ant_val_log = antecedent_val

    log_event(_logger, logging.DEBUG, "era5_scene_values",
              scene_id=scene_id, t2m=round(t2m_val_log, 2),
              ssrd=round(ssrd_val_log, 2), ssrd_antecedent=round(ant_val_log, 2))

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
            "t2m_scene": round(t2m_val_log, 2),
            "ssrd_scene": round(ssrd_val_log, 2),
            "ssrd_antecedent_72h_mean": round(ant_val_log, 2),
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
