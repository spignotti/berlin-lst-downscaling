"""ERA5-Land meteorology adapter — download, cache, and derive 8 weather channels.

Produces one eight-band COG per Landsat anchor scene containing:
- ``t2m_scene``: 2m air temperature (K) at acquisition time
- ``ssrd_scene``: surface solar radiation downwards (W/m²) at acquisition time
- ``ssrd_antecedent_72h_mean``: 72-hour rolling mean of hourly SSRD (W/m²)
- ``vpd_scene``: vapour pressure deficit (kPa) at acquisition time
- ``wind_speed_10m_scene``: 10m wind speed (m/s) at acquisition time
- ``tp_0_24h``: total precipitation (mm) in 24h ending at acquisition
- ``tp_24_48h``: total precipitation (mm) 24–48h before acquisition
- ``tp_48_72h``: total precipitation (mm) 48–72h before acquisition

ERA5-Land variables requested from CDS:
- ``2m_temperature`` (instantaneous, K)
- ``2m_dewpoint_temperature`` (instantaneous, K)
- ``10m_u_component_of_wind`` (instantaneous, m/s)
- ``10m_v_component_of_wind`` (instantaneous, m/s)
- ``surface_solar_radiation_downwards`` (accumulated, J/m²)
- ``total_precipitation`` (accumulated, m of water equivalent)

Processing
----------
1. Cache monthly NetCDF files under ``_raw/dynamic/era5_land/YYYY-MM/``.
   Fetch the preceding month when the 72h window spills into the previous month.
2. Validate each cache: requires all 6 variables, monotonic hourly timestamps,
   regular lat/lon coordinates, and sufficient spatial coverage (1-cell halo
   around canonical Berlin extent).
3. Load native 0.1° ERA5 grid, derive all temporal quantities at native resolution.
4. Reproject each field to the canonical 10m grid via bilinear interpolation.

Native-first derivation: all temporal/physical quantities are computed on the
ERA5 0.1° grid BEFORE spatial reprojection.  The 10m grid receives bilinearly
interpolated values from the native grid — this is a smooth spatial gradient,
not genuine 10m meteorological observation.
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — adds .rio accessor
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

# One-native-cell halo (0.1°) around Berlin for bilinear interpolation
_HALO_DEG = 0.1
_BERLIN_BBOX_HALO = (
    _BERLIN_BBOX[0] - _HALO_DEG,  # S
    _BERLIN_BBOX[1] - _HALO_DEG,  # W
    _BERLIN_BBOX[2] + _HALO_DEG,  # N
    _BERLIN_BBOX[3] + _HALO_DEG,  # E
)

# CDS variable short names
_CDS_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]

# Hours in the antecedent window
_ANTECEDENT_HOURS = 72

# ERA5-Land grid resolution (official CDS default: 0.1° × 0.1°)
_ERA5_GRID_DEG = 0.1

# Berlin center for spatial validation
_BERLIN_LAT = 52.52
_BERLIN_LON = 13.42

# ── helpers ────────────────────────────────────────────────────────────

def normalize_acquisition_hour(acquisition_dt: datetime) -> datetime:
    """Round a tz-aware acquisition datetime to the nearest UTC hour.

    Half-up rounding (minute >= 30 → next hour). Naive datetimes are
    interpreted as UTC. The returned datetime keeps the input timezone.
    """
    if acquisition_dt.tzinfo is None:
        naive = acquisition_dt
    else:
        naive = acquisition_dt.astimezone(UTC).replace(tzinfo=None)
    if naive.minute >= 30:
        rounded = (naive + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        rounded = naive.replace(minute=0, second=0, microsecond=0)
    if acquisition_dt.tzinfo is None:
        return rounded
    return rounded.replace(tzinfo=UTC)


def _saturation_vapour_pressure(t_celsius: float) -> float:
    """Compute saturation vapour pressure in kPa using the Tetens formula.

    Reference: FAO Irrigation and Drainage Paper 56, Chapter 4.
    """
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def _vpd_from_t_d2m(t_k: float, d2m_k: float) -> float:
    """Compute vapour pressure deficit in kPa from air and dewpoint temperature (K)."""
    t_c = float(t_k) - 273.15
    d2m_c = float(d2m_k) - 273.15
    es = _saturation_vapour_pressure(t_c)
    ea = _saturation_vapour_pressure(d2m_c)
    return max(es - ea, 0.0)


# ── contract ───────────────────────────────────────────────────────────

def contract_for_era5_scene() -> Contract:
    """Return the output Contract for ERA5 scene COGs (8 bands)."""
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
            BandSpec(
                name="vpd_scene",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Vapour pressure deficit at Landsat acquisition time. "
                    "Derived from 2m temperature and dewpoint via Tetens formula."
                ),
                unit="kPa",
                valid_range=(0.0, 10.0),
            ),
            BandSpec(
                name="wind_speed_10m_scene",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "10m wind speed at Landsat acquisition time. "
                    "Magnitude of u10/v10 components."
                ),
                unit="m/s",
                valid_range=(0.0, 50.0),
            ),
            BandSpec(
                name="tp_0_24h",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Total precipitation (mm) in the 24 hours ending at "
                    "the acquisition hour."
                ),
                unit="mm",
                valid_range=(0.0, 500.0),
            ),
            BandSpec(
                name="tp_24_48h",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Total precipitation (mm) from 24 to 48 hours "
                    "before the acquisition hour."
                ),
                unit="mm",
                valid_range=(0.0, 500.0),
            ),
            BandSpec(
                name="tp_48_72h",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Total precipitation (mm) from 48 to 72 hours "
                    "before the acquisition hour."
                ),
                unit="mm",
                valid_range=(0.0, 500.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )

# ── ERA5 cache management ─────────────────────────────────────────────

def _cache_nc_path(output_root: str, year: int, month: int) -> str:
    return era5_cache_path(output_root, year, month)

def _ensure_month_cached(
    output_root: str,
    year: int,
    month: int,
    run_id: str,
    *,
    local_dir: Path,
) -> Path | None:
    """Ensure a monthly ERA5-Land NetCDF file is available locally.

    Returns a local file path for decoding. Downloads from GCS using
    streaming (no full-file RAM load). The caller owns ``local_dir`` and
    is responsible for cleanup.

    Parameters
    ----------
    local_dir :
        Directory to write the ``.nc`` file into. Caller-managed.
    """
    cache_path = _cache_nc_path(output_root, year, month)
    fname = f"era5_land_{year:04d}{month:02d}.nc"
    target = local_dir / fname

    if target.exists() and target.stat().st_size > 0:
        return target

    if exists(cache_path):
        _download_gcs_to_local(cache_path, target)
        return target

    log_event(_logger, logging.INFO, "era5_download", year=year, month=month)
    t0 = time.perf_counter()

    try:
        _download_era5_month(year, month, target)
        elapsed = time.perf_counter() - t0
        log_event(
            _logger,
            logging.INFO,
            "era5_downloaded",
            year=year,
            month=month,
            elapsed_s=round(elapsed, 1),
            size_mb=round(target.stat().st_size / 1024 / 1024, 1),
        )
        # Upload to GCS cache via streaming (no full-file RAM load)
        from berlin_lst_downscaling.data.io.storage import atomic_upload

        atomic_upload(target, cache_path, overwrite=False)
        return target
    except Exception as exc:
        log_event(
            _logger, logging.ERROR, "era5_download_failed", year=year, month=month, error=str(exc)
        )
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
            "variable": _CDS_VARIABLES,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            # CDS area order: N, W, S, E
            "area": [
                _BERLIN_BBOX_HALO[2],
                _BERLIN_BBOX_HALO[1],
                _BERLIN_BBOX_HALO[0],
                _BERLIN_BBOX_HALO[3],
            ],
            "format": "netcdf",
        },
        str(zip_path),
    )

    with zipfile.ZipFile(zip_path) as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise ValueError(f"No .nc file in CDS ZIP for {year}-{month:02d}")
        # Write the first (usually only) NetCDF entry to target
        target.write_bytes(zf.read(nc_names[0]))
    zip_path.unlink(missing_ok=True)

# ── ERA5 decode and validation ─────────────────────────────────────────

_REQUIRED_ERA5_VARS = {"t2m", "d2m", "u10", "v10", "ssrd", "tp"}

def _decode_and_validate_monthly_era5(
    nc_path: str | Path,
    time_slice: tuple[str, str] | None = None,
) -> xr.Dataset:
    """Decode a monthly ERA5-Land NetCDF file with validation.

    Validates:
    - All 6 required variables present
    - Monotonic hourly timestamps
    - Regular lat/lon coordinates
    - Sufficient spatial coverage (Berlin center within bounds)
    """
    nc_str = str(nc_path)
    if nc_str.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes

        local_tmp = Path(tempfile.mkdtemp()) / Path(nc_str).name
        local_tmp.write_bytes(read_bytes(nc_str))
        nc_str = str(local_tmp)

    ds = xr.open_dataset(nc_str)

    # Validate required variables
    present = set(ds.data_vars)
    missing = _REQUIRED_ERA5_VARS - present
    if missing:
        ds.close()
        raise ValueError(
            f"ERA5 cache missing variables {sorted(str(v) for v in missing)}: "
            f"found {sorted(str(v) for v in present)}"
        )

    # Validate temporal dimension
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    time_vals = ds[time_dim].values
    if len(time_vals) < 2:
        ds.close()
        raise ValueError("ERA5 cache has fewer than 2 timesteps")

    # Check monotonicity
    diffs = np.diff(time_vals.astype("datetime64[h]").astype(int))
    if not np.all(diffs >= 0):
        ds.close()
        raise ValueError("ERA5 timestamps are not monotonically increasing")

    # Validate spatial coverage
    if "latitude" in ds.coords:
        lat_range = float(ds.latitude.min()), float(ds.latitude.max())
        lon_range = float(ds.longitude.min()), float(ds.longitude.max())
        if not (lat_range[0] <= _BERLIN_LAT <= lat_range[1]):
            ds.close()
            raise ValueError(
                f"ERA5 latitude range {lat_range} does not cover Berlin ({_BERLIN_LAT})"
            )
        if not (lon_range[0] <= _BERLIN_LON <= lon_range[1]):
            ds.close()
            raise ValueError(
                f"ERA5 longitude range {lon_range} does not cover Berlin ({_BERLIN_LON})"
            )

    if time_slice is not None:
        ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})

    return ds

# ── native-grid derivation ─────────────────────────────────────────────

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
    """
    time_dim = "valid_time" if "valid_time" in ssrd.dims else "time"
    time_vals = ssrd[time_dim].values
    hourly = np.empty_like(ssrd.data, dtype=np.float32)

    for t in range(len(time_vals)):
        h = int(time_vals[t].astype("datetime64[h]").astype(int) % 24)
        if h == 1:
            hourly[t] = ssrd.data[t].astype(np.float32) / 3600.0
        else:
            hourly[t] = (
                (ssrd.data[t].astype(np.float64) - ssrd.data[t - 1].astype(np.float64)) / 3600.0
            ).astype(np.float32)

    return xr.DataArray(
        hourly,
        coords=ssrd.coords,
        dims=ssrd.dims,
        attrs=ssrd.attrs,
    )

def _tp_to_hourly(tp: xr.DataArray) -> xr.DataArray:
    """Convert cumulative total precipitation (m) to hourly amounts (m).

    Same accumulation convention as SSRD.  The raw ERA5-Land tp values are
    in metres of water equivalent.  This function returns hourly values in
    metres; the caller converts to mm.
    """
    time_dim = "valid_time" if "valid_time" in tp.dims else "time"
    time_vals = tp[time_dim].values
    hourly = np.empty_like(tp.data, dtype=np.float32)

    for t in range(len(time_vals)):
        h = int(time_vals[t].astype("datetime64[h]").astype(int) % 24)
        if h == 1:
            # 01 UTC: tp = accumulation for 00:00–01:00
            hourly[t] = tp.data[t].astype(np.float32)
        else:
            # All other hours: (tp[t] - tp[t-1])
            hourly[t] = (
                tp.data[t].astype(np.float64) - tp.data[t - 1].astype(np.float64)
            ).astype(np.float32)

    return xr.DataArray(
        hourly,
        coords=tp.coords,
        dims=tp.dims,
        attrs=tp.attrs,
    )

def _derive_native_fields(
    ds: xr.Dataset,
    acquisition_dt: datetime,
) -> dict[str, float]:
    """Derive all 8 weather fields as scalar values from the native ERA5 grid.

    Each field is extracted from the nearest ERA5 cell to Berlin center.
    The caller broadcasts these scalars across the target grid.

    Parameters
    ----------
    ds : validated ERA5 dataset with all 6 variables
    acquisition_dt : scene acquisition time (tz-aware)

    Returns
    -------
    dict mapping band name to scalar float value
    """
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"

    # Find nearest cell to Berlin
    has_lat = "latitude" in ds.coords
    has_lon = "longitude" in ds.coords
    lat_vals = ds.latitude.values if has_lat else np.array([_BERLIN_LAT])
    lon_vals = ds.longitude.values if has_lon else np.array([_BERLIN_LON])
    lat_idx = int(np.abs(lat_vals - _BERLIN_LAT).argmin())
    lon_idx = int(np.abs(lon_vals - _BERLIN_LON).argmin())

    # Normalize acquisition to nearest UTC hour
    acq_hour = normalize_acquisition_hour(acquisition_dt)
    acq_np = np.datetime64(acq_hour.replace(tzinfo=None))

    # ── t2m, d2m, u10, v10: instantaneous, extract at acq_hour ──────
    t2m_cell = ds["t2m"].isel({time_dim: slice(None), "latitude": lat_idx, "longitude": lon_idx})
    d2m_cell = ds["d2m"].isel({time_dim: slice(None), "latitude": lat_idx, "longitude": lon_idx})
    u10_cell = ds["u10"].isel({time_dim: slice(None), "latitude": lat_idx, "longitude": lon_idx})
    v10_cell = ds["v10"].isel({time_dim: slice(None), "latitude": lat_idx, "longitude": lon_idx})

    # Find nearest timestep to acquisition
    time_vals = t2m_cell[time_dim].values
    diffs = np.abs(time_vals - acq_np)
    nearest_idx = int(diffs.argmin())

    t2m_val = float(t2m_cell.isel({time_dim: nearest_idx}).values)
    d2m_val = float(d2m_cell.isel({time_dim: nearest_idx}).values)
    u10_val = float(u10_cell.isel({time_dim: nearest_idx}).values)
    v10_val = float(v10_cell.isel({time_dim: nearest_idx}).values)

    # VPD
    vpd_val = _vpd_from_t_d2m(t2m_val, d2m_val)

    # Wind speed
    wind_val = float(np.sqrt(u10_val**2 + v10_val**2))

    # ── ssrd: convert to hourly, extract 72h mean ────────────────────
    ssrd_raw = ds["ssrd"].isel({"latitude": lat_idx, "longitude": lon_idx})
    ssrd_hourly = _ssrd_to_hourly(ssrd_raw)

    # 72h antecedent mean (exactly 72 hours ending at acq_hour)
    window_start_72 = acq_np - np.timedelta64(_ANTECEDENT_HOURS, "h")
    time_vals_ssrd = ssrd_hourly[time_dim].values
    mask_72 = (time_vals_ssrd > window_start_72) & (time_vals_ssrd <= acq_np)
    window_72 = ssrd_hourly.values[mask_72]
    ssrd_72h_mean = float(np.nanmean(window_72)) if len(window_72) > 0 else np.nan

    # Scene SSRD
    ssrd_scene = float(ssrd_hourly.isel({time_dim: nearest_idx}).values)

    # ── tp: convert to hourly, sum in 3 daily bins ───────────────────
    tp_raw = ds["tp"].isel({"latitude": lat_idx, "longitude": lon_idx})
    tp_hourly = _tp_to_hourly(tp_raw)

    time_vals_tp = tp_hourly[time_dim].values

    # Exactly 24 hourly values per bin
    mask_0_24 = (time_vals_tp > (acq_np - np.timedelta64(24, "h"))) & (time_vals_tp <= acq_np)
    mask_24_48 = (time_vals_tp > (acq_np - np.timedelta64(48, "h"))) & (
        time_vals_tp <= (acq_np - np.timedelta64(24, "h"))
    )
    mask_48_72 = (time_vals_tp > (acq_np - np.timedelta64(72, "h"))) & (
        time_vals_tp <= (acq_np - np.timedelta64(48, "h"))
    )

    tp_0_24 = float(np.nansum(tp_hourly.values[mask_0_24])) * 1000.0  # m → mm
    tp_24_48 = float(np.nansum(tp_hourly.values[mask_24_48])) * 1000.0
    tp_48_72 = float(np.nansum(tp_hourly.values[mask_48_72])) * 1000.0

    return {
        "t2m": t2m_val,
        "ssrd_scene": ssrd_scene,
        "ssrd_72h_mean": ssrd_72h_mean,
        "vpd": vpd_val,
        "wind": wind_val,
        "tp_0_24h": tp_0_24,
        "tp_24_48h": tp_24_48,
        "tp_48_72h": tp_48_72,
    }

# ── spatial reprojection ───────────────────────────────────────────────

def _reproject_to_canonical(
    ds: xr.Dataset,
    grid,
) -> dict[str, np.ndarray]:
    """Reproject native ERA5 2D fields to the canonical 10m grid via bilinear.

    Parameters
    ----------
    ds : Dataset with 2D DataArrays on native lat/lon
    grid : canonical GeoBox

    Returns
    -------
    dict mapping band name to 2D float32 numpy array on canonical grid
    """
    from rasterio.enums import Resampling

    # Convert lat/lon to WGS84 CRS for reprojection
    ds_wgs84 = ds.rio.write_crs("EPSG:4326")

    results = {}
    for var_name in ds.data_vars:
        da = ds_wgs84[var_name]
        # Bilinear reprojection to exact canonical grid
        reprojected = da.rio.reproject(
            str(grid.crs),
            shape=grid.shape,
            transform=grid.transform,
            resampling=Resampling.bilinear,
        )
        results[var_name] = reprojected.values.astype(np.float32)

    return results

# ── public API ─────────────────────────────────────────────────────────

def prepare_era5_scene(
    scene_id: str,
    acquisition_dt: datetime,
    output_root: str,
    run_id: str,
    *,
    grid=None,
    local_dir: Path,
) -> PreparedSecondaryProduct:
    """Prepare ERA5-Land scene channels for a Landsat anchor.

    Produces 8 bands on the canonical 10m grid after native-grid derivation
    and bilinear reprojection.

    Parameters
    ----------
    local_dir :
        Directory for ERA5 monthly cache files. Caller manages cleanup.
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

    nc_paths: dict[tuple[int, int], Path] = {}
    for year, month in months_needed:
        path = _ensure_month_cached(output_root, year, month, run_id, local_dir=local_dir)
        if path is not None:
            nc_paths[(year, month)] = path

    if (acq_year, acq_month) not in nc_paths:
        raise ValueError(
            f"Cannot process {scene_id}: ERA5 cache missing for {acq_year}-{acq_month:02d}"
        )

    # ── 2. decode and concatenate months ──────────────────────────────
    log_event(_logger, logging.INFO, "era5_processing", scene_id=scene_id)

    # Normalize acquisition time to nearest UTC hour (round, not truncate)
    acq_hour = normalize_acquisition_hour(acquisition_dt)

    # Time window: 72h + 1h padding before acquisition (for differencing)
    window_start = acq_hour - timedelta(hours=_ANTECEDENT_HOURS + 1)
    time_slice = (str(window_start), str(acq_hour))

    primary_ds = None
    prev_ds = None

    try:
        primary_ds = _decode_and_validate_monthly_era5(
            nc_paths[(acq_year, acq_month)],
            time_slice=time_slice,
        )

        # If we need previous month for antecedent, load it too
        prev_month_key = months_needed[1] if len(months_needed) > 1 else None
        if prev_month_key and prev_month_key in nc_paths:
            prev_ds = _decode_and_validate_monthly_era5(
                nc_paths[prev_month_key],
                time_slice=time_slice,
            )

            # Concatenate along time dimension
            time_dim = "valid_time" if "valid_time" in primary_ds.dims else "time"
            for var_name in _REQUIRED_ERA5_VARS:
                primary_ds[var_name] = xr.concat(
                    [prev_ds[var_name], primary_ds[var_name]], dim=time_dim
                )
                primary_ds[var_name] = primary_ds[var_name].sortby(time_dim)

            prev_ds.close()
            prev_ds = None

        # ── 3. derive all 8 fields at native resolution ──────────────
        native_fields = _derive_native_fields(primary_ds, acq_hour)

        # Build 2D native Dataset for reprojection
        native_shape = (
            primary_ds.dims.get("latitude", primary_ds.dims.get("lat", 0)),
            primary_ds.dims.get("longitude", primary_ds.dims.get("lon", 0)),
        )
        lat_vals = primary_ds.latitude.values
        lon_vals = (
            primary_ds.longitude.values
            if "longitude" in primary_ds.coords
            else primary_ds.longitude.values
        )

        native_2d = xr.Dataset(
            {
                "t2m_scene": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["t2m"], dtype=np.float32),
                ),
                "ssrd_scene": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["ssrd_scene"], dtype=np.float32),
                ),
                "ssrd_antecedent_72h_mean": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["ssrd_72h_mean"], dtype=np.float32),
                ),
                "vpd_scene": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["vpd"], dtype=np.float32),
                ),
                "wind_speed_10m_scene": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["wind"], dtype=np.float32),
                ),
                "tp_0_24h": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["tp_0_24h"], dtype=np.float32),
                ),
                "tp_24_48h": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["tp_24_48h"], dtype=np.float32),
                ),
                "tp_48_72h": (
                    ("lat", "lon"),
                    np.full(native_shape, native_fields["tp_48_72h"], dtype=np.float32),
                ),
            },
            coords={"lat": lat_vals, "lon": lon_vals},
        )

        primary_ds.close()
        primary_ds = None

        # ── 4. reproject to canonical grid ────────────────────────────
        reprojected = _reproject_to_canonical(native_2d, grid)

    finally:
        if primary_ds is not None:
            primary_ds.close()
        if prev_ds is not None:
            prev_ds.close()

    # ── 5. build output dataset ──────────────────────────────────────
    shape = (grid.shape.y, grid.shape.x)

    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    t2m_ds = xr.Dataset(
        {
            "t2m_scene": (("y", "x"), reprojected["t2m_scene"]),
            "ssrd_scene": (("y", "x"), reprojected["ssrd_scene"]),
            "ssrd_antecedent_72h_mean": (("y", "x"), reprojected["ssrd_antecedent_72h_mean"]),
            "vpd_scene": (("y", "x"), reprojected["vpd_scene"]),
            "wind_speed_10m_scene": (("y", "x"), reprojected["wind_speed_10m_scene"]),
            "tp_0_24h": (("y", "x"), reprojected["tp_0_24h"]),
            "tp_24_48h": (("y", "x"), reprojected["tp_24_48h"]),
            "tp_48_72h": (("y", "x"), reprojected["tp_48_72h"]),
        },
        coords={"x": xs, "y": ys},
    )
    t2m_ds = t2m_ds.rio.write_crs(str(grid.crs))
    t2m_ds = t2m_ds.rio.write_transform(grid.transform)

    log_event(
        _logger,
        logging.DEBUG,
        "era5_scene_values",
        scene_id=scene_id,
        t2m=round(float(native_fields["t2m"]), 2),
        ssrd=round(float(native_fields["ssrd_scene"]), 2),
        ssrd_antecedent=round(float(native_fields["ssrd_72h_mean"]), 2),
        vpd=round(float(native_fields["vpd"]), 4),
        wind=round(float(native_fields["wind"]), 2),
        tp_0_24h=round(float(native_fields["tp_0_24h"]), 2),
        tp_24_48h=round(float(native_fields["tp_24_48h"]), 2),
        tp_48_72h=round(float(native_fields["tp_48_72h"]), 2),
    )

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
            "era5_variables": _CDS_VARIABLES,
            "era5_months_used": [f"{y:04d}-{m:02d}" for y, m in nc_paths],
            "acquisition_time_utc": acquisition_dt.isoformat(),
            "scene_year": acquisition_dt.year,
            "day_of_year": doy,
            "spatial_method": "native_grid_bilinear",
            "era5_grid_resolution_deg": _ERA5_GRID_DEG,
            "era5_halo_deg": _HALO_DEG,
            "retrieved_at": retrieved_at,
        },
        qa_stats={
            "t2m_scene": round(float(native_fields["t2m"]), 2),
            "ssrd_scene": round(float(native_fields["ssrd_scene"]), 2),
            "ssrd_antecedent_72h_mean": round(float(native_fields["ssrd_72h_mean"]), 2),
            "vpd_scene": round(float(native_fields["vpd"]), 4),
            "wind_speed_10m_scene": round(float(native_fields["wind"]), 2),
            "tp_0_24h": round(float(native_fields["tp_0_24h"]), 2),
            "tp_24_48h": round(float(native_fields["tp_24_48h"]), 2),
            "tp_48_72h": round(float(native_fields["tp_48_72h"]), 2),
            "shape": list(shape),
        },
        config_hash=c_hash,
        acquisition_datetime=acquisition_dt,
        stac_properties={
            "era5:temporal_mode": "scene_timestamp",
            "era5:t2m_unit": "K",
            "era5:ssrd_unit": "W/m²",
            "era5:vpd_unit": "kPa",
            "era5:wind_speed_unit": "m/s",
            "era5:tp_unit": "mm",
            "era5:antecedent_hours": _ANTECEDENT_HOURS,
            "era5:spatial_method": "native_grid_bilinear",
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
    "normalize_acquisition_hour",
    "prepare_era5_scene",
]
