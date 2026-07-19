"""DWD station adapter — download and compare temperature observations.

Downloads official DWD historical hourly temperature ZIPs for Berlin stations
using stdlib HTTP + ZIP + CSV.  Station data is used ONLY for validation/QA,
never as a model channel.

Berlin stations (verified active 2017–2025):
- 00427 (Berlin-Tegel): 1973–present
- 00403 (Berlin-Dahlem): 2002–present
- 00433 (Berlin-Tempelhof): 1951–present (SD ends 2022)
- 00400 (Berlin-Alexanderplatz): 1991–present

Data source: https://opendata.dwd.de/climate_environment/CDC/
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

# ── Berlin stations ────────────────────────────────────────────────────

柏林_STATIONS = {
    "00427": "Berlin-Tegel",
    "00403": "Berlin-Dahlem",
    "00433": "Berlin-Tempelhof",
    "00400": "Berlin-Alexanderplatz",
}

# DWD base URLs for hourly temperature
_HIST_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/air_temperature/historical/"
)
_RECENT_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/air_temperature/recent/"
)


@dataclass
class DwdStationRecord:
    """A single hourly temperature observation from a DWD station."""

    station_id: str
    timestamp: datetime  # UTC
    temperature: float  # °C (DWD convention: TMK)


@dataclass
class DwdStationData:
    """Full hourly record for a station."""

    station_id: str
    records: list[DwdStationRecord] = field(default_factory=list)

    def lookup_hour(self, utc_dt: datetime) -> float | None:
        """Find the temperature for the nearest full UTC hour.

        Returns °C or None if no matching record.
        """
        for rec in self.records:
            if (
                rec.timestamp.year == utc_dt.year
                and rec.timestamp.month == utc_dt.month
                and rec.timestamp.day == utc_dt.day
                and rec.timestamp.hour == utc_dt.hour
            ):
                return rec.temperature
        return None


def download_dwd_station(
    station_id: str,
    year_range: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
) -> DwdStationData:
    """Download and parse DWD hourly temperature data for a station.

    Tries the historical archive first, then recent if available.
    """
    data = DwdStationData(station_id=station_id)

    # Try historical
    try:
        records = _fetch_dwd_station(station_id, _HIST_URL, year_range)
        data.records = records
        log_event(
            _logger, logging.INFO, "dwd_downloaded", station_id=station_id, n_records=len(records)
        )
        return data
    except Exception as exc:
        log_event(
            _logger, logging.WARNING, "dwd_hist_failed", station_id=station_id, error=str(exc)
        )

    # Fallback to recent
    try:
        records = _fetch_dwd_station(station_id, _RECENT_URL, year_range)
        data.records = records
        log_event(
            _logger,
            logging.INFO,
            "dwd_recent_downloaded",
            station_id=station_id,
            n_records=len(records),
        )
    except Exception as exc:
        log_event(
            _logger, logging.WARNING, "dwd_recent_failed", station_id=station_id, error=str(exc)
        )

    return data


def _fetch_dwd_station(
    station_id: str,
    base_url: str,
    year_range: tuple[int, int] | None = None,
) -> list[DwdStationRecord]:
    """Fetch DWD hourly temperature CSV from the CDC archive."""
    # Construct expected filename pattern
    # Historical: stundenwerte_TU_{station}_YYYYMMDD_YYYYMMDD_hist.zip
    # Recent: stundenwerte_TU_{station}_akt.zip
    if year_range is not None:
        start, end = year_range
        url = f"{base_url}stundenwerte_TU_{station_id}_{start:04d}0101_{end:04d}1231_hist.zip"
    else:
        url = f"{base_url}stundenwerte_TU_{station_id}_akt.zip"

    log_event(_logger, logging.DEBUG, "dwd_fetch", url=url)

    response = urlopen(url, timeout=60)  # noqa: S310 — DWD CDC is known safe
    zip_bytes = response.read()

    return _parse_dwd_zip(zip_bytes, station_id, year_range)


def _parse_dwd_zip(
    zip_bytes: bytes,
    station_id: str,
    year_range: tuple[int, int] | None = None,
) -> list[DwdStationRecord]:
    """Parse DWD hourly temperature data from a ZIP archive."""
    records: list[DwdStationRecord] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Find the data CSV file (not the station description)
        csv_names = [n for n in z.namelist() if n.endswith(".csv") and "Beschreibung" not in n]
        if not csv_names:
            raise ValueError(f"No CSV found in DWD ZIP for {station_id}")

        with z.open(csv_names[0]) as f:
            content = f.read().decode("latin-1")

        # Parse CSV (semicolon-separated)
        reader = csv.DictReader(io.StringIO(content), delimiter=";")

        for row in reader:
            try:
                # Date column: "MESSDATUM" or "date"
                date_val = row.get("MESSDATUM") or row.get("date", "")
                temp_val = row.get("TMK") or row.get("tt_tu", "")

                if not date_val or not temp_val:
                    continue

                # DWD date format: YYYYMMDDHH
                dt = datetime.strptime(date_val.strip(), "%Y%m%d%H%M")
                dt = dt.replace(tzinfo=UTC)

                temp = float(temp_val.strip())

                # Filter by year range
                if year_range is not None:
                    if dt.year < year_range[0] or dt.year > year_range[1]:
                        continue

                records.append(
                    DwdStationRecord(
                        station_id=station_id,
                        timestamp=dt,
                        temperature=temp,
                    )
                )
            except (ValueError, KeyError):
                continue

    return records


# ── DWD vs ERA5 comparison ────────────────────────────────────────────


@dataclass
class DwdComparisonResult:
    """Result of comparing DWD station temperatures with ERA5."""

    station_id: str
    station_name: str
    n_matched: int
    bias: float  # ERA5 - DWD in °C
    mae: float
    rmse: float
    dwd_mean: float
    era5_mean: float


def compare_era5_with_dwd(
    era5_t2m_by_hour: dict[datetime, float],
    dwd_data: DwdStationData,
    station_name: str,
) -> DwdComparisonResult:
    """Compare ERA5 2m temperature with DWD station observations.

    Parameters
    ----------
    era5_t2m_by_hour :
        Dict mapping UTC datetime → ERA5 t2m in Kelvin.
    dwd_data :
        DWD station data.
    station_name :
        Human-readable station name.
    """
    diffs: list[float] = []
    dwd_vals: list[float] = []
    era5_vals: list[float] = []

    for utc_dt, era5_k in era5_t2m_by_hour.items():
        dwd_c = dwd_data.lookup_hour(utc_dt)
        if dwd_c is None:
            continue

        era5_c = era5_k - 273.15  # K → °C
        diff = era5_c - dwd_c
        diffs.append(diff)
        dwd_vals.append(dwd_c)
        era5_vals.append(era5_c)

    if not diffs:
        return DwdComparisonResult(
            station_id=dwd_data.station_id,
            station_name=station_name,
            n_matched=0,
            bias=0.0,
            mae=0.0,
            rmse=0.0,
            dwd_mean=0.0,
            era5_mean=0.0,
        )

    import math

    import numpy as np

    diffs_arr = np.array(diffs)
    bias = float(np.mean(diffs_arr))
    mae = float(np.mean(np.abs(diffs_arr)))
    rmse = float(math.sqrt(np.mean(diffs_arr**2)))

    return DwdComparisonResult(
        station_id=dwd_data.station_id,
        station_name=station_name,
        n_matched=len(diffs),
        bias=round(bias, 3),
        mae=round(mae, 3),
        rmse=round(rmse, 3),
        dwd_mean=round(float(np.mean(dwd_vals)), 2),
        era5_mean=round(float(np.mean(era5_vals)), 2),
    )


def dwd_comparison_to_dict(result: DwdComparisonResult) -> dict:
    """Convert a comparison result to a JSON-serializable dict."""
    return {
        "station_id": result.station_id,
        "station_name": result.station_name,
        "n_matched": result.n_matched,
        "bias_celsius": result.bias,
        "mae_celsius": result.mae,
        "rmse_celsius": result.rmse,
        "dwd_mean_celsius": result.dwd_mean,
        "era5_mean_celsius": result.era5_mean,
    }


__all__ = [
    "DwdStationRecord",
    "DwdStationData",
    "download_dwd_station",
    "compare_era5_with_dwd",
    "dwd_comparison_to_dict",
    "柏林_STATIONS",
]
