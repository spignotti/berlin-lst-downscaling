"""DWD station adapter — download and compare temperature observations.

Downloads official DWD historical hourly temperature ZIPs for Berlin stations
using stdlib HTTP + ZIP + CSV.  Station data is used ONLY for validation/QA,
never as a model channel.

Berlin stations (verified active 2017–2025):
- 00427 (Berlin-Tegel): 1973–present
- 00403 (Berlin-Dahlem): 2002–present
- 00433 (Berlin-Tempelhof): 1951–present
- 00400 (Berlin-Alexanderplatz): 1991–present

Data source: https://opendata.dwd.de/climate_environment/CDC/
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.request import urlopen

from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

柏林_STATIONS = {
    "00427": "Berlin-Tegel",
    "00403": "Berlin-Dahlem",
    "00433": "Berlin-Tempelhof",
    "00400": "Berlin-Alexanderplatz",
}

_HIST_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/air_temperature/historical/"
)
_RECENT_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/air_temperature/recent/"
)

# Pattern for historical ZIPs: stundenwerte_TU_{station}_{start}_{end}_hist.zip
_HIST_ZIP_RE = re.compile(
    r"stundenwerte_TU_(\d{5})_(\d{8})_(\d{8})_hist\.zip"
)


@dataclass
class DwdStationRecord:
    """A single hourly temperature observation from a DWD station."""

    station_id: str
    timestamp: datetime  # UTC
    temperature: float  # °C


@dataclass
class DwdStationData:
    """Full hourly record for a station."""

    station_id: str
    records: list[DwdStationRecord] = field(default_factory=list)

    def lookup_hour(self, utc_dt: datetime) -> float | None:
        """Find the temperature for the exact UTC hour, or None."""
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
) -> DwdStationData:
    """Download and parse DWD hourly temperature data for a station.

    Finds the correct ZIP via directory listing rather than guessing filenames.
    """
    data = DwdStationData(station_id=station_id)

    # Try to find the right historical ZIP
    try:
        records = _fetch_dwd_historical(station_id, year_range)
        data.records = records
        log_event(_logger, logging.INFO, "dwd_downloaded",
                  station_id=station_id, n_records=len(records))
        return data
    except Exception as exc:
        log_event(_logger, logging.WARNING, "dwd_hist_failed",
                  station_id=station_id, error=str(exc))

    # Fallback to recent
    try:
        records = _fetch_dwd_recent(station_id, year_range)
        data.records = records
        log_event(_logger, logging.INFO, "dwd_recent_downloaded",
                  station_id=station_id, n_records=len(records))
    except Exception as exc:
        log_event(_logger, logging.WARNING, "dwd_recent_failed",
                  station_id=station_id, error=str(exc))

    return data


def _fetch_dwd_historical(
    station_id: str,
    year_range: tuple[int, int] | None = None,
) -> list[DwdStationRecord]:
    """Discover and fetch the correct historical ZIP from the DWD index."""
    # Fetch the directory listing page
    response = urlopen(_HIST_BASE, timeout=30)  # noqa: S310
    listing = response.read().decode("latin-1")

    # Find ZIPs matching this station
    matches = []
    for m in _HIST_ZIP_RE.finditer(listing):
        sid, start_str, end_str = m.group(1), m.group(2), m.group(3)
        if sid != station_id:
            continue
        start_year = int(start_str[:4])
        end_year = int(end_str[:4])

        # Check overlap with requested year range
        if year_range is not None:
            req_start, req_end = year_range
            if end_year < req_start or start_year > req_end:
                continue

        matches.append((start_str, end_str))

    if not matches:
        raise ValueError(
            f"No historical ZIP found for station {station_id} "
            f"in range {year_range}"
        )

    # Use the most recent matching archive
    start_str, end_str = sorted(matches)[-1]
    url = (
        f"{_HIST_BASE}"
        f"stundenwerte_TU_{station_id}_{start_str}_{end_str}_hist.zip"
    )

    log_event(_logger, logging.DEBUG, "dwd_fetch", url=url)
    response = urlopen(url, timeout=60)  # noqa: S310
    zip_bytes = response.read()

    return _parse_dwd_zip(zip_bytes, station_id, year_range)


def _fetch_dwd_recent(
    station_id: str,
    year_range: tuple[int, int] | None = None,
) -> list[DwdStationRecord]:
    """Fetch the 'recent' ZIP for a station."""
    url = f"{_RECENT_BASE}stundenwerte_TU_{station_id}_akt.zip"
    log_event(_logger, logging.DEBUG, "dwd_fetch_recent", url=url)
    response = urlopen(url, timeout=60)  # noqa: S310
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
        csv_names = [
            n for n in z.namelist()
            if n.endswith(".csv") and "Beschreibung" not in n
        ]
        if not csv_names:
            raise ValueError(f"No CSV found in DWD ZIP for {station_id}")

        with z.open(csv_names[0]) as f:
            content = f.read().decode("latin-1")

        reader = csv.DictReader(io.StringIO(content), delimiter=";")

        for row in reader:
            try:
                date_val = row.get("MESSDATUM") or row.get("date", "")
                temp_val = row.get("TMK") or row.get("tt_tu", "")

                if not date_val or not temp_val:
                    continue

                # DWD date format: YYYYMMDDHH
                dt = datetime.strptime(date_val.strip(), "%Y%m%d%H%M")
                dt = dt.replace(tzinfo=UTC)

                temp = float(temp_val.strip())

                if year_range is not None:
                    if dt.year < year_range[0] or dt.year > year_range[1]:
                        continue

                records.append(DwdStationRecord(
                    station_id=station_id,
                    timestamp=dt,
                    temperature=temp,
                ))
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
    """Compare ERA5 2m temperature with DWD station observations."""
    import math

    import numpy as np

    diffs: list[float] = []
    dwd_vals: list[float] = []
    era5_vals: list[float] = []

    for utc_dt, era5_k in era5_t2m_by_hour.items():
        dwd_c = dwd_data.lookup_hour(utc_dt)
        if dwd_c is None:
            continue

        era5_c = era5_k - 273.15  # K → °C
        diffs.append(era5_c - dwd_c)
        dwd_vals.append(dwd_c)
        era5_vals.append(era5_c)

    if not diffs:
        return DwdComparisonResult(
            station_id=dwd_data.station_id,
            station_name=station_name,
            n_matched=0, bias=0.0, mae=0.0, rmse=0.0,
            dwd_mean=0.0, era5_mean=0.0,
        )

    diffs_arr = np.array(diffs)
    return DwdComparisonResult(
        station_id=dwd_data.station_id,
        station_name=station_name,
        n_matched=len(diffs),
        bias=round(float(np.mean(diffs_arr)), 3),
        mae=round(float(np.mean(np.abs(diffs_arr))), 3),
        rmse=round(float(math.sqrt(np.mean(diffs_arr**2))), 3),
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
