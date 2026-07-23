"""DWD station acquisition via wetterdienst — validation-only boundary.

Acquires hourly 2 m air temperature observations from official DWD Open
Data (CDC) for stations inside the Berlin AOI. Used exclusively to
sanity-check the ERA5-Land ``t2m_scene`` channel in the published
dynamic products; never fed into training, normalisation, or the
downstream model.

Returns
-------
- :class:`DwdFetchResult` with ``inventory`` (stations inside AOI,
  filtered to the query window) and ``observations`` (hourly values
  with DWD quality level and source period).
"""

from __future__ import annotations

import os

# decision: wetterdienst's Settings (``extra="forbid"`` pydantic default)
# reads every env var, so Berlin-LST ``.env`` (GCP, W&B, Earthdata)
# crashes Settings construction. Strip conflicting keys at import and
# override Settings.model_config to ``extra="ignore"``, ``env_file=None``.
# Both steps are no-ops once wetterdienst is already configured this way.
_SAFE_KEYS = ("google_application_credentials", "wandb_api_key", "earthdata_token")
for _key in _SAFE_KEYS:
    os.environ.pop(_key, None)

import logging  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Iterable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import wetterdienst.settings as _wd_settings  # noqa: E402
from pydantic_settings import SettingsConfigDict  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from berlin_lst_downscaling.data.io import log_event  # noqa: E402

# Apply the wetterdienst Settings patch exactly once per process.
_model_config = _wd_settings.Settings.model_config
if isinstance(_model_config, dict):
    _current_extra = _model_config.get("extra")
else:
    _current_extra = getattr(_model_config, "extra", None)
if _current_extra != "ignore":
    _wd_settings.Settings.model_config = SettingsConfigDict(
        env_file=None,
        env_ignore_empty=True,
        env_prefix="WD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

_logger = logging.getLogger(__name__)

@dataclass
class DwdStationInventory:
    """One DWD station inside the AOI."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    height: float | None
    start_date: datetime | None
    end_date: datetime | None
    in_request_window: bool

@dataclass
class DwdObservation:
    """One hourly DWD temperature observation."""

    station_id: str
    timestamp_utc: datetime
    temperature_c: float
    quality: float | None
    dwd_period: str  # "historical" or "recent"

@dataclass
class DwdFetchResult:
    """Combined DWD fetch result."""

    inventory: list[DwdStationInventory] = field(default_factory=list)
    observations: list[DwdObservation] = field(default_factory=list)
    inventory_df: pd.DataFrame = field(
        default_factory=lambda: _inventory_to_dataframe([]),
    )
    observations_df: pd.DataFrame = field(
        default_factory=lambda: _observations_to_dataframe([]),
    )

    @property
    def station_ids(self) -> list[str]:
        return [s.station_id for s in self.inventory]

def _observations_to_dataframe(observations: list[DwdObservation]) -> pd.DataFrame:
    if not observations:
        return pd.DataFrame(
            columns=["station_id", "timestamp_utc", "temperature_c", "quality", "dwd_period"],
        )
    return pd.DataFrame(
        [
            {
                "station_id": o.station_id,
                "timestamp_utc": o.timestamp_utc,
                "temperature_c": o.temperature_c,
                "quality": o.quality,
                "dwd_period": o.dwd_period,
            }
            for o in observations
        ]
    )

def _inventory_to_dataframe(inventory: list[DwdStationInventory]) -> pd.DataFrame:
    if not inventory:
        return pd.DataFrame(
            columns=[
                "station_id",
                "name",
                "latitude",
                "longitude",
                "height",
                "start_date",
                "end_date",
                "in_request_window",
            ],
        )
    return pd.DataFrame(
        [
            {
                "station_id": s.station_id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "height": s.height,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "in_request_window": s.in_request_window,
            }
            for s in inventory
        ]
    )

def _load_aoi_polygon(aoi_uri: str):
    """Load the Berlin AOI polygon and ensure WGS84."""
    gdf = gpd.read_file(aoi_uri)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:25833")
    crs = gdf.crs
    if crs is None or crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf.union_all()

def _bbox_for_polygon(geom) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    # Pad 0.2° to cover stations right on the AOI edge.
    return (minx - 0.2, miny - 0.2, maxx + 0.2, maxy + 0.2)

def _in_window(
    start: datetime | None,
    end: datetime | None,
    query_start: datetime,
    query_end: datetime,
) -> bool:
    """Return True if [start, end] overlaps the query window."""
    if start is not None and start > query_end:
        return False
    if end is not None and end < query_start:
        return False
    return True

def _request_period_bounds(period: str) -> tuple[datetime, datetime]:
    """Return the (start, end) UTC bounds a wetterdienst period covers."""
    now = datetime.now(UTC)
    if period == "historical":
        return datetime(1678, 1, 1, tzinfo=UTC), now.replace(month=1, day=1, tzinfo=UTC)
    if period == "recent":
        return (now - timedelta(days=500)).replace(hour=0, minute=0, second=0, microsecond=0), now
    msg = f"Unsupported DWD period: {period}"
    raise ValueError(msg)

def _wetterdienst_request(
    period: str,
    start_utc: datetime,
    end_utc: datetime,
    station_ids: list[str] | None = None,
):
    """Build a DwdObservationRequest filtered to *station_ids*."""
    from wetterdienst.provider.dwd.observation import DwdObservationRequest
    from wetterdienst.settings import Settings

    settings = Settings(
        ts_shape="long",
        ts_humanize=False,
        ts_convert_units=False,
        cache_disable=False,
    )
    req = DwdObservationRequest(
        parameters=("hourly", "temperature_air", "temperature_air_mean_2m"),
        periods=period,
        start_date=start_utc,
        end_date=end_utc,
        settings=settings,
    )
    if station_ids:
        req = req.filter_by_station_id(station_ids)
    return req

def _collect_stations(
    period: str,
    aoi_geom,
    query_start: datetime,
    query_end: datetime,
) -> list[DwdStationInventory]:
    bbox = _bbox_for_polygon(aoi_geom)
    req = _wetterdienst_request(period, query_start, query_end)
    req = req.filter_by_bbox(*bbox)  # type: ignore[attr-defined]
    df = req.df
    keep_cols = ["station_id", "name", "latitude", "longitude", "height", "start_date", "end_date"]
    df = df.select(keep_cols)
    rows: list[DwdStationInventory] = []
    for record in df.iter_rows(named=True):
        lat = float(record["latitude"])
        lon = float(record["longitude"])
        if not Point(lon, lat).within(aoi_geom):
            continue
        sid = str(record["station_id"]).zfill(5)
        rows.append(
            DwdStationInventory(
                station_id=sid,
                name=str(record["name"]),
                latitude=lat,
                longitude=lon,
                height=float(record["height"]) if record["height"] is not None else None,
                start_date=record["start_date"],
                end_date=record["end_date"],
                in_request_window=_in_window(
                    record["start_date"],
                    record["end_date"],
                    query_start,
                    query_end,
                ),
            )
        )
    return rows

def _collect_observations(
    period: str,
    station_ids: Iterable[str],
    query_start: datetime,
    query_end: datetime,
) -> list[DwdObservation]:
    station_ids = sorted(set(station_ids))
    if not station_ids:
        return []
    period_start, period_end = _request_period_bounds(period)
    window_start = max(query_start, period_start)
    window_end = min(query_end, period_end)
    if window_start > window_end:
        return []
    req = _wetterdienst_request(period, window_start, window_end, station_ids)
    try:
        lf = req.values.all().df  # type: ignore[attr-defined]
        df = lf.collect() if hasattr(lf, "collect") else lf  # type: ignore[attr-defined]
    except Exception as exc:
        log_event(_logger, logging.WARNING, "dwd_values_failed", period=period, error=str(exc))
        return []
    keep = {"station_id", "date", "value", "quality"}
    if not keep.issubset(set(df.columns)):
        log_event(
            _logger,
            logging.WARNING,
            "dwd_values_schema_unexpected",
            period=period,
            columns=list(df.columns),
        )
        return []
    df = df.select(list(keep))
    out: list[DwdObservation] = []
    for record in df.iter_rows(named=True):
        ts = record["date"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        else:
            ts = ts.astimezone(UTC)
        value = record["value"]
        if value is None or pd.isna(value):
            continue
        out.append(
            DwdObservation(
                station_id=str(record["station_id"]).zfill(5),
                timestamp_utc=ts,
                temperature_c=float(value),
                quality=float(record["quality"]) if record["quality"] is not None else None,
                dwd_period=period,
            )
        )
    return out

def _merge_historical_recent(
    historical: list[DwdObservation],
    recent: list[DwdObservation],
) -> list[DwdObservation]:
    """Prefer historical over recent for duplicate (station, timestamp)."""
    by_key: dict[tuple[str, datetime], DwdObservation] = {}
    for obs in historical:
        by_key[(obs.station_id, obs.timestamp_utc)] = obs
    for obs in recent:
        by_key.setdefault((obs.station_id, obs.timestamp_utc), obs)
    return list(by_key.values())

def fetch_dwd_temperature(
    aoi_uri: str,
    start_utc: datetime,
    end_utc: datetime,
    periods: list[str] | None = None,
) -> DwdFetchResult:
    """Fetch DWD hourly temperature observations for stations inside the AOI.

    Parameters
    ----------
    aoi_uri : Local path or ``gs://...`` URI to the AOI polygon.
    start_utc, end_utc : Inclusive query window (UTC).
    periods : wetterdienst periods to query. Defaults to ``["historical", "recent"]``.
    """
    from berlin_lst_downscaling.data.io.storage import exists as uri_exists
    from berlin_lst_downscaling.data.io.storage import read_bytes

    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=UTC)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=UTC)
    if periods is None:
        periods = ["historical", "recent"]

    if aoi_uri.startswith("gs://"):
        if not uri_exists(aoi_uri):
            raise FileNotFoundError(f"AOI not found at {aoi_uri}")
        tmp = Path(tempfile.mkdtemp(prefix="dwd_aoi_")) / "aoi.geojson"
        tmp.write_bytes(read_bytes(aoi_uri))
        aoi_uri = str(tmp)

    aoi_geom = _load_aoi_polygon(aoi_uri)

    inventory_rows: dict[str, DwdStationInventory] = {}
    for period in periods:
        log_event(_logger, logging.INFO, "dwd_stations_query", period=period)
        for station in _collect_stations(period, aoi_geom, start_utc, end_utc):
            existing = inventory_rows.get(station.station_id)
            if existing is None:
                inventory_rows[station.station_id] = station
                continue
            # Prefer the metadata row with the broader temporal coverage.
            if (station.start_date, station.end_date) < (existing.start_date, existing.end_date):
                inventory_rows[station.station_id] = station

    stations_in_window = [s for s in inventory_rows.values() if s.in_request_window]
    station_ids = [s.station_id for s in stations_in_window]
    log_event(
        _logger,
        logging.INFO,
        "dwd_stations_selected",
        n_stations=len(inventory_rows),
        n_in_window=len(stations_in_window),
    )

    historical_obs: list[DwdObservation] = []
    recent_obs: list[DwdObservation] = []
    if "historical" in periods:
        historical_obs = _collect_observations("historical", station_ids, start_utc, end_utc)
    if "recent" in periods:
        recent_obs = _collect_observations("recent", station_ids, start_utc, end_utc)

    observations = _merge_historical_recent(historical_obs, recent_obs)
    log_event(
        _logger,
        logging.INFO,
        "dwd_observations_collected",
        historical=len(historical_obs),
        recent=len(recent_obs),
        merged=len(observations),
    )

    return DwdFetchResult(
        inventory=list(inventory_rows.values()),
        observations=observations,
        inventory_df=_inventory_to_dataframe(list(inventory_rows.values())),
        observations_df=_observations_to_dataframe(observations),
    )

__all__ = [
    "DwdFetchResult",
    "DwdObservation",
    "DwdStationInventory",
    "fetch_dwd_temperature",
]