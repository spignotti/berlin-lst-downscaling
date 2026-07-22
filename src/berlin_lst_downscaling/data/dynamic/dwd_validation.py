"""DWD-vs-ERA5 validation pipeline.

Reads published dynamic-pipeline outputs (COG provenance + ledger),
acquires DWD hourly air-temperature observations for the AOI, joins
them at every Landsat anchor's normalised UTC hour, and emits a
reproducible comparison table and QA report.

This module is read-only against the published dynamic pipeline and
never feeds DWD data into training, normalisation, or the downstream
model.
"""
# decision: ignore pyright's overly conservative narrowing for pandas
# Series arithmetic (``Series | int | Unknown`` for ``int(Series.sum())``).
# ``pandas-stubs`` is not installed in this project; correctness is
# covered by the actual validation run, not static types.

from __future__ import annotations

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportReturnType=false
# pyright: reportAssignmentType=false
import os

# decision: see berlin_lst_downscaling.data.dynamic.dwd — strip env vars
# wetterdienst's pydantic-settings rejects before importing it. Guard
# here so the module can also be imported by tests / notebooks without
# first importing dwd.
for _key in (
    "google_application_credentials",
    "wandb_api_key",
    "earthdata_token",
):
    os.environ.pop(_key, None)

import json  # noqa: E402
import logging  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from hashlib import sha256  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from berlin_lst_downscaling.common.config import BERLIN_BBOX  # noqa: E402
from berlin_lst_downscaling.data.dynamic.dwd import (  # noqa: E402
    DwdFetchResult,
    fetch_dwd_temperature,
)
from berlin_lst_downscaling.data.dynamic.era5 import normalize_acquisition_hour  # noqa: E402
from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors  # noqa: E402
from berlin_lst_downscaling.data.dynamic.paths import (  # noqa: E402
    dwd_comparison_path,
    dwd_completion_path,
    dwd_observations_path,
    dwd_provenance_path,
    dwd_qa_report_path,
    dwd_run_dir,
    dwd_station_inventory_path,
)
from berlin_lst_downscaling.data.io import atomic_write, exists, log_event  # noqa: E402
from berlin_lst_downscaling.data.io.storage import read_bytes  # noqa: E402
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger  # noqa: E402

_logger = logging.getLogger(__name__)

_ERA5_LEDGER_SOURCE = "era5_land"


@dataclass
class Era5AnchorValue:
    """One Landsat anchor with the ERA5 t2m value at its normalised hour."""

    scene_id: str
    acquisition_utc: datetime
    era5_t2m_k: float | None
    era5_role: str | None
    era5_source_root: str | None
    era5_provenance_uri: str | None


@dataclass
class ValidationSummary:
    """Aggregated validation outcomes."""

    n_anchors: int
    n_anchors_with_era5: int
    n_dwd_stations: int
    n_dwd_stations_in_window: int
    n_observations: int
    n_historical_observations: int
    n_recent_observations: int
    n_pairs_total: int
    n_pairs_matched: int
    n_pairs_dwd_missing: int
    n_pairs_era5_missing: int
    n_pairs_provisional: int
    bias_celsius: float | None
    mae_celsius: float | None
    rmse_celsius: float | None
    per_station: list[dict[str, Any]] = field(default_factory=list)


def _atomic_write_parquet(df: pd.DataFrame, uri: str) -> None:
    """Write a Parquet file locally or to GCS via ``atomic_write``."""
    buf = df.to_parquet(index=False)
    if isinstance(buf, bytes):
        payload: bytes | str = buf
    else:
        payload = buf.getvalue()  # type: ignore[union-attr]
    atomic_write(uri, payload, overwrite=True)


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri).decode("utf-8"))


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def load_anchors_from_manifest(
    manifest_uri: str,
    dataset_roles: list[str] | None = None,
    scene_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load every Landsat anchor scene from the published manifest.

    Returns a DataFrame with ``scene_id``, ``acquisition_utc``,
    ``role``, and the manifest's raw ``year``.

    ``scene_ids`` filters the manifest to a fixed set of scenes before
    dataset-role filtering. Used by the bounded DWD smoke.
    """
    report = load_landsat_anchors(manifest_uri, scene_ids=scene_ids)
    if not report.ok:
        msg = f"Manifest load failed: {report.errors}"
        raise RuntimeError(msg)
    rows = [
        {
            "scene_id": s.scene_id,
            "acquisition_utc": _ensure_utc(s.acquisition_datetime),
            "role": s.role,
            "year": s.year,
        }
        for s in report.scenes
    ]
    df = pd.DataFrame(rows)
    if dataset_roles is not None:
        df = df[df["role"].isin(dataset_roles)]  # type: ignore[assignment]
    return df


def _load_era5_anchors(
    dynamic_full_root: str,
    dynamic_inference_root: str | None,
    scene_ids: list[str] | None = None,
) -> list[Era5AnchorValue]:
    """Load published ERA5 anchor t2m values from the dynamic ledgers.

    ``scene_ids`` optionally restricts to a fixed set of scenes (matches
    against ``item_key`` in the published provenance) — used by the
    bounded DWD smoke to skip full-period queries.
    """
    anchors: list[Era5AnchorValue] = []
    roots: list[tuple[str, str]] = []
    if exists(dynamic_full_root):
        roots.append((dynamic_full_root, "anchor"))
    if dynamic_inference_root and exists(dynamic_inference_root):
        roots.append((dynamic_inference_root, "inference"))

    scene_id_filter = set(scene_ids) if scene_ids else None

    for root, default_role in roots:
        ledger_uri = f"{root.rstrip('/')}/_state/dynamic/ledger.parquet"
        if not exists(ledger_uri):
            log_event(_logger, logging.WARNING, "validation_ledger_missing",
                      root=root)
            continue
        ledger = SecondaryLedger.open(ledger_uri)
        for row in ledger.items_for_source(_ERA5_LEDGER_SOURCE):
            if row.status != "done" or not row.provenance_uri:
                continue
            if not exists(row.provenance_uri):
                continue
            try:
                prov = _read_json(row.provenance_uri)
            except Exception as exc:
                log_event(_logger, logging.WARNING, "validation_provenance_failed",
                          scene_id=row.item_id, error=str(exc))
                continue
            qa_stats = prov.get("qa_stats") or {}
            t2m_val = qa_stats.get("t2m_scene")
            if t2m_val is None:
                continue
            acquisition = prov.get("acquisition_datetime")
            if acquisition is None:
                continue
            try:
                acq_dt = _ensure_utc(datetime.fromisoformat(acquisition))
            except ValueError:
                continue
            scene_id = prov.get("item_key") or row.item_id.replace(
                f"{_ERA5_LEDGER_SOURCE}_", "", 1,
            )
            if scene_id_filter is not None and scene_id not in scene_id_filter:
                continue
            role = row.role or default_role
            anchors.append(
                Era5AnchorValue(
                    scene_id=scene_id,
                    acquisition_utc=acq_dt,
                    era5_t2m_k=float(t2m_val),
                    era5_role=role,
                    era5_source_root=root,
                    era5_provenance_uri=row.provenance_uri,
                )
            )
    return anchors


def _join_anchors_with_dwd(
    anchors: list[Era5AnchorValue],
    dwd_obs_df: pd.DataFrame,
    *,
    include_provisional: bool = True,
) -> pd.DataFrame:
    """Left-join DWD observations onto anchors at the normalised UTC hour."""
    if not anchors:
        return pd.DataFrame(
            columns=[
                "scene_id", "acquisition_utc", "match_hour_utc", "era5_t2m_k",
                "era5_role", "dwd_period", "dwd_period_kind",
                "station_id", "dwd_temperature_c", "dwd_quality", "match_state",
            ],
        )
    rows = []
    dwd_by_station: dict[str, pd.DataFrame] = {}
    if not dwd_obs_df.empty:
        dwd_obs_df = dwd_obs_df.assign(
            timestamp_utc=pd.to_datetime(dwd_obs_df["timestamp_utc"], utc=True),  # type: ignore[arg-type]
        )
        for sid, sub in dwd_obs_df.groupby("station_id"):  # type: ignore[union-attr]
            dwd_by_station[str(sid)] = sub.sort_values("timestamp_utc")

    for anchor in anchors:
        match_hour = normalize_acquisition_hour(anchor.acquisition_utc)
        for sid, sub in dwd_by_station.items():
            match = sub[sub["timestamp_utc"] == match_hour]
            if match.empty:
                rows.append(
                    {
                        "scene_id": anchor.scene_id,
                        "acquisition_utc": anchor.acquisition_utc,
                        "match_hour_utc": match_hour,
                        "era5_t2m_k": anchor.era5_t2m_k,
                        "era5_role": anchor.era5_role,
                        "era5_source_root": anchor.era5_source_root,
                        "station_id": sid,
                        "dwd_period": None,
                        "dwd_period_kind": None,
                        "dwd_temperature_c": None,
                        "dwd_quality": None,
                        "match_state": "dwd_missing",
                    }
                )
                continue
            for _, obs_row in match.iterrows():
                period = str(obs_row["dwd_period"])
                is_provisional = period == "recent"
                if is_provisional and not include_provisional:
                    state = "dwd_provisional"
                else:
                    state = "matched"
                rows.append(
                    {
                        "scene_id": anchor.scene_id,
                        "acquisition_utc": anchor.acquisition_utc,
                        "match_hour_utc": match_hour,
                        "era5_t2m_k": anchor.era5_t2m_k,
                        "era5_role": anchor.era5_role,
                        "era5_source_root": anchor.era5_source_root,
                        "station_id": sid,
                        "dwd_period": period,
                        "dwd_period_kind": "historical" if period == "historical" else "recent",
                        "dwd_temperature_c": float(obs_row["temperature_c"]),
                        "dwd_quality": (
                            float(obs_row["quality"])  # type: ignore[arg-type]
                            if bool(pd.notna(obs_row["quality"]))  # type: ignore[arg-type]
                            else None
                        ),
                        "match_state": state,
                    }
                )
    return pd.DataFrame(rows)


def _summarise(comparison_df: pd.DataFrame, fetch: DwdFetchResult) -> ValidationSummary:
    matched = comparison_df[comparison_df["match_state"] == "matched"]
    diffs_c = (
        matched["era5_t2m_k"].astype(float) - 273.15
        - matched["dwd_temperature_c"].astype(float)
    )
    bias = float(diffs_c.mean()) if not diffs_c.empty else None
    mae = float(np.mean(np.abs(diffs_c))) if not diffs_c.empty else None
    rmse = float(np.sqrt(np.mean(diffs_c.to_numpy() ** 2))) if not diffs_c.empty else None

    per_station: list[dict[str, Any]] = []
    if not matched.empty:
        matched = matched.assign(_diff_c=diffs_c)
        for sid, sub in matched.groupby("station_id"):
            station = next(
                (s for s in fetch.inventory if s.station_id == sid), None,
            )
            diffs = sub["_diff_c"]
            per_station.append(
                {
                    "station_id": sid,
                    "station_name": station.name if station else None,
                    "latitude": station.latitude if station else None,
                    "longitude": station.longitude if station else None,
                    "n_matched": int(len(sub)),
                    "bias_celsius": round(float(diffs.mean()), 3),
                    "mae_celsius": round(float(np.mean(np.abs(diffs))), 3),
                    "rmse_celsius": round(float(np.sqrt(np.mean(diffs.to_numpy() ** 2))), 3),
                    "min_quality": (
                        float(sub["dwd_quality"].min())  # type: ignore[arg-type]
                        if bool(sub["dwd_quality"].notna().any())  # type: ignore[arg-type]
                        else None
                    ),
                    "max_quality": (
                        float(sub["dwd_quality"].max())  # type: ignore[arg-type]
                        if bool(sub["dwd_quality"].notna().any())  # type: ignore[arg-type]
                        else None
                    ),
                }
            )
    n_obs_hist = int(fetch.observations_df["dwd_period"].eq("historical").sum()) \
        if not fetch.observations_df.empty else 0
    n_obs_recent = int(fetch.observations_df["dwd_period"].eq("recent").sum()) \
        if not fetch.observations_df.empty else 0

    if comparison_df.empty:
        n_pairs_total = 0
        n_pairs_matched = 0
        n_pairs_dwd_missing = 0
        n_pairs_era5_missing = 0
        n_pairs_provisional = 0
        n_anchors_with_era5 = 0
    else:
        n_pairs_total = int(len(comparison_df))
        n_pairs_matched = int(comparison_df["match_state"].eq("matched").sum())
        n_pairs_dwd_missing = int(
            comparison_df["match_state"].eq("dwd_missing").sum(),
        )
        n_pairs_era5_missing = int(
            comparison_df["era5_t2m_k"].isna().sum(),
        )
        n_pairs_provisional = int(
            comparison_df["match_state"].eq("dwd_provisional").sum(),
        )
        n_anchors_with_era5 = int(
            comparison_df.dropna(subset=["era5_t2m_k"])["scene_id"].nunique(),
        )

    return ValidationSummary(
        n_anchors=int(comparison_df["scene_id"].nunique()),
        n_anchors_with_era5=n_anchors_with_era5,
        n_dwd_stations=len(fetch.inventory),
        n_dwd_stations_in_window=sum(1 for s in fetch.inventory if s.in_request_window),
        n_observations=int(len(fetch.observations_df)),
        n_historical_observations=n_obs_hist,
        n_recent_observations=n_obs_recent,
        n_pairs_total=n_pairs_total,
        n_pairs_matched=n_pairs_matched,
        n_pairs_dwd_missing=n_pairs_dwd_missing,
        n_pairs_era5_missing=n_pairs_era5_missing,
        n_pairs_provisional=n_pairs_provisional,
        bias_celsius=round(bias, 3) if bias is not None else None,
        mae_celsius=round(mae, 3) if mae is not None else None,
        rmse_celsius=round(rmse, 3) if rmse is not None else None,
        per_station=per_station,
    )


def _write_text(path: str, text: str) -> None:
    atomic_write(path, text, overwrite=True)


def _hash_uri(uri: str) -> str:
    """SHA-256 of an arbitrary URI's content (or path string if missing)."""
    try:
        return sha256(read_bytes(uri)).hexdigest()
    except Exception:
        return sha256(uri.encode("utf-8")).hexdigest()


def run_dwd_validation(
    cfg: object,
    run_id: str,
) -> int:
    """Execute the DWD-vs-ERA5 validation pipeline.

    Returns 0 on success, 1 on a hard error (missing inputs, IO failure).
    QA degradations (missing DWD observations, unmatched hours) are
    reported in the QA report, not as failures.
    """
    from omegaconf import OmegaConf

    cfg_dict_raw: Mapping[str, object] | Any
    if OmegaConf.is_config(cfg):
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict_raw = cfg
    cfg_dict: dict[str, object] = {str(k): v for k, v in cfg_dict_raw.items()}

    def _str(value: object, default: str = "") -> str:
        return str(value) if value not in (None, "") else default

    manifest_uri = _str(cfg_dict.get("manifest_uri"))
    output_root = _str(cfg_dict.get("output_root"))
    if not output_root:
        raise ValueError("output_root is required")
    aoi_uri = _str(
        cfg_dict.get("aoi_uri"),
        "data/boundaries/berlin_landesgrenze.geojson",
    )
    start_date = _ensure_utc(datetime.fromisoformat(
        _str(cfg_dict.get("start_date")).replace("Z", "+00:00"),
    ))
    end_date = _ensure_utc(datetime.fromisoformat(
        _str(cfg_dict.get("end_date")).replace("Z", "+00:00"),
    ))
    periods_raw = cfg_dict.get("periods") or ["historical", "recent"]
    periods = [str(p) for p in periods_raw]  # type: ignore[union-attr]

    scene_ids_raw = cfg_dict.get("scene_ids") or []
    scene_ids = [str(s) for s in scene_ids_raw] if scene_ids_raw else None  # type: ignore[union-attr]

    if not manifest_uri:
        log_event(_logger, logging.ERROR, "validation_manifest_missing")
        return 1

    anchors_df = load_anchors_from_manifest(manifest_uri, scene_ids=scene_ids)
    log_event(_logger, logging.INFO, "validation_anchors_loaded",
              n_anchors=int(len(anchors_df)))

    full_root = _str(cfg_dict.get("dynamic_full_root"))
    inference_root_raw = cfg_dict.get("dynamic_inference_root")
    inference_root_str = str(inference_root_raw) if inference_root_raw else None
    era5_anchors = _load_era5_anchors(full_root, inference_root_str, scene_ids=scene_ids)
    log_event(_logger, logging.INFO, "validation_era5_loaded",
              n_era5_anchors=len(era5_anchors),
              full_root=full_root,
              inference_root=inference_root_str)

    if not era5_anchors:
        log_event(_logger, logging.ERROR, "validation_no_era5_anchors")
        return 1

    # Expand the DWD window to cover all anchor times plus a 1-hour safety
    # margin to absorb the half-up rounding used in normalize_acquisition_hour.
    anchor_min = min((a.acquisition_utc for a in era5_anchors), default=start_date)
    anchor_max = max((a.acquisition_utc for a in era5_anchors), default=end_date)
    query_start = min(start_date, anchor_min - timedelta(hours=1))
    query_end = max(end_date, anchor_max + timedelta(hours=1))

    fetch = fetch_dwd_temperature(
        aoi_uri=aoi_uri,
        start_utc=query_start,
        end_utc=query_end,
        periods=periods,
    )

    # Persist raw snapshot
    inventory_df = fetch.inventory_df.copy()
    observations_df = fetch.observations_df.copy()
    _atomic_write_parquet(inventory_df, dwd_station_inventory_path(output_root, run_id))
    _atomic_write_parquet(observations_df, dwd_observations_path(output_root, run_id))

    comparison_df = _join_anchors_with_dwd(
        era5_anchors,
        observations_df,
        include_provisional=True,
    )
    _atomic_write_parquet(comparison_df, dwd_comparison_path(output_root, run_id))

    summary = _summarise(comparison_df, fetch)
    report = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest_uri": manifest_uri,
        "output_root": output_root,
        "aoi_uri": aoi_uri,
        "bbox_wgs84": list(BERLIN_BBOX),
        "query_window_utc": [
            query_start.isoformat(),
            query_end.isoformat(),
        ],
        "dwd_periods_requested": periods,
        "dynamic_full_root": full_root,
        "dynamic_inference_root": inference_root_str,
        "wetterdienst_version": _wetterdienst_version(),
        "validation_method": "historical_precedence_over_recent",
        "metric_basis": (
            "All matched DWD observations (historical quality-controlled "
            "plus recent provisional). Recent observations are flagged in "
            "the per-row match_state and excluded from headline metrics "
            "by downstream consumers."
        ),
        "summary": {
            "n_anchors": summary.n_anchors,
            "n_anchors_with_era5": summary.n_anchors_with_era5,
            "n_dwd_stations": summary.n_dwd_stations,
            "n_dwd_stations_in_window": summary.n_dwd_stations_in_window,
            "n_observations": summary.n_observations,
            "n_historical_observations": summary.n_historical_observations,
            "n_recent_observations": summary.n_recent_observations,
            "n_pairs_total": summary.n_pairs_total,
            "n_pairs_matched": summary.n_pairs_matched,
            "n_pairs_dwd_missing": summary.n_pairs_dwd_missing,
            "n_pairs_era5_missing": summary.n_pairs_era5_missing,
            "n_pairs_provisional": summary.n_pairs_provisional,
            "bias_celsius_era5_minus_dwd": summary.bias_celsius,
            "mae_celsius": summary.mae_celsius,
            "rmse_celsius": summary.rmse_celsius,
        },
        "per_station": summary.per_station,
        "stations_outside_window": [
            {
                "station_id": s.station_id,
                "name": s.name,
                "start_date": s.start_date.isoformat() if s.start_date else None,
                "end_date": s.end_date.isoformat() if s.end_date else None,
            }
            for s in fetch.inventory
            if not s.in_request_window
        ],
    }
    _write_text(dwd_qa_report_path(output_root, run_id), json.dumps(report, indent=2))

    provenance = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest_uri": manifest_uri,
        "manifest_sha256": _hash_uri(manifest_uri),
        "aoi_uri": aoi_uri,
        "aoi_sha256": _hash_uri(aoi_uri),
        "output_root": output_root,
        "dynamic_full_root": full_root,
        "dynamic_inference_root": inference_root_str,
        "wetterdienst_version": _wetterdienst_version(),
        "dwd_periods": periods,
        "dwd_query_window_utc": [
            query_start.isoformat(),
            query_end.isoformat(),
        ],
        "station_ids": fetch.station_ids,
        "anchor_count": summary.n_anchors,
        "observation_count": summary.n_observations,
    }
    _write_text(dwd_provenance_path(output_root, run_id), json.dumps(provenance, indent=2))
    _write_text(
        dwd_completion_path(output_root, run_id),
        json.dumps(
            {"published_at": datetime.now(UTC).isoformat(), "run_id": run_id},
            indent=2,
        ),
    )

    log_event(
        _logger, logging.INFO, "validation_done",
        n_pairs=summary.n_pairs_total,
        n_matched=summary.n_pairs_matched,
        bias=summary.bias_celsius,
        mae=summary.mae_celsius,
        rmse=summary.rmse_celsius,
        output=dwd_run_dir(output_root, run_id),
    )
    return 0


def _wetterdienst_version() -> str | None:
    try:
        from wetterdienst import __version__ as version  # type: ignore
    except ImportError:
        return None
    return str(version)


__all__ = [
    "Era5AnchorValue",
    "ValidationSummary",
    "load_anchors_from_manifest",
    "run_dwd_validation",
]
