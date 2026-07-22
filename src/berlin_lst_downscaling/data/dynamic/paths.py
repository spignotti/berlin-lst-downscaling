"""Deterministic output paths for dynamic scene products.

Layout
------
Raw ERA5-Land cache:
  _raw/dynamic/era5_land/YYYY-MM/

Per-scene dynamic products:
  ard/dynamic/{source}/{scene_id}/
    ├─ {source}_{scene_id}.tif          # final COG
    ├─ {source}_{scene_id}.stac.json    # STAC Item metadata
    ├─ provenance.json                  # source/archive provenance
    └─ complete.json                    # publication marker (written last)

QA:
  qa/dynamic/{run_id}/report.json

Ledger:
  _state/dynamic/ledger.parquet

DWD validation runs:
  _raw/dwd/<run_id>/                   # station inventory + hourly observations
  runs/<run_id>/                       # anchor comparison, report, provenance, complete
"""

from __future__ import annotations

# ── layout constants ──────────────────────────────────────────────────

_DYNAMIC_ROOT = "ard/dynamic"
_RAW_ROOT = "_raw/dynamic"
_QA_ROOT = "qa/dynamic"
_STATE_ROOT = "_state/dynamic"
_VALIDATION_RAW_ROOT = "_raw/dwd"
_VALIDATION_RUNS_ROOT = "runs/dwd"


# ── raw ERA5 cache ───────────────────────────────────────────────────


def era5_cache_path(root: str, year: int, month: int) -> str:
    """Return the ERA5-Land NetCDF file path for a given month."""
    return (
        f"{root.rstrip('/')}/{_RAW_ROOT}/era5_land/"
        f"{year:04d}-{month:02d}/era5_land_{year:04d}{month:02d}.nc"
    )


# ── per-scene product paths ──────────────────────────────────────────


def scene_product_dir(root: str, source: str, scene_id: str) -> str:
    """Return the product directory for a scene-keyed dynamic source."""
    return f"{root.rstrip('/')}/{_DYNAMIC_ROOT}/{source}/{scene_id}"


# ── QA and ledger ────────────────────────────────────────────────────


def qa_dir(root: str, run_id: str) -> str:
    """Return the QA report directory for a dynamic run."""
    return f"{root.rstrip('/')}/{_QA_ROOT}/{run_id}"


def qa_report_path(root: str, run_id: str) -> str:
    """Return the persisted QA report URI for a dynamic run."""
    return f"{qa_dir(root, run_id)}/report.json"


def ledger_path(root: str) -> str:
    """Return the dynamic ledger Parquet path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/ledger.parquet"


# ── DWD validation run paths ────────────────────────────────────────


def _dwd_raw_dir(root: str, run_id: str) -> str:
    """Return the raw DWD snapshot directory for a validation run."""
    return f"{root.rstrip('/')}/{_VALIDATION_RAW_ROOT}/{run_id}"


def dwd_station_inventory_path(root: str, run_id: str) -> str:
    """Return the URI of the DWD station inventory Parquet file."""
    return f"{_dwd_raw_dir(root, run_id)}/station_inventory.parquet"


def dwd_observations_path(root: str, run_id: str) -> str:
    """Return the URI of the merged DWD observations Parquet file."""
    return f"{_dwd_raw_dir(root, run_id)}/dwd_hourly_observations.parquet"


def dwd_run_dir(root: str, run_id: str) -> str:
    """Return the run-scoped DWD validation output directory."""
    return f"{root.rstrip('/')}/{_VALIDATION_RUNS_ROOT}/{run_id}"


def dwd_comparison_path(root: str, run_id: str) -> str:
    """Return the URI of the per-anchor comparison Parquet file."""
    return f"{dwd_run_dir(root, run_id)}/anchor_comparison.parquet"


def dwd_qa_report_path(root: str, run_id: str) -> str:
    """Return the URI of the DWD validation QA report."""
    return f"{dwd_run_dir(root, run_id)}/report.json"


def dwd_provenance_path(root: str, run_id: str) -> str:
    """Return the URI of the DWD validation provenance file."""
    return f"{dwd_run_dir(root, run_id)}/provenance.json"


def dwd_completion_path(root: str, run_id: str) -> str:
    """Return the URI of the DWD validation completion marker."""
    return f"{dwd_run_dir(root, run_id)}/complete.json"


__all__ = [
    "dwd_comparison_path",
    "dwd_completion_path",
    "dwd_observations_path",
    "dwd_provenance_path",
    "dwd_qa_report_path",
    "dwd_run_dir",
    "dwd_station_inventory_path",
    "era5_cache_path",
    "ledger_path",
    "qa_dir",
    "qa_report_path",
    "scene_product_dir",
]
