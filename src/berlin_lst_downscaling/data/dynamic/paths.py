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
"""

from __future__ import annotations

# ── layout constants ──────────────────────────────────────────────────

_DYNAMIC_ROOT = "ard/dynamic"
_RAW_ROOT = "_raw/dynamic"
_QA_ROOT = "qa/dynamic"
_STATE_ROOT = "_state/dynamic"


# ── raw ERA5 cache ───────────────────────────────────────────────────


def era5_cache_dir(root: str, year: int, month: int) -> str:
    """Return the ERA5-Land cache directory for a given month."""
    return f"{root.rstrip('/')}/{_RAW_ROOT}/era5_land/{year:04d}-{month:02d}"


def era5_cache_path(root: str, year: int, month: int) -> str:
    """Return the ERA5-Land NetCDF file path for a given month."""
    return f"{era5_cache_dir(root, year, month)}/era5_land_{year:04d}{month:02d}.nc"


# ── per-scene product paths ──────────────────────────────────────────


def scene_product_dir(root: str, source: str, scene_id: str) -> str:
    """Return the product directory for a scene-keyed dynamic source."""
    return f"{root.rstrip('/')}/{_DYNAMIC_ROOT}/{source}/{scene_id}"


def scene_product_cog(root: str, source: str, scene_id: str) -> str:
    """Return the final COG URI for a scene-keyed dynamic product."""
    return f"{scene_product_dir(root, source, scene_id)}/{source}_{scene_id}.tif"


def scene_product_stac(root: str, source: str, scene_id: str) -> str:
    """Return the STAC Item URI for a scene-keyed dynamic product."""
    return f"{scene_product_dir(root, source, scene_id)}/{source}_{scene_id}.stac.json"


def scene_product_provenance(root: str, source: str, scene_id: str) -> str:
    """Return the provenance URI for a scene-keyed dynamic product."""
    return f"{scene_product_dir(root, source, scene_id)}/provenance.json"


def scene_product_completion(root: str, source: str, scene_id: str) -> str:
    """Return the completion marker URI for a scene-keyed dynamic product."""
    return f"{scene_product_dir(root, source, scene_id)}/complete.json"


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


__all__ = [
    "era5_cache_dir",
    "era5_cache_path",
    "scene_product_dir",
    "scene_product_cog",
    "scene_product_stac",
    "scene_product_provenance",
    "scene_product_completion",
    "qa_dir",
    "qa_report_path",
    "ledger_path",
]
