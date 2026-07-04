"""Szenen-Selektion & Kopplung — Landsat anchor, S2 match, ECOSTRESS subset.

Builds an ARD manifest (Parquet) per :ref:`ard-manifest-schema`:
- Landsat scenes as anchors (May–Sep, configurable years)
- Best Sentinel-2 match per anchor: score = clear_frac − λ·Δt/3
- ECOSTRESS subset only on anchor days, ±2 h local time

Usage::

    from berlin_lst_downscaling.data.selection import build_manifest, run_scan
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/selection/smoke_jul2024.yaml")
    result = build_manifest(cfg)          # returns ManifestResult
    report = run_scan(cfg)               # returns ScanReport
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# ── public API ───────────────────────────────────────────────────────────────
from berlin_lst_downscaling.data.selection.anchors import build_anchors
from berlin_lst_downscaling.data.selection.clear_frac import compute_clear_frac
from berlin_lst_downscaling.data.selection.couple import couple_all
from berlin_lst_downscaling.data.selection.ecostress import search_ecostress
from berlin_lst_downscaling.data.selection.ecostress_subset import (
    build_ecostress_subset,
)
from berlin_lst_downscaling.data.selection.manifest import write_manifest
from berlin_lst_downscaling.data.selection.s2_search import (
    match_s2_candidates,
    match_s2_candidates_with_clear_frac,
)
from berlin_lst_downscaling.data.selection.scan import run_scan

__all__ = [
    # Core entry-points
    "build_anchors",
    "match_s2_candidates",
    "match_s2_candidates_with_clear_frac",
    "couple_all",
    "build_ecostress_subset",
    "compute_clear_frac",
    "write_manifest",
    "run_scan",
    "search_ecostress",
    # Dataclasses
    "Anchor",
    "S2Candidate",
    "CoupledPair",
    "DroppedPair",
    "ECOSTRESSMatch",
    "ScanReport",
    "ManifestResult",
]


# ── shared dataclasses ────────────────────────────────────────────────────────


@dataclass
class Anchor:
    """A Landsat scene used as coupling anchor."""

    scene_id: str
    source: Literal["landsat-c2-l2"]
    year: int
    datetime: datetime          # UTC acquisition datetime
    date: str                   # ISO date string "YYYY-MM-DD"
    cloud_cover: float | None  # eo:cloud_cover from STAC
    sun_azimuth: float | None
    sun_elevation: float | None
    item_href: str | None       # signed PC asset URL (for mode=full pipeline)


@dataclass
class S2Candidate:
    """A Sentinel-2 L2A scene within ±window_days of an anchor."""

    scene_id: str
    source: Literal["sentinel-2-l2a"]
    year: int
    datetime: datetime          # UTC acquisition datetime
    date: str                   # ISO date string "YYYY-MM-DD"
    dt_days: float             # |s2.datetime − anchor.datetime| in days
    cloud_cover: float | None
    item_href: str | None
    clear_frac: float | None = None  # computed after pixel load; None = not yet computed


@dataclass
class CoupledPair:
    """A Landsat anchor successfully coupled with one S2 scene."""

    anchor: Anchor
    s2: S2Candidate
    clear_frac: float           # fraction of clear pixels inside AOI ∩ l8_clear ∩ s2_clear
    score: float                # clear_frac − λ·(dt_days/3)
    ecostress: list[ECOSTRESSMatch]  # may be empty


@dataclass
class DroppedPair:
    """A Landsat anchor with no S2 candidate above the clear_frac threshold."""

    anchor: Anchor
    reason: str                # e.g. "no_s2_in_window" | "no_s2_above_threshold"
    max_clear_frac: float       # best clear_frac found, or 0.0 if none


@dataclass
class ECOSTRESSMatch:
    """An ECOSTRESS granule matched to a Landsat anchor day."""

    granule_id: str
    source: Literal["ecostress"]
    year: int
    datetime: datetime          # UTC acquisition datetime
    date: str                   # ISO date string
    dt_hours: float            # |ecostress.datetime − anchor.datetime| in hours (local)
    mgrs_tile: str | None
    overlap_frac: float         # fraction of Berlin bbox overlapped
    clear_frac: float | None   # fraction of cloud==0 inside Berlin; None if not computed


@dataclass
class ScanReport:
    """Volume scan result — counts + estimated data volume."""

    n_landsat_total: int
    n_landsat_may_sep: int
    n_s2_candidates: int       # sum across all anchor windows (approx)
    n_landsat_coupled: int     # estimated from median clear_frac
    n_landsat_dropped: int
    n_ecostress_matches: int
    est_landsat_gb: float
    est_s2_gb: float
    est_ecostress_gb: float
    est_total_gb: float
    metadata_json: str         # path to data/ard/scan_report.json


@dataclass
class ManifestResult:
    """Result of a full coupling run."""

    n_anchors: int
    n_coupled: int
    n_dropped: int
    n_ecostress: int
    manifest_path: str         # path to written Parquet
