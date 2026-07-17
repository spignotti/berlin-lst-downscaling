"""Szenen-Selektion & Kopplung — Landsat anchor, S2 match, ECOSTRESS subset.

Builds an ARD manifest (Parquet) per :ref:`ard-manifest-schema`:
- Landsat scenes as anchors (May–Sep, configurable years)
- Best Sentinel-2 match per anchor: score = clear_frac − λ·Δt/3
- ECOSTRESS subset only on anchor days, ±2 h local time

Usage::

    from berlin_lst_downscaling.data.selection import build_anchors, run_scan

    # smoke test
    cfg = OmegaConf.load("configs/selection/smoke_2024_mai_sep.yaml")
    anchors, anchor_stats = build_anchors(cfg)

    # volume scan
    report = run_scan(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScanReport:
    """Volume scan result — counts + estimated data volume."""

    n_landsat_total: int
    n_landsat_may_sep: int
    n_s2_candidates: int  # sum across all anchor windows (approx)
    n_landsat_coupled: int  # estimated from median clear_frac
    n_landsat_dropped: int
    n_ecostress_matches: int
    est_landsat_gb: float
    est_s2_gb: float
    est_ecostress_gb: float
    est_total_gb: float
    metadata_json: str  # path to data/ard/scan_report.json


@dataclass
class ManifestResult:
    """Result of a full coupling run."""

    n_anchors: int
    n_coupled: int
    n_dropped: int
    n_ecostress: int
    manifest_path: str  # path to written Parquet


# ── submodule imports (after dataclasses are defined to avoid circular imports) ──
# noqa: E402 — submodules return dicts; dataclasses defined here to break import cycle

from berlin_lst_downscaling.data.selection.anchors import build_anchors  # noqa: E402
from berlin_lst_downscaling.data.selection.couple import couple_all  # noqa: E402
from berlin_lst_downscaling.data.selection.ecostress import search_ecostress  # noqa: E402
from berlin_lst_downscaling.data.selection.ecostress_subset import (  # noqa: E402
    build_ecostress_subset,
)
from berlin_lst_downscaling.data.selection.manifest import write_bundle  # noqa: E402
from berlin_lst_downscaling.data.selection.s2_search import (  # noqa: E402
    match_s2_candidates,
    match_s2_candidates_with_clear_frac,
)
from berlin_lst_downscaling.data.selection.scan import run_scan  # noqa: E402

__all__ = [
    # Core entry-points
    "build_anchors",
    "match_s2_candidates",
    "match_s2_candidates_with_clear_frac",
    "couple_all",
    "build_ecostress_subset",
    "write_bundle",
    "run_scan",
    "search_ecostress",
    # Dataclasses
    "ScanReport",
    "ManifestResult",
]
