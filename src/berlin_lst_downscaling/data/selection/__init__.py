"""Szenen-Selektion & Kopplung — Landsat anchor + Sentinel-2 matching.

Builds the canonical v3 manifest bundle:

- ``manifest.parquet`` — one unique executable scene per row.
- ``pairings.parquet`` — one Landsat→Sentinel-2 relation per anchor.
- ``manifest_report.json`` — publication gate with hashes, counts, and policy.
"""
from __future__ import annotations

from berlin_lst_downscaling.data.selection.anchors import build_anchors
from berlin_lst_downscaling.data.selection.couple import couple_all
from berlin_lst_downscaling.data.selection.ecostress import search_ecostress
from berlin_lst_downscaling.data.selection.manifest import write_bundle
from berlin_lst_downscaling.data.selection.s2_search import (
    match_s2_candidates,
    match_s2_candidates_with_clear_frac,
)

__all__ = [
    "build_anchors",
    "couple_all",
    "match_s2_candidates",
    "match_s2_candidates_with_clear_frac",
    "search_ecostress",
    "write_bundle",
]
