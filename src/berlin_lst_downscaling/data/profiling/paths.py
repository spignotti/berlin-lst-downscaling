"""Deterministic output paths for WB2c-1 profiling artifacts.

Layout
------
Profiling root (fixed, overwrite):
  gs://berlin-lst-data/profiling/wb2c-1/

Artifacts:
  profiles.parquet    — per-asset profiling results
  profiles.csv        — same content, human-readable
  summary.json        — aggregated summary (written last)
  notion-summary.md   — condensed Notion-ready summary
"""

from __future__ import annotations

# ── layout constants ──────────────────────────────────────────────────

_PROFILING_ROOT = "gs://berlin-lst-data/profiling/wb2c-1"

# ── artifact paths ────────────────────────────────────────────────────


def profiles_parquet_path(root: str = _PROFILING_ROOT) -> str:
    """Return the URI of the profiles Parquet file."""
    return f"{root.rstrip('/')}/profiles.parquet"


def profiles_csv_path(root: str = _PROFILING_ROOT) -> str:
    """Return the URI of the profiles CSV file."""
    return f"{root.rstrip('/')}/profiles.csv"


def summary_json_path(root: str = _PROFILING_ROOT) -> str:
    """Return the URI of the summary JSON file (written last)."""
    return f"{root.rstrip('/')}/summary.json"


def notion_summary_path(root: str = _PROFILING_ROOT) -> str:
    """Return the URI of the Notion-ready summary Markdown file."""
    return f"{root.rstrip('/')}/notion-summary.md"


__all__ = [
    "profiles_parquet_path",
    "profiles_csv_path",
    "summary_json_path",
    "notion_summary_path",
]
