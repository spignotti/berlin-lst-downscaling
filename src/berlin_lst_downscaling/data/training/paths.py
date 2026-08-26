"""Deterministic output paths for WB2c-4 training-data artifacts.

Layout (published under the canonical root, e.g. ``training/v1``)::

    <root>/<scene_id>/
        ├─ <scene_id>.training_eligible_100m.tif   # uint8 0/1 eligibility COG
        ├─ provenance.json                         # policy, inputs, counts
        └─ complete.json                           # publication marker

Top-level release artifacts (written last, after every scene):
    <root>/manifest.parquet, manifest.csv          # per-scene manifest
    <root>/cells.parquet, cells.csv                # eligible-cell index
    <root>/scaler.json                             # train-only scaler
    <root>/complete.json                           # release completion marker

Ledger:
    <root>/_state/training/ledger.parquet

Run QA reports:
    <root>/qa/training/<run_id>/report.json
"""

from __future__ import annotations

# decision: thin f-string helpers (like data/features/paths.py) — pathlib
# strips the ``gs://`` double slash.

_STATE_ROOT = "_state/training"
_QA_ROOT = "qa/training"

_MASK_NAME = "training_eligible_100m"


def training_scene_dir(root: str, scene_id: str) -> str:
    """Return the product directory for one scene's training artifacts."""
    return f"{root.rstrip('/')}/{scene_id}"


def eligibility_cog(root: str, scene_id: str) -> str:
    """Return the 100 m eligibility-mask COG URI for a scene."""
    return f"{training_scene_dir(root, scene_id)}/{scene_id}.{_MASK_NAME}.tif"


def eligibility_provenance(root: str, scene_id: str) -> str:
    """Return the provenance URI for a scene's eligibility artifact."""
    return f"{training_scene_dir(root, scene_id)}/provenance.json"


def eligibility_completion(root: str, scene_id: str) -> str:
    """Return the per-scene completion-marker URI."""
    return f"{training_scene_dir(root, scene_id)}/complete.json"


def ledger_path(root: str) -> str:
    """Return the training ledger Parquet path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/ledger.parquet"


def manifest_parquet(root: str) -> str:
    """Return the scene-manifest Parquet URI."""
    return f"{root.rstrip('/')}/manifest.parquet"


def manifest_csv(root: str) -> str:
    """Return the scene-manifest CSV URI."""
    return f"{root.rstrip('/')}/manifest.csv"


def cells_parquet(root: str) -> str:
    """Return the eligible-cell index Parquet URI."""
    return f"{root.rstrip('/')}/cells.parquet"


def cells_csv(root: str) -> str:
    """Return the eligible-cell index CSV URI."""
    return f"{root.rstrip('/')}/cells.csv"


def scaler_json(root: str) -> str:
    """Return the scaler JSON URI."""
    return f"{root.rstrip('/')}/scaler.json"


def release_completion(root: str) -> str:
    """Return the top-level release completion-marker URI."""
    return f"{root.rstrip('/')}/complete.json"


def qa_report_dir(root: str, run_id: str) -> str:
    """Return the QA report directory for a training run."""
    return f"{root.rstrip('/')}/{_QA_ROOT}/{run_id}"


def qa_report_path(root: str, run_id: str) -> str:
    """Return the persisted QA report URI for a training run."""
    return f"{qa_report_dir(root, run_id)}/report.json"


__all__ = [
    "cells_csv",
    "cells_parquet",
    "eligibility_cog",
    "eligibility_completion",
    "eligibility_provenance",
    "ledger_path",
    "manifest_csv",
    "manifest_parquet",
    "qa_report_dir",
    "qa_report_path",
    "release_completion",
    "scaler_json",
    "training_scene_dir",
]
