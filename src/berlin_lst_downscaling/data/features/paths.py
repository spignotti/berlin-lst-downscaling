"""Deterministic output paths for scene feature stacks.

Layout (published under the canonical root, e.g. ``features/v1``)::

    <root>/<scene_id>/
        ├─ <scene_id>.tif                 # 28-band float32 feature COG
        ├─ <scene_id>.feature_valid.tif   # uint8 0/1 validity mask COG
        ├─ <scene_id>.stac.json           # STAC Item (data + mask assets)
        ├─ provenance.json                # source/transform provenance
        └─ complete.json                  # publication marker (written last)

Ledger:
    <root>/_state/features/ledger.parquet

Run QA reports:
    <root>/qa/features/<run_id>/report.json
"""

from __future__ import annotations

# decision: thin f-string helpers (like data/secondary/paths.py) — pathlib
# strips the ``gs://`` double slash.

_STATE_ROOT = "_state/features"
_QA_ROOT = "qa/features"


def scene_product_dir(root: str, scene_id: str) -> str:
    """Return the product directory for one feature stack."""
    return f"{root.rstrip('/')}/{scene_id}"


def feature_cog(root: str, scene_id: str) -> str:
    """Return the 28-band feature COG URI."""
    return f"{scene_product_dir(root, scene_id)}/{scene_id}.tif"


def feature_mask_cog(root: str, scene_id: str) -> str:
    """Return the feature_valid mask COG URI."""
    return f"{scene_product_dir(root, scene_id)}/{scene_id}.feature_valid.tif"


def feature_stac(root: str, scene_id: str) -> str:
    """Return the STAC Item URI for a feature stack."""
    return f"{scene_product_dir(root, scene_id)}/{scene_id}.stac.json"


def feature_provenance(root: str, scene_id: str) -> str:
    """Return the provenance URI for a feature stack."""
    return f"{scene_product_dir(root, scene_id)}/provenance.json"


def feature_completion(root: str, scene_id: str) -> str:
    """Return the completion-marker URI for a feature stack."""
    return f"{scene_product_dir(root, scene_id)}/complete.json"


def ledger_path(root: str) -> str:
    """Return the features ledger Parquet path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/ledger.parquet"


def qa_report_dir(root: str, run_id: str) -> str:
    """Return the QA report directory for a features run."""
    return f"{root.rstrip('/')}/{_QA_ROOT}/{run_id}"


def qa_report_path(root: str, run_id: str) -> str:
    """Return the persisted QA report URI for a features run."""
    return f"{qa_report_dir(root, run_id)}/report.json"


__all__ = [
    "feature_cog",
    "feature_completion",
    "feature_mask_cog",
    "feature_provenance",
    "feature_stac",
    "ledger_path",
    "qa_report_dir",
    "qa_report_path",
    "scene_product_dir",
]
