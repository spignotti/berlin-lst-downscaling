"""Deterministic output paths for secondary-data artefacts.

Layout
------
_raw/secondary/{source}/{period}/
_staging/secondary/{source}/{run_id}/
ard/static/{category}/{source}/{vintage}/
ard/dynamic/meteorology/{year}/{scene_id}/
qa/secondary/{run_id}/
"""

from __future__ import annotations


def raw_dir(root: str, source: str, period: str) -> str:
    """Return the raw-data directory for a source and period.

    ``root`` is typically ``cfg.output_root`` (local or ``gs://bucket/...``).

    .. code-block:: text

        <root>/_raw/secondary/versiegelung/2021/
    """
    return f"{root.rstrip('/')}/_raw/secondary/{source}/{period}"


def staging_dir(root: str, source: str, run_id: str) -> str:
    """Return the staging directory for a source and run.

    .. code-block:: text

        <root>/_staging/secondary/versiegelung/run_abc/
    """
    return f"{root.rstrip('/')}/_staging/secondary/{source}/{run_id}"


def static_dir(root: str, category: str, source: str, vintage: str) -> str:
    """Return the static-output directory for a source vintage.

    .. code-block:: text

        <root>/ard/static/morphology/versiegelung/2021/
    """
    return f"{root.rstrip('/')}/ard/static/{category}/{source}/{vintage}"


def qa_dir(root: str, run_id: str) -> str:
    """Return the QA report directory for a run.

    .. code-block:: text

        <root>/qa/secondary/run_abc/
    """
    return f"{root.rstrip('/')}/qa/secondary/{run_id}"


def ledger_path(root: str) -> str:
    """Return the ledger Parquet path under *root*."""
    return f"{root.rstrip('/')}/ledger.parquet"


__all__ = [
    "raw_dir",
    "staging_dir",
    "static_dir",
    "qa_dir",
    "ledger_path",
]
