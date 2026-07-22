"""Deterministic output paths for secondary-data artefacts.

Layout
------
Pipeline A (source products):
  _raw/static/sources/{source}/{revision}/
  _staging/static/sources/{source}/{revision}/
  ard/static/sources/{source}/{revision}/
    ├─ <source>_<revision>.tif            # final COG
    ├─ <source>_<revision>.stac.json      # STAC Item metadata
    ├─ provenance.json                   # source/archive provenance
    └─ complete.json                     # publication marker (written last)

Pipeline B (derived geometry):
  ard/static/derived/{category}/{geometry_id}/
    ├─ <product>_<geometry_id>.tif
    ├─ <product>_<geometry_id>.stac.json
    ├─ provenance.json
    └─ complete.json

QA / state:
  qa/static/sources/{run_id}/report.json
  qa/static/derived/{run_id}/report.json
  _state/static/sources/ledger.parquet
  _state/static/derived/ledger.parquet
"""

from __future__ import annotations

# decision: keep per-file helpers thin (string f-strings) rather than
# a Path-based abstraction, because ``gs://`` URIs lose the double slash
# under pathlib.

_STATIC_ROOT = "ard/static"
_RAW_ROOT = "_raw/secondary"
_QA_ROOT = "qa/secondary"


def raw_dir(root: str, source: str, period: str) -> str:
    """Return the raw-data directory for a source and period.

    ``root`` is typically ``cfg.output_root`` (local or ``gs://bucket/...``).

    .. code-block:: text

        <root>/_raw/secondary/versiegelung/2021/
    """
    return f"{root.rstrip('/')}/{_RAW_ROOT}/{source}/{period}"


def static_dir(root: str, category: str, source: str, vintage: str) -> str:
    """Return the static-output directory for a source vintage.

    .. code-block:: text

        <root>/ard/static/morphology/versiegelung/2021/
    """
    return f"{root.rstrip('/')}/{_STATIC_ROOT}/{category}/{source}/{vintage}"


def product_dir(root: str, category: str, source: str, vintage: str) -> str:
    """Alias for :func:`static_dir` — the canonical product directory.

    Returns ``ard/static/{category}/{source}/{vintage}``.
    """
    return static_dir(root, category, source, vintage)


def product_cog_path(root: str, category: str, source: str, vintage: str) -> str:
    """Return the final COG URI for a product."""
    return f"{product_dir(root, category, source, vintage)}/{source}_{vintage}.tif"


def product_stac_path(root: str, category: str, source: str, vintage: str) -> str:
    """Return the final STAC Item JSON URI for a product."""
    return f"{product_dir(root, category, source, vintage)}/{source}_{vintage}.stac.json"


def product_provenance_path(
    root: str, category: str, source: str, vintage: str,
) -> str:
    """Return the final provenance.json URI for a product."""
    return f"{product_dir(root, category, source, vintage)}/provenance.json"


def product_completion_path(
    root: str, category: str, source: str, vintage: str,
) -> str:
    """Return the publication marker URI for a product.

    Written **last** after all other artifacts are in place.  Its absence
    means the product is not yet considered final by :func:`reconcile`.
    """
    return f"{product_dir(root, category, source, vintage)}/complete.json"


def qa_dir(root: str, run_id: str) -> str:
    """Return the QA report directory for a run."""
    return f"{root.rstrip('/')}/{_QA_ROOT}/{run_id}"


def qa_report_path(root: str, run_id: str) -> str:
    """Return the persisted QA report URI for a run."""
    return f"{qa_dir(root, run_id)}/report.json"


def ledger_path(root: str) -> str:
    """Return the ledger Parquet path under *root*."""
    return f"{root.rstrip('/')}/ledger.parquet"


__all__ = [
    "raw_dir",
    "static_dir",
    "product_dir",
    "product_cog_path",
    "product_stac_path",
    "product_provenance_path",
    "product_completion_path",
    "qa_dir",
    "qa_report_path",
    "ledger_path",
]


# ── Pipeline A / B helpers ──────────────────────────────────────────

_SOURCES_ROOT = "ard/static/sources"
_DERIVED_ROOT = "ard/static/derived"
_STATE_ROOT = "_state/static"


def source_product_dir(root: str, source: str, revision: str) -> str:
    """Return the Pipeline A source product directory.

    .. code-block:: text

        <root>/ard/static/sources/imperviousness/2016/
    """
    return f"{root.rstrip('/')}/{_SOURCES_ROOT}/{source}/{revision}"


def source_product_cog(root: str, source: str, revision: str) -> str:
    """Return the final COG URI for a Pipeline A source product."""
    return f"{source_product_dir(root, source, revision)}/{source}_{revision}.tif"


def derived_product_dir(root: str, product: str, geometry_id: str) -> str:
    """Return the Pipeline B derived product directory.

    .. code-block:: text

        <root>/ard/static/derived/combined_dsm/dgm1-2021__lod2-2024__vh-2020/
    """
    return f"{root.rstrip('/')}/{_DERIVED_ROOT}/{product}/{geometry_id}"


def derived_product_cog(root: str, product: str, geometry_id: str) -> str:
    """Return the final COG URI for a Pipeline B derived product."""
    return f"{derived_product_dir(root, product, geometry_id)}/{product}_{geometry_id}.tif"


def derived_ledger_path(root: str) -> str:
    """Return the Pipeline B ledger path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/derived/ledger.parquet"


def source_ledger_path(root: str) -> str:
    """Return the Pipeline A ledger path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/sources/ledger.parquet"
