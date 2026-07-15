"""Deterministic output paths for secondary-data artefacts.

Layout
------
_raw/secondary/{source}/{period}/
_staging/secondary/{source}/{run_id}/
ard/static/{category}/{source}/{vintage}/
  ├─ <source>_<vintage>.tif            # final COG
  ├─ <source>_<vintage>.stac.json      # STAC Item metadata
  ├─ provenance.json                   # source/archive provenance
  └─ complete.json                     # publication marker (written last)
ard/dynamic/meteorology/{year}/{scene_id}/   # future: scene-keyed products
qa/secondary/{run_id}/report.json
ledger.parquet

Future dynamic sources (e.g. ERA5, scene-level shadows) use the same
artifact pattern under ``ard/dynamic/meteorology/{source}/{scene_id}/``.
The product identity is ``source + scene_id`` for those; the artifact
helpers below accept a generic ``item_key`` to cover both shapes.
"""

from __future__ import annotations

# decision: keep per-file helpers thin (string f-strings) rather than
# a Path-based abstraction, because ``gs://`` URIs lose the double slash
# under pathlib.

_STATIC_ROOT = "ard/static"
_DYNAMIC_ROOT = "ard/dynamic"
_RAW_ROOT = "_raw/secondary"
_STAGING_ROOT = "_staging/secondary"
_QA_ROOT = "qa/secondary"


def raw_dir(root: str, source: str, period: str) -> str:
    """Return the raw-data directory for a source and period.

    ``root`` is typically ``cfg.output_root`` (local or ``gs://bucket/...``).

    .. code-block:: text

        <root>/_raw/secondary/versiegelung/2021/
    """
    return f"{root.rstrip('/')}/{_RAW_ROOT}/{source}/{period}"


def staging_dir(root: str, source: str, run_id: str) -> str:
    """Return the staging directory for a source and run.

    .. code-block:: text

        <root>/_staging/secondary/versiegelung/run_abc/
    """
    return f"{root.rstrip('/')}/{_STAGING_ROOT}/{source}/{run_id}"


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


def dynamic_dir(root: str, source: str, scene_id: str) -> str:
    """Return the directory for a scene-keyed dynamic product.

    Used for future dynamic sources (ERA5, scene-level shadows).
    Currently not consumed by the active runners.
    """
    return f"{root.rstrip('/')}/{_DYNAMIC_ROOT}/{source}/{scene_id}"


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
    "staging_dir",
    "static_dir",
    "product_dir",
    "product_cog_path",
    "product_stac_path",
    "product_provenance_path",
    "product_completion_path",
    "dynamic_dir",
    "qa_dir",
    "qa_report_path",
    "ledger_path",
]
