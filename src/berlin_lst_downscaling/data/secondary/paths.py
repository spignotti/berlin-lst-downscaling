"""Deterministic output paths for secondary-data artefacts.

Pipeline A (source products) layout::

    ard/static/sources/{source}/{revision}/
        ├─ <source>_<revision>.tif            # final COG
        ├─ <source>_<revision>.stac.json      # STAC Item metadata
        ├─ provenance.json
        └─ complete.json                      # written last

Pipeline B (derived geometry) layout::

    ard/static/derived/{product}/{geometry_id}/
        ├─ <product>_<geometry_id>.tif
        ├─ <product>_<geometry_id>.stac.json
        ├─ provenance.json
        └─ complete.json
"""

from __future__ import annotations

# decision: keep per-file helpers thin (string f-strings) rather than a
# Path-based abstraction, because ``gs://`` URIs lose the double slash
# under pathlib.

_RAW_ROOT = "_raw/secondary"
_SOURCES_ROOT = "ard/static/sources"
_DERIVED_ROOT = "ard/static/derived"
_STATE_ROOT = "_state/static"

def raw_dir(root: str, source: str, period: str) -> str:
    """Return the raw staging directory for a source and revision/period."""
    return f"{root.rstrip('/')}/{_RAW_ROOT}/{source}/{period}"

def source_product_dir(root: str, source: str, revision: str) -> str:
    """Return the Pipeline A source product directory."""
    return f"{root.rstrip('/')}/{_SOURCES_ROOT}/{source}/{revision}"

def source_product_cog(root: str, source: str, revision: str) -> str:
    """Return the final COG URI for a Pipeline A source product."""
    return f"{source_product_dir(root, source, revision)}/{source}_{revision}.tif"

def derived_product_dir(root: str, product: str, geometry_id: str) -> str:
    """Return the Pipeline B derived product directory."""
    return f"{root.rstrip('/')}/{_DERIVED_ROOT}/{product}/{geometry_id}"

def derived_product_cog(root: str, product: str, geometry_id: str) -> str:
    """Return the final COG URI for a Pipeline B derived product."""
    return f"{derived_product_dir(root, product, geometry_id)}/{product}_{geometry_id}.tif"

def derived_ledger_path(root: str) -> str:
    """Return the Pipeline B ledger path."""
    return f"{root.rstrip('/')}/{_STATE_ROOT}/derived/ledger.parquet"

__all__ = [
    "raw_dir",
    "source_product_dir",
    "source_product_cog",
    "derived_product_dir",
    "derived_product_cog",
    "derived_ledger_path",
]