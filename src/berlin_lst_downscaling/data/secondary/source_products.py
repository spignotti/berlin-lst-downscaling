"""Resolve finalized Pipeline A source products.

Used exclusively by Pipeline B (derived geometry) to ensure it only
operates on published, validated source products.  Rejects any source
that lacks complete artifacts (COG + STAC + provenance + completion marker).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from berlin_lst_downscaling.data.io.storage import exists, read_bytes


@dataclass
class ResolvedSource:
    """A validated source product path for Pipeline B consumption."""

    source: str
    revision: str
    cog_uri: str
    stac_uri: str
    provenance_uri: str
    completion_uri: str
    config_hash: str


@dataclass
class ResolutionReport:
    """Result of resolving all Pipeline A sources."""

    resolved: list[ResolvedSource] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and len(self.resolved) > 0


# Sources consumed by Pipeline B — only geometry-relevant upstreams.
_REQUIRED_SOURCES: dict[str, list[str]] = {
    "terrain_height": ["2021"],
    "vegetation_height": ["2020"],
    "lod2_morphology": ["2024"],
}


def resolve_source_products(
    source_root: str,
    sources: dict[str, str] | None = None,
) -> ResolutionReport:
    """Resolve required source products under *source_root*.

    Parameters
    ----------
    source_root :
        Root URI of finalized Pipeline A output (local or ``gs://…``).
    sources :
        Optional ``{source: revision}`` mapping.  If ``None``, resolves
        all required sources.
    """
    report = ResolutionReport()
    to_check = sources or _REQUIRED_SOURCES

    for src, revisions in to_check.items():
        for rev in revisions if isinstance(revisions, list) else [revisions]:
            r = _check_source(source_root, src, rev)
            if r is not None:
                report.resolved.append(r)
            else:
                report.errors.append(f"{src}/{rev}: missing artifacts at {source_root}")

    return report


def _check_source(
    source_root: str,
    source: str,
    revision: str,
) -> ResolvedSource | None:
    """Check that a source/revision has all four artifacts."""
    base = f"{source_root.rstrip('/')}/ard/static/sources/{source}/{revision}"
    cog = f"{base}/{source}_{revision}.tif"
    stac = f"{base}/{source}_{revision}.stac.json"
    prov = f"{base}/provenance.json"
    comp = f"{base}/complete.json"

    if not all(exists(u) for u in [cog, stac, prov, comp]):
        return None

    prov_data = json.loads(read_bytes(prov))
    config_hash = prov_data.get("config_hash", "")
    if not config_hash:
        return None

    return ResolvedSource(
        source=source,
        revision=revision,
        cog_uri=cog,
        stac_uri=stac,
        provenance_uri=prov,
        completion_uri=comp,
        config_hash=config_hash,
    )


__all__ = [
    "ResolutionReport",
    "ResolvedSource",
    "resolve_source_products",
]
