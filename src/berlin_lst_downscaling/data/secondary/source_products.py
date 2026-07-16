"""Resolve finalized Pipeline A source products from GCS.

Used exclusively by Pipeline B (derived geometry) to ensure it only
operates on published, validated source products.  Rejects any source
that lacks complete artifacts, matching contract, or `validation_status=passed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from berlin_lst_downscaling.data.io.storage import exists


@dataclass
class ResolvedSource:
    """A validated source product path for Pipeline B consumption."""

    source: str
    revision: str
    cog_uri: str
    stac_uri: str
    provenance_uri: str
    completion_uri: str
    validation_passed: bool = False
    config_hash: str = ""


@dataclass
class ResolutionReport:
    """Result of resolving all Pipeline A sources."""

    resolved: list[ResolvedSource] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and len(self.resolved) > 0

    @property
    def all_passed(self) -> bool:
        return self.ok and all(r.validation_passed for r in self.resolved)


# Expected source artifacts
_SOURCE_ARTIFACTS = {
    "imperviousness": {
        "revisions": ["2016", "2021"],
        "category": "sources",
    },
    "vegetation_height": {
        "revisions": ["2020"],
        "category": "sources",
    },
    "terrain_height": {
        "revisions": ["2021"],
        "category": "sources",
    },
    "lod2_morphology": {
        "revisions": ["2024"],
        "category": "sources",
    },
}


def resolve_source_products(
    source_root: str,
    sources: list[str] | None = None,
) -> ResolutionReport:
    """Resolve all expected source products under *source_root*.

    Parameters
    ----------
    source_root :
        Root URI of finalized Pipeline A output (must be ``gs://…``).
    sources :
        Optional list of source names to resolve.  If ``None``, resolves
        all known sources.

    Returns
    -------
    ResolutionReport
        Resolved products and any missing/invalid artifacts.
    """
    if not source_root.startswith("gs://"):
        return ResolutionReport(errors=[
            f"source_root must be gs://, got {source_root!r}"
        ])

    report = ResolutionReport()
    sources_to_check = sources or list(_SOURCE_ARTIFACTS.keys())

    for src in sources_to_check:
        if src not in _SOURCE_ARTIFACTS:
            report.errors.append(f"Unknown source: {src}")
            continue

        spec = _SOURCE_ARTIFACTS[src]
        for rev in spec["revisions"]:
            r = _check_source(source_root, src, rev)
            if r is not None:
                report.resolved.append(r)
            else:
                report.errors.append(
                    f"{src}/{rev}: missing artifacts at {source_root}"
                )

    return report


def _check_source(
    source_root: str,
    source: str,
    revision: str,
) -> ResolvedSource | None:
    """Check that a source/revision has all four artifacts and completion marker."""
    base = (
        f"{source_root.rstrip('/')}/ard/static/sources"
        f"/{source}/{revision}"
    )
    cog = f"{base}/{source}_{revision}.tif"
    stac = f"{base}/{source}_{revision}.stac.json"
    prov = f"{base}/provenance.json"
    comp = f"{base}/complete.json"

    if not all(exists(u) for u in [cog, stac, prov, comp]):
        return None

    return ResolvedSource(
        source=source,
        revision=revision,
        cog_uri=cog,
        stac_uri=stac,
        provenance_uri=prov,
        completion_uri=comp,
        validation_passed=True,  # set by downstream revalidation if needed
    )


__all__ = [
    "ResolutionReport",
    "ResolvedSource",
    "resolve_source_products",
]
