"""Geometry resolver — validate and resolve static geometry products for shadows.

For dynamic shadow computation, the existing horizon cubes (36-band building
and vegetation horizons) plus the component DSMs must be validated as
published artifacts.  This module resolves them by geometry_id and validates
their completeness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from berlin_lst_downscaling.data.io.storage import exists, read_bytes


@dataclass
class ResolvedGeometry:
    """Validated static geometry products for shadow computation."""

    geometry_id: str
    # Source product COGs
    terrain_cog: str
    vegetation_height_cog: str
    lod2_cog: str
    # Derived product COGs
    building_dsm_cog: str
    vegetation_dsm_cog: str
    combined_dsm_cog: str
    # Horizon cubes
    horizon_building_cog: str
    horizon_vegetation_cog: str
    # Upstream hashes (from provenance)
    terrain_hash: str
    vh_hash: str
    lod2_hash: str


@dataclass
class GeometryResolutionReport:
    """Result of resolving geometry artifacts."""

    resolved: ResolvedGeometry | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.resolved is not None and len(self.errors) == 0


def resolve_geometry(
    source_root: str,
    derived_root: str,
    geometry_id: str,
) -> GeometryResolutionReport:
    """Resolve and validate all static geometry products.

    Parameters
    ----------
    source_root :
        Root URI of finalized Pipeline A output (local or ``gs://``).
    derived_root :
        Root URI of finalized Pipeline B output (local or ``gs://``).
    geometry_id :
        Frozen geometry key, e.g. ``"dgm1-2021__lod2-2024__vh-2020"``.
    """
    errors: list[str] = []

    # ── source products ──────────────────────────────────────────────
    src_base = f"{source_root.rstrip('/')}/ard/static/sources"
    src_products = {
        "terrain_height": ("terrain_height", "2021"),
        "vegetation_height": ("vegetation_height", "2020"),
        "lod2_morphology": ("lod2_morphology", "2024"),
    }

    src_uris: dict[str, str] = {}
    src_hashes: dict[str, str] = {}

    for name, (source, revision) in src_products.items():
        cog = f"{src_base}/{source}/{revision}/{source}_{revision}.tif"
        prov = f"{src_base}/{source}/{revision}/provenance.json"
        comp = f"{src_base}/{source}/{revision}/complete.json"

        if not all(exists(u) for u in [cog, prov, comp]):
            errors.append(f"Source product incomplete: {source}/{revision}")
            continue

        src_uris[name] = cog

        # Source products are required to carry a non-empty config_hash.
        prov_data = json.loads(read_bytes(prov))
        config_hash = prov_data.get("config_hash", "")
        if not config_hash:
            errors.append(f"Source product missing config_hash: {source}/{revision}")
            continue
        src_hashes[name] = config_hash

    if src_uris.get("terrain_height") is None:
        errors.append("terrain_height required for grid inference")

    # ── derived products ─────────────────────────────────────────────
    derived_base = f"{derived_root.rstrip('/')}/ard/static/derived"
    derived_products = [
        "building_dsm",
        "vegetation_dsm",
        "combined_dsm",
        "horizon_building",
        "horizon_vegetation",
    ]

    derived_uris: dict[str, str] = {}
    for name in derived_products:
        cog = f"{derived_base}/{name}/{geometry_id}/{name}_{geometry_id}.tif"
        comp = f"{derived_base}/{name}/{geometry_id}/complete.json"

        if not all(exists(u) for u in [cog, comp]):
            errors.append(f"Derived product incomplete: {name}/{geometry_id}")
            continue

        derived_uris[name] = cog

    # ── build resolved report ────────────────────────────────────────
    if errors:
        return GeometryResolutionReport(errors=errors)

    resolved = ResolvedGeometry(
        geometry_id=geometry_id,
        terrain_cog=src_uris["terrain_height"],
        vegetation_height_cog=src_uris["vegetation_height"],
        lod2_cog=src_uris["lod2_morphology"],
        building_dsm_cog=derived_uris["building_dsm"],
        vegetation_dsm_cog=derived_uris["vegetation_dsm"],
        combined_dsm_cog=derived_uris["combined_dsm"],
        horizon_building_cog=derived_uris["horizon_building"],
        horizon_vegetation_cog=derived_uris["horizon_vegetation"],
        terrain_hash=src_hashes.get("terrain_height", ""),
        vh_hash=src_hashes.get("vegetation_height", ""),
        lod2_hash=src_hashes.get("lod2_morphology", ""),
    )

    return GeometryResolutionReport(resolved=resolved, errors=errors)


__all__ = [
    "ResolvedGeometry",
    "GeometryResolutionReport",
    "resolve_geometry",
]
