"""Geometry resolver — load carry-forward mapping and resolve per-scene geometry.

Reads the published ``geometry_mapping.json`` once per run, validates it,
then resolves the correct horizon cubes for each scene based on its year.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from hashlib import sha256

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.io.storage import exists, read_bytes

_logger = logging.getLogger(__name__)

# ── geometry mapping ────────────────────────────────────────────────────

@dataclass
class GeometryMapping:
    """Parsed and validated geometry_mapping.json."""

    uri: str
    content_hash: str
    version: str
    rule: str
    year_to_vintage: dict[int, int]
    vintages: dict[int, dict]
    building_horizons: dict[int, str]  # vintage -> horizon COG URI
    vegetation_horizon_uri: str  # fixed VH-2020 horizon

@dataclass
class SceneGeometry:
    """Resolved geometry for one scene."""

    scene_year: int
    building_vintage: int
    building_geometry_id: str
    building_horizon_uri: str
    vegetation_horizon_uri: str

@dataclass
class GeometryMappingReport:
    """Result of loading and validating the geometry mapping."""

    mapping: GeometryMapping | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mapping is not None and len(self.errors) == 0


def load_geometry_mapping(uri: str) -> GeometryMappingReport:
    """Load, parse, and validate the published geometry_mapping.json.

    Validates:
    - JSON is readable and well-formed.
    - Version field present.
    - Carry-forward rule present.
    - All years 2017–2026 are covered.
    - No future vintage is assigned to any year.
    - All referenced building horizon COGs and completion markers exist.
    - Vegetation horizon (VH-2020) exists.
    """
    errors: list[str] = []

    try:
        raw = read_bytes(uri)
    except Exception as exc:
        return GeometryMappingReport(errors=[f"Cannot read mapping: {uri}: {exc}"])

    content_hash = sha256(raw).hexdigest()[:16]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return GeometryMappingReport(errors=[f"Invalid JSON in mapping: {exc}"])

    version = data.get("version", "")
    rule = data.get("rule", "")

    if not version:
        errors.append("mapping missing 'version'")
    if not rule:
        errors.append("mapping missing 'rule'")

    ytv = data.get("year_to_vintage", {})
    vintages = data.get("vintages", {})

    # Check year coverage
    required_years = set(range(2017, 2027))
    covered_years = {int(y) for y in ytv.keys()}
    missing = required_years - covered_years
    if missing:
        errors.append(f"mapping missing years: {sorted(missing)}")

    # Check no future vintage assignment
    for year_str, vintage_str in ytv.items():
        year = int(year_str)
        vintage = int(vintage_str)
        if vintage > year:
            errors.append(f"year {year} maps to future vintage {vintage}")

    # Resolve building horizons per vintage
    building_horizons: dict[int, str] = {}
    for vintage_str, vdata in vintages.items():
        vintage = int(vintage_str)
        geom_id = vdata.get("geometry_id", "")

        # Building horizon
        horizon_uri = (
            f"gs://berlin-lst-data/static/derived/full/ard/static/derived/"
            f"horizon_building/{geom_id}/horizon_building_{geom_id}.tif"
        )
        horizon_comp = (
            f"gs://berlin-lst-data/static/derived/full/ard/static/derived/"
            f"horizon_building/{geom_id}/complete.json"
        )
        if not exists(horizon_uri) or not exists(horizon_comp):
            errors.append(f"Building horizon incomplete: vintage={vintage} geom_id={geom_id}")
        else:
            building_horizons[vintage] = horizon_uri

    # Vegetation horizon (fixed VH-2020)
    vh_geom_id = "dgm1-2021__lod2-2024__vh-2020"
    vh_uri = (
        f"gs://berlin-lst-data/static/derived/full/ard/static/derived/"
        f"horizon_vegetation/{vh_geom_id}/horizon_vegetation_{vh_geom_id}.tif"
    )
    vh_comp = (
        f"gs://berlin-lst-data/static/derived/full/ard/static/derived/"
        f"horizon_vegetation/{vh_geom_id}/complete.json"
    )
    if not exists(vh_uri) or not exists(vh_comp):
        errors.append(f"Vegetation horizon incomplete: geom_id={vh_geom_id}")

    if errors:
        return GeometryMappingReport(errors=errors)

    mapping = GeometryMapping(
        uri=uri,
        content_hash=content_hash,
        version=version,
        rule=rule,
        year_to_vintage={int(y): int(v) for y, v in ytv.items()},
        vintages={int(k): v for k, v in vintages.items()},
        building_horizons=building_horizons,
        vegetation_horizon_uri=vh_uri,
    )

    log_event(
        _logger,
        logging.INFO,
        "geometry_mapping_loaded",
        uri=uri,
        version=version,
        content_hash=content_hash,
        n_vintages=len(building_horizons),
    )

    return GeometryMappingReport(mapping=mapping)


def resolve_scene_geometry(
    scene_year: int,
    mapping: GeometryMapping,
) -> SceneGeometry:
    """Resolve geometry for a specific scene year using the carry-forward mapping.

    Raises ValueError if the scene year is not covered by the mapping.
    """
    vintage = mapping.year_to_vintage.get(scene_year)
    if vintage is None:
        raise ValueError(f"Scene year {scene_year} not covered by geometry mapping")

    vdata = mapping.vintages.get(vintage, {})
    geometry_id = vdata.get("geometry_id", "")
    building_horizon = mapping.building_horizons.get(vintage, "")

    if not building_horizon:
        raise ValueError(
            f"No building horizon for vintage {vintage} (scene year {scene_year})"
        )

    return SceneGeometry(
        scene_year=scene_year,
        building_vintage=vintage,
        building_geometry_id=geometry_id,
        building_horizon_uri=building_horizon,
        vegetation_horizon_uri=mapping.vegetation_horizon_uri,
    )

__all__ = [
    "GeometryMapping",
    "GeometryMappingReport",
    "SceneGeometry",
    "load_geometry_mapping",
    "resolve_scene_geometry",
]
