"""Configuration fingerprinting for dynamic scene products.

Every dynamic product carries a config hash that ties the output to the
exact set of inputs: manifest hash, geometry ID, channel definitions,
and temporal mode.
"""

from __future__ import annotations

import json
from hashlib import sha256

# ── channel definitions ──────────────────────────────────────────────

ERA5_CHANNELS = ("t2m_scene", "ssrd_scene", "ssrd_antecedent_72h_mean")
SHADOW_CHANNELS = ("shadow_building", "shadow_vegetation")

# ── geometry temporal mode ────────────────────────────────────────────

GEOMETRY_TEMPORAL_MODE = "retrospective_static"
# LoD2-2024, DGM-2021, vegetation height-2020
GEOMETRY_VINTAGES = {
    "lod2_morphology": "2024",
    "terrain_height": "2021",
    "vegetation_height": "2020",
}

def config_hash_for_dynamic(
    manifest_hash: str,
    geometry_id: str,
    era5_cache_root: str,
    antecedent_hours: int = 72,
) -> str:
    """Return a stable SHA-256 fingerprint of the dynamic config.

    Covers all parameters that change the dynamic product output:
    manifest identity, geometry version, ERA5 channel set, and
    antecedent window.
    """
    payload = json.dumps(
        {
            "manifest_hash": manifest_hash,
            "geometry_id": geometry_id,
            "era5_channels": list(ERA5_CHANNELS),
            "shadow_channels": list(SHADOW_CHANNELS),
            "era5_cache_root": era5_cache_root,
            "antecedent_hours": antecedent_hours,
            "geometry_temporal_mode": GEOMETRY_TEMPORAL_MODE,
            "geometry_vintages": GEOMETRY_VINTAGES,
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]

def config_hash_for_era5(
    manifest_hash: str,
    geometry_id: str,
    era5_cache_root: str,
    antecedent_hours: int = 72,
) -> str:
    """Config hash for ERA5 scene products specifically.

    Includes processing parameters that affect output values (CDS area order,
    grid resolution, cell selection strategy). Shadow products use a separate
    hash and are unaffected.
    """
    payload = json.dumps(
        {
            "manifest_hash": manifest_hash,
            "geometry_id": geometry_id,
            "era5_channels": list(ERA5_CHANNELS),
            "era5_cache_root": era5_cache_root,
            "antecedent_hours": antecedent_hours,
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]

__all__ = [
    "ERA5_CHANNELS",
    "SHADOW_CHANNELS",
    "GEOMETRY_TEMPORAL_MODE",
    "GEOMETRY_VINTAGES",
    "config_hash_for_dynamic",
    "config_hash_for_era5",
]