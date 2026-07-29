"""Configuration fingerprinting for dynamic scene products.

Every dynamic product carries a config hash that ties the output to the
exact set of inputs: manifest hash, geometry mapping hash, channel definitions,
and temporal mode.
"""

from __future__ import annotations

import json
from hashlib import sha256

# ── channel definitions ──────────────────────────────────────────────

ERA5_CHANNELS = (
    "t2m_scene",
    "ssrd_scene",
    "ssrd_antecedent_72h_mean",
    "vpd_scene",
    "wind_speed_10m_scene",
    "tp_0_24h",
    "tp_24_48h",
    "tp_48_72h",
)
SHADOW_CHANNELS = ("shadow_building", "shadow_vegetation")


def config_hash_for_dynamic(
    manifest_hash: str,
    geometry_mapping_hash: str,
    era5_cache_root: str,
    antecedent_hours: int = 72,
) -> str:
    """Return a stable SHA-256 fingerprint of the dynamic config.

    Covers all parameters that change the dynamic product output:
    manifest identity, geometry mapping version, ERA5 channel set, and
    antecedent window.
    """
    payload = json.dumps(
        {
            "manifest_hash": manifest_hash,
            "geometry_mapping_hash": geometry_mapping_hash,
            "era5_channels": list(ERA5_CHANNELS),
            "shadow_channels": list(SHADOW_CHANNELS),
            "era5_cache_root": era5_cache_root,
            "antecedent_hours": antecedent_hours,
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]

def config_hash_for_era5(
    manifest_hash: str,
    geometry_mapping_hash: str,
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
            "geometry_mapping_hash": geometry_mapping_hash,
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
    "config_hash_for_dynamic",
    "config_hash_for_era5",
]
