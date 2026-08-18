"""Feature-schema fingerprinting — config hash for scene feature stacks.

The config hash ties every published stack to the exact set of inputs and
processing semantics: manifest identity, geometry mapping, ledger
fingerprints, the AOI mask, the channel schema, and the vegetation
carry-forward policy. Any change invalidates the published stacks via the
``config_changed`` reconcile path.
"""

from __future__ import annotations

import json
from hashlib import sha256

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION,
)


def config_hash_for_features(
    *,
    manifest_hash: str,
    geometry_mapping_hash: str,
    ard_ledger_hash: str,
    static_derived_ledger_hash: str,
    dynamic_ledger_hash: str,
    aoi_fingerprint: str,
    vegetation_carry_forward_geometry_id: str,
) -> str:
    """Return a stable SHA-256 fingerprint of the feature-stack config.

    Covers everything that changes the stack output: upstream ledger
    identities, the exact AOI mask, the vegetation carry-forward target,
    and the channel schema (names + formula version). The vegetation
    policy is encoded by its geometry id so changing the carry-forward
    vintage reprocesses every scene.
    """
    payload = json.dumps(
        {
            "channel_order": list(FEATURE_CHANNEL_NAMES),
            "schema_version": FEATURE_SCHEMA_VERSION,
            "manifest_hash": manifest_hash,
            "geometry_mapping_hash": geometry_mapping_hash,
            "ard_ledger_hash": ard_ledger_hash,
            "static_derived_ledger_hash": static_derived_ledger_hash,
            "dynamic_ledger_hash": dynamic_ledger_hash,
            "aoi_fingerprint": aoi_fingerprint,
            "vegetation_carry_forward_geometry_id": vegetation_carry_forward_geometry_id,
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]


__all__ = ["config_hash_for_features"]
