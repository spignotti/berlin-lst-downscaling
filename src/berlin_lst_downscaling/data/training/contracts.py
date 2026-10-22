"""WB2c-4 training-data contract — splits, support rule, cell IDs, scaler policy.

The training handoff is defined by a small set of user-mandated policy
decisions plus the immutable Feature Release V3 input basis:

- **Temporal split contract:** 2017-2023 train, 2024 validation, 2025
  test, 2026 inference (metadata-only, deferred).
- **Eligibility rule:** a 100 m cell is ``training_eligible`` iff the
  Landsat target cell is valid **and** all 100 of its 10 m feature
  subpixels are ``feature_valid`` (strict 100/100 support; edge-truncated
  cells can never be eligible).
- **Stable cell identity:** a spatial ID derived deterministically from
  the canonical EPSG:25833 100 m grid (origin ``(369190, 5838410)``),
  independent of any scene.
- **Scaler policy:** z-score for continuous channels, log1p then z-score
  for the three precipitation channels, identity (0/1) for shadows; fit
  exclusively on train-split eligible cells.

Everything below is hashed into ``training_policy_hash`` so any policy
change invalidates the published artifacts via the ledger's
``config_changed`` reconcile path.
"""

from __future__ import annotations

import json
from hashlib import sha256

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_CHANNELS,
    FEATURE_SCHEMA_VERSION,
    FeatureSchema,
    feature_schema,
)

# ── schema version ────────────────────────────────────────────────────

# Bump when the training-data contract (splits, eligibility rule, cell-ID
# formula, scaler policy) changes in a way that invalidates published
# artifacts. The value feeds the policy hash.
TRAINING_SCHEMA_VERSION: int = 1

# ── immutable input basis ─────────────────────────────────────────────

# Pinned feature-release hash. The name stays EXPECTED_V3_CONFIG_HASH.
# The value is feature release v5 (area-mean Landsat ARD, imperviousness v3).
# Feature releases v1 and v2 stay rejected.
EXPECTED_V3_CONFIG_HASH: str = "2d2086f9d36e48fe"

# ── temporal split contract (user-mandated) ───────────────────────────

# 2017-2023 train, 2024 validation, 2025 test, 2026 inference.
_TRAIN_YEARS = tuple(range(2017, 2024))
SPLIT_BY_YEAR: dict[int, str] = {
    **{year: "train" for year in _TRAIN_YEARS},
    2024: "validation",
    2025: "test",
    2026: "inference",
}

# ── eligibility rule ──────────────────────────────────────────────────

# Strict support: 100 of 100 feature subpixels must be valid (user-mandated).
SUPPORT_PIXELS: int = 100

# ── exclusion reasons ─────────────────────────────────────────────────

# Scenes with zero training-eligible cells are excluded with this reason
# (no sparse category, no percentage threshold — user-mandated).
NO_ELIGIBLE_CELLS_REASON: str = "no_eligible_cells"

# 2026 scenes are metadata-only: no feature materialisation, no eligibility
# mask, no cells, no scaler contribution (user decision, 2026-08-26).
INFERENCE_DEFERRED_REASON: str = "inference_deferred"

# USGS surface temperature on these two 27 Aug 2022 path-193 scenes is
# physically implausible for that day (clear-pixel means below 280 K while
# the air temperature was about 19–24 °C). They stay out of the training
# release even where a patch would otherwise clear the support gate.
IMPLAUSIBLE_LST_REASON: str = "usgs_st_implausible_20220827"
IMPLAUSIBLE_LST_SCENE_IDS: frozenset[str] = frozenset(
    {
        "LC09_L2SP_193023_20220827_02_T2",
        "LC09_L2SP_193024_20220827_02_T2",
    }
)

# ── canonical 100 m grid (EPSG:25833) ─────────────────────────────────

# Origin of the canonical nested grid at 100 m (common/grid.py). The
# eligibility mask is written on this lattice; a bbox-snapped analysis
# grid stays on the same lattice, so global row/col are well-defined.
CANON_GRID_ORIGIN_X: float = 369190.0
CANON_GRID_ORIGIN_Y: float = 5838410.0
CELL_SIZE_M: int = 100


def cell_id(row: int, col: int) -> str:
    """Return the stable spatial cell ID for a **global** canonical 100 m cell.

    ``row``/``col`` are indices into the canonical EPSG:25833 100 m grid
    (not analysis-relative). The ID encodes the cell's top-left corner
    easting/northing, so it is deterministic, scene-independent, and
    reproducible from the canonical grid alone.
    """
    easting = CANON_GRID_ORIGIN_X + col * CELL_SIZE_M
    northing = CANON_GRID_ORIGIN_Y - row * CELL_SIZE_M
    return f"E{int(easting)}N{int(northing)}"


# ── scaler policy (user-mandated, applied in WB2c-4 P2) ───────────────

# The three ERA5 precipitation channels: log1p then z-score.
PRECIP_CHANNELS: tuple[str, ...] = ("tp_0_24h", "tp_24_48h", "tp_48_72h")

# Shadow channels are 0/1 and stay unchanged.
SHADOW_CHANNELS: tuple[str, ...] = ("shadow_building", "shadow_vegetation")


def precip_channels_for(schema: FeatureSchema) -> tuple[str, ...]:
    """Precipitation channels for a feature roster.

    features/v5 keeps the three windows. features/v6 has the single
    ``tp_0_72h`` sum, still ``log1p`` then z-score.
    """
    if schema.version == FEATURE_SCHEMA_VERSION:
        return PRECIP_CHANNELS
    return tuple(name for name in schema.names if name == "tp_0_72h" or name.startswith("tp_"))


def shadow_channels_for(schema: FeatureSchema) -> tuple[str, ...]:
    """Shadow channels of a roster. v5 returns the published constant."""
    if schema.version == FEATURE_SCHEMA_VERSION:
        return SHADOW_CHANNELS
    return tuple(name for name in schema.names if name.startswith("shadow_"))


def scaler_transform_for(channel_name: str, schema: FeatureSchema | None = None) -> str:
    """Return ``zscore``, ``log1p_zscore``, or ``identity`` for one channel."""
    resolved = schema or feature_schema(FEATURE_SCHEMA_VERSION)
    if channel_name in precip_channels_for(resolved):
        return "log1p_zscore"
    if channel_name in shadow_channels_for(resolved):
        return "identity"
    return "zscore"


# ── policy hash ───────────────────────────────────────────────────────


def split_for_year(year: int) -> str:
    """Return the temporal split for a scene year (train/validation/test/inference)."""
    split = SPLIT_BY_YEAR.get(year)
    if split is None:
        raise ValueError(f"year {year} outside the temporal contract (2017-2026)")
    return split


def training_policy_hash(
    *,
    v3_config_hash: str = EXPECTED_V3_CONFIG_HASH,
    feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    schema: FeatureSchema | None = None,
) -> str:
    """Return a stable SHA-256 fingerprint of the training-data policy.

    Covers every policy decision that changes the published handoff: the
    temporal split mapping, the strict support rule, the cell-ID formula,
    the scaler transformations, the canonical channel order, the feature
    schema version, the pinned V3 config hash, and the implausible-LST
    scene denylist. Any change invalidates the published artifacts via
    the ledger's ``config_changed`` reconcile.

    The default arguments reproduce the published training/v2 hash. A v6
    ``schema`` records that roster and its precipitation transform.
    """
    if schema is not None and schema.version != FEATURE_SCHEMA_VERSION:
        return _training_policy_hash_v6(
            schema,
            v3_config_hash=v3_config_hash,
            feature_schema_version=schema.version,
        )
    channel_schema = [
        {
            "name": ch.name,
            "valid_range": list(ch.valid_range) if ch.valid_range else None,
        }
        for ch in FEATURE_CHANNELS
    ]
    payload = json.dumps(
        {
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "v3_config_hash": v3_config_hash,
            "feature_schema_version": feature_schema_version,
            "channel_order": list(FEATURE_CHANNEL_NAMES),
            "channel_schema": channel_schema,
            "splits_by_year": SPLIT_BY_YEAR,
            "support_pixels": SUPPORT_PIXELS,
            "excluded_scenes": {
                IMPLAUSIBLE_LST_REASON: sorted(IMPLAUSIBLE_LST_SCENE_IDS),
            },
            "cell_id_formula": (
                "E{origin_x + col*100}N{origin_y - row*100} on canonical EPSG:25833"
            ),
            "scaler_policy": {
                "zscore": "all channels except precip and shadows",
                "log1p_then_zscore": list(PRECIP_CHANNELS),
                "identity": list(SHADOW_CHANNELS),
                "variance": "population (ddof=0)",
            },
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]


def _training_policy_hash_v6(
    schema: FeatureSchema,
    *,
    v3_config_hash: str,
    feature_schema_version: int,
) -> str:
    """Policy fingerprint for a non-v5 roster. Not the published v2 hash."""
    channel_schema = [
        {
            "name": channel.name,
            "valid_range": list(channel.valid_range) if channel.valid_range else None,
        }
        for channel in schema.channels
    ]
    payload = json.dumps(
        {
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "v3_config_hash": v3_config_hash,
            "feature_schema_version": feature_schema_version,
            "channel_order": list(schema.names),
            "channel_schema": channel_schema,
            "splits_by_year": {str(year): split for year, split in SPLIT_BY_YEAR.items()},
            "support_pixels": SUPPORT_PIXELS,
            "excluded_scenes": {
                IMPLAUSIBLE_LST_REASON: sorted(IMPLAUSIBLE_LST_SCENE_IDS),
            },
            "cell_id_formula": (
                "E{origin_x + col*100}N{origin_y - row*100} on canonical EPSG:25833"
            ),
            "scaler_policy": {
                "zscore": "all channels except precip and shadows",
                "log1p_then_zscore": list(precip_channels_for(schema)),
                "identity": list(shadow_channels_for(schema)),
                "variance": "population (ddof=0)",
            },
        },
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]


__all__ = [
    "CANON_GRID_ORIGIN_X",
    "CANON_GRID_ORIGIN_Y",
    "CELL_SIZE_M",
    "EXPECTED_V3_CONFIG_HASH",
    "IMPLAUSIBLE_LST_REASON",
    "IMPLAUSIBLE_LST_SCENE_IDS",
    "INFERENCE_DEFERRED_REASON",
    "NO_ELIGIBLE_CELLS_REASON",
    "PRECIP_CHANNELS",
    "SHADOW_CHANNELS",
    "precip_channels_for",
    "scaler_transform_for",
    "shadow_channels_for",
    "SPLIT_BY_YEAR",
    "SUPPORT_PIXELS",
    "TRAINING_SCHEMA_VERSION",
    "cell_id",
    "split_for_year",
    "training_policy_hash",
]
