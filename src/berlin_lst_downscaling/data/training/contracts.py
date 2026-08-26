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
)

# ── schema version ────────────────────────────────────────────────────

# Bump when the training-data contract (splits, eligibility rule, cell-ID
# formula, scaler policy) changes in a way that invalidates published
# artifacts. The value feeds the policy hash.
TRAINING_SCHEMA_VERSION: int = 1

# ── immutable input basis ─────────────────────────────────────────────

# Feature Release V3 is the only permitted input basis (user-mandated;
# V1/V2 must never be consumed). Pinned from the live features ledger —
# every V3 row carries this config hash.
EXPECTED_V3_CONFIG_HASH: str = "d9eb25995b2f4911"

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
) -> str:
    """Return a stable SHA-256 fingerprint of the training-data policy.

    Covers every policy decision that changes the published handoff: the
    temporal split mapping, the strict support rule, the cell-ID formula,
    the scaler transformations, the canonical channel order, the feature
    schema version, and the pinned V3 config hash. Any change invalidates
    the published artifacts via the ledger's ``config_changed`` reconcile.
    """
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


__all__ = [
    "CANON_GRID_ORIGIN_X",
    "CANON_GRID_ORIGIN_Y",
    "CELL_SIZE_M",
    "EXPECTED_V3_CONFIG_HASH",
    "INFERENCE_DEFERRED_REASON",
    "NO_ELIGIBLE_CELLS_REASON",
    "PRECIP_CHANNELS",
    "SHADOW_CHANNELS",
    "SPLIT_BY_YEAR",
    "SUPPORT_PIXELS",
    "TRAINING_SCHEMA_VERSION",
    "cell_id",
    "split_for_year",
    "training_policy_hash",
]
