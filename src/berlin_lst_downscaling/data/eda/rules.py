"""Fixed EDA rules. These thresholds are not tuned on a result."""

from __future__ import annotations

import json
from hashlib import sha256

from berlin_lst_downscaling.data.stadtstruktur.class_review import (
    PROPOSAL_B,
    PROPOSAL_B_GROUPS,
    SPARSE_AREA_KM2,
    SPARSE_EFFECTIVE_CELLS,
    assert_crosswalk_closed,
    scene_vintage,
)

RELEASE_ID = "eda-v1"
OUTPUT_URI = "gs://berlin-lst-training-data/evaluation/eda/v2"
TRAINING_ROOT = "gs://berlin-lst-training-data/training/v2"
FEATURES_ROOT = "gs://berlin-lst-training-data/features/v5"
STADTSTRUKTUR_ROOT = "gs://berlin-lst-training-data/evaluation/stadtstruktur/v1"
PATCH_INDEX_ROOT = "gs://berlin-lst-training-data/training/patch-index/v2"
V3_MANIFEST_URI = (
    "gs://berlin-lst-training-data/manifests/v3/"
    "2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
)

# Meteorological seasons from the acquisition month.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

# Hot and cold cells use the scene's own empirical quantiles.
HOT_QUANTILE = 0.90
COLD_QUANTILE = 0.10

# A channel is coarse-scale when its median within-scene std is below this
# fraction of the between-scene std of scene means. ERA5 is expected to trip
# this. It is a label, not a reason to drop the channel.
COARSE_WITHIN_TO_BETWEEN = 0.01
FLAT_STD = 1e-6

# Within-scene correlations need this many eligible cells.
MIN_WITHIN_SCENE_N = 30

# 4-neighborhood. Diagonals are not neighbors.
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1))

STAT_COLUMNS: tuple[str, ...] = (
    "n",
    "mean",
    "std",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "min",
    "max",
)

GAP_CLASSES: tuple[str, ...] = (
    "target_valid_support_100",
    "target_valid_support_lt_100",
    "target_invalid_support_100",
    "edge_lt_100",
)

EVAL_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
BANNED_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".svg",
    ".html",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
)

KELVIN_OFFSET = 273.15


def season_of_month(month: int | None) -> str:
    """Return the meteorological season for a calendar month."""
    if month is None:
        return "unknown"
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    return "unknown"


def proposal_b_fingerprint() -> str:
    """Fingerprint the approved typ-to-group map."""
    assert_crosswalk_closed()
    payload = json.dumps(
        [{"group": PROPOSAL_B[typ], "typ": typ} for typ in sorted(PROPOSAL_B)],
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()[:16]


def group_role(group_id: str) -> str:
    """Return the pre-registered role of one proposal-B group."""
    for spec in PROPOSAL_B_GROUPS:
        if spec.group_id == group_id:
            return spec.role
    raise KeyError(group_id)


def group_label(group_id: str) -> str:
    """Return the pre-registered label of one proposal-B group."""
    for spec in PROPOSAL_B_GROUPS:
        if spec.group_id == group_id:
            return spec.label
    raise KeyError(group_id)


__all__ = [
    "BANNED_SUFFIXES",
    "COARSE_WITHIN_TO_BETWEEN",
    "COLD_QUANTILE",
    "EVAL_SPLITS",
    "FEATURES_ROOT",
    "FLAT_STD",
    "GAP_CLASSES",
    "HOT_QUANTILE",
    "KELVIN_OFFSET",
    "MIN_WITHIN_SCENE_N",
    "NEIGHBOR_OFFSETS",
    "OUTPUT_URI",
    "PATCH_INDEX_ROOT",
    "PROPOSAL_B",
    "RELEASE_ID",
    "SEASON_MONTHS",
    "SPARSE_AREA_KM2",
    "SPARSE_EFFECTIVE_CELLS",
    "STADTSTRUKTUR_ROOT",
    "STAT_COLUMNS",
    "TRAINING_ROOT",
    "V3_MANIFEST_URI",
    "group_label",
    "group_role",
    "proposal_b_fingerprint",
    "scene_vintage",
    "season_of_month",
]
