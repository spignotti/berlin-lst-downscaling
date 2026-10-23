"""TESSERA arm leaves the v6 main arm byte-for-byte unaffected."""

from __future__ import annotations

from pathlib import Path

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.training.contracts import training_policy_hash

_REPO = Path(__file__).resolve().parents[1]


def test_published_v5_hashes_still_frozen() -> None:
    assert training_policy_hash() == "08c2ec4d5a257c7e"


def test_v6_channel_list_unchanged() -> None:
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    assert schema.n_channels == 22
    assert "canopy_cover_fraction" in schema.names
    assert "building_height_std" not in schema.names
    assert "vegetation_height_max" not in schema.names
    assert "building_height_mean" in schema.names
    assert "shadow_building" in schema.names


def test_main_arm_configs_do_not_mention_tessera() -> None:
    for name in ("configs/modeling/v6.yaml", "configs/features/v6.yaml"):
        text = (_REPO / name).read_text()
        assert "tessera" not in text.lower()
        assert "TESSERA" not in text
    # Tessera arm lives in its own configs only.
    tessera_modeling = (_REPO / "configs" / "modeling" / "tessera_arm.yaml").read_text()
    assert "tessera" in tessera_modeling.lower()
    # features/v6 roster helper still reports the 28-channel published list.
    assert len(FEATURE_CHANNEL_NAMES) == 28
