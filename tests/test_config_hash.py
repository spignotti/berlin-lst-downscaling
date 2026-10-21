"""Config hashes move when a relevant input moves, and stay put otherwise."""

from __future__ import annotations

import pytest

from berlin_lst_downscaling.data.dynamic.schema import (
    config_hash_for_dynamic,
    config_hash_for_era5,
)
from berlin_lst_downscaling.data.features.schema import config_hash_for_features
from berlin_lst_downscaling.data.training.contracts import SPLIT_BY_YEAR, training_policy_hash


def _feature_kwargs(**overrides: object) -> dict[str, object]:
    """Return a complete, stable feature-hash payload."""
    payload: dict[str, object] = {
        "manifest_hash": "manifest",
        "geometry_mapping_hash": "geometry",
        "ard_ledger_hash": "ard",
        "static_sources_ledger_hash": "sources",
        "static_derived_ledger_hash": "derived",
        "dynamic_ledger_hash": "dynamic",
        "aoi_fingerprint": "aoi",
        "vegetation_carry_forward_geometry_id": "geom-2020",
        "lod_coverage_fingerprints": {"2017": "cover-a", "2021": "cover-b"},
        "lod_cog_fingerprints": {"2017": "cog-a", "2021": "cog-b"},
    }
    payload.update(overrides)
    return payload


def test_feature_config_hash_is_stable_and_input_sensitive() -> None:
    """Reordering dict keys does not change the hash. Changing an input does."""
    original = config_hash_for_features(**_feature_kwargs())  # type: ignore[arg-type]
    again = config_hash_for_features(**_feature_kwargs())  # type: ignore[arg-type]
    reordered = config_hash_for_features(  # type: ignore[arg-type]
        **_feature_kwargs(
            lod_coverage_fingerprints={"2021": "cover-b", "2017": "cover-a"},
            lod_cog_fingerprints={"2021": "cog-b", "2017": "cog-a"},
        )
    )
    changed = config_hash_for_features(  # type: ignore[arg-type]
        **_feature_kwargs(manifest_hash="other-manifest")
    )

    assert original == again == reordered
    assert changed != original
    assert len(original) == 16
    assert all(char in "0123456789abcdef" for char in original)


def test_training_policy_hash_covers_the_split_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy hash is stable, and a split-map edit changes it."""
    baseline = training_policy_hash()
    assert training_policy_hash() == baseline
    assert training_policy_hash(v3_config_hash="other") != baseline

    monkeypatch.setitem(SPLIT_BY_YEAR, 2025, "validation")
    assert training_policy_hash() != baseline


def test_dynamic_and_era5_hashes_diverge_when_their_inputs_diverge() -> None:
    """ERA5 and the full dynamic fingerprint are different hashes of the same ids."""
    dynamic = config_hash_for_dynamic("manifest", "geometry", "cache", 72)
    era5 = config_hash_for_era5("manifest", "geometry", "cache", 72)
    assert dynamic == config_hash_for_dynamic("manifest", "geometry", "cache", 72)
    assert dynamic != era5
    assert dynamic != config_hash_for_dynamic("other", "geometry", "cache", 72)
    assert era5 != config_hash_for_era5("manifest", "geometry", "cache", 48)
