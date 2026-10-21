"""Temporal split contract: 2017-2023 train, 2024 val, 2025 test, 2026 inference."""

from __future__ import annotations

import pytest

from berlin_lst_downscaling.data.training.contracts import SPLIT_BY_YEAR, split_for_year
from berlin_lst_downscaling.data.training.release import _validate_split_leakage
from berlin_lst_downscaling.data.training.report import SceneTrainingResult


def _scene(scene_id: str, year: int, s2_scene_id: str, split: str) -> SceneTrainingResult:
    """Build one scene row for the leakage check."""
    return SceneTrainingResult(
        scene_id=scene_id,
        year=year,
        s2_scene_id=s2_scene_id,
        split=split,
        status="done",
        sensor="landsat-8",
    )


def test_years_map_to_one_split_each() -> None:
    """Every contract year has exactly one split, and the year sets are disjoint."""
    assert set(SPLIT_BY_YEAR) == set(range(2017, 2027))
    assert len(SPLIT_BY_YEAR) == len(set(SPLIT_BY_YEAR))
    for year in range(2017, 2024):
        assert split_for_year(year) == "train"
    assert split_for_year(2024) == "validation"
    assert split_for_year(2025) == "test"
    assert split_for_year(2026) == "inference"
    with pytest.raises(ValueError, match="2017-2026"):
        split_for_year(2016)
    with pytest.raises(ValueError, match="2017-2026"):
        split_for_year(2027)


def test_one_sentinel2_scene_stays_in_one_non_inference_split() -> None:
    """A shared Sentinel-2 scene may repeat inside one split, not across splits."""
    same_split = [
        _scene("LC08_2018", 2018, "S2_SHARED", "train"),
        _scene("LC08_2022", 2022, "S2_SHARED", "train"),
    ]
    _validate_split_leakage(same_split)

    leaked = [
        _scene("LC08_2018", 2018, "S2_SHARED", "train"),
        _scene("LC09_2024", 2024, "S2_SHARED", "validation"),
    ]
    with pytest.raises(RuntimeError, match="temporal leakage"):
        _validate_split_leakage(leaked)

    # 2026 inference is metadata-only and is excluded from the leakage check.
    with_inference = [
        _scene("LC08_2018", 2018, "S2_SHARED", "train"),
        _scene("LC09_2026", 2026, "S2_SHARED", "inference"),
    ]
    _validate_split_leakage(with_inference)
