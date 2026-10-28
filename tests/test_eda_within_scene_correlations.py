"""Vectorized within-scene correlations match the previous per-scene loop."""

from __future__ import annotations

import math

import numpy as np
import pytest

from berlin_lst_downscaling.data.eda.rules import MIN_WITHIN_SCENE_N
from berlin_lst_downscaling.data.eda.stats import (
    pearson,
    self_check,
    spearman,
    within_scene_correlations,
)


def _reference_within_scene_correlations(
    scene_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | int]:
    """The previous loop implementation, kept as the equivalence reference."""
    ids = np.asarray(scene_ids).ravel()
    pair_left = np.asarray(left, dtype=np.float64).ravel()
    pair_right = np.asarray(right, dtype=np.float64).ravel()
    pearsons: list[float] = []
    spearmans: list[float] = []
    used = 0
    skipped = 0
    for scene_id in dict.fromkeys(ids.tolist()):
        mask = ids == scene_id
        ok = np.isfinite(pair_left[mask]) & np.isfinite(pair_right[mask])
        if int(np.sum(ok)) < MIN_WITHIN_SCENE_N:
            skipped += 1
            continue
        used += 1
        scene_left = pair_left[mask][ok]
        scene_right = pair_right[mask][ok]
        left_corr = pearson(scene_left, scene_right)
        right_corr = spearman(scene_left, scene_right)
        if math.isfinite(left_corr):
            pearsons.append(left_corr)
        if math.isfinite(right_corr):
            spearmans.append(right_corr)
    return {
        "n_scenes_used": used,
        "n_scenes_skipped": skipped,
        "median_pearson": float(np.median(pearsons)) if pearsons else math.nan,
        "median_spearman": float(np.median(spearmans)) if spearmans else math.nan,
    }


def _fixture(seed: int, string_ids: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    sizes = [int(n) for n in rng.integers(5, 400, size=60)]
    # Below the minimum, exactly at it, and one scene with no finite pair at all.
    sizes += [MIN_WITHIN_SCENE_N - 1, MIN_WITHIN_SCENE_N, 50, 80, 80, 120]
    for scene, size in enumerate(sizes):
        x = rng.normal(300.0, 5.0, size)
        y = 0.6 * x + rng.normal(0.0, 3.0, size)
        if scene % 4 == 0:
            # Tied values exercise average ranks.
            x = np.round(x)
            y = np.round(y / 2.0) * 2.0
        if scene == len(sizes) - 4:
            y[:] = math.nan
        if scene == len(sizes) - 3:
            x[:] = 5.0
        if scene == len(sizes) - 2:
            y = rng.integers(0, 3, size).astype(np.float64)
        holes = rng.random(size) < 0.08
        x[holes & (rng.random(size) < 0.5)] = math.nan
        y[holes & (rng.random(size) < 0.5)] = math.inf
        y[holes & (rng.random(size) < 0.2)] = -math.inf
        ids.append(np.full(size, scene))
        lefts.append(x)
        rights.append(y)
    order = rng.permutation(sum(sizes))
    scene_ids = np.concatenate(ids)[order]
    if string_ids:
        scene_ids = np.array([f"LC09_scene_{value:03d}" for value in scene_ids])
    return scene_ids, np.concatenate(lefts)[order], np.concatenate(rights)[order]


def _assert_same(actual: dict[str, float | int], expected: dict[str, float | int]) -> None:
    assert set(actual) == set(expected)
    assert actual["n_scenes_used"] == expected["n_scenes_used"]
    assert actual["n_scenes_skipped"] == expected["n_scenes_skipped"]
    assert isinstance(actual["n_scenes_used"], int)
    assert isinstance(actual["n_scenes_skipped"], int)
    for key in ("median_pearson", "median_spearman"):
        assert isinstance(actual[key], float)
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-10, atol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("string_ids", [False, True])
def test_matches_reference_loop(seed: int, string_ids: bool) -> None:
    scene_ids, left, right = _fixture(seed, string_ids)
    expected = _reference_within_scene_correlations(scene_ids, left, right)
    assert expected["n_scenes_skipped"] >= 2
    _assert_same(within_scene_correlations(scene_ids, left, right), expected)


def test_constant_scenes_only_give_nan_medians() -> None:
    scene_ids = np.repeat(np.array(["a", "b"]), 40)
    left = np.concatenate([np.full(40, 0.1), np.arange(40, dtype=np.float64)])
    right = np.concatenate([np.arange(40, dtype=np.float64), np.full(40, 7.3)])
    result = within_scene_correlations(scene_ids, left, right)
    assert result["n_scenes_used"] == 2
    assert result["n_scenes_skipped"] == 0
    assert math.isnan(float(result["median_pearson"]))
    assert math.isnan(float(result["median_spearman"]))


def test_inexact_constant_scene_is_nan() -> None:
    # 300.15 is not exact in binary; float cancellation must not yield a finite r.
    result = within_scene_correlations(
        np.zeros(50, dtype=np.int64), np.full(50, 300.15), np.arange(50, dtype=np.float64)
    )
    assert result["n_scenes_used"] == 1
    assert math.isnan(float(result["median_pearson"]))
    assert math.isnan(float(result["median_spearman"]))


def test_all_scenes_skipped_and_empty_input() -> None:
    small = within_scene_correlations(np.array([1, 2, 1]), np.ones(3), np.ones(3))
    assert small["n_scenes_used"] == 0
    assert small["n_scenes_skipped"] == 2
    assert math.isnan(float(small["median_pearson"]))
    empty = within_scene_correlations(np.array([]), np.array([]), np.array([]))
    _assert_same(
        empty, _reference_within_scene_correlations(np.array([]), np.array([]), np.array([]))
    )


def test_handles_many_large_scenes() -> None:
    rng = np.random.default_rng(42)
    n_scenes, per_scene = 200, 5_000
    scene_ids = rng.permutation(np.repeat(np.arange(n_scenes), per_scene))
    left = rng.normal(size=scene_ids.size)
    right = 0.5 * left + rng.normal(size=scene_ids.size)
    result = within_scene_correlations(scene_ids, left, right)
    assert result["n_scenes_used"] == n_scenes
    assert result["n_scenes_skipped"] == 0
    assert 0.3 < float(result["median_pearson"]) < 0.6
    assert 0.3 < float(result["median_spearman"]) < 0.6


@pytest.mark.parametrize(
    "scene_ids",
    [
        np.array([1.0, 1.0, math.nan, 2.0]),
        np.array(["a", "a", None, "b"], dtype=object),
    ],
)
def test_missing_scene_id_raises(scene_ids: np.ndarray) -> None:
    values = np.arange(scene_ids.size, dtype=np.float64)
    with pytest.raises(ValueError, match="missing scene ID"):
        within_scene_correlations(scene_ids, values, values)


def test_self_check_passes() -> None:
    assert self_check() == []
