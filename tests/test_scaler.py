"""Train-only scaler: holdout scenes contribute nothing; variance is population."""

from __future__ import annotations

import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.training.report import SceneTrainingResult
from berlin_lst_downscaling.data.training.scaler import (
    _apply_transform,
    _Welford,
    fit_scaler,
)


def _scene(split: str, year: int, eligible_cells: int) -> SceneTrainingResult:
    """Build one scene row for the scaler filter."""
    return SceneTrainingResult(
        scene_id=f"LC08_{year}",
        year=year,
        s2_scene_id=f"S2_{year}",
        split=split,
        status="done",
        sensor="landsat-8",
        eligible_cells=eligible_cells,
    )


def test_welford_population_std_matches_a_hand_calculation() -> None:
    """Two updates equal one batch, and std uses population variance (ddof=0)."""
    values = np.array([1.0, 3.0, 5.0])
    batched = _Welford()
    batched.update(np.array([1.0, 3.0]))
    batched.update(np.array([5.0]))
    once = _Welford()
    once.update(values)

    assert batched.count == once.count == 3
    assert batched.mean == pytest.approx(3.0)
    assert batched.mean == pytest.approx(once.mean)
    assert batched.std() == pytest.approx(float(np.sqrt(np.var(values, ddof=0))))
    assert _Welford().std() == 0.0


def test_precip_transform_is_log1p_before_the_zscore() -> None:
    """Precipitation is log1p'd; other continuous channels are left unchanged."""
    values = np.array([0.0, np.e - 1.0, 1.0])
    np.testing.assert_allclose(_apply_transform(values, "log1p_zscore"), np.log1p(values))
    np.testing.assert_array_equal(_apply_transform(values, "zscore"), values)


def test_scaler_fit_reads_only_train_scenes_with_eligible_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation, test, inference, and empty train scenes never open a raster."""

    def _refuse_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scaler opened a raster for a scene that must not contribute")

    monkeypatch.setattr(
        "berlin_lst_downscaling.data.training.scaler.rasterio.open",
        _refuse_open,
    )
    grid = GeoBox.from_bbox(
        (369190.0, 5838310.0, 369290.0, 5838410.0),
        crs="EPSG:25833",
        resolution=10,
    )
    results = [
        _scene("validation", 2024, eligible_cells=50),
        _scene("test", 2025, eligible_cells=50),
        _scene("inference", 2026, eligible_cells=50),
        _scene("train", 2019, eligible_cells=0),
    ]
    payload = fit_scaler(
        results,
        features_root="unused",
        output_root="unused",
        grid_10m=grid,
    )
    assert all(channel["count"] == 0 for channel in payload["channels"])
