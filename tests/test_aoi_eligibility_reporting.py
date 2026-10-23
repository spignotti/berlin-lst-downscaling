"""AC6: offline AOI reporting regressions, independent of S2/selection tests."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.diagnostics import feature_correctness as diagnostics
from berlin_lst_downscaling.data.diagnostics.feature_correctness import (
    aoi_fraction_on_grid,
    exclusion_reason_counts,
    main,
    rollup_exclusions,
    scan_feature_stack,
)
from berlin_lst_downscaling.data.eda.frames import eligibility_rate_table, eligibility_summary
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNELS
from berlin_lst_downscaling.data.qa.inventory import ResolvedScene
from berlin_lst_downscaling.data.training.eligibility import compute_eligibility

_ROOT = Path(__file__).resolve().parents[1]


def _write_tif(path: Path, data: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = data[np.newaxis] if data.ndim == 2 else data
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:25833",
        transform=transform,
    ) as dst:
        dst.write(array)


def _scene(tmp_path: Path) -> tuple[str, str, str]:
    """Two eligible, one partial-AOI excluded, two exterior excluded, one bad target."""
    transform = Affine(10, 0, 1000, 0, -10, 2000)
    aoi = np.zeros((20, 30), dtype=np.uint8)
    aoi[:, :10] = 1
    aoi[:5, 10:20] = 1
    stack = np.full((len(FEATURE_CHANNELS), 20, 30), np.nan, dtype=np.float32)
    for index, channel in enumerate(FEATURE_CHANNELS):
        assert channel.valid_range is not None
        low, high = channel.valid_range
        stack[index, aoi == 1] = (low + high) / 2
    flag = np.zeros((2, 3), dtype=np.uint8)
    flag[1, 2] = 1
    _write_tif(tmp_path / "data/boundaries/aoi_10m.tif", aoi, transform)
    _write_tif(tmp_path / "features.tif", stack, transform)
    _write_tif(
        tmp_path / "lst.tif", np.full((2, 3), 300, dtype=np.float32), transform * Affine.scale(10)
    )
    _write_tif(tmp_path / "flag.tif", flag, transform * Affine.scale(10))
    return tuple(str(tmp_path / name) for name in ("features.tif", "lst.tif", "flag.tif"))


def _edge_scene(tmp_path: Path, height: int, width: int) -> tuple[str, str, str]:
    transform = Affine(10, 0, 1000, 0, -10, 2000)
    shape100 = ((height + 9) // 10, (width + 9) // 10)
    stack = np.zeros((len(FEATURE_CHANNELS), height, width), dtype=np.float32)
    for index, channel in enumerate(FEATURE_CHANNELS):
        assert channel.valid_range is not None
        low, high = channel.valid_range
        stack[index] = (low + high) / 2
    _write_tif(tmp_path / "features.tif", stack, transform)
    _write_tif(
        tmp_path / "lst.tif", np.full(shape100, 300, np.float32), transform * Affine.scale(10)
    )
    _write_tif(tmp_path / "flag.tif", np.zeros(shape100, np.uint8), transform * Affine.scale(10))
    _write_tif(tmp_path / "aoi.tif", np.ones((height, width), np.uint8), transform)
    return tuple(str(tmp_path / name) for name in ("features.tif", "lst.tif", "flag.tif"))


def _eda_operator():
    spec = importlib.util.spec_from_file_location(
        "run_training_eda", _ROOT / "scripts/operators/run_training_eda.py"
    )
    assert spec is not None and spec.loader is not None
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    return operator


def test_primary_denominator_excludes_exterior_target_valid_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _scene(tmp_path)
    monkeypatch.chdir(tmp_path)
    counted = scan_feature_stack(*paths)
    assert counted["target_valid_cells"] == 3  # Exterior target-valid cells are secondary only.
    assert counted["eligible_cells"] == 2
    assert counted["excluded_cells"] == 1
    assert counted["target_valid_aoi_effective_cells"] == pytest.approx(2.5)
    assert counted["eligibility_rate"] == pytest.approx(2 / 2.5)
    assert counted["exclusion_rate"] == pytest.approx(0.5 / 2.5)
    assert counted["per_channel"]["B02"] == 1
    rectangle = counted["full_rectangle"]
    assert rectangle["target_valid_cells"] == 5
    assert rectangle["eligible_cells"] == 2
    assert rectangle["excluded_cells"] == 3
    assert rectangle["exclusion_rate"] == pytest.approx(3 / 5)


def test_windowed_and_in_memory_counts_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _scene(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(diagnostics, "_WINDOW", 10)  # Exercise multiple windows.
    with rasterio.open(paths[0]) as src:
        bands = src.read()
        fraction = aoi_fraction_on_grid(
            "data/boundaries/aoi_10m.tif", GeoBox(src.shape, src.transform, src.crs)
        )
    target = np.ones((2, 3), dtype=bool)
    target[1, 2] = False
    memory = exclusion_reason_counts(target, bands, aoi_fraction=fraction)
    windowed = scan_feature_stack(*paths)
    for key, value in memory.items():
        assert windowed[key] == value
    assert fraction.tolist() == [[1, 0.5, 0], [1, 0, 0]]
    assert windowed["per_channel_aoi_effective_cells"]["B02"] == 0.5
    assert windowed["full_rectangle"]["per_channel"]["B02"] == 3


def test_aoi_extent_zeroes_are_included_in_fraction(tmp_path: Path) -> None:
    """Nodata=0 and cells outside the raster must not inflate partial support."""
    transform = Affine(10, 0, 1000, 0, -10, 2000)
    path = tmp_path / "offset-aoi.tif"
    _write_tif(path, np.ones((10, 10), dtype=np.uint8), transform * Affine.translation(5, 0))
    with rasterio.open(path, "r+") as dst:
        dst.nodata = 0
    grid = GeoBox((20, 30), transform, "EPSG:25833")
    fraction = aoi_fraction_on_grid(str(path), grid)
    assert fraction.tolist() == [[0.5, 0.5, 0], [0, 0, 0]]


def test_partial_edge_aoi_fraction_is_retained(tmp_path: Path) -> None:
    transform = Affine(10, 0, 1000, 0, -10, 2000)
    mask = np.zeros((10, 15), dtype=np.uint8)
    mask[:, 10:] = 1
    path = tmp_path / "edge-aoi.tif"
    _write_tif(path, mask, transform)
    fraction = aoi_fraction_on_grid(str(path), GeoBox(mask.shape, transform, "EPSG:25833"))
    assert fraction.tolist() == [[0, 0.5]]


def test_eda_retains_target_valid_edge_records(tmp_path: Path) -> None:
    operator = _eda_operator()
    gaps = tmp_path / "gaps"
    gaps.mkdir()
    np.savez(
        gaps / "edge.npz",
        target_valid_support_100=np.array([[0, 0]], dtype=np.int32),
        target_valid_support_lt_100=np.empty((0, 2), dtype=np.int32),
        target_valid_edge=np.array([[0, 1]], dtype=np.int32),
    )
    markers = [{"scene_id": "edge", "year": 2024, "split": "train", "season": "JJA"}]
    table = operator._eligibility_rates(tmp_path, markers, np.array([[0, 0.5]]))
    pooled = table.loc[table["grain"] == "all"].iloc[0]
    assert pooled["target_valid_cells"] == 1
    assert pooled["target_valid_aoi_effective_cells"] == 0.5
    assert pooled["excluded_cells"] == 1
    assert pooled["exclusion_rate"] == 1
    assert pooled["full_rectangle_target_valid_cells"] == 2
    assert pooled["full_rectangle_eligible_cells"] == 1


@pytest.mark.parametrize("height,width", [(10, 15), (15, 10), (15, 15), (3, 7)])
def test_partial_edges_are_reported_but_never_training_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, height: int, width: int
) -> None:
    paths = _edge_scene(tmp_path, height, width)
    monkeypatch.setattr(diagnostics, "_WINDOW", 10)
    with rasterio.open(paths[0]) as src:
        grid = GeoBox(src.shape, src.transform, src.crs)
        bands = src.read()
    fraction = aoi_fraction_on_grid(str(tmp_path / "aoi.tif"), grid)
    assert fraction.shape == tuple(grid.zoom_out(10).shape)
    assert fraction.sum() == pytest.approx(height * width / 100)
    target = np.ones(fraction.shape, dtype=bool)
    memory = exclusion_reason_counts(target, bands, aoi_fraction=fraction)
    scanned = scan_feature_stack(*paths, aoi_mask=str(tmp_path / "aoi.tif"))
    for key, value in memory.items():
        assert scanned[key] == value
    expected_eligible = (height // 10) * (width // 10)
    expected_edge = fraction.size - expected_eligible
    assert scanned["eligible_cells"] == expected_eligible
    assert scanned["excluded_cells"] == scanned["edge_truncated_cells"] == expected_edge
    assert scanned["full_rectangle"]["edge_truncated_cells"] == expected_edge
    assert scanned["incomplete_support_cells"] == 0
    assert scanned["per_channel"]["B02"] == 0  # Geometry is not a channel defect.
    assert scanned["target_valid_aoi_effective_cells"] == pytest.approx(height * width / 100)
    assert scanned["excluded_aoi_effective_cells"] == pytest.approx(
        height * width / 100 - expected_eligible
    )
    assert scanned["dropped_edge_rows"] == scanned["dropped_edge_cols"] == 0
    rolled = rollup_exclusions([{**scanned, "year": 2024}])
    assert rolled["edge_truncated_cells"] == expected_edge
    assert rolled["full_rectangle"]["edge_truncated_cells"] == expected_edge

    mask = tmp_path / "feature_valid.tif"
    _write_tif(mask, np.ones((height, width), dtype=np.uint8), grid.transform)
    scene = ResolvedScene(
        scene_id="edge",
        year=2024,
        s2_scene_id="fake",
        geometry_id="",
        landsat_cog=paths[1],
        landsat_flag=paths[2],
        s2_cog="",
        s2_flag="",
        dynamic={},
        static_derived={},
        static_derived_meta={},
        static_sources={},
    )
    training = compute_eligibility(scene=scene, analysis_10=grid, mask_uri=str(mask))
    assert training.errors == []
    assert training.eligible_cells == expected_eligible == scanned["eligible_cells"]
    assert not np.any(training.eligible[training.present < 100])
    report = eligibility_summary(training.target_valid, training.eligible, fraction)
    assert report["target_valid_aoi_effective_cells"] == scanned["target_valid_aoi_effective_cells"]
    assert report["exclusion_rate"] == scanned["exclusion_rate"]


def test_synthetic_bottom_edge_preserves_153_aoi_pixels(tmp_path: Path) -> None:
    paths = _edge_scene(tmp_path, 11, 160)
    with rasterio.open(paths[0]) as src:
        grid = GeoBox(src.shape, src.transform, src.crs)
    aoi = np.zeros((11, 160), dtype=np.uint8)
    aoi[-1, :153] = 1
    _write_tif(tmp_path / "aoi.tif", aoi, grid.transform)
    fraction = aoi_fraction_on_grid(str(tmp_path / "aoi.tif"), grid)
    assert fraction.shape == (2, 16)
    assert fraction.sum() * 100 == pytest.approx(153)
    scanned = scan_feature_stack(*paths, aoi_mask=str(tmp_path / "aoi.tif"))
    assert scanned["target_valid_aoi_effective_cells"] == pytest.approx(1.53)
    assert scanned["excluded_aoi_effective_cells"] == pytest.approx(1.53)
    assert scanned["exclusion_rate"] == 1
    assert scanned["eligible_cells"] == 0


def test_eda_refreshes_legacy_cache_and_persists_only_valid_edge_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = _eda_operator()
    paths = _edge_scene(tmp_path, 15, 15)
    with rasterio.open(paths[0]) as src:
        grid = GeoBox(src.shape, src.transform, src.crs)
    with rasterio.open(paths[2], "r+") as dst:
        dst.write(np.array([[0, 0], [0, 1]], dtype=np.uint8), 1)
    fraction = aoi_fraction_on_grid(str(tmp_path / "aoi.tif"), grid)
    output = tmp_path / "eda"
    gaps = output / "gaps"
    gaps.mkdir(parents=True)
    arrays = {name: np.empty((0, 2), dtype=np.int32) for name in operator.GAP_CLASSES}
    arrays["target_valid_support_100"] = np.array([[0, 0]], dtype=np.int32)
    arrays["edge_lt_100"] = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
    np.savez(gaps / "edge.npz", **arrays)
    marker = {
        "scene_id": "edge",
        "year": 2024,
        "season": "JJA",
        "split": "train",
        "empty": False,
        "provenance_target_valid": 3,
        "provenance_eligible": 1,
    }
    (gaps / "edge.json").write_text(json.dumps(marker))
    observations = pd.DataFrame({"scene_id": ["edge"]})
    operator._write_frame(observations, output / "observations/edge.parquet")
    with pytest.raises(ValueError, match="missing target_valid_edge"):
        operator._eligibility_rates(output, [marker], fraction)

    calls = []

    def fake_scan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            gap_counts={name: len(pairs) for name, pairs in arrays.items()},
            gap_cells=arrays,
            observations=observations,
            provenance_target_valid=3,
            provenance_eligible=1,
        )

    monkeypatch.setattr(operator, "scan_scene", fake_scan)
    monkeypatch.setattr(operator, "canon_grid_10m", lambda: grid)
    monkeypatch.setattr(
        operator,
        "read_json",
        lambda _: {
            "inputs": {
                "landsat_cog": paths[1],
                "landsat_flag": paths[2],
                "feature_valid": "unused-local",
            }
        },
    )
    row = pd.Series(
        {
            "scene_id": "edge",
            "year": 2024,
            "split": "train",
            "s2_scene_id": "fake",
            "eligible_cells": 1,
            "feature_stack": paths[0],
        }
    )
    calendar = pd.DataFrame({"scene_id": ["edge"], "month": [7]})
    operator._scan_one(output, row, calendar, str(tmp_path))
    assert len(calls) == 1
    with np.load(gaps / "edge.npz") as persisted:
        assert persisted["target_valid_edge"].tolist() == [[0, 1], [1, 0]]
        assert persisted["edge_lt_100"].tolist() == [[0, 1], [1, 0], [1, 1]]
    table = operator._eligibility_rates(output, [marker], fraction)
    pooled = table.loc[table["grain"] == "all"].iloc[0]
    diagnostic = scan_feature_stack(*paths, aoi_mask=str(tmp_path / "aoi.tif"))
    for key in ("target_valid_cells", "excluded_cells", "eligible_cells", "exclusion_rate"):
        assert pooled[key] == diagnostic[key]
    assert pooled["target_valid_aoi_effective_cells"] == 2
    with pytest.raises(ValueError, match="disagrees with provenance"):
        operator._eligibility_rates(output, [{**marker, "provenance_target_valid": 4}], fraction)
    operator._scan_one(output, row, calendar, str(tmp_path))
    assert len(calls) == 1  # New-schema cache can still be reused.


def test_reporting_leaves_training_100_of_100_eligibility_untouched(tmp_path: Path) -> None:
    paths = _scene(tmp_path)
    with rasterio.open(paths[0]) as src:
        grid = GeoBox(src.shape, src.transform, src.crs)
        valid = np.isfinite(src.read()).all(axis=0).astype(np.uint8)
    mask = tmp_path / "feature_valid.tif"
    _write_tif(mask, valid, grid.transform)
    scene = ResolvedScene(
        scene_id="local",
        year=2024,
        s2_scene_id="fake",
        geometry_id="",
        landsat_cog=paths[1],
        landsat_flag=paths[2],
        s2_cog="",
        s2_flag="",
        dynamic={},
        static_derived={},
        static_derived_meta={},
        static_sources={},
    )
    result = compute_eligibility(scene=scene, analysis_10=grid, mask_uri=str(mask))
    assert result.errors == []
    original_target = result.target_valid.copy()
    original_eligible = result.eligible.copy()
    fraction = aoi_fraction_on_grid(str(tmp_path / "data/boundaries/aoi_10m.tif"), grid)
    counted = eligibility_summary(result.target_valid, result.eligible, fraction)
    assert result.target_valid_cells == 5
    assert result.eligible_cells == counted["eligible_cells"] == 2
    assert counted["full_rectangle"]["eligible_cells"] == 2
    assert int((result.eligible & (fraction > 0)).sum()) == result.eligible_cells
    assert result.support.tolist() == [[100, 50, 0], [100, 0, 0]]
    np.testing.assert_array_equal(result.target_valid, original_target)
    np.testing.assert_array_equal(result.eligible, original_eligible)


def test_reporting_does_not_gate_training_eligible_exterior_cells() -> None:
    target = np.ones((1, 2), dtype=bool)
    eligible = np.ones((1, 2), dtype=bool)
    counted = eligibility_summary(target, eligible, np.array([[1, 0]]))
    assert counted["eligible_cells"] == 1
    assert counted["full_rectangle"]["eligible_cells"] == 2
    assert eligible.tolist() == [[True, True]]


def test_empty_aoi_and_invalid_targets_have_no_rate() -> None:
    for target, fraction in (
        (np.ones((1, 1), dtype=bool), np.zeros((1, 1))),
        (np.zeros((1, 1), dtype=bool), np.ones((1, 1))),
    ):
        counted = eligibility_summary(target, np.zeros((1, 1), dtype=bool), fraction)
        assert counted["target_valid_aoi_effective_cells"] == 0
        assert counted["eligibility_rate"] is None
        assert counted["exclusion_rate"] is None


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_aoi_fractions_are_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        eligibility_summary(np.array([True]), np.array([True]), np.array([bad]))


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        eligibility_summary(np.ones((2, 2)), np.ones((2, 2)), np.ones((1, 2)))


def test_rollups_and_eda_pool_areas_rather_than_scene_rates() -> None:
    def _counts(year: int, fraction: np.ndarray) -> dict[str, object]:
        target = np.ones((1, 2), dtype=bool)
        summary = eligibility_summary(target, np.array([[True, False]]), fraction)
        return {
            "scene_id": str(year),
            "year": year,
            "split": "train",
            "season": "JJA",
            **summary,
            "per_channel": {"B02": 1},
            "per_channel_aoi_effective_cells": {"B02": float(fraction[0, 1])},
            "full_rectangle": {**summary["full_rectangle"], "per_channel": {"B02": 1}},
        }

    rows = [_counts(2021, np.array([[1, 0.5]])), _counts(2022, np.array([[0.1, 1]]))]
    rolled = rollup_exclusions(rows)
    assert rolled["exclusion_rate"] == pytest.approx(1.5 / 2.6)
    assert rolled["by_year"]["2021"]["exclusion_rate"] == pytest.approx(0.5 / 1.5)
    assert rolled["full_rectangle"]["exclusion_rate"] == 0.5
    eda = eligibility_rate_table(pd.DataFrame(rows))
    pooled = eda.loc[eda["grain"] == "all"].iloc[0]
    assert pooled["exclusion_rate"] == rolled["exclusion_rate"]
    assert pooled["target_valid_aoi_effective_cells"] == 2.6
    assert pooled["full_rectangle_target_valid_cells"] == 4
    assert pooled["full_rectangle_exclusion_rate"] == 0.5


def test_operator_writes_primary_aoi_and_secondary_rectangle_reports(tmp_path: Path) -> None:
    paths = _scene(tmp_path)
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "local",
                    "year": 2024,
                    **dict(
                        zip(("feature_stack", "landsat_cog", "landsat_flag"), paths, strict=True)
                    ),
                }
            ]
        )
    )
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--scope",
                "eligibility-gaps",
                "--scenes-json",
                str(scenes),
                "--yes",
                "--aoi-mask",
                str(tmp_path / "data/boundaries/aoi_10m.tif"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())["eligibility_gaps"]
    assert summary["target_valid_cells"] == 3
    assert summary["exclusion_rate"] == 0.2
    assert summary["full_rectangle"]["exclusion_rate"] == 0.6
    assert summary["by_year"]["2024"]["exclusion_rate"] == 0.2
    row = json.loads((output / "results.jsonl").read_text())
    assert row["target_valid_aoi_effective_cells"] == 2.5
    markdown = (output / "summary.md").read_text()
    assert "Primary denominator: target-valid ∩ AOI" in markdown
    assert "Secondary: full rectangle" in markdown


def test_eda_operator_builds_rates_from_local_gap_pairs(tmp_path: Path) -> None:
    operator = _eda_operator()
    gaps = tmp_path / "gaps"
    gaps.mkdir()
    np.savez(
        gaps / "local.npz",
        target_valid_support_100=np.array([[0, 0], [1, 0]]),
        target_valid_support_lt_100=np.array([[0, 1], [0, 2], [1, 1]]),
        target_valid_edge=np.empty((0, 2), dtype=np.int32),
    )
    fraction = np.array([[1, 0.5, 0], [1, 0, 0]])
    markers = [{"scene_id": "local", "year": 2024, "split": "train", "season": "JJA"}]
    table = operator._eligibility_rates(tmp_path, markers, fraction)
    pooled = table.loc[table["grain"] == "all"].iloc[0]
    assert pooled["eligible_cells"] == 2
    assert pooled["target_valid_cells"] == 3
    assert pooled["exclusion_rate"] == 0.2
    assert pooled["full_rectangle_target_valid_cells"] == 5
    assert pooled["full_rectangle_exclusion_rate"] == 0.6
    with np.load(gaps / "local.npz") as raw:
        assert raw["target_valid_support_lt_100"].shape == (3, 2)
