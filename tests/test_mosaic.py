"""Offline tests for the per-overpass 10 m mosaic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from berlin_lst_downscaling.data.training.contracts import CANON_GRID_ORIGIN_X, CANON_GRID_ORIGIN_Y
from berlin_lst_downscaling.modeling.mosaic import (
    acquisition_year,
    group_scenes_by_overpass,
    mosaic_overpass,
    parse_landsat_scene_id,
    write_overpass_mosaic,
)

_OX = CANON_GRID_ORIGIN_X
_OY = CANON_GRID_ORIGIN_Y


def _affine(col0: int, row0: int, *, x_shift: float = 0.0) -> Affine:
    return Affine(10.0, 0.0, _OX + col0 * 10.0 + x_shift, 0.0, -10.0, _OY - row0 * 10.0)


def _write(path: Path, values: np.ndarray, transform: Affine, *, crs: str = "EPSG:25833") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def test_scene_ids_group_by_sensor_path_and_date() -> None:
    parsed = parse_landsat_scene_id("LC09_L2SP_193023_20250608_20250610_02_T1")
    assert parsed.overpass_id == "LC09_193_20250608"
    assert parsed.row == "023"
    assert acquisition_year("LC09_L2SP_193024_20250608_02_T1") == 2025
    groups = group_scenes_by_overpass(
        [
            "LC09_L2SP_193024_20250608",
            "LC09_L2SP_193023_20250608",
            "LC08_L2SP_193023_20250608_02_T1",
            "LC09_L2SP_192023_20250608_20250610_02_T1",
        ]
    )
    assert groups["LC09_193_20250608"] == (
        "LC09_L2SP_193023_20250608",
        "LC09_L2SP_193024_20250608",
    )
    assert groups["LC08_193_20250608"] == ("LC08_L2SP_193023_20250608_02_T1",)
    assert "LC09_192_20250608" in groups
    with pytest.raises(ValueError, match="same WRS row"):
        group_scenes_by_overpass(["LC09_L2SP_193023_20250608", "LC09_L2SP_193023_20250608_02_T1"])
    with pytest.raises(ValueError, match="scene id"):
        parse_landsat_scene_id("not-a-scene")


def test_mosaic_means_the_overlap_and_ignores_nodata(tmp_path: Path) -> None:
    left = tmp_path / "LC09_L2SP_193023_20250608.lst10m.tif"
    right = tmp_path / "LC09_L2SP_193024_20250608.lst10m.tif"
    left_values = np.full((20, 20), 300, dtype=np.float32)
    right_values = np.full((20, 20), 310, dtype=np.float32)
    left_values[:, 10:] = np.nan
    _write(left, left_values, _affine(0, 0))
    _write(right, right_values, _affine(10, 0))
    mosaic = mosaic_overpass([str(right), str(left)])
    assert mosaic.overpass_id == "LC09_193_20250608"
    assert mosaic.array.shape == (20, 30)
    assert np.all(mosaic.array[:, :10] == np.float32(300))
    assert np.all(mosaic.array[:, 10:20] == np.float32(310))
    assert np.all(mosaic.array[:, 20:] == np.float32(310))
    pairs = mosaic.qa["pairs"]
    assert isinstance(pairs, list) and len(pairs) == 1
    assert pairs[0]["overlap_count"] == 0
    assert pairs[0]["mean_abs_difference"] is None
    # Both finite on the overlap: the mean is 305 and the absolute difference is 10.
    left_full = np.full((20, 20), 300, dtype=np.float32)
    _write(left, left_full, _affine(0, 0))
    both = mosaic_overpass([str(left), str(right)])
    assert np.all(both.array[:, :10] == np.float32(300))
    assert np.allclose(both.array[:, 10:20], 305)
    assert np.all(both.array[:, 20:] == np.float32(310))
    pair = both.qa["pairs"]
    assert isinstance(pair, list)
    assert pair[0]["overlap_count"] == 200
    assert pair[0]["mean_abs_difference"] == pytest.approx(10)
    assert pair[0]["max_abs_difference"] == pytest.approx(10)


def test_mosaic_refuses_a_crs_or_lattice_mismatch(tmp_path: Path) -> None:
    good = tmp_path / "LC09_L2SP_193023_20250608.lst10m.tif"
    bad = tmp_path / "LC09_L2SP_193024_20250608.lst10m.tif"
    values = np.full((16, 16), 300, dtype=np.float32)
    _write(good, values, _affine(0, 0))
    _write(bad, values, _affine(0, 0), crs="EPSG:4326")
    with pytest.raises(ValueError, match="EPSG:25833"):
        mosaic_overpass([str(good), str(bad)])
    shifted = tmp_path / "LC09_L2SP_193024_20250608.lst10m.tif"
    _write(shifted, values, _affine(0, 0, x_shift=3.0))
    with pytest.raises(ValueError, match="integer"):
        mosaic_overpass([str(good), str(shifted)])
    other_day = tmp_path / "LC09_L2SP_193024_20250609.lst10m.tif"
    _write(other_day, values, _affine(16, 0))
    with pytest.raises(ValueError, match="not one overpass"):
        mosaic_overpass([str(good), str(other_day)])


def test_write_overpass_mosaic(tmp_path: Path) -> None:
    scene = tmp_path / "LC08_L2SP_193023_20240102.lst10m.tif"
    _write(scene, np.full((32, 32), 290, dtype=np.float32), _affine(0, 0))
    tif_path, json_path = write_overpass_mosaic([str(scene)], str(tmp_path / "out"))
    assert tif_path.endswith("overpass/LC08_193_20240102.lst10m.tif")
    with rasterio.open(tif_path) as src:
        assert src.crs.to_epsg() == 25833
        written = src.read(1)
    assert np.allclose(written, 290)
    payload = json.loads(Path(json_path).read_text())
    assert payload["pairs"] == []
    assert payload["overpass_id"] == "LC08_193_20240102"
