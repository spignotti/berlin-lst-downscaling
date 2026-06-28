"""Unit tests for the ARD QA module.

These tests do NOT require GCS, GEE, or any external services.
They create synthetic rasters with rasterio and verify the QA logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard_qa import (
    check_grid_conformity,
    compute_radiometric_stats,
    generate_qa_report,
)
from berlin_lst_downscaling.data.grid_spec import GridSpec, make_grid_spec

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_synthetic_raster(
    tmp_path: Path,
    filename: str = "test.tif",
    *,
    data: np.ndarray | None = None,
    transform: Affine | None = None,
    crs: str = "EPSG:25833",
    nodata: float | None = None,
    height: int = 10,
    width: int = 10,
    dtype: str = "float32",
) -> Path:
    """Create a small GeoTIFF for testing."""
    path = tmp_path / filename

    if data is None:
        data = np.random.rand(1, height, width).astype(np.float32) * 300

    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": dtype,
        "crs": crs,
        "transform": transform or Affine(10.0, 0, 368000, 0, -10.0, 5839000),
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data if data.dtype == dtype else data.astype(dtype))

    return path


def _make_spec() -> GridSpec:
    return make_grid_spec(
        origin_x=368000.0,
        origin_y=5839000.0,
        aoi_25833=(368002.320, 5797523.137, 417784.933, 5839258.933),
        wgs84_bbox=(13.0471, 52.3122, 13.7937, 52.6970),
    )


# ── check_grid_conformity ────────────────────────────────────────────────


def test_check_grid_conformity_match(tmp_path: Path) -> None:
    """A raster matching the spec should pass all checks."""
    spec = _make_spec()
    path = _make_synthetic_raster(
        tmp_path,
        transform=Affine(
            100.0, 0, spec.origin_x,
            0, -100.0, spec.origin_y,
        ),
        height=spec.height_100m,
        width=spec.width_100m,
    )
    result = check_grid_conformity(path, spec, target_resolution=100)
    assert result["crs_match"] is True
    assert result["resolution_match"] is True
    assert result["origin_ok"] is True
    assert result["bounds_ok"] is True


def test_check_grid_conformity_wrong_crs(tmp_path: Path) -> None:
    """Wrong CRS should fail crs_match."""
    spec = _make_spec()
    path = _make_synthetic_raster(tmp_path, crs="EPSG:4326")
    result = check_grid_conformity(path, spec, target_resolution=10)
    assert result["crs_match"] is False


def test_check_grid_conformity_wrong_resolution(tmp_path: Path) -> None:
    """Wrong resolution should fail resolution_match."""
    spec = _make_spec()
    path = _make_synthetic_raster(
        tmp_path,
        transform=Affine(30.0, 0, 368000, 0, -30.0, 5839000),
    )
    result = check_grid_conformity(path, spec, target_resolution=10)
    assert result["resolution_match"] is False


def test_check_grid_conformity_wrong_origin(tmp_path: Path) -> None:
    """Misaligned origin should fail origin_ok."""
    spec = _make_spec()
    path = _make_synthetic_raster(
        tmp_path,
        transform=Affine(10.0, 0, 368005, 0, -10.0, 5839005),
    )
    result = check_grid_conformity(path, spec, target_resolution=10)
    assert result["origin_ok"] is False


def test_check_grid_conformity_wrong_bounds(tmp_path: Path) -> None:
    """Bounds outside AOI should fail bounds_ok."""
    spec = _make_spec()
    path = _make_synthetic_raster(
        tmp_path,
        transform=Affine(10.0, 0, 100000, 0, -10.0, 100000),
    )
    result = check_grid_conformity(path, spec, target_resolution=10)
    assert result["bounds_ok"] is False


# ── compute_radiometric_stats ────────────────────────────────────────────


def test_compute_radiometric_stats_known_values(tmp_path: Path) -> None:
    """Known array values produce correct min/max/mean/std."""
    data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    path = _make_synthetic_raster(
        tmp_path, data=data, height=3, width=3,
        transform=Affine(10.0, 0, 0, 0, -10.0, 0),
    )
    stats = compute_radiometric_stats(path)
    band_key = list(stats.keys())[0]
    b = stats[band_key]
    assert b["valid"] is True
    assert b["min"] == 1.0
    assert b["max"] == 9.0
    assert b["mean"] == 5.0
    assert b["nodata_pct"] == 0.0


def test_compute_radiometric_stats_with_nodata(tmp_path: Path) -> None:
    """Nodata pixels are excluded from statistics."""
    data = np.full((1, 4, 4), -9999.0, dtype=np.float32)
    data[0, 0, 0] = 100.0
    data[0, 1, 1] = 200.0
    path = _make_synthetic_raster(
        tmp_path, data=data, height=4, width=4, nodata=-9999.0,
        transform=Affine(10.0, 0, 0, 0, -10.0, 0),
    )
    stats = compute_radiometric_stats(path)
    band_key = list(stats.keys())[0]
    b = stats[band_key]
    assert b["valid"] is True
    assert b["min"] == 100.0
    assert b["max"] == 200.0
    assert b["mean"] == 150.0
    assert 0.8 < b["nodata_pct"] < 0.9  # 14/16 pixels = 87.5%


def test_compute_radiometric_stats_all_nodata(tmp_path: Path) -> None:
    """100% nodata band returns valid=False."""
    data = np.full((1, 4, 4), -9999.0, dtype=np.float32)
    path = _make_synthetic_raster(
        tmp_path, data=data, height=4, width=4, nodata=-9999.0,
        transform=Affine(10.0, 0, 0, 0, -10.0, 0),
    )
    stats = compute_radiometric_stats(path)
    band_key = list(stats.keys())[0]
    b = stats[band_key]
    assert b["valid"] is False
    assert b["nodata_pct"] == 1.0
    assert b["min"] is None


def test_compute_radiometric_stats_multi_band(tmp_path: Path) -> None:
    """Multi-band raster returns stats for each band."""
    data = np.zeros((3, 5, 5), dtype=np.float32)
    data[0] = 10.0
    data[1] = 20.0
    data[2] = 30.0
    path = _make_synthetic_raster(
        tmp_path, data=data, height=5, width=5,
        transform=Affine(10.0, 0, 0, 0, -10.0, 0),
    )
    stats = compute_radiometric_stats(path)
    assert len(stats) == 3
    for b in stats.values():
        assert b["valid"] is True
        assert b["nodata_pct"] == 0.0


# ── generate_qa_report ───────────────────────────────────────────────────


def test_generate_qa_report_integration(tmp_path: Path) -> None:
    """Full QA report is a valid dict with expected keys."""
    spec = _make_spec()
    cfg = OmegaConf.create({
        "ard": {"process": {"qa": {"nodata_threshold": 0.95}}},
    })
    path = _make_synthetic_raster(tmp_path)
    report = generate_qa_report(path, spec, target_resolution=10, cfg=cfg)
    assert "grid_conformity" in report
    assert "radiometric_stats" in report
    assert "qa_passed" in report
    assert isinstance(report["grid_conformity"], dict)
    assert isinstance(report["radiometric_stats"], dict)


def test_generate_qa_report_skip_grid_check(tmp_path: Path) -> None:
    """Skip grid check returns checked=False and passes on valid data."""
    spec = _make_spec()
    cfg = OmegaConf.create({
        "ard": {"process": {"qa": {"nodata_threshold": 0.95}}},
    })
    path = _make_synthetic_raster(tmp_path)
    report = generate_qa_report(
        path, spec, target_resolution=0, cfg=cfg, skip_grid_check=True,
    )
    assert report["grid_conformity"]["checked"] is False
    assert report["qa_passed"] is True  # has valid data


def test_generate_qa_report_skip_grid_no_data(tmp_path: Path) -> None:
    """Skip grid check with all-nodata raster fails qa_passed."""
    spec = _make_spec()
    cfg = OmegaConf.create({
        "ard": {"process": {"qa": {"nodata_threshold": 0.95}}},
    })
    data = np.full((1, 4, 4), np.nan, dtype=np.float32)
    path = _make_synthetic_raster(
        tmp_path, data=data, height=4, width=4,
        nodata=np.nan,
        transform=Affine(10.0, 0, 0, 0, -10.0, 0),
        crs="EPSG:4326",
    )
    report = generate_qa_report(
        path, spec, target_resolution=0, cfg=cfg, skip_grid_check=True,
    )
    assert report["grid_conformity"]["checked"] is False
    assert report["qa_passed"] is False  # no valid data
