"""Unit tests for the ARD processor module.

These tests do NOT require GCS, GEE, or any external services.
GCS-related functions are tested with mock/dry-run paths.
Reprojection logic is tested with tiny in-memory synthetic rasters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.ard_processor import (
    _compute_target_dims,
    _parse_gcs_uri,
    _parse_scene_id,
    _parse_year,
    _reproject_and_regrid,
    _resolve_nodata,
    _resolve_years,
    process_scene,
)
from berlin_lst_downscaling.data.grid_spec import GridSpec, make_grid_spec

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_spec() -> GridSpec:
    return make_grid_spec(
        origin_x=368000.0,
        origin_y=5839000.0,
        aoi_25833=(368002.320, 5797523.137, 417784.933, 5839258.933),
        wgs84_bbox=(13.0471, 52.3122, 13.7937, 52.6970),
    )


def _make_cfg() -> DictConfig:
    return OmegaConf.create({
        "ard": {
            "time": {"start_year": 2017, "end_year": 2025},
            "output": {
                "bucket": "test-bucket",
                "cog": {
                    "tile_size": 512,
                    "overview_levels": 2,
                    "overview_resampling": "BILINEAR",
                    "nodata": "nan",
                    "dtype": "float32",
                    "compression": "ZSTD",
                },
            },
            "process": {
                "temp_dir": "/tmp/ard_test",
                "resampling": "bilinear",
                "sources": {
                    "landsat": {
                        "gcs_prefix": "ard/dynamic/landsat",
                        "output_prefix": "ard/processed/landsat",
                        "target_resolution": 100,
                    },
                    "sentinel2": {
                        "gcs_prefix": "ard/dynamic/sentinel2",
                        "output_prefix": "ard/processed/sentinel2",
                        "target_resolution": 10,
                    },
                    "ecostress": {
                        "gcs_prefix": "ard/validation/ecostress",
                        "output_prefix": "ard/processed/ecostress",
                        "target_resolution": None,
                    },
                },
                "qa": {"nodata_threshold": 0.95, "quicklook": False},
            },
        },
    })


def _make_tiny_raster(
    tmp_path: Path,
    filename: str = "src.tif",
    *,
    crs: str = "EPSG:25833",
    transform: Affine | None = None,
    height: int = 4,
    width: int = 4,
    bands: int = 2,
    nodata: float | None = None,
    descriptions: list[str] | None = None,
) -> Path:
    """Create a tiny (4×4) multi-band raster for fast tests.

    Args:
        descriptions: Optional band description strings (one per band).
    """
    path = tmp_path / filename
    data = np.random.rand(bands, height, width).astype(np.float32) * 100
    if nodata is not None:
        data[data < 10] = nodata

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": "float32",
        "crs": crs,
        "transform": transform or Affine(
            100.0, 0, 368500,
            0, -100.0, 5838500,
        ),
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions):
                dst.set_band_description(i + 1, desc)
    return path


# ── _parse_gcs_uri ───────────────────────────────────────────────────────


def test_parse_gcs_uri_standard() -> None:
    bucket, path = _parse_gcs_uri("gs://my-bucket/some/path/file.tif")
    assert bucket == "my-bucket"
    assert path == "some/path/file.tif"


def test_parse_gcs_uri_no_scheme() -> None:
    bucket, path = _parse_gcs_uri("my-bucket/file.tif")
    assert bucket == "my-bucket"
    assert path == "file.tif"


# ── _parse_scene_id ──────────────────────────────────────────────────────


def test_parse_scene_id_landsat() -> None:
    cfg = _make_cfg()
    uri = "gs://bucket/ard/dynamic/landsat/2023/LC08_123456_LST.tif"
    scene_id = _parse_scene_id(uri, "landsat", cfg)
    assert scene_id == "LC08_123456"


def test_parse_scene_id_sentinel2() -> None:
    cfg = _make_cfg()
    uri = "gs://bucket/ard/dynamic/sentinel2/2023/S2A_MSIL2A_20230601T100031.tif"
    scene_id = _parse_scene_id(uri, "sentinel2", cfg)
    assert scene_id == "S2A_MSIL2A_20230601T100031"


def test_parse_scene_id_ecostress() -> None:
    cfg = _make_cfg()
    uri = "gs://bucket/ard/validation/ecostress/2023/ECO_L2T_LSTE_12345_COG.tif"
    scene_id = _parse_scene_id(uri, "ecostress", cfg)
    assert scene_id == "ECO_L2T_LSTE_12345"


# ── _parse_year ──────────────────────────────────────────────────────────


def test_parse_year_found() -> None:
    assert _parse_year("gs://bucket/ard/dynamic/landsat/2023/scene.tif") == 2023


def test_parse_year_not_found() -> None:
    assert _parse_year("gs://bucket/scene.tif") == 0


# ── _resolve_years ───────────────────────────────────────────────────────


def test_resolve_years_explicit() -> None:
    cfg = _make_cfg()
    assert _resolve_years(cfg, 2023) == [2023]


def test_resolve_years_range() -> None:
    cfg = OmegaConf.create({"ard": {"time": {"start_year": 2020, "end_year": 2022}}})
    assert _resolve_years(cfg, None) == [2020, 2021, 2022]


# ── _resolve_nodata ──────────────────────────────────────────────────────


def test_resolve_nodata_nan() -> None:
    import math
    assert math.isnan(_resolve_nodata("nan"))


def test_resolve_nodata_number() -> None:
    assert _resolve_nodata(-9999) == -9999.0


# ── _compute_target_dims (Landsat / S2) ──────────────────────────────────


def test_compute_target_dims_landsat(tmp_path: Path) -> None:
    """Landsat (EPSG:25833, 100m) produces origin-aligned output within AOI."""
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        transform=Affine(100.0, 0, 370000, 0, -100.0, 5835000),
        crs="EPSG:25833",
        height=5,
        width=5,
    )
    with rasterio.open(src) as src_ds:
        transform, width, height = _compute_target_dims(
            src_ds, spec, "EPSG:25833", dst_resolution=100.0,
        )
    assert width > 0
    assert height > 0
    # Origin should be aligned to canonical grid
    assert (transform.c - spec.origin_x) % 100 == 0
    assert (spec.origin_y - transform.f) % 100 == 0


def test_compute_target_dims_sentinel2(tmp_path: Path) -> None:
    """Sentinel-2 (EPSG:25833, 10m) produces origin-aligned output."""
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        transform=Affine(10.0, 0, 370000, 0, -10.0, 5835000),
        crs="EPSG:25833",
        height=20,
        width=20,
    )
    with rasterio.open(src) as src_ds:
        transform, width, height = _compute_target_dims(
            src_ds, spec, "EPSG:25833", dst_resolution=10.0,
        )
    assert width > 0
    assert height > 0
    assert (transform.c - spec.origin_x) % 10 == 0
    assert (spec.origin_y - transform.f) % 10 == 0


def test_compute_target_dims_ecostress(tmp_path: Path) -> None:
    """ECOSTRESS passthrough is handled upstream;
    calling this function with dst_resolution=None raises ValueError."""
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        transform=Affine(0.001, 0, 13.2, 0, -0.001, 52.6),
        crs="EPSG:4326",
        height=5,
        width=5,
    )
    with rasterio.open(src) as src_ds:
        with pytest.raises(ValueError, match="dst_resolution=None"):
            _compute_target_dims(
                src_ds, spec, "EPSG:25833", dst_resolution=None,
            )


def test_compute_target_dims_no_overlap(tmp_path: Path) -> None:
    """Scene bounds outside AOI raises ValueError."""
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        transform=Affine(100.0, 0, 100000, 0, -100.0, 100000),
        crs="EPSG:25833",
        height=1,
        width=1,
    )
    with rasterio.open(src) as src_ds:
        with pytest.raises(ValueError, match="do not overlap"):
            _compute_target_dims(
                src_ds, spec, "EPSG:25833", dst_resolution=100.0,
            )


# ── _reproject_and_regrid ────────────────────────────────────────────────


def test_reproject_and_regrid_same_crs(tmp_path: Path) -> None:
    """Reprojecting with same CRS + new transform works (regrid only)."""
    spec = _make_spec()
    src = _make_tiny_raster(tmp_path, crs="EPSG:25833", bands=1)
    dst = tmp_path / "out.tif"

    cog_cfg = OmegaConf.create({
        "tile_size": 64,
        "nodata": "nan",
        "compression": "ZSTD",
        "dtype": "float32",
    })

    result = _reproject_and_regrid(
        src, dst,
        dst_crs="EPSG:25833",
        dst_resolution=100.0,
        dst_dtype="float32",
        resampling_name="bilinear",
        spec=spec,
        cog_cfg=cog_cfg,
    )

    assert result.exists()
    with rasterio.open(result) as out:
        assert out.crs.to_string() == "EPSG:25833"
        assert out.count == 1
        assert out.width > 0
        assert out.height > 0


def test_reproject_and_regrid_ecostress(tmp_path: Path) -> None:
    """ECOSTRESS passthrough preserves native CRS (no reprojection)."""
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 13.2, 0, -0.01, 52.6),
        bands=1,
    )
    dst = tmp_path / "out.tif"

    cog_cfg = OmegaConf.create({
        "tile_size": 64,
        "nodata": "nan",
        "compression": "ZSTD",
        "dtype": "float32",
    })

    result = _reproject_and_regrid(
        src, dst,
        dst_crs="EPSG:25833",
        dst_resolution=None,
        dst_dtype="float32",
        resampling_name="bilinear",
        spec=spec,
        cog_cfg=cog_cfg,
    )

    assert result.exists()
    with rasterio.open(result) as out:
        # Native CRS preserved (not reprojected to 25833)
        assert "4326" in str(out.crs)
        assert out.width > 0
        assert out.height > 0


def test_reproject_and_regrid_mask_names(tmp_path: Path) -> None:
    """Mask/flag bands get nearest-neighbour resampling without errors.

    Creates a multi-band raster with band descriptions matching
    ``cloud_mask`` and ``SCL`` patterns, then verifies that
    ``_reproject_and_regrid`` completes without crashing and produces
    a valid COG with the correct number of bands.
    """
    spec = _make_spec()
    src = _make_tiny_raster(
        tmp_path,
        crs="EPSG:25833",
        bands=4,
        height=8,
        width=8,
        descriptions=["B2", "cloud_mask", "SCL", "B3"],
    )
    dst = tmp_path / "out.tif"

    cog_cfg = OmegaConf.create({
        "tile_size": 16,
        "nodata": "nan",
        "compression": "ZSTD",
        "dtype": "float32",
    })

    result = _reproject_and_regrid(
        src, dst,
        dst_crs="EPSG:25833",
        dst_resolution=100.0,
        dst_dtype="float32",
        resampling_name="bilinear",
        spec=spec,
        cog_cfg=cog_cfg,
    )

    assert result.exists()
    with rasterio.open(result) as out:
        assert out.count == 4
        assert out.crs.to_string() == "EPSG:25833"
        assert out.width > 0
        assert out.height > 0
        assert out.descriptions == ("B2", "cloud_mask", "SCL", "B3")


# ── process_scene dry-run ────────────────────────────────────────────────


def test_process_scene_dry_run(tmp_path: Path) -> None:
    """Dry-run mode returns plan dict with no side effects."""
    cfg = _make_spec()
    spec = cfg

    # Override temp dir for safety
    dc = _make_cfg()
    dc.ard.process.temp_dir = str(tmp_path / "ard_temp")

    uri = "gs://test-bucket/ard/dynamic/landsat/2023/LC08_TEST_LST.tif"
    result = process_scene(uri, "landsat", spec, dc, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["scene_id"] == "LC08_TEST"
    assert result["source"] == "landsat"
    assert "output_path" in result
    assert not list(tmp_path.iterdir()), "Dry run created no files"


# ── QA failure → no upload (regression) ─────────────────────────────────


def test_process_scene_fails_before_upload_on_low_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    """Low AOI coverage must raise before the COG is uploaded to GCS."""
    import shutil as _shutil

    from berlin_lst_downscaling.data import ard_processor

    spec = _make_spec()
    dc = _make_cfg()
    dc.ard.process.temp_dir = str(tmp_path / "ard_temp")
    dc.ard.process.qa.min_aoi_coverage = 0.80  # tight threshold

    # ── Build a tiny 200×200 m raster inside the AOI corner ──
    # 2×2 pixels at 100 m anchored at the AOI origin (368000, 5839000)
    # gives 4 valid pixels inside an AOI of ~2×10⁹ m² → coverage ~0%.
    far_transform = Affine(100.0, 0, 368000.0, 0, -100.0, 5797723.0)
    far_input = _make_tiny_raster(
        tmp_path,
        filename="far_input.tif",
        crs="EPSG:25833",
        transform=far_transform,
        height=2,
        width=2,
        bands=2,
        nodata=np.nan,
        descriptions=["LST", "cloud_mask"],
    )

    # ── Monkeypatch I/O helpers ──
    upload_calls: list[tuple[str, str]] = []

    def fake_download(gcs_uri: str, local_path: Path) -> Path:
        # Copy the synthetic raster to the path _download_from_gcs would write.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(far_input, local_path)
        return local_path

    def fake_upload(local_path: Path, gcs_path: str, bucket_name: str) -> str:  # noqa: ARG001
        upload_calls.append((str(local_path), gcs_path))
        return f"gs://{bucket_name}/{gcs_path}"

    monkeypatch.setattr(ard_processor, "_download_from_gcs", fake_download)
    monkeypatch.setattr(ard_processor, "_upload_to_gcs", fake_upload)

    uri = "gs://test-bucket/ard/dynamic/landsat/2023/LC08_FAR_LST.tif"
    result = process_scene(uri, "landsat", spec, dc, dry_run=False)

    # Status reflects the QA failure…
    assert result["status"] == "error"
    assert "aoi_coverage" in result["error"]
    # …the QA report is attached for debugging…
    assert "qa_report" in result
    assert result["qa_report"]["qa_passed"] is False
    # …and crucially no COG/QA/STAC was ever uploaded to GCS.
    assert upload_calls == [], f"Unexpected uploads: {upload_calls}"
