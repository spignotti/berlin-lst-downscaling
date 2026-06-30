"""Unit tests for the STAC writer module.

Tests solar position, config hashing, overflight datetime parsing,
and the full STAC item structure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine

from berlin_lst_downscaling.data.stac_writer import (
    _config_hash,
    _parse_overflight_datetime,
    _solar_position,
    write_stac_item,
)

# Module-level constant avoids UP017 on every ``tzinfo=`` kwarg
_UTC = timezone.utc  # noqa: UP017

# ── Solar position ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lat", "lon", "dt", "expected_elev_low", "expected_az_low"),
    [
        # Berlin, summer solstice noon UTC → high elevation, azimuth south
        (
            52.52, 13.405,
            datetime(2023, 6, 21, 12, 0, 0, tzinfo=_UTC),
            55, 170,
        ),
        # Berlin, winter solstice noon UTC → low elevation
        (
            52.52, 13.405,
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=_UTC),
            10, 160,
        ),
        # Equator, equinox noon → elevation near 90°, azimuth variable
        (
            0.0, 0.0,
            datetime(2023, 3, 20, 12, 0, 0, tzinfo=_UTC),
            85, 160,
        ),
    ],
)
def test_solar_position(
    lat: float,
    lon: float,
    dt: datetime,
    expected_elev_low: float,
    expected_az_low: float,
) -> None:
    az, elev = _solar_position(lat, lon, dt)
    # Elevation should be in a reasonable range (> lower bound)
    assert elev >= expected_elev_low, f"Expected elev >= {expected_elev_low}, got {elev}"
    assert 0 <= az < 360, f"Azimuth {az} out of [0, 360)"
    assert 0 <= elev <= 90, f"Elevation {elev} out of [0, 90]"


def test_solar_position_deterministic() -> None:
    dt = datetime(2023, 6, 21, 12, 0, 0, tzinfo=_UTC)
    a1, e1 = _solar_position(52.52, 13.405, dt)
    a2, e2 = _solar_position(52.52, 13.405, dt)
    assert a1 == a2
    assert e1 == e2


def test_solar_position_nighttime() -> None:
    """Berlin midnight in December → no sun (elevation < 0 or negative)."""
    dt = datetime(2023, 12, 21, 0, 0, 0, tzinfo=_UTC)
    az, elev = _solar_position(52.52, 13.405, dt)
    assert elev < 0, f"Expected negative elevation at midnight, got {elev}"
    assert isinstance(az, float)


# ── Config hash ─────────────────────────────────────────────────────────


def test_config_hash() -> None:
    """Config hash: deterministic, 12-char, sensitive to value diff, order-independent."""
    # Deterministic + 12 chars
    cfg = {"ard": {"time": {"start_year": 2017, "end_year": 2025}}}
    h1, h2 = _config_hash(cfg), _config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 12

    # Sensitive to value diff
    cfg_diff = {"ard": {"time": {"start_year": 2020}}}
    assert _config_hash(cfg) != _config_hash(cfg_diff)

    # Order-independent
    cfg_a = {"ard": {"a": 1, "b": 2}}
    cfg_b = {"ard": {"b": 2, "a": 1}}
    assert _config_hash(cfg_a) == _config_hash(cfg_b)


# ── Overflight datetime ─────────────────────────────────────────────────


def test_overflight_sentinel2_full() -> None:
    dt = _parse_overflight_datetime(
        "20230601T100029_20230601T100029_T33UUT", "sentinel2", 2023
    )
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 6
    assert dt.day == 1
    assert dt.hour == 10
    assert dt.minute == 0
    assert dt.second == 29
    assert dt.tzinfo is not None


def test_overflight_landsat_date() -> None:
    dt = _parse_overflight_datetime(
        "LC08_L2SP_193023_20230511_20230516_02_T1", "landsat", 2023
    )
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 5
    assert dt.day == 11
    assert dt.hour == 10  # approximate overpass time
    assert dt.minute == 0
    assert dt.tzinfo is not None


def test_overflight_ecostress() -> None:
    dt = _parse_overflight_datetime(
        "ECO_L2T_LSTE_27651_001_20230712T124527_0713_01", "ecostress", 2023
    )
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 7
    assert dt.day == 12
    assert dt.hour == 12
    assert dt.minute == 45
    assert dt.second == 27


def test_overflight_fallback() -> None:
    """Unknown source / unparseable date falls back to July 1 of given year."""
    dt = _parse_overflight_datetime("NO_MATCH", "unknown_source", 2020)
    assert dt is not None
    assert dt.year == 2020
    assert dt.month == 7
    assert dt.day == 1


# ── Full STAC item ──────────────────────────────────────────────────────


def _write_tiny_cog(
    tmp_path: Path,
    name: str = "output.tif",
    bands: int = 1,
) -> Path:
    """Write a tiny single-band COG with known bounds (blocksize ≥128 for COG)."""
    path = tmp_path / name
    data = np.ones((bands, 4, 4), dtype=np.float32) * 300.0  # Kelvin-ish
    transform = Affine(100.0, 0, 368000, 0, -100.0, 5839000)
    profile = {
        "driver": "COG",
        "height": 4,
        "width": 4,
        "count": bands,
        "dtype": "float32",
        "crs": "EPSG:25833",
        "transform": transform,
        "nodata": float("nan"),
        "compress": "LZW",
        "blocksize": 128,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_stac_item_contains_all_scope_fields(tmp_path: Path) -> None:
    """Verify all Scope-specified metadata fields are present."""
    cog = _write_tiny_cog(tmp_path, name="output.tif")
    input_cog = _write_tiny_cog(tmp_path, name="input.tif")

    config_dict: dict[str, Any] = {
        "ard": {"time": {"start_year": 2017, "end_year": 2025}},
    }

    item = write_stac_item(
        cog_path=cog,
        scene_id="LC08_L2SP_193023_20230511_20230516_02_T1",
        source="landsat",
        year=2023,
        qa_report={"cloud_fraction": 0.15},
        output_bucket="test-bucket",
        output_prefix="ard/processed/landsat",
        input_cog_path=input_cog,
        config_dict=config_dict,
        collection_id="LANDSAT/LC08/C02/T1_L2",
    )

    props = item["properties"]

    # Core STAC fields
    assert "datetime" in props, "missing datetime"
    assert item["stac_version"] == "1.1.0"
    assert item["type"] == "Feature"

    # Scope-specified fields
    assert "overflight_datetime" in props
    assert props["overflight_datetime"] == "2023-05-11T10:00:00+00:00"

    assert "constellation" in props
    assert "instruments" in props
    assert "gsd" in props
    assert "processing:level" in props
    assert "processing:version" in props
    assert "processing:config_hash" in props
    assert len(props["processing:config_hash"]) == 12

    assert "processing:collection_id" in props
    assert props["processing:collection_id"] == "LANDSAT/LC08/C02/T1_L2"

    assert "proj:transform" in props
    assert len(props["proj:transform"]) == 6, "proj:transform must be [a,b,c,d,e,f]"
    assert props["proj:transform"][0] == 100.0  # x-resolution

    assert "cloud_fraction" in props
    assert props["cloud_fraction"] == 0.15

    # Solar angles should be present when input_cog_path is given
    assert "sun_azimuth" in props
    assert "sun_elevation" in props
    # Berlin (reprojected from EPSG:25833), 2023-05-11 10:00 UTC → daytime
    assert 0 <= props["sun_azimuth"] < 360
    assert props["sun_elevation"] > 30, (
        f"Expected daytime elevation > 30°, got {props['sun_elevation']}"
    )

    # Assets
    assert "cog" in item["assets"]
    assert "qa-json" in item["assets"]
    assert item["assets"]["cog"]["href"].startswith("gs://")

    # Links
    assert len(item["links"]) == 1
    assert item["links"][0]["rel"] == "self"
    assert item["links"][0]["href"].endswith("_stac.json")

    # Geometry
    assert "geometry" in item
    assert "bbox" in item
    assert "crs" in item
    assert "EPSG:25833" in item["crs"]

    # Check JSON serializable (no NaNs in properties)
    raw = json.dumps(item, default=str)
    parsed = json.loads(raw)
    assert parsed["type"] == "Feature"


def test_stac_item_without_input_cog(tmp_path: Path) -> None:
    """STAC item still works when input_cog_path is not given."""
    cog = _write_tiny_cog(tmp_path)

    item = write_stac_item(
        cog_path=cog,
        scene_id="LC08_TEST",
        source="landsat",
        year=2023,
        qa_report=None,
        output_bucket="test-bucket",
        output_prefix="ard/processed/landsat",
    )

    assert item["id"] == "LC08_TEST"
    # Solar angles not present without input COG
    assert "sun_azimuth" not in item["properties"]
    assert "sun_elevation" not in item["properties"]
    assert "processing:config_hash" not in item["properties"]
    assert "processing:collection_id" not in item["properties"]


def test_stac_item_sentinel2(tmp_path: Path) -> None:
    """Verify full timestamp for S2 scenes."""
    cog = _write_tiny_cog(tmp_path)
    input_cog = _write_tiny_cog(tmp_path, name="input.tif")

    item = write_stac_item(
        cog_path=cog,
        scene_id="20230601T100029_20230601T100029_T33UUT",
        source="sentinel2",
        year=2023,
        qa_report={"cloud_fraction": 0.0},
        output_bucket="test-bucket",
        output_prefix="ard/processed/sentinel2",
        input_cog_path=input_cog,
        config_dict={},
        collection_id="COPERNICUS/S2_SR_HARMONIZED",
    )

    dt = item["properties"]["datetime"]
    assert "T10:00:29" in dt, f"Expected T10:00:29 in datetime, got {dt}"
    assert item["properties"]["start_datetime"] == dt
    assert item["properties"]["overflight_datetime"] == dt


def test_stac_item_ecostress(tmp_path: Path) -> None:
    """Verify ECOSTRESS scene with full timestamp."""
    cog = _write_tiny_cog(tmp_path)
    input_cog = _write_tiny_cog(tmp_path, name="input.tif")

    item = write_stac_item(
        cog_path=cog,
        scene_id="ECO_L2T_LSTE_27651_001_20230712T124527_0713_01",
        source="ecostress",
        year=2023,
        qa_report=None,
        output_bucket="test-bucket",
        output_prefix="ard/processed/ecostress",
        input_cog_path=input_cog,
        config_dict={},
        collection_id="ECO_L2T_LSTE/002",
    )
    assert item["properties"]["processing:collection_id"] == "ECO_L2T_LSTE/002"
    assert "T12:45:27" in item["properties"]["overflight_datetime"]
