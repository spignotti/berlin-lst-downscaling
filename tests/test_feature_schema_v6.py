"""features/v6 roster: canopy cover, SVF radius diagnostic, ERA5 switch.

The published features/v5 hashes and the 28-channel roster stay put.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from hydra import compose, initialize_config_dir
from odc.geo.geobox import GeoBox
from rasterio.crs import CRS

from berlin_lst_downscaling.data.dynamic.era5 import (
    apply_era5_representation,
    reduce_era5_channels,
)
from berlin_lst_downscaling.data.dynamic.schema import config_hash_for_era5
from berlin_lst_downscaling.data.eda.frames import feature_names_in, feature_scale_table
from berlin_lst_downscaling.data.features.contracts import (
    CANOPY_COVER_THRESHOLD_M_DEFAULT,
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    era5_representation_for_schema,
    feature_schema,
    resolve_era5_representation,
    schema_from_provenance,
)
from berlin_lst_downscaling.data.features.schema import config_hash_for_features
from berlin_lst_downscaling.data.features.stack import assemble_named_channels
from berlin_lst_downscaling.data.secondary.canopy_cover import (
    canopy_cover_fraction,
    canopy_cover_fraction_on_grid,
)
from berlin_lst_downscaling.data.secondary.svf_radius_diagnostic import (
    SVF_DIAGNOSTIC_RADII_M,
    diagnose_svf_radii,
    synthetic_dsm,
)
from berlin_lst_downscaling.data.secondary.vegetation_height import contract_for_vegetation_height
from berlin_lst_downscaling.data.training.contracts import (
    precip_channels_for,
    scaler_transform_for,
    training_policy_hash,
)
from berlin_lst_downscaling.modeling.channels import (
    feature_order_for,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.guards import contract_invariants

_REPO = Path(__file__).resolve().parents[1]


def _feature_hash() -> str:
    return config_hash_for_features(
        manifest_hash="m",
        geometry_mapping_hash="g",
        ard_ledger_hash="a",
        static_sources_ledger_hash="s",
        static_derived_ledger_hash="d",
        dynamic_ledger_hash="y",
        aoi_fingerprint="o",
        vegetation_carry_forward_geometry_id="v",
        lod_coverage_fingerprints={"2024": "c"},
        lod_cog_fingerprints={"2024": "k"},
    )


def _compose(config_dir: str, config_name: str):
    with initialize_config_dir(config_dir=str(_REPO / config_dir), version_base=None):
        return compose(config_name=config_name)


def test_published_v5_hashes_and_roster_are_unchanged() -> None:
    assert training_policy_hash() == "08c2ec4d5a257c7e"
    # Default path is the 28-channel roster plus the data-correctness
    # policy keys. It does not include the v6 canopy threshold or ERA5 switch.
    assert _feature_hash() == "a8d231b313a7ad73"
    assert config_hash_for_era5("m", "g", "root") == "f381861cc6b460da"
    assert "building_height_std" in FEATURE_CHANNEL_NAMES
    assert "building_height_max" in FEATURE_CHANNEL_NAMES
    assert "vegetation_height_max" in FEATURE_CHANNEL_NAMES
    assert feature_schema(FEATURE_SCHEMA_VERSION).n_channels == len(FEATURE_CHANNEL_NAMES)
    v5_yaml = (_REPO / "configs" / "features" / "v5.yaml").read_text()
    assert "feature_schema_version" not in v5_yaml
    assert "canopy_cover_fraction" not in v5_yaml


def test_v6_roster_is_derived_and_drops_the_decided_channels() -> None:
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    assert schema.n_channels == len(schema.names)
    assert schema.names_in_family("era5") == (
        "t2m_scene",
        "ssrd_scene",
        "wind_speed_10m_scene",
        "tp_0_72h",
    )
    removed = {
        "building_height_std",
        "building_height_max",
        "vegetation_height_max",
        "ssrd_antecedent_72h_mean",
        "vpd_scene",
        "tp_0_24h",
        "tp_24_48h",
        "tp_48_72h",
    }
    assert removed.isdisjoint(schema.names)
    assert "canopy_cover_fraction" in schema.names
    assert "vegetation_height_mean" in schema.names
    assert schema.canopy_cover_threshold_m == CANOPY_COVER_THRESHOLD_M_DEFAULT
    doc = (_REPO / "docs" / "features-v6.md").read_text()
    positions = [doc.index(f"`{name}`") for name in schema.names]
    assert positions == sorted(positions)
    v6_yaml = (_REPO / "configs" / "features" / "v6.yaml").read_text()
    assert "n_active_channels" not in v6_yaml


def test_v6_configs_resolve_scene_value_and_the_registry_width() -> None:
    features = _compose("configs/features", "v6")
    assert int(features.feature_schema_version) == FEATURE_SCHEMA_VERSION_V6
    assert features.era5_representation == "scene_value"
    assert float(features.canopy_cover_threshold_m) == 2.5
    modeling = _compose("configs/modeling", "v6")
    contract_invariants(modeling)
    selection = selection_from_config(modeling)
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    assert modeling.data.n_active_channels is None
    assert selection.n_active == schema.n_channels
    assert selection.names == schema.names
    assert feature_order_for(selection) == "v6_registry"
    derived = (_REPO / "configs" / "static_derived" / "v6.yaml").read_text()
    assert "svf_max_radius: 3" in derived
    text = (_REPO / "configs" / "features" / "v6.yaml").read_text()
    assert "2017-2026-s2-mosaic-v10" in text
    assert "manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet" in text
    assert "static/derived/lod2-area-exact-v2" in text
    assert "dynamic/lod2-area-exact-v2" in text
    assert "static/sources/lod2-area-exact-v2" in text
    assert 'output_root: "gs://berlin-lst-training-data/features/v6"' in text
    assert "qa-avg-20261019" not in text


def test_canopy_cover_fraction_uses_a_configurable_threshold() -> None:
    height = np.zeros((20, 10), dtype=np.float32)
    height[:10, :] = 3.0
    height[10:, :] = 2.5
    default = canopy_cover_fraction(height)
    stricter = canopy_cover_fraction(height, threshold_m=3.0)
    assert default.shape == (2, 1)
    assert default[0, 0] == pytest.approx(1.0)
    assert default[1, 0] == pytest.approx(0.0)
    assert stricter[0, 0] == pytest.approx(0.0)
    height[0, 0] = np.nan
    with_gap = canopy_cover_fraction(height[:10, :])
    assert with_gap[0, 0] == pytest.approx(0.99)
    with pytest.raises(ValueError):
        canopy_cover_fraction(height, threshold_m=-1.0)


def test_canopy_cover_on_a_windowed_raster_matches_the_block_mean(tmp_path: Path) -> None:
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)
    height_dm = np.zeros((20, 20), dtype=np.float32)
    height_dm[:10, :10] = 40.0  # 4 m, above 2.5 m
    height_dm[:10, 10:] = 10.0  # 1 m, below the threshold
    profile = {
        "driver": "GTiff",
        "width": 20,
        "height": 20,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(25833),
        "transform": transform,
        "nodata": -9999.0,
    }
    path = tmp_path / "canopy_cover_sample.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(height_dm, 1)
    grid = GeoBox(shape=(2, 2), affine=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0), crs="EPSG:25833")
    with rasterio.open(path) as src:
        cover = canopy_cover_fraction_on_grid(src, grid, threshold_m=2.5, scale_to_m=0.1)
    assert cover[0, 0] == pytest.approx(1.0)
    assert cover[0, 1] == pytest.approx(0.0)
    assert contract_for_vegetation_height().output_bands.__len__() == 2
    with_canopy = contract_for_vegetation_height(canopy_cover_threshold_m=2.5)
    assert [band.name for band in with_canopy.output_bands][-1] == "canopy_cover_fraction"


def test_era5_v6_channels_and_representation_switch() -> None:
    shape = (2, 3)
    bands = {
        "t2m_scene": np.array([[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]], dtype=np.float32),
        "ssrd_scene": np.full(shape, 100.0, dtype=np.float32),
        "ssrd_antecedent_72h_mean": np.full(shape, 50.0, dtype=np.float32),
        "vpd_scene": np.full(shape, 1.0, dtype=np.float32),
        "wind_speed_10m_scene": np.array([[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]], dtype=np.float32),
        "tp_0_24h": np.full(shape, 1.0, dtype=np.float32),
        "tp_24_48h": np.full(shape, 2.0, dtype=np.float32),
        "tp_48_72h": np.full(shape, 3.0, dtype=np.float32),
    }
    target = feature_schema(FEATURE_SCHEMA_VERSION_V6).names_in_family("era5")
    reduced = reduce_era5_channels(bands, target_names=target)
    assert tuple(reduced) == target
    assert reduced["tp_0_72h"][0, 0] == pytest.approx(6.0)
    assert "vpd_scene" not in reduced
    scene = apply_era5_representation(reduced, None)
    assert np.all(scene["t2m_scene"] == pytest.approx(282.5))
    assert np.all(scene["wind_speed_10m_scene"] == pytest.approx(3.0))
    bilinear = apply_era5_representation(reduced, "bilinear")
    assert bilinear["t2m_scene"][0, 0] == pytest.approx(280.0)
    assert bilinear["t2m_scene"][1, 2] == pytest.approx(285.0)
    mask = np.array([[True, False, False], [False, False, False]])
    masked = apply_era5_representation(reduced, "scene_value", valid_mask=mask)
    assert np.all(masked["t2m_scene"] == pytest.approx(280.0))
    v6 = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    assert resolve_era5_representation(None) == "scene_value"
    assert era5_representation_for_schema(v6, None) == "scene_value"
    assert era5_representation_for_schema(feature_schema(FEATURE_SCHEMA_VERSION), None) is None
    assert precip_channels_for(v6) == ("tp_0_72h",)
    assert scaler_transform_for("tp_0_72h", v6) == "log1p_zscore"
    assert scaler_transform_for("t2m_scene", v6) == "zscore"
    assert scaler_transform_for("shadow_building", v6) == "identity"
    assert scaler_transform_for("tp_0_24h") == "log1p_zscore"
    ordered = assemble_named_channels(
        {name: np.zeros((1, 1), dtype=np.float32) for name in v6.names},
        v6,
    )
    assert len(ordered) == v6.n_channels


def test_svf_radius_diagnostic_reports_correlation_and_runtime() -> None:
    report = diagnose_svf_radii(
        synthetic_dsm((48, 48)),
        radii_m=(30.0, 50.0),
        reference_pixels=1000,
    )
    assert [summary.radius_m for summary in report.summaries] == [30.0, 50.0]
    assert report.summaries[0].radius_cells == 3
    assert report.summaries[0].n_finite == 48 * 48
    assert report.summaries[0].estimated_full_grid_s > 0
    assert report.correlation[(30.0, 50.0)] is not None
    assert "read-only" in report.format_report().lower() or "Runtime estimate" in report.note
    assert SVF_DIAGNOSTIC_RADII_M == (30.0, 50.0, 100.0, 200.0)
    with pytest.raises(ValueError):
        diagnose_svf_radii(np.zeros((600, 600), dtype=np.float32), radii_m=(30.0,))


def test_svf_diagnostic_cli_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = set(tmp_path.iterdir())
    script = _REPO / "scripts" / "validators" / "diagnose_svf_radius.py"
    spec = importlib.util.spec_from_file_location("diagnose_svf_radius", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose_svf_radius.py", "--synthetic", "--radii-m", "30", "50"],
    )
    assert module.main() == 0
    captured = capsys.readouterr().out
    assert "30" in captured
    assert "estimated_full_s" in captured
    assert set(tmp_path.iterdir()) == before


def test_schema_from_provenance_reads_the_published_28_channel_versions() -> None:
    for version in (2, 3, FEATURE_SCHEMA_VERSION):
        schema = schema_from_provenance({"schema_version": version})
        assert schema.version == version
        assert schema.names == FEATURE_CHANNEL_NAMES
    v6 = schema_from_provenance(
        {
            "schema_version": FEATURE_SCHEMA_VERSION_V6,
            "canopy_cover_threshold_m": CANOPY_COVER_THRESHOLD_M_DEFAULT,
        }
    )
    assert v6.n_channels == feature_schema(FEATURE_SCHEMA_VERSION_V6).n_channels
    assert "canopy_cover_fraction" in v6.names
    with pytest.raises(ValueError, match="unknown feature schema version"):
        schema_from_provenance({"schema_version": 1})
    with pytest.raises(ValueError, match="unknown feature schema version"):
        feature_schema(3)


def test_scene_value_is_the_aoi_mean_of_the_bilinear_field() -> None:
    field = np.array([[10.0, 30.0], [1000.0, 1000.0]], dtype=np.float32)
    aoi = np.array([[True, True], [False, False]])
    written = apply_era5_representation({"t2m_scene": field}, "scene_value", valid_mask=aoi)
    assert float(written["t2m_scene"][1, 1]) == pytest.approx(20.0)
    again = apply_era5_representation(written, "scene_value", valid_mask=aoi)
    assert np.all(again["t2m_scene"] == pytest.approx(20.0))
    with pytest.raises(ValueError, match="AOI file fingerprint"):
        config_hash_for_era5("m", "g", "root", channel_set="v6", representation="scene_value")
    v6_hash = config_hash_for_era5(
        "m",
        "g",
        "root",
        channel_set="v6",
        representation="scene_value",
        aoi_fingerprint="aoi-a",
    )
    other_aoi = config_hash_for_era5(
        "m",
        "g",
        "root",
        channel_set="v6",
        representation="scene_value",
        aoi_fingerprint="aoi-b",
    )
    bilinear_hash = config_hash_for_era5(
        "m", "g", "root", channel_set="v6", representation="bilinear"
    )
    assert v6_hash != bilinear_hash
    assert v6_hash != other_aoi
    assert config_hash_for_era5("m", "g", "root") == "f381861cc6b460da"
    assert config_hash_for_era5("m", "g", "root", aoi_fingerprint="aoi-a") == "f381861cc6b460da"


def test_eda_tables_follow_the_roster_in_the_frame() -> None:
    v6_names = list(feature_schema(FEATURE_SCHEMA_VERSION_V6).names)
    assert feature_names_in(pd.DataFrame(columns=["scene_id", "lst_k", *v6_names])) == v6_names
    assert feature_names_in(pd.DataFrame(columns=["scene_id", *FEATURE_CHANNEL_NAMES])) == list(
        FEATURE_CHANNEL_NAMES
    )
    frame = pd.DataFrame(
        {
            "scene_id": ["a", "a", "b"],
            "canopy_cover_fraction": [0.0, 1.0, 0.5],
            "tp_0_72h": [0.0, 1.0, 2.0],
            "building_height_std": [9.0, 9.0, 9.0],
        }
    )
    names = feature_names_in(frame)
    assert names == ["canopy_cover_fraction", "tp_0_72h"]
    table = feature_scale_table(frame)
    assert set(table["channel"]) == {"canopy_cover_fraction", "tp_0_72h"}
