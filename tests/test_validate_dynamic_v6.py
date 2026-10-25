"""validate_dynamic v6 roster, GDAL auth, and dynamic fail-closed exit code.

Covers task acceptance criteria 1-5 (offline, synthetic local COGs only):
- v6 contract passes on a synthetic 4-band constant COG
- v6 contract fails on an 8-band COG and on a non-constant band
- 8-band (v5) contract still passes for dynamic/full
- the ADC helper is called before gs:// COG opens
- the dynamic pipeline exits non-zero with a failed ledger row left behind
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

_REPO = Path(__file__).resolve().parents[1]
_VALIDATOR_PATH = _REPO / "scripts/validators/validate_dynamic.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_dynamic_v6_test", _VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cog(path: Path, names: list[str], arrays: list[np.ndarray]) -> str:
    assert len(names) == len(arrays)
    h, w = arrays[0].shape
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": len(names),
        "dtype": "float32",
        "crs": CRS.from_epsg(25833),
        "transform": Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        "nodata": float("nan"),
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(zip(names, arrays, strict=True)):
            dst.write(arr.astype(np.float32), i + 1)
            dst.set_band_description(i + 1, name)
    return str(path)


def _v6_names() -> list[str]:
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )

    return list(feature_schema(FEATURE_SCHEMA_VERSION_V6).names_in_family("era5"))


def _v6_constants(value_map: dict[str, float] | None = None) -> list[np.ndarray]:
    defaults = {
        "t2m_scene": 280.0,
        "ssrd_scene": 500.0,
        "wind_speed_10m_scene": 3.0,
        "tp_0_72h": 10.0,
    }
    if value_map:
        defaults.update(value_map)
    names = _v6_names()
    return [np.full((16, 16), defaults[n], dtype=np.float32) for n in names]


def test_v6_roster_comes_from_src_not_a_duplicate_list() -> None:
    """The validator's v6 roster must equal the src feature-schema roster."""
    module = _load_validator()
    names, _units, _ranges = module._v6_era5_contract()
    assert names == _v6_names()
    assert len(names) == 4
    # The validator module must not hard-code a second copy of the 4-band list.
    text = _VALIDATOR_PATH.read_text()
    assert text.count("tp_0_72h") <= 3, "v6 roster looks duplicated in the validator"


def test_v6_contract_passes_on_constant_cog(tmp_path: Path) -> None:
    module = _load_validator()
    names = _v6_names()
    cog = _write_cog(tmp_path / "era5_v6.tif", names, _v6_constants())
    assert module._validate_cog_bands(cog, channel_set="v6") == []


def test_v6_contract_fails_on_8band_cog(tmp_path: Path) -> None:
    """A v6 validation of an 8-band published COG is an error, never a pass."""
    module = _load_validator()
    names8 = list(module._ERA5_BAND_NAMES)
    assert len(names8) == 8
    arrays = [
        np.full((16, 16), 280.0 if n == "t2m_scene" else 1.0, dtype=np.float32) for n in names8
    ]
    cog = _write_cog(tmp_path / "era5_v5.tif", names8, arrays)
    errors = module._validate_cog_bands(cog, channel_set="v6")
    assert any("band count" in e for e in errors)


def test_v6_contract_fails_on_non_constant_band(tmp_path: Path) -> None:
    module = _load_validator()
    names = _v6_names()
    # 128x128 so the 1/32 overview window (4x4) still resolves the gradient;
    # a tiny raster would average it away into a single overview pixel.
    arrays = [np.full((128, 128), 280.0, dtype=np.float32) for _ in names]
    ramp = np.tile(np.linspace(270.0, 285.0, 128, dtype=np.float32), (128, 1))
    arrays[names.index("t2m_scene")] = ramp
    for n, arr in zip(names, arrays, strict=True):
        if n != "t2m_scene":
            arr[:] = {"ssrd_scene": 500.0, "wind_speed_10m_scene": 3.0, "tp_0_72h": 10.0}[n]
    cog = _write_cog(tmp_path / "era5_v6_ramp.tif", names, arrays)
    errors = module._validate_cog_bands(cog, channel_set="v6")
    assert any("not spatially constant" in e for e in errors)


def test_v6_contract_fails_outside_valid_range(tmp_path: Path) -> None:
    module = _load_validator()
    names = _v6_names()
    arrays = _v6_constants({"t2m_scene": 100.0})  # below the (200, 350) K range
    cog = _write_cog(tmp_path / "era5_v6_cold.tif", names, arrays)
    errors = module._validate_cog_bands(cog, channel_set="v6")
    assert any("outside" in e for e in errors)


def test_v5_contract_still_passes_for_full(tmp_path: Path) -> None:
    """The published dynamic/full 8-band contract keeps validating as v5."""
    module = _load_validator()
    names8 = list(module._ERA5_BAND_NAMES)
    in_range = {
        "t2m_scene": 280.0,
        "ssrd_scene": 500.0,
        "ssrd_antecedent_72h_mean": 200.0,
        "vpd_scene": 1.0,
        "wind_speed_10m_scene": 3.0,
        "tp_0_24h": 1.0,
        "tp_24_48h": 2.0,
        "tp_48_72h": 3.0,
    }
    arrays = [np.full((16, 16), in_range[n], dtype=np.float32) for n in names8]
    cog = _write_cog(tmp_path / "era5_v5_ok.tif", names8, arrays)
    assert module._validate_cog_bands(cog) == []
    assert module._validate_cog_bands(cog, channel_set="v5") == []


def test_v5_rejects_v6_cog_as_mismatch(tmp_path: Path) -> None:
    """Validating a 4-band v6 COG against the default 8-band contract fails."""
    module = _load_validator()
    names = _v6_names()
    cog = _write_cog(tmp_path / "era5_v6_as_v5.tif", names, _v6_constants())
    errors = module._validate_cog_bands(cog, channel_set="v5")
    assert any("band count" in e for e in errors)


def test_unknown_channel_set_is_fail_closed(tmp_path: Path) -> None:
    module = _load_validator()
    names = _v6_names()
    cog = _write_cog(tmp_path / "era5_x.tif", names, _v6_constants())
    errors = module._validate_cog_bands(cog, channel_set="v9")
    assert any("unknown era5 channel set" in e for e in errors)
    prov_errors = module._validate_era5_provenance("gs://nope/prov.json", channel_set="v9")
    assert any("unknown era5 channel set" in e for e in prov_errors)


def test_provenance_channel_set_mismatch_is_an_error(monkeypatch) -> None:
    import json

    import berlin_lst_downscaling.data.io.storage as storage

    module = _load_validator()
    v6_prov = json.dumps(
        {
            "source_metadata": {
                "era5_variables": list(module._ERA5_INPUT_VARIABLES),
                "channel_set": "v6",
            }
        }
    ).encode()
    monkeypatch.setattr(storage, "read_bytes", lambda uri: v6_prov)
    assert module._validate_era5_provenance("gs://x/prov.json", channel_set="v6") == []
    assert module._validate_era5_provenance("gs://x/prov.json", channel_set="v5") != []

    v5_prov = json.dumps(
        {"source_metadata": {"era5_variables": list(module._ERA5_INPUT_VARIABLES)}}
    ).encode()
    monkeypatch.setattr(storage, "read_bytes", lambda uri: v5_prov)
    assert module._validate_era5_provenance("gs://x/prov.json", channel_set="v5") == []
    assert module._validate_era5_provenance("gs://x/prov.json", channel_set="v6") != []


def test_main_calls_ensure_gdal_gcs_auth(monkeypatch, tmp_path: Path) -> None:
    """A Mac run without exported GOOGLE_APPLICATION_CREDENTIALS still works."""
    import berlin_lst_downscaling.data.io.gdal_gcs as gdal_gcs

    module = _load_validator()
    calls: list[bool] = []
    monkeypatch.setattr(gdal_gcs, "ensure_gdal_gcs_auth", lambda: calls.append(True) or None)
    # main() imports the helper from the same module path, so patch there too.
    monkeypatch.setattr(
        "berlin_lst_downscaling.data.io.gdal_gcs.ensure_gdal_gcs_auth",
        lambda: calls.append(True) or None,
    )
    monkeypatch.setattr(module, "_read_ledger", lambda root: _empty_ledger())
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--output-root", str(tmp_path), "--expected-role", "anchor", "--progress-only"],
    )
    assert module.main() == 0
    assert calls, "ensure_gdal_gcs_auth was not called"


def _empty_ledger() -> dict:
    return {
        "total_rows": 0,
        "counts": {},
        "non_done": [],
        "role_counts": {},
        "missing_role": 0,
    }


def _dynamic_cfg(tmp_path: Path):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "output_root": str(tmp_path),
            "manifest_uri": "dummy",
            "geometry_mapping_uri": "dummy",
            "scene_ids": [],
            "years": [],
            "dataset_role": "anchor",
            "expected_scene_count": None,
            "era5_channel_set": "v5",
            "era5_representation": "bilinear",
            "derived_root": "",
        }
    )


def _fake_ok_manifest():
    from types import SimpleNamespace

    return SimpleNamespace(ok=True, scenes=[], errors=[], total_rows=0, manifest_hash="h")


def _fake_ok_mapping():
    from types import SimpleNamespace

    return SimpleNamespace(
        ok=True,
        errors=[],
        mapping=SimpleNamespace(
            uri="dummy",
            content_hash="m",
            version=1,
            year_to_vintage={},
            vegetation_horizon_hash="",
        ),
    )


def _seed_ledger(output_root: Path, statuses: list[tuple[str, str, str, str]]) -> None:
    from berlin_lst_downscaling.data.dynamic.paths import ledger_path
    from berlin_lst_downscaling.data.secondary.ledger import (
        SecondaryLedger,
        SecondaryLedgerRow,
    )

    led = SecondaryLedger.open(ledger_path(str(output_root)))
    for item_id, source, period, status in statuses:
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source=source,
                period_or_vintage=period,
                status=status,
                run_id="seed",
                role="anchor",
            )
        )


def test_pipeline_exits_nonzero_when_a_ledger_row_is_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from berlin_lst_downscaling.data.dynamic import pipeline as dynamic_pipeline

    monkeypatch.setattr(
        dynamic_pipeline, "load_landsat_anchors", lambda *a, **k: _fake_ok_manifest()
    )
    monkeypatch.setattr(
        dynamic_pipeline, "load_geometry_mapping", lambda *a, **k: _fake_ok_mapping()
    )
    _seed_ledger(
        tmp_path,
        [
            ("era5_land_S1", "era5_land", "S1", "failed"),
            ("shadow_building_S1", "shadow_building", "S1", "done"),
        ],
    )
    assert dynamic_pipeline.run_dynamic(_dynamic_cfg(tmp_path), run_id="t1") == 1


def test_pipeline_exits_zero_when_all_rows_are_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from berlin_lst_downscaling.data.dynamic import pipeline as dynamic_pipeline
    from berlin_lst_downscaling.data.dynamic.paths import ledger_path
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    monkeypatch.setattr(
        dynamic_pipeline, "load_landsat_anchors", lambda *a, **k: _fake_ok_manifest()
    )
    monkeypatch.setattr(
        dynamic_pipeline, "load_geometry_mapping", lambda *a, **k: _fake_ok_mapping()
    )
    _seed_ledger(
        tmp_path,
        [
            ("era5_land_S1", "era5_land", "S1", "done"),
            ("shadow_building_S1", "shadow_building", "S1", "done"),
        ],
    )
    assert dynamic_pipeline.run_dynamic(_dynamic_cfg(tmp_path), run_id="t2") == 0
    led = SecondaryLedger.open(ledger_path(str(tmp_path)))
    assert dynamic_pipeline.find_failed_ledger_rows(led) == []


def test_find_failed_ledger_rows_lists_failed_and_exporting(tmp_path: Path) -> None:
    from berlin_lst_downscaling.data.dynamic import pipeline as dynamic_pipeline

    _seed_ledger(
        tmp_path,
        [
            ("a", "era5_land", "S1", "failed"),
            ("b", "era5_land", "S2", "exporting"),
            ("c", "era5_land", "S3", "done"),
        ],
    )
    from berlin_lst_downscaling.data.dynamic.paths import ledger_path
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    led = SecondaryLedger.open(ledger_path(str(tmp_path)))
    errors = dynamic_pipeline.find_failed_ledger_rows(led)
    assert len(errors) == 2
    assert any("status=failed" in e for e in errors)
    assert any("status=exporting" in e for e in errors)
