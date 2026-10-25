"""ARD v10 coverage-gate exclusions + validator gs:// auth (offline, small)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
_VALIDATE_ARD = _REPO / "scripts/validators/validate_ard.py"

GATE_ERROR_1 = (
    "Sentinel-2 mosaic AOI coverage failed: missing 560178/8922403 pixels "
    "(0.06277345) > max_missing_aoi_fraction=0.005; "
    "item_ids=['S2A_MSIL2A_20210814T102031_R065_T32UQD_20210814T215405']"
)
GATE_ERROR_2 = (
    "Sentinel-2 mosaic AOI coverage failed: missing 1629098/8922403 pixels "
    "(0.18257300) > max_missing_aoi_fraction=0.005; item_ids=['S2B_X']"
)


def _load_validate_ard(name: str):
    spec = importlib.util.spec_from_file_location(name, _VALIDATE_ARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── GDAL GCS auth helper ─────────────────────────────────────────────


def test_gdal_gcs_patch_points_at_adc_only_when_needed(tmp_path, monkeypatch) -> None:
    from berlin_lst_downscaling.data.io import gdal_gcs

    adc = tmp_path / "adc.json"
    adc.write_text("{}")

    # Already-authenticated env stays untouched.
    assert gdal_gcs.gdal_gcs_env_patch({"GOOGLE_APPLICATION_CREDENTIALS": "/x"}, None) == {}
    assert (
        gdal_gcs.gdal_gcs_env_patch(
            {"GS_SECRET_ACCESS_KEY": "a", "GS_ACCESS_KEY_ID": "b"}, str(adc)
        )
        == {}
    )
    assert gdal_gcs.gdal_gcs_env_patch({"GS_OAUTH2_REFRESH_TOKEN": "t"}, str(adc)) == {}

    # No credentials + existing ADC file -> point GDAL at the file path only.
    patch = gdal_gcs.gdal_gcs_env_patch({}, str(adc))
    assert patch == {"GOOGLE_APPLICATION_CREDENTIALS": str(adc)}

    # No ADC file -> no change (validator still runs; COG opens fail loudly).
    assert gdal_gcs.gdal_gcs_env_patch({}, str(tmp_path / "missing.json")) == {}

    # ensure_gdal_gcs_auth mutates only the env var, never secret content.
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GS_OAUTH2_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("GS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setattr(gdal_gcs, "default_adc_candidate", lambda: str(adc))
    assert gdal_gcs.ensure_gdal_gcs_auth() == str(adc)
    import os

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(adc)


# ── coverage-gate classifier ─────────────────────────────────────────


def test_parse_coverage_gate_error_with_task_numbers() -> None:
    from berlin_lst_downscaling.data.ard.coverage_gate import parse_coverage_gate_error

    parsed = parse_coverage_gate_error(GATE_ERROR_1)
    assert parsed is not None
    assert parsed["aoi_missing_px"] == 560178
    assert parsed["aoi_total_px"] == 8922403
    assert parsed["missing_aoi_fraction"] == 0.06277345
    assert parsed["max_missing_aoi_fraction"] == 0.005

    assert parse_coverage_gate_error("Stage download failed: boom") is None
    assert parse_coverage_gate_error(None) is None


def test_collect_gate_exclusions_splits_s2_gate_from_real_failures() -> None:
    from berlin_lst_downscaling.data.ard.coverage_gate import collect_gate_exclusions

    rows = [
        {
            "scene_id": "S1",
            "source": "sentinel-2-l2a",
            "status": "failed",
            "last_error": GATE_ERROR_1,
        },
        {"scene_id": "S2", "source": "sentinel-2-l2a", "status": "failed", "last_error": "boom"},
        {
            "scene_id": "L1",
            "source": "landsat-c2-l2",
            "status": "failed",
            "last_error": GATE_ERROR_1,
        },
        {"scene_id": "L2", "source": "landsat-c2-l2", "status": "done", "last_error": None},
    ]
    exclusions, real = collect_gate_exclusions(rows)
    assert [r["scene_id"] for r in exclusions] == ["S1"]
    assert sorted(r["scene_id"] for r in real) == ["L1", "S2"]


def test_excluded_pairings_drop_out_without_replacement() -> None:
    import pyarrow as pa

    from berlin_lst_downscaling.data.ard.coverage_gate import excluded_pairings

    pairings = pa.table(
        {
            "landsat_scene_id": ["A1", "A2", "A3"],
            "sentinel2_scene_id": ["S1", "S2", "S1"],
        }
    )
    out = excluded_pairings(pairings, {"S1"})
    assert out == [
        {"landsat_scene_id": "A1", "sentinel2_scene_id": "S1"},
        {"landsat_scene_id": "A3", "sentinel2_scene_id": "S1"},
    ]


# ── validate_ard gate behavior ───────────────────────────────────────


def _write_manifest_ledger(tmp_path, ledger_rows):
    manifest = pa.table(
        {
            "scene_id": ["L1", "S1", "E1"],
            "source": ["landsat-c2-l2", "sentinel-2-l2a", "ecostress"],
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(manifest, manifest_path)
    ledger = pa.table(
        {
            "scene_id": [r[0] for r in ledger_rows],
            "source": [r[1] for r in ledger_rows],
            "status": [r[2] for r in ledger_rows],
            "path_cog": [r[3] for r in ledger_rows],
            "path_flag": [None, None, None],
            "path_stac": ["s", "s", "s"],
            "schema_version": [7, 10, 7],
            "last_error": [r[4] for r in ledger_rows],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    pq.write_table(ledger, ledger_path)
    return manifest_path, ledger_path


def _run_validate_ard(monkeypatch, capsys, manifest_path, ledger_path, module_name):
    import berlin_lst_downscaling.data.io as io_module

    module = _load_validate_ard(module_name)
    # All artifacts exist; per-scene content checks are out of scope here.
    monkeypatch.setattr(io_module, "exists", lambda uri: True)
    monkeypatch.setattr(module, "_validate_s2_six_bands", lambda cog, res: None)
    monkeypatch.setattr(module, "_validate_stac", lambda *a: None)
    monkeypatch.setattr(module, "_validate_provenance", lambda *a: None)
    monkeypatch.setattr(module, "_validate_completion", lambda *a: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_ard.py", "--ledger", str(ledger_path), "--manifest", str(manifest_path)],
    )
    rc = module.main()
    out = capsys.readouterr().out
    return rc, out


def test_validate_ard_gate_exclusion_passes(tmp_path, monkeypatch, capsys) -> None:
    manifest_path, ledger_path = _write_manifest_ledger(
        tmp_path,
        [
            ("L1", "landsat-c2-l2", "done", "cog", None),
            ("S1", "sentinel-2-l2a", "failed", None, GATE_ERROR_1),
            ("E1", "ecostress", "done", "cog", None),
        ],
    )
    rc, out = _run_validate_ard(
        monkeypatch, capsys, manifest_path, ledger_path, "validate_ard_gate_pass"
    )
    assert rc == 0
    assert "gate exclusions" in out
    assert "S1" in out


def test_validate_ard_real_s2_failure_still_fails(tmp_path, monkeypatch, capsys) -> None:
    manifest_path, ledger_path = _write_manifest_ledger(
        tmp_path,
        [
            ("L1", "landsat-c2-l2", "done", "cog", None),
            ("S1", "sentinel-2-l2a", "failed", None, "Stage download failed: boom"),
            ("E1", "ecostress", "done", "cog", None),
        ],
    )
    rc, _ = _run_validate_ard(
        monkeypatch, capsys, manifest_path, ledger_path, "validate_ard_gate_real"
    )
    assert rc == 1


def test_validate_ard_landsat_nondone_still_fails(tmp_path, monkeypatch, capsys) -> None:
    manifest_path, ledger_path = _write_manifest_ledger(
        tmp_path,
        [
            ("L1", "landsat-c2-l2", "failed", None, "boom"),
            ("S1", "sentinel-2-l2a", "failed", None, GATE_ERROR_2),
            ("E1", "ecostress", "done", "cog", None),
        ],
    )
    rc, out = _run_validate_ard(
        monkeypatch, capsys, manifest_path, ledger_path, "validate_ard_gate_landsat"
    )
    assert rc == 1
    assert "landsat-c2-l2: 0/1 done" in out


# ── selection footprint estimate (future selections) ─────────────────


def test_selection_footprint_missing_fraction() -> None:
    from berlin_lst_downscaling.data.selection.s2_coverage import footprint_missing_fraction

    aoi = np.ones((4, 4), dtype=bool)
    footprint = np.ones((4, 4), dtype=bool)
    footprint[0, 0] = False
    result = footprint_missing_fraction(footprint, aoi)
    assert result["aoi_total_px"] == 16
    assert result["aoi_missing_px"] == 1
    assert result["missing_aoi_fraction"] == 1 / 16


def test_selection_gate_is_opt_in_and_legacy_unchanged() -> None:
    from omegaconf import OmegaConf

    from berlin_lst_downscaling.data.selection.couple import couple_one_anchor
    from berlin_lst_downscaling.data.selection.s2_coverage import DROP_REASON
    from berlin_lst_downscaling.data.selection.schema import policy_fingerprint

    anchor = {"scene_id": "A1"}
    candidates = [
        {
            "scene_id": "S-good",
            "dt_days": 1.0,
            "clear_frac": 0.9,
            "landsat_clear_px": 10,
            "joint_clear_px": 9,
            "s2_aoi_missing_frac": 0.001,
        },
        {
            "scene_id": "S-hole",
            "dt_days": 0.1,
            "clear_frac": 0.95,
            "landsat_clear_px": 10,
            "joint_clear_px": 9,
            "s2_aoi_missing_frac": 0.18,
        },
    ]
    legacy_cfg = OmegaConf.create(
        {
            "sentinel2": {
                "min_clear_frac": 0.05,
                "score": {"lambda": 0.1},
                "mosaic": {"enabled": True},
            }
        }
    )
    gated_cfg = OmegaConf.create(
        {
            "sentinel2": {
                "min_clear_frac": 0.05,
                "score": {"lambda": 0.1},
                "mosaic": {"enabled": True, "max_missing_aoi_fraction": 0.005},
            }
        }
    )
    # Legacy default: annotation only — the hole candidate still wins on score.
    legacy = couple_one_anchor(anchor, [dict(c) for c in candidates], legacy_cfg)
    assert legacy["status"] == "coupled"
    assert legacy["s2"]["scene_id"] == "S-hole"

    # Opt-in gate: the hole candidate never reaches scoring.
    gated = couple_one_anchor(anchor, [dict(c) for c in candidates], gated_cfg)
    assert gated["status"] == "coupled"
    assert gated["s2"]["scene_id"] == "S-good"

    # Gate drops the anchor when every candidate misses the footprint.
    only_hole = couple_one_anchor(anchor, [dict(candidates[1])], gated_cfg)
    assert only_hole["status"] == "dropped"
    assert only_hole["reason"] == DROP_REASON

    # Policy fingerprint: absent key preserves the legacy hash.
    base = OmegaConf.create(
        {
            "years": [2024],
            "months": [6],
            "bbox": [0, 0, 1, 1],
            "landsat": {"anchor": {"min_clear_frac": 0.05}},
            "sentinel2": {
                "min_clear_frac": 0.05,
                "window_days": 3,
                "score": {"lambda": 0.1},
                "mosaic": {"enabled": True},
            },
        }
    )
    assert policy_fingerprint(base) == policy_fingerprint(
        OmegaConf.create(
            {
                "years": [2024],
                "months": [6],
                "bbox": [0, 0, 1, 1],
                "landsat": {"anchor": {"min_clear_frac": 0.05}},
                "sentinel2": {
                    "min_clear_frac": 0.05,
                    "window_days": 3,
                    "score": {"lambda": 0.1},
                    "mosaic": {"enabled": True},
                },
            }
        )
    )
    with_threshold = OmegaConf.merge(
        base, {"sentinel2": {"mosaic": {"max_missing_aoi_fraction": 0.005}}}
    )
    assert policy_fingerprint(with_threshold) != policy_fingerprint(base)


# ── downstream consistency: features count minus gate exclusions ─────


def test_expected_features_count_subtracts_gate_exclusions(tmp_path) -> None:
    from berlin_lst_downscaling.data.selection.bundle_counts import (
        expected_features_count_for_bundle,
    )

    manifest = pa.table(
        {
            "scene_id": ["A17", "A25", "A26", "S1", "S2", "S3"],
            "source": [
                "landsat-c2-l2",
                "landsat-c2-l2",
                "landsat-c2-l2",
                "sentinel-2-l2a",
                "sentinel-2-l2a",
                "sentinel-2-l2a",
            ],
            "role": ["anchor", "anchor", "anchor", "predictor", "predictor", "predictor"],
            "year": [2017, 2025, 2026, 2017, 2025, 2026],
        }
    )
    pairings = pa.table(
        {
            "landsat_scene_id": ["A17", "A25", "A26"],
            "sentinel2_scene_id": ["S1", "S2", "S3"],
        }
    )
    ard = pa.table(
        {
            "scene_id": ["S1", "S2", "S3"],
            "source": ["sentinel-2-l2a", "sentinel-2-l2a", "sentinel-2-l2a"],
            "status": ["done", "failed", "failed"],
            "last_error": [None, GATE_ERROR_2, GATE_ERROR_2],
        }
    )
    pq.write_table(manifest, tmp_path / "manifest.parquet")
    pq.write_table(pairings, tmp_path / "pairings.parquet")
    ard_path = tmp_path / "ard_ledger.parquet"
    pq.write_table(ard, ard_path)

    expected, assessable, exclusions = expected_features_count_for_bundle(
        str(tmp_path / "manifest.parquet"), str(ard_path)
    )
    # Second value is the assessable 2017-2025 universe (2), not the
    # every-anchor total (3): the 2026 exclusion must not move the count.
    assert assessable == 2
    assert expected == 1
    assert exclusions == [{"landsat_scene_id": "A25", "sentinel2_scene_id": "S2"}]


# ── downstream chain: inventory exclusion + launchers + training ──────

GATE_S2_ID = "S2A_MSIL2A_20210814T102031_R065_T32UQD_20210814T215405"


def _load_validator_module(name: str, filename: str):
    path = _REPO / "scripts/validators" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_inventory_marks_gate_excluded_pairing_excluded_not_failed() -> None:
    """A pairing whose S2 partner failed the gate is excluded, not failed.

    The pipeline maps non-assessable inventory scenes to ``status="excluded"``
    without touching the ledger; ``failed`` only comes from ``_process_scene``
    exceptions on assessable scenes.
    """
    from types import SimpleNamespace

    from berlin_lst_downscaling.data.qa.inventory import _resolve_scene

    cog = "gs://berlin-lst-training-data/ard/full/x/landsat-c2-l2/2021/L1/cog.tif"
    flag = "gs://berlin-lst-training-data/ard/full/x/landsat-c2-l2/2021/L1/flag.tif"
    ard_rows = {
        ("landsat-c2-l2", "L1"): {
            "source": "landsat-c2-l2",
            "scene_id": "L1",
            "status": "done",
            "path_cog": cog,
            "path_flag": flag,
        },
        ("sentinel-2-l2a", GATE_S2_ID): {
            "source": "sentinel-2-l2a",
            "scene_id": GATE_S2_ID,
            "status": "failed",
            "last_error": GATE_ERROR_1,
            "path_cog": None,
            "path_flag": None,
        },
    }
    mapping = SimpleNamespace(year_to_vintage={2021: 2021}, vintages={2021: {"geometry_id": "g"}})
    resolved = _resolve_scene(
        ls_id="L1",
        year=2021,
        pairings={"L1": {"sentinel2_scene_id": GATE_S2_ID}},
        manifest_rows={"L1": {"scene_id": "L1", "year": 2021}},
        ard_rows=ard_rows,
        derived_rows={},
        source_rows={},
        dynamic_rows={},
        mapping=mapping,
        ard_root="gs://berlin-lst-training-data/ard/full/v10",
        dynamic_root="gs://berlin-lst-training-data/dynamic/v6",
    )
    assert resolved.exclusion_reason == "ard sentinel2 not done"
    assert not resolved.assessable
    # Sync guard: the training validator allowlists exactly this reason.
    training_module = _load_validator_module(
        "validate_training_data_fx", "validate_training_data.py"
    )
    assert training_module._S2_GATE_EXCLUSION_REASON == resolved.exclusion_reason


def test_features_run_skips_gate_excluded_pairing_without_ledger_row(tmp_path, monkeypatch) -> None:
    """The features run loop never processes or ledger-writes excluded scenes."""
    from types import SimpleNamespace

    import berlin_lst_downscaling.data.features.pipeline as fp
    from berlin_lst_downscaling.data.qa.inventory import InventoryReport, ResolvedScene

    out_root = tmp_path / "features"
    excluded = ResolvedScene(
        scene_id="L1",
        year=2021,
        s2_scene_id=GATE_S2_ID,
        geometry_id="g",
        landsat_cog="a",
        landsat_flag="b",
        s2_cog="",
        s2_flag="",
        dynamic={},
        static_derived={},
        static_derived_meta={},
        static_sources={},
        exclusion_reason="ard sentinel2 not done",
    )
    done = ResolvedScene(
        scene_id="L2",
        year=2021,
        s2_scene_id="S2",
        geometry_id="g",
        landsat_cog="a",
        landsat_flag="b",
        s2_cog="c",
        s2_flag="d",
        dynamic={},
        static_derived={},
        static_derived_meta={},
        static_sources={},
    )
    inventory = InventoryReport(
        scenes=[excluded, done],
        total_pairings=2,
        assessed=1,
        excluded=1,
        exclusion_reasons={"ard sentinel2 not done": 1},
        static_sources={},
        fingerprints={
            "manifest": "m",
            "ard_ledger": "a",
            "static_derived_ledger": "d",
            "dynamic_ledger": "y",
        },
        errors=[],
    )
    monkeypatch.setattr(fp, "build_inventory", lambda **kwargs: inventory)
    monkeypatch.setattr(
        fp,
        "load_geometry_mapping",
        lambda uri, **kwargs: SimpleNamespace(
            ok=True,
            mapping=SimpleNamespace(vintages={2024: {"geometry_id": "g-veg"}}, content_hash="m"),
        ),
    )
    monkeypatch.setattr(fp, "_resolve_vegetation_carry_forward", lambda *args: "gs://x/veg.tif")
    monkeypatch.setattr(fp, "read_bytes", lambda uri: b"aoi")
    monkeypatch.setattr(
        fp,
        "analysis_grid_10m",
        lambda bbox: SimpleNamespace(
            shape=SimpleNamespace(x=8, y=8),
            transform=SimpleNamespace(xoff=0.0, yoff=0.0),
            crs="EPSG:25833",
        ),
    )
    import numpy as _np

    monkeypatch.setattr(fp, "load_aoi_mask_on_grid", lambda *args: _np.ones((8, 8), dtype=bool))
    monkeypatch.setattr(fp, "resolve_lod_coverage_artifacts", lambda root: {})
    monkeypatch.setattr(fp, "config_hash_for_features", lambda **kwargs: "testhash")
    monkeypatch.setattr(
        fp,
        "feature_schema",
        lambda *args, **kwargs: SimpleNamespace(version=5, names=(), canopy_cover_threshold_m=None),
    )
    monkeypatch.setattr(fp, "era5_representation_for_schema", lambda *args: "scene_value")
    monkeypatch.setattr(fp, "_manifest_datetimes", lambda uri: {})

    processed: list[str] = []

    def _fake_process_scene(*, scene, **kwargs):
        processed.append(scene.scene_id)
        return fp.SceneFeatureResult(
            scene_id=scene.scene_id,
            year=scene.year,
            s2_scene_id=scene.s2_scene_id,
            geometry_id=scene.geometry_id,
            status="done",
            config_hash="testhash",
            coverage={"feature_valid_px": 1, "inside_aoi_px": 8, "outside_aoi_px": 0},
        )

    monkeypatch.setattr(fp, "_process_scene", _fake_process_scene)

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "output_root": str(out_root),
            "manifest_uri": "bundle/manifest.parquet",
            "ard_root": "ard",
            "static_sources_root": "src",
            "static_derived_root": "derived",
            "dynamic_root": "dyn",
            "geometry_mapping_uri": "mapping.json",
            "aoi_mask_uri": "aoi.tif",
            "scene_ids": [],
            "bbox": None,
            "vegetation_carry_forward_vintage": 2024,
            "expected_scene_count": None,
            "feature_schema_version": 5,
            "canopy_cover_threshold_m": 2.5,
            "era5_representation": "scene_value",
        }
    )
    report = fp.run_features(cfg, run_id="testexcl")
    assert report.ok
    assert report.excluded == 1
    assert processed == ["L2"]
    by_id = {s.scene_id: s for s in report.scenes}
    assert by_id["L1"].status == "excluded"
    assert by_id["L1"].exclusion_reason == "ard sentinel2 not done"
    # No ledger row was written for the excluded pairing (nor, with
    # _process_scene mocked, for any scene — the point is the excluded
    # scene never reaches the ledger-writing path).
    from berlin_lst_downscaling.data.features.paths import ledger_path
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    led = SecondaryLedger.open(ledger_path(str(out_root)))
    assert led.get("feature_L1", "feature_stack", "L1") is None


def test_excluded_landsat_scene_ids_returns_all_years_sorted(tmp_path) -> None:
    """The shared helper lists every excluded Landsat anchor (all years)."""
    from berlin_lst_downscaling.data.selection.bundle_counts import (
        excluded_landsat_scene_ids,
    )

    manifest = pa.table(
        {
            "scene_id": ["A21", "A25", "A26", "S1", "S2", "S3"],
            "source": [
                "landsat-c2-l2",
                "landsat-c2-l2",
                "landsat-c2-l2",
                "sentinel-2-l2a",
                "sentinel-2-l2a",
                "sentinel-2-l2a",
            ],
            "role": ["anchor", "anchor", "anchor", "predictor", "predictor", "predictor"],
            "year": [2021, 2025, 2026, 2021, 2025, 2026],
        }
    )
    pairings = pa.table(
        {
            "landsat_scene_id": ["A21", "A25", "A26"],
            "sentinel2_scene_id": ["S1", "S2", "S3"],
        }
    )
    ard = pa.table(
        {
            "scene_id": ["S1", "S2", "S3"],
            "source": ["sentinel-2-l2a", "sentinel-2-l2a", "sentinel-2-l2a"],
            "status": ["failed", "done", "failed"],
            "last_error": [GATE_ERROR_1, None, GATE_ERROR_2],
        }
    )
    pq.write_table(manifest, tmp_path / "manifest.parquet")
    pq.write_table(pairings, tmp_path / "pairings.parquet")
    ard_path = tmp_path / "ard_ledger.parquet"
    pq.write_table(ard, ard_path)

    assert excluded_landsat_scene_ids(str(tmp_path / "manifest.parquet"), str(ard_path)) == [
        "A21",
        "A26",
    ]


def _write_training_release(tmp_path, rows: list[dict]):
    """Write a minimal local training release (manifest.parquet + .csv)."""
    import csv as _csv

    fields = [
        "scene_id",
        "year",
        "sensor",
        "s2_scene_id",
        "split",
        "status",
        "eligible_cells",
        "exclusion_reason",
        "feature_stack",
        "feature_valid",
        "eligibility_mask",
    ]
    pq.write_table(
        pa.table({k: [r[k] for r in rows] for k in fields}), tmp_path / "manifest.parquet"
    )
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training_row(
    sid: str,
    year: int,
    split: str,
    status: str,
    cells: int,
    reason: str,
    s2: str = "S-X",
) -> dict:
    return {
        "scene_id": sid,
        "year": year,
        "sensor": "LC08",
        "s2_scene_id": s2,
        "split": split,
        "status": status,
        "eligible_cells": cells,
        "exclusion_reason": reason,
        "feature_stack": "",
        "feature_valid": "",
        "eligibility_mask": "",
    }


def test_training_manifest_accepts_proven_gate_exclusion(tmp_path) -> None:
    """A proven gate-excluded row passes with reason and zero cells."""
    module = _load_validator_module("validate_training_data_manifest", "validate_training_data.py")
    _write_training_release(
        tmp_path,
        [
            _training_row("A21", 2021, "train", "published", 5, ""),
            _training_row("B21", 2021, "train", "excluded", 0, "ard sentinel2 not done"),
            _training_row("A26", 2026, "inference", "excluded", 0, "inference_deferred"),
        ],
    )
    errors: list[str] = []
    module._check_manifest(str(tmp_path), errors, [], None, s2_gate_excluded={"B21"})
    assert errors == []


def test_training_manifest_rejects_unproven_gate_exclusion(tmp_path) -> None:
    """A gate reason without --ard-ledger proof (or for a scene not in the
    proven set) is a hard error — exclusions are never silent."""
    module = _load_validator_module("validate_training_data_manifest2", "validate_training_data.py")
    _write_training_release(
        tmp_path,
        [
            _training_row("A21", 2021, "train", "published", 5, ""),
            _training_row("B21", 2021, "train", "excluded", 0, "ard sentinel2 not done"),
        ],
    )
    errors: list[str] = []
    module._check_manifest(str(tmp_path), errors, [], None, s2_gate_excluded=None)
    assert any("without --ard-ledger proof" in e for e in errors)

    errors2: list[str] = []
    module._check_manifest(str(tmp_path), errors2, [], None, s2_gate_excluded={"OTHER"})
    assert any("without --ard-ledger proof" in e for e in errors2)


def test_training_manifest_rejects_gate_exclusion_with_published_status(tmp_path) -> None:
    """A gate reason on a non-excluded row is a hard error."""
    module = _load_validator_module("validate_training_data_manifest4", "validate_training_data.py")
    _write_training_release(
        tmp_path,
        [
            _training_row("A21", 2021, "train", "published", 5, ""),
            _training_row("B21", 2021, "train", "published", 5, "ard sentinel2 not done"),
        ],
    )
    errors: list[str] = []
    module._check_manifest(str(tmp_path), errors, [], None, s2_gate_excluded={"B21"})
    assert any("must have status 'excluded'" in e for e in errors)


def test_training_manifest_rejects_gate_exclusion_publishing_refs(tmp_path) -> None:
    """A gate-excluded row carrying asset refs is a hard error (no mask
    is published for these pairings)."""
    module = _load_validator_module("validate_training_data_manifest5", "validate_training_data.py")
    row = _training_row("B21", 2021, "train", "excluded", 0, "ard sentinel2 not done")
    row["eligibility_mask"] = "training/v3/L2021B/mask.tif"
    _write_training_release(
        tmp_path,
        [_training_row("A21", 2021, "train", "published", 5, ""), row],
    )
    errors: list[str] = []
    module._check_manifest(str(tmp_path), errors, [], None, s2_gate_excluded={"B21"})
    assert any("publishes eligibility_mask" in e for e in errors)


def test_training_manifest_rejects_unknown_exclusion_reason(tmp_path) -> None:
    """A non-gate, non-contract exclusion reason still fails."""
    module = _load_validator_module("validate_training_data_manifest3", "validate_training_data.py")
    _write_training_release(
        tmp_path,
        [
            _training_row("A21", 2021, "train", "published", 5, ""),
            _training_row("B21", 2021, "train", "excluded", 0, "mystery_reason"),
        ],
    )
    errors: list[str] = []
    module._check_manifest(str(tmp_path), errors, [], None, s2_gate_excluded={"B21"})
    assert any("expected 'no_eligible_cells'" in e for e in errors)


def _training_chain_fixture(tmp_path):
    """Full-size local fixtures: 12-pairing bundle + release with the same
    exclusion shape as production (1 gate-excluded, 2 denylist, 1 inference)."""
    from berlin_lst_downscaling.data.training.contracts import (
        IMPLAUSIBLE_LST_REASON,
        IMPLAUSIBLE_LST_SCENE_IDS,
        INFERENCE_DEFERRED_REASON,
    )

    deny = sorted(IMPLAUSIBLE_LST_SCENE_IDS)
    anchors: list[tuple[str, int, str]] = []
    for year in (2017, 2018, 2019, 2020, 2021, 2023, 2024, 2025, 2026):
        anchors.append((f"L{year}", year, f"S{year}"))
    anchors.append(("L2021B", 2021, "S2021G"))
    anchors.append((deny[0], 2022, "S2022a"))
    anchors.append((deny[1], 2022, "S2022b"))
    assert len(anchors) == 12

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "scene_id": [a[0] for a in anchors],
                "source": ["landsat-c2-l2"] * len(anchors),
                "role": ["anchor"] * len(anchors),
                "year": [a[1] for a in anchors],
            }
        ),
        bundle_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "landsat_scene_id": [a[0] for a in anchors],
                "sentinel2_scene_id": [a[2] for a in anchors],
            }
        ),
        bundle_dir / "pairings.parquet",
    )
    ard_path = tmp_path / "ard_ledger.parquet"
    pq.write_table(
        pa.table(
            {
                "scene_id": [a[2] for a in anchors],
                "source": ["sentinel-2-l2a"] * len(anchors),
                "status": ["failed" if a[0] == "L2021B" else "done" for a in anchors],
                "last_error": [GATE_ERROR_1 if a[0] == "L2021B" else None for a in anchors],
            }
        ),
        ard_path,
    )

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "complete.json").write_text('{"policy_hash": "test"}')
    rows = []
    splits = {
        2017: "train",
        2018: "train",
        2019: "train",
        2020: "train",
        2021: "train",
        2022: "train",
        2023: "train",
        2024: "validation",
        2025: "test",
        2026: "inference",
    }
    for sid, year, s2 in anchors:
        if sid == "L2021B":
            rows.append(
                _training_row(sid, year, "train", "excluded", 0, "ard sentinel2 not done", s2)
            )
        elif sid in set(deny):
            rows.append(
                _training_row(sid, year, "train", "excluded", 0, IMPLAUSIBLE_LST_REASON, s2)
            )
        elif year == 2026:
            rows.append(
                _training_row(sid, year, "inference", "excluded", 0, INFERENCE_DEFERRED_REASON, s2)
            )
        else:
            rows.append(_training_row(sid, year, splits[year], "published", 5, "", s2))
    _write_training_release(release_dir, rows)
    return bundle_dir, ard_path, release_dir


def test_training_main_skips_cog_check_for_proven_exclusions(tmp_path, monkeypatch, capsys) -> None:
    """main() with --ard-ledger: proven exclusions skip the mask check and
    the run passes; the release manifest row count stays one-per-pairing."""
    import berlin_lst_downscaling.data.io as io_module

    module = _load_validator_module("validate_training_data_main", "validate_training_data.py")
    bundle_dir, ard_path, release_dir = _training_chain_fixture(tmp_path)

    monkeypatch.setattr(io_module, "exists", lambda uri: True)
    monkeypatch.setattr(module, "exists", lambda uri: True)
    monkeypatch.setattr(module, "_check_v3_source", lambda root, errors, **kwargs: None)
    checked: list[str] = []

    def _fake_cogs(root, scene_ids, errors, expected):
        checked.extend(scene_ids)
        return {}, {}

    monkeypatch.setattr(module, "_check_eligibility_cogs", _fake_cogs)
    monkeypatch.setattr(module, "_check_cells", lambda *args: 0)
    monkeypatch.setattr(module, "_check_scaler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_training_data.py",
            "--release-root",
            str(release_dir),
            "--expected-scenes-from-manifest",
            str(bundle_dir / "manifest.parquet"),
            "--ard-ledger",
            str(ard_path),
        ],
    )
    assert module.main() == 0
    out = capsys.readouterr().out
    # 8 published scenes are mask-checked; the gate exclusion (like the
    # denylist and inference scenes) publishes no mask and is skipped.
    assert "L2021B" not in checked
    assert len(checked) == 8
    assert "S2 coverage-gate excluded pairings (1)" in out


def test_training_ard_ledger_requires_manifest_flag(monkeypatch) -> None:
    """--ard-ledger without --expected-scenes-from-manifest is rejected."""
    import pytest

    module = _load_validator_module("validate_training_data_flag", "validate_training_data.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_training_data.py",
            "--release-root",
            "somewhere",
            "--ard-ledger",
            "ard.parquet",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_v6_launchers_pass_ard_ledger_for_gate_exclusions() -> None:
    """Step 6/7 launchers derive the v10 ledger URI from ARD_ROOT (no
    hard-coded run id) and pass it to the manifest-derived validators."""
    scripts = _REPO / ".opencode/skills/google-access/scripts"
    for name in ("run-features-v6-vm.sh", "run-training-v3-vm.sh"):
        text = (scripts / name).read_text()
        assert 'ARD_LEDGER="${ARD_ROOT}/ledger.parquet"' in text
        assert "--ard-ledger" in text
    features = (scripts / "run-features-v6-vm.sh").read_text()
    assert "--ard-ledger" in features and '"$ARD_LEDGER"' in features
    training = (scripts / "run-training-v3-vm.sh").read_text()
    assert "--ard-ledger" in training and '"$ARD_LEDGER"' in training
