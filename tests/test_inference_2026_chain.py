"""Inference-2026 chain: inventory opt-in, publish guard, configs, preflight.

The default path stays the 2017-2025 training universe. Nothing here reads
GCS or the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard.publish_guard import (
    FEATURES_V6_ROOT,
    INFERENCE_2026_ARD_ROOT,
    INFERENCE_2026_DYNAMIC_ROOT,
    INFERENCE_2026_FEATURES_ROOT,
    INFERENCE_2026_MANIFEST_ROOT,
    SENTINEL2_V10_ROOT,
    FrozenPrefixError,
    assert_feature_output_root,
    assert_manifests_output_root,
)
from berlin_lst_downscaling.data.qa.inventory import (
    INFERENCE_EXCLUSION_REASON,
    _resolve_scene,
)
from berlin_lst_downscaling.data.selection.bundle_counts import (
    expected_counts,
    expected_features_count_for_bundle,
)

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST_URI = f"{INFERENCE_2026_MANIFEST_ROOT}/manifest.parquet"
_CUTOFF = "2026-09-30T23:59:59Z"
_OLD_V4 = "manifests/v4/2017-2026-cutoff-20260717T235959Z"
_GATE_ERROR = (
    "Sentinel-2 mosaic AOI coverage failed: missing 1629098/8922403 pixels "
    "(0.18257300) > max_missing_aoi_fraction=0.005; item_ids=['S2B_X']"
)
_DYNAMIC_SOURCES = ("era5_land", "shadow_building", "shadow_vegetation")


def _resolve(
    scene_id: str,
    year: int,
    *,
    role: str = "anchor",
    include_inference_scenes: bool | None = None,
    landsat_done: bool = True,
):
    s2_id = f"S2_{scene_id}"
    ard_rows: dict = {}
    if landsat_done:
        ard_rows[("landsat-c2-l2", scene_id)] = {
            "status": "done",
            "path_cog": f"gs://berlin-lst-training-data/ard/x/{scene_id}/cog.tif",
            "path_flag": f"gs://berlin-lst-training-data/ard/x/{scene_id}/flag.tif",
        }
    ard_rows[("sentinel-2-l2a", s2_id)] = {
        "status": "done",
        "path_cog": f"gs://berlin-lst-training-data/ard/x/{s2_id}/cog.tif",
        "path_flag": f"gs://berlin-lst-training-data/ard/x/{s2_id}/flag.tif",
    }
    dynamic_rows = {
        (source, scene_id): {
            "status": "done",
            "role": role,
            "output_uri": (f"gs://berlin-lst-training-data/dynamic/x/{source}/{scene_id}.tif"),
        }
        for source in _DYNAMIC_SOURCES
    }
    mapping = SimpleNamespace(
        year_to_vintage={year: 2024},
        vintages={2024: {"geometry_id": "g"}},
    )
    kwargs = {
        "ls_id": scene_id,
        "year": year,
        "pairings": {scene_id: {"sentinel2_scene_id": s2_id}},
        "manifest_rows": {scene_id: {"scene_id": scene_id, "year": year}},
        "ard_rows": ard_rows,
        "derived_rows": {},
        "source_rows": {},
        "dynamic_rows": dynamic_rows,
        "mapping": mapping,
        "ard_root": "gs://berlin-lst-training-data/ard/x",
        "dynamic_root": "gs://berlin-lst-training-data/dynamic/x",
    }
    if include_inference_scenes is not None:
        kwargs["include_inference_scenes"] = include_inference_scenes
    return _resolve_scene(**kwargs)


def test_inventory_excludes_2026_and_inference_role_by_default() -> None:
    """Default and explicit false match the training exclusion."""
    for flag in (None, False):
        scene = _resolve("L2026", 2026, role="inference", include_inference_scenes=flag)
        assert scene.exclusion_reason == INFERENCE_EXCLUSION_REASON
        assert scene.landsat_cog == ""
        assert scene.s2_cog == ""
        assert scene.dynamic == {}
        assert not scene.assessable

    role = _resolve("L2025", 2025, role="inference")
    assert role.exclusion_reason == INFERENCE_EXCLUSION_REASON
    assert role.dynamic == {}
    assert role.landsat_cog.endswith("/cog.tif")
    assert not role.assessable


def test_inventory_include_inference_scenes_admits_2026() -> None:
    admitted = _resolve("L2026", 2026, role="inference", include_inference_scenes=True)
    assert admitted.assessable
    assert admitted.exclusion_reason is None
    assert set(admitted.dynamic) == set(_DYNAMIC_SOURCES)

    also = _resolve("L2025", 2025, role="inference", include_inference_scenes=True)
    assert also.assessable
    assert set(also.dynamic) == set(_DYNAMIC_SOURCES)

    gap = _resolve("L2026", 2026, include_inference_scenes=True, landsat_done=False)
    assert gap.exclusion_reason == "ard landsat not done"
    assert not gap.assessable


def test_features_publish_guard_refuses_inference_writes_into_v6() -> None:
    assert_feature_output_root(FEATURES_V6_ROOT, schema_version=5)
    assert_feature_output_root(FEATURES_V6_ROOT, schema_version=5, include_inference_scenes=False)
    with pytest.raises(FrozenPrefixError, match="must not write there"):
        assert_feature_output_root(
            FEATURES_V6_ROOT, schema_version=5, include_inference_scenes=True
        )
    assert_feature_output_root(INFERENCE_2026_FEATURES_ROOT, schema_version=5)
    assert_feature_output_root(
        INFERENCE_2026_FEATURES_ROOT,
        schema_version=5,
        include_inference_scenes=True,
    )
    with pytest.raises(FrozenPrefixError):
        assert_feature_output_root(FEATURES_V6_ROOT, schema_version=4)
    with pytest.raises(FrozenPrefixError):
        assert_feature_output_root(INFERENCE_2026_FEATURES_ROOT, schema_version=4)
    for frozen in ("features/v3", "features/v4", "features/v5"):
        with pytest.raises(FrozenPrefixError):
            assert_feature_output_root(
                f"gs://berlin-lst-training-data/{frozen}",
                schema_version=5,
                include_inference_scenes=True,
            )


def test_selection_manifest_root_accepts_inference_bundle() -> None:
    assert_manifests_output_root(INFERENCE_2026_MANIFEST_ROOT)
    pipeline = (_REPO / "src/berlin_lst_downscaling/data/selection/pipeline.py").read_text()
    assert "cutoff-20260717T235959Z" not in pipeline


def _anchors(*rows: tuple[str, int]):
    manifest = pa.table(
        {
            "scene_id": [scene_id for scene_id, _year in rows],
            "source": ["landsat-c2-l2"] * len(rows),
            "role": ["anchor"] * len(rows),
            "year": [year for _scene_id, year in rows],
        }
    )
    pairings = pa.table(
        {
            "landsat_scene_id": [scene_id for scene_id, _year in rows],
            "sentinel2_scene_id": [f"S_{scene_id}" for scene_id, _year in rows],
        }
    )
    return manifest, pairings


def test_dynamic_expected_counts_keep_training_window_by_default() -> None:
    manifest, pairings = _anchors(("A25", 2025), ("A26", 2026))
    assert expected_counts(manifest, pairings) == (1, 2)
    assert expected_counts(manifest, pairings, include_inference_scenes=False) == (1, 2)
    assert expected_counts(manifest, pairings, include_inference_scenes=True) == (1, 2)

    only_2026, pairings_2026 = _anchors(("A26", 2026))
    assert expected_counts(only_2026, pairings_2026) == (0, 1)
    assert expected_counts(only_2026, pairings_2026, include_inference_scenes=True) == (1, 1)


def test_features_count_window_follows_include_inference_scenes(tmp_path) -> None:
    manifest = pa.table(
        {
            "scene_id": ["A17", "A25", "A26", "S1", "S2", "S3"],
            "source": ["landsat-c2-l2"] * 3 + ["sentinel-2-l2a"] * 3,
            "role": ["anchor"] * 3 + ["predictor"] * 3,
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
            "source": ["sentinel-2-l2a"] * 3,
            "status": ["done", "failed", "failed"],
            "last_error": [None, _GATE_ERROR, _GATE_ERROR],
        }
    )
    pq.write_table(manifest, tmp_path / "manifest.parquet")
    pq.write_table(pairings, tmp_path / "pairings.parquet")
    ard_path = tmp_path / "ard_ledger.parquet"
    pq.write_table(ard, ard_path)
    manifest_uri = str(tmp_path / "manifest.parquet")

    expected, assessable, exclusions = expected_features_count_for_bundle(
        manifest_uri, str(ard_path)
    )
    assert assessable == 2
    assert expected == 1
    assert [row["landsat_scene_id"] for row in exclusions] == ["A25"]

    inferred, inferred_universe, inferred_excluded = expected_features_count_for_bundle(
        manifest_uri, str(ard_path), include_inference_scenes=True
    )
    assert inferred_universe == 1
    assert inferred == 0
    assert [row["landsat_scene_id"] for row in inferred_excluded] == ["A26"]


def _compose(config_dir: str, config_name: str):
    GlobalHydra.instance().clear()
    config_path = str((_REPO / config_dir).resolve())
    with initialize_config_dir(config_dir=config_path, version_base=None):
        return compose(config_name=config_name)


def test_selection_ard_dynamic_features_inference_2026_configs() -> None:
    selection = _compose("configs/selection", "inference_2026_v4")
    assert [int(year) for year in selection.years] == [2026]
    assert [int(month) for month in selection.months] == [5, 6, 7, 8, 9]
    assert str(selection.cutoff_utc) == _CUTOFF
    assert float(selection.landsat.anchor.min_aoi_overlap_frac) == 0.05
    assert selection.sentinel2.mosaic.enabled is True
    assert str(selection.output_root).rstrip("/") == INFERENCE_2026_MANIFEST_ROOT
    assert str(selection.checkpoint_dir).rstrip("/") == (
        f"{INFERENCE_2026_MANIFEST_ROOT}/checkpoints"
    )
    assert _OLD_V4 not in str(selection.output_root)

    s2 = _compose("configs/ard", "sentinel2_mosaic_v10_inference_2026")
    assert list(s2.sources) == ["sentinel-2-l2a"]
    assert int(s2.sentinel2.schema_version) == 10
    assert s2.sentinel2.mosaic.enabled is True
    assert float(s2.sentinel2.mosaic.max_missing_aoi_fraction) == 0.005
    assert str(s2.manifest_uri).rstrip("/") == _MANIFEST_URI
    assert str(s2.output_root).rstrip("/") == INFERENCE_2026_ARD_ROOT

    landsat = _compose("configs/ard", "landsat_v10_inference_2026")
    assert list(landsat.sources) == ["landsat-c2-l2"]
    assert str(landsat.manifest_uri).rstrip("/") == _MANIFEST_URI
    assert str(landsat.output_root).rstrip("/") == INFERENCE_2026_ARD_ROOT

    dynamic = _compose("configs/dynamic", "v6_inference_2026")
    assert [int(year) for year in dynamic.years] == [2026]
    assert str(dynamic.dataset_role) == "inference"
    assert dynamic.expected_scene_count is None
    assert str(dynamic.era5_channel_set) == "v6"
    assert str(dynamic.era5_representation) == "scene_value"
    assert str(dynamic.manifest_uri).rstrip("/") == _MANIFEST_URI
    assert str(dynamic.output_root).rstrip("/") == INFERENCE_2026_DYNAMIC_ROOT
    assert str(dynamic.source_root).rstrip("/").endswith("static/sources/lod2-area-exact-v2")
    assert str(dynamic.derived_root).rstrip("/").endswith("static/derived/lod2-area-exact-v2")

    features = _compose("configs/features", "v6_inference_2026")
    assert features.include_inference_scenes is True
    assert int(features.feature_schema_version) == 5
    assert features.expected_scene_count is None
    assert str(features.era5_representation) == "scene_value"
    assert float(features.canopy_cover_threshold_m) == 2.5
    assert str(features.manifest_uri).rstrip("/") == _MANIFEST_URI
    assert str(features.ard_root).rstrip("/") == INFERENCE_2026_ARD_ROOT
    assert str(features.dynamic_root).rstrip("/") == INFERENCE_2026_DYNAMIC_ROOT
    assert str(features.output_root).rstrip("/") == INFERENCE_2026_FEATURES_ROOT
    assert _OLD_V4 not in str(features.manifest_uri)

    # The isolated runner merges _base with the named file and ignores defaults.
    base = OmegaConf.load(_REPO / "configs/features/_base.yaml")
    shallow = OmegaConf.merge(
        base, OmegaConf.load(_REPO / "configs/features/v6_inference_2026.yaml")
    )
    assert shallow.include_inference_scenes is True
    assert shallow.expected_scene_count is None
    assert int(shallow.feature_schema_version) == 5
    assert str(shallow.output_root).rstrip("/") == INFERENCE_2026_FEATURES_ROOT
    assert str(shallow.ard_root).rstrip("/") == INFERENCE_2026_ARD_ROOT
    assert _OLD_V4 not in str(shallow.manifest_uri)

    published = _compose("configs/features", "v6")
    assert published.get("include_inference_scenes") in (None, False)
    assert str(published.output_root).rstrip("/") == FEATURES_V6_ROOT
    assert _OLD_V4 in str(published.manifest_uri)


def test_features_training_pin_rejects_inference_root() -> None:
    """The training release checks the exact features/v6 root before any IO."""
    from berlin_lst_downscaling.data.training.release import (
        _verify_v3_release,
        _verify_v6_release,
    )

    with pytest.raises(RuntimeError, match="not the pinned feature release"):
        _verify_v6_release(
            features_root=INFERENCE_2026_FEATURES_ROOT,
            expected_scene_count=None,
            scene_ids=[],
        )
    with pytest.raises(RuntimeError, match="not the pinned feature release"):
        _verify_v3_release(
            features_root=INFERENCE_2026_FEATURES_ROOT,
            expected_scene_count=None,
            scene_ids=[],
        )


def _load_preflight():
    path = _REPO / "scripts/operators/preflight_republish_v6.py"
    spec = importlib.util.spec_from_file_location("preflight_inference_2026", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inference_upstream(preflight) -> set[str]:
    from berlin_lst_downscaling.data.ard.publish_guard import STATIC_V6_ROOTS

    sources = next(root for root in STATIC_V6_ROOTS if "/static/sources/" in root)
    derived = next(root for root in STATIC_V6_ROOTS if "/static/derived/" in root)
    present = {
        f"{sources}/ledger.parquet",
        f"{derived}/_state/static/derived/ledger.parquet",
        "gs://berlin-lst-training-data/static/geometry_vintages/v1/geometry_mapping.json",
        f"{SENTINEL2_V10_ROOT}/ledger.parquet",
    }
    present |= {uri for uri, _label in preflight.v6_vintage_horizon_uris()}
    present |= {uri for uri, _label in preflight.v6_vintage_product_uris()}
    return present


def test_preflight_inference_2026_needs_published_inputs_only(monkeypatch) -> None:
    import berlin_lst_downscaling.data.io as io

    preflight = _load_preflight()
    present = _inference_upstream(preflight)
    assert _MANIFEST_URI not in present
    monkeypatch.setattr(io, "exists", lambda uri: str(uri) in present)
    assert preflight._failures_for("inference_2026", None) == []

    published = set(present)
    published.add(_MANIFEST_URI)
    monkeypatch.setattr(io, "exists", lambda uri: str(uri) in published)
    refused = preflight._failures_for("inference_2026", None)
    assert any("already published" in finding for finding in refused)

    monkeypatch.setattr(io, "exists", lambda uri: False)
    missing = preflight._failures_for("inference_2026", None)
    assert any("missing upstream" in finding for finding in missing)
    assert not any("already published" in finding for finding in missing)


def test_preflight_inference_2026_refuses_features_v6_and_wrong_config() -> None:
    preflight = _load_preflight()
    configs = preflight._inference_2026_configs()
    configs["features"] = OmegaConf.merge(configs["features"], {"output_root": FEATURES_V6_ROOT})
    failures = preflight._check_inference_2026_configs(configs)
    assert any("features/v6" in finding for finding in failures)

    assert (
        preflight.main(
            [
                "--step",
                "inference_2026",
                "--config-dir",
                "configs/dynamic",
                "--config-name",
                "v6",
            ]
        )
        == 1
    )


def test_ard_launcher_chains_inference_2026_without_the_training_manifest() -> None:
    text = (_REPO / "scripts/runners/vm/run-inference-2026-chain-vm.sh").read_text()
    assert _OLD_V4 not in text
    assert "configs/dynamic/inference_2026.yaml" not in text
    assert "--expected-scenes 324" not in text
    assert "--expected-scenes-from-manifest" in text
    assert "--include-inference-scenes" in text
    assert "--expected-role inference" in text
    assert "ard/full/2017-2026-s2-mosaic-v10/ledger.parquet" in text
    assert "qa-avg" not in text
    assert "vm_stop" in text
    remote = text.split('REMOTE_CMD="', 1)[1].split('"\n', 1)[0]
    assert "'" not in remote
    assert 'S2_CONFIG="sentinel2_mosaic_v10_inference_2026"' in text
    assert 'LANDSAT_CONFIG="landsat_v10_inference_2026"' in text
    selection = text.split('SELECTION_CMD="', 1)[1].split('"\n', 1)[0]
    assert "build_manifest.py" in selection
    assert remote.startswith("${SELECTION_CMD}")
    steps = [
        "seed_ard_v10_reuse.py",
        "${S2_CONFIG}",
        "${LANDSAT_CONFIG}",
        "run_dynamic.py",
        "run_features_isolated.py",
    ]
    positions = [remote.index(step) for step in steps]
    assert positions == sorted(positions)
    assert remote.count(" && ") == 4
    assert text.index("validate_ard.py") < text.index("validate_dynamic.py")
    assert text.index("validate_dynamic.py") < text.index("validate_feature_stacks.py")
    assert text.index("validate_feature_stacks.py") < text.index("vm_stop")
    assert "POLL_MAX_SECONDS=172800" in text

    dynamic = (_REPO / "scripts/validators/validate_dynamic.py").read_text()
    features = (_REPO / "scripts/validators/validate_feature_stacks.py").read_text()
    for source in (dynamic, features):
        assert "--include-inference-scenes" in source
        assert "include_inference_scenes=args.include_inference_scenes" in source
