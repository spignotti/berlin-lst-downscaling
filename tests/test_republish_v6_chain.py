"""v6 republish chain: root wiring, fail-closed preflights, publish guards.

Covers task acceptance criteria 2 (chain wiring), 3 (fail-closed before
write), and 4 (publish_guard additions) with offline fakes.
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]

MANIFEST_V4_ROOT = "gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z"
MANIFEST_V4_URI = f"{MANIFEST_V4_ROOT}/manifest.parquet"
ARD_V10_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
STATIC_SOURCES_V6 = "gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
STATIC_DERIVED_V6 = "gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
DYNAMIC_V6 = "gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2"
FEATURES_V6 = "gs://berlin-lst-training-data/features/v6"
TRAINING_V3 = "gs://berlin-lst-training-data/training/v3"
PATCH_INDEX_V3 = "gs://berlin-lst-training-data/training/patch-index/v3"
EDA_V3 = "gs://berlin-lst-training-data/evaluation/eda/v3"


def _load(config_dir: str, config_name: str):
    base = _REPO / config_dir / "_base.yaml"
    over = _REPO / config_dir / f"{config_name}.yaml"
    if base.exists():
        cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.load(over))
    else:
        cfg = OmegaConf.load(over)
    with suppress(Exception):
        OmegaConf.resolve(cfg)
    return cfg


def _roots_of(cfg, keys: list[str]) -> dict[str, str]:
    return {key: str(cfg.get(key) or "") for key in keys}


# ── 1. chain wiring ───────────────────────────────────────────────────


def test_v6_chain_root_wiring_matches_step_to_step() -> None:
    selection = _load("configs/selection", "full_2017_2026_v4")
    ard = _load("configs/ard", "sentinel2_mosaic_v10")
    static_sources = _load("configs/static_sources", "v6")
    static_derived = _load("configs/static_derived", "v6")
    dynamic = _load("configs/dynamic", "v6")
    features = _load("configs/features", "v6")
    training = _load("configs/training", "v3")
    patch = _load("configs/training", "patch_index_v3")
    eda = _load("configs/eda", "v3")

    # Step 1 -> step 2: ARD reads the v4 bundle written by selection.
    assert str(selection["output_root"]).rstrip("/") == MANIFEST_V4_ROOT
    assert str(ard["manifest_uri"]).rstrip("/") == MANIFEST_V4_URI
    assert str(ard["output_root"]).rstrip("/") == ARD_V10_ROOT

    # Step 3 -> step 4: derived reads the v6 sources output.
    assert str(static_sources["source_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(static_derived["source_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(static_derived["derived_root"]).rstrip("/") == STATIC_DERIVED_V6

    # Step 4/3 -> step 5: dynamic reads the v6 static roots and the v4 manifest.
    assert str(dynamic["source_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(dynamic["derived_root"]).rstrip("/") == STATIC_DERIVED_V6
    assert str(dynamic["manifest_uri"]).rstrip("/") == MANIFEST_V4_URI
    assert str(dynamic["output_root"]).rstrip("/") == DYNAMIC_V6

    # Step 2/3/4/5 -> step 6: features reads ARD v10, the v6 static/dynamic
    # roots, and the v4 manifest.
    assert str(features["manifest_uri"]).rstrip("/") == MANIFEST_V4_URI
    assert str(features["ard_root"]).rstrip("/") == ARD_V10_ROOT
    assert str(features["static_sources_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(features["static_derived_root"]).rstrip("/") == STATIC_DERIVED_V6
    assert str(features["dynamic_root"]).rstrip("/") == DYNAMIC_V6
    assert str(features["output_root"]).rstrip("/") == FEATURES_V6

    # Step 6 (+ARD v10, v4 manifest, v6 statics) -> step 7: training v3.
    assert str(training["features_root"]).rstrip("/") == FEATURES_V6
    assert str(training["ard_root"]).rstrip("/") == ARD_V10_ROOT
    assert str(training["manifest_uri"]).rstrip("/") == MANIFEST_V4_URI
    assert str(training["static_sources_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(training["static_derived_root"]).rstrip("/") == STATIC_DERIVED_V6
    assert str(training["dynamic_root"]).rstrip("/") == DYNAMIC_V6
    assert str(training["output_root"]).rstrip("/") == TRAINING_V3

    # Step 7 -> step 8 -> step 9.
    assert str(patch["source_root"]).rstrip("/") == TRAINING_V3
    assert str(patch["output_root"]).rstrip("/") == PATCH_INDEX_V3
    assert str(eda["training_root"]).rstrip("/") == TRAINING_V3
    assert str(eda["features_root"]).rstrip("/") == FEATURES_V6
    assert str(eda["v3_manifest"]).rstrip("/") == MANIFEST_V4_URI
    assert str(eda["patch_index_root"]).rstrip("/") == PATCH_INDEX_V3
    assert str(eda["upload_uri"]).rstrip("/") == EDA_V3


def test_no_v6_v3_config_references_a_published_old_root() -> None:
    old_roots = [
        "features/v3",
        "features/v4",
        "features/v5",
        "training/v1",
        "training/v2",
        "training/patch-index/v1",
        "training/patch-index/v2",
        "manifests/v3",
        "s2-harmonized-v9",
        "2017-2026-qa-avg-20261019",
        "2017-2026-qa-20261019",
        "ard/full/2017-2026-cutoff-20260717T235959Z",
        "static/sources/full",
        "static/derived/full",
        "dynamic/full",
        "static/sources/imperviousness-v3",
    ]
    checked = {
        "configs/ard/sentinel2_mosaic_v10.yaml": ["manifest_uri", "output_root"],
        "configs/dynamic/v6.yaml": [
            "manifest_uri",
            "source_root",
            "derived_root",
            "output_root",
        ],
        "configs/features/v6.yaml": [
            "manifest_uri",
            "ard_root",
            "static_sources_root",
            "static_derived_root",
            "dynamic_root",
            "output_root",
        ],
        "configs/training/v3.yaml": [
            "manifest_uri",
            "ard_root",
            "static_sources_root",
            "static_derived_root",
            "dynamic_root",
            "features_root",
            "output_root",
        ],
        "configs/training/patch_index_v3.yaml": ["source_root", "output_root"],
        "configs/eda/v3.yaml": [
            "training_root",
            "features_root",
            "v3_manifest",
            "patch_index_root",
            "upload_uri",
        ],
        "configs/selection/full_2017_2026_v4.yaml": ["output_root", "checkpoint_dir"],
        "configs/static_sources/v6.yaml": ["source_root"],
        "configs/static_derived/v6.yaml": ["source_root", "derived_root"],
    }
    for rel, keys in checked.items():
        cfg = OmegaConf.load(_REPO / rel)
        for key in keys:
            value = str(cfg.get(key) or "")
            for old in old_roots:
                assert old not in value, f"{rel}:{key} references old root {old!r}: {value!r}"


def test_published_configs_keep_their_old_roots() -> None:
    v5 = OmegaConf.load(_REPO / "configs/features/v5.yaml")
    assert "features/v5" in str(v5["output_root"])
    v2 = OmegaConf.load(_REPO / "configs/training/v2.yaml")
    assert "training/v2" in str(v2["output_root"])
    assert "features/v5" in str(v2["features_root"])
    pi2 = OmegaConf.load(_REPO / "configs/training/patch_index_v2.yaml")
    assert "training/patch-index/v2" in str(pi2["output_root"])


# ── 2. fail-closed preflights (fakes) ─────────────────────────────────


def _import_preflight():
    import importlib.util

    path = _REPO / "scripts/operators/preflight_republish_v6.py"
    spec = importlib.util.spec_from_file_location("preflight_republish_v6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_io(monkeypatch: pytest.MonkeyPatch, present: set[str], markers: dict[str, str]):
    import berlin_lst_downscaling.data.io as io

    monkeypatch.setattr(io, "exists", lambda uri: str(uri) in present)
    monkeypatch.setattr(io, "read_bytes", lambda uri: markers[str(uri)].encode())


def test_preflight_fails_closed_without_upstream_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import berlin_lst_downscaling.data.io as io

    monkeypatch.setattr(io, "exists", lambda uri: False)
    monkeypatch.setattr(io, "read_bytes", lambda uri: (_ for _ in ()).throw(FileNotFoundError(uri)))
    preflight = _import_preflight()
    for step, config_dir, config_name in (
        ("ard_v10", "configs/ard", "sentinel2_mosaic_v10"),
        ("features_v6", "configs/features", "v6"),
        ("training_v3", "configs/training", "v3"),
        ("patch_index_v3", "configs/training", "patch_index_v3"),
        ("eda_v3", "configs/eda", "v3"),
    ):
        cfg = preflight._load_config(config_dir, config_name)
        assert preflight._failures_for(step, cfg), f"{step} should fail with no upstream"


def test_preflight_passes_with_all_upstream_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    present = {
        MANIFEST_V4_URI,
        f"{MANIFEST_V4_ROOT}/pairings.parquet",
        f"{MANIFEST_V4_ROOT}/manifest_report.json",
        "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019/ledger.parquet",
        f"{ARD_V10_ROOT}/ledger.parquet",
        f"{STATIC_SOURCES_V6}/ledger.parquet",
        f"{STATIC_DERIVED_V6}/_state/static/derived/ledger.parquet",
        f"{DYNAMIC_V6}/_state/dynamic/ledger.parquet",
        f"{FEATURES_V6}/ledger.parquet",
        f"{TRAINING_V3}/manifest.parquet",
        f"{PATCH_INDEX_V3}/complete.json",
        "gs://berlin-lst-training-data/static/geometry_vintages/v1/geometry_mapping.json",
    }
    markers = {
        f"{TRAINING_V3}/complete.json": json.dumps({"policy_hash": "abc123"}),
        f"{PATCH_INDEX_V3}/complete.json": json.dumps(
            {"index_policy_hash": "i", "source_policy_hash": "s"}
        ),
    }
    _patch_io(monkeypatch, present, markers)
    for step, config_dir, config_name in (
        ("ard_v10", "configs/ard", "sentinel2_mosaic_v10"),
        ("static_derived_v6", "configs/static_derived", "v6"),
        ("dynamic_v6", "configs/dynamic", "v6"),
        ("features_v6", "configs/features", "v6"),
    ):
        cfg = preflight._load_config(config_dir, config_name)
        assert preflight._failures_for(step, cfg) == [], f"{step} should pass, got failures"


def test_preflight_selection_refuses_an_existing_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    _patch_io(monkeypatch, {MANIFEST_V4_URI}, {})
    cfg = preflight._load_config("configs/selection", "full_2017_2026_v4")
    failures = preflight._failures_for("selection_v4", cfg)
    assert any("already published" in f for f in failures)


def test_preflight_patch_index_requires_training_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    _patch_io(monkeypatch, set(), {})
    cfg = preflight._load_config("configs/training", "patch_index_v3")
    failures = preflight._failures_for("patch_index_v3", cfg)
    assert any("training/v3 complete.json" in f for f in failures)


# ── 3. publish guards ─────────────────────────────────────────────────


def test_publish_guard_blocks_old_into_new_and_keeps_old_blocks() -> None:
    from berlin_lst_downscaling.data.ard.publish_guard import (
        FrozenPrefixError,
        assert_feature_output_root,
        assert_manifests_output_root,
        assert_patch_index_output_root,
        assert_sentinel2_output_root,
        assert_training_output_root,
    )

    # New v10 root: schema 10 allowed, legacy schemas refused.
    assert_sentinel2_output_root(ARD_V10_ROOT, schema_version=10)
    with pytest.raises(FrozenPrefixError):
        assert_sentinel2_output_root(ARD_V10_ROOT, schema_version=9)
    # Existing blocks stay put (only additions allowed).
    with pytest.raises(FrozenPrefixError):
        assert_sentinel2_output_root(
            "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019",
            schema_version=10,
        )
    with pytest.raises(FrozenPrefixError):
        assert_sentinel2_output_root(
            "gs://berlin-lst-training-data/ard/full/2017-2026-s2-harmonized-v9",
            schema_version=10,
        )

    # New v6 root: schema 5 allowed (explicit or unknown), older refused.
    assert_feature_output_root(FEATURES_V6, schema_version=5)
    assert_feature_output_root(FEATURES_V6)
    with pytest.raises(FrozenPrefixError):
        assert_feature_output_root(FEATURES_V6, schema_version=4)
    with pytest.raises(FrozenPrefixError):
        assert_feature_output_root("gs://berlin-lst-training-data/features/v5")

    # Published training / patch-index / manifest roots stay blocked.
    with pytest.raises(FrozenPrefixError):
        assert_training_output_root("gs://berlin-lst-training-data/training/v2")
    assert_training_output_root(TRAINING_V3)
    with pytest.raises(FrozenPrefixError):
        assert_patch_index_output_root("gs://berlin-lst-training-data/training/patch-index/v2")
    assert_patch_index_output_root(PATCH_INDEX_V3)
    with pytest.raises(FrozenPrefixError):
        assert_manifests_output_root(
            "gs://berlin-lst-training-data/manifests/v3/"
            "2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
        )
    assert_manifests_output_root(MANIFEST_V4_ROOT)


# ── 4. SVF radius stays configurable in one place ─────────────────────


def test_svf_radius_is_in_the_derived_config_hash() -> None:
    from berlin_lst_downscaling.data.secondary.svf import config_hash_for_svf

    assert config_hash_for_svf(3, 16, "upstream") != config_hash_for_svf(5, 16, "upstream")
    derived = OmegaConf.load(_REPO / "configs/static_derived/v6.yaml")
    assert int(derived["svf_max_radius"]) == 3
    assert int(derived["svf_n_directions"]) == 16
    text = (_REPO / "configs/static_derived/v6.yaml").read_text()
    assert text.count("svf_max_radius") == 1


# ── 5. Bugbot round-trip regressions ──────────────────────────────────


def _small_grid():
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from rasterio.crs import CRS

    return GeoBox((4, 4), Affine(100, 0, 369190, 0, -100, 5838410), CRS.from_epsg(25833))


def _eligibility_result():
    import numpy as np

    from berlin_lst_downscaling.data.training.eligibility import EligibilityResult

    return EligibilityResult(
        scene_id="S",
        year=2024,
        s2_scene_id="S2",
        eligible=np.ones((4, 4), dtype=bool),
        target_valid=np.ones((4, 4), dtype=bool),
        support=np.full((4, 4), 100, dtype=np.int32),
        present=np.full((4, 4), 100, dtype=np.int32),
        target_valid_cells=16,
        eligible_cells=16,
    )


def test_publish_eligibility_v6_provenance_gate(tmp_path, monkeypatch) -> None:
    """A v6 feature provenance publishes only with the v6 schema (Bugbot).

    The default path still demands the pinned v5 hash + 28-channel order.
    """
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_CHANNEL_NAMES,
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )
    from berlin_lst_downscaling.data.training import product as training_product

    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    v6_hash = "shared-v6-hash"
    v6_prov = {"config_hash": v6_hash, "channel_order": list(schema.names)}
    real_read = training_product.read_bytes

    def fake_read(uri):
        if str(uri) == "prov":
            return json.dumps(v6_prov).encode()
        return real_read(uri)

    monkeypatch.setattr(training_product, "read_bytes", fake_read)
    out = training_product.publish_eligibility(
        result=_eligibility_result(),
        grid_100m=_small_grid(),
        output_root=str(tmp_path),
        run_id="r",
        policy_hash="p",
        v3_config_hash=v6_hash,
        feature_valid_uri="f",
        feature_provenance_uri="prov",
        landsat_cog_uri="l",
        landsat_flag_uri="lf",
        geometry_id="g",
        schema=schema,
    )
    assert Path(out.completion_uri).exists()

    with pytest.raises(ValueError, match="Feature Release V3"):
        training_product.publish_eligibility(
            result=_eligibility_result(),
            grid_100m=_small_grid(),
            output_root=str(tmp_path / "v3"),
            run_id="r",
            policy_hash="p",
            v3_config_hash=v6_hash,
            feature_valid_uri="f",
            feature_provenance_uri="prov",
            landsat_cog_uri="l",
            landsat_flag_uri="lf",
            geometry_id="g",
        )

    v5_prov = {"config_hash": "pinned-v5", "channel_order": list(FEATURE_CHANNEL_NAMES)}

    def fake_read_v5(uri):
        if str(uri) == "prov":
            return json.dumps(v5_prov).encode()
        return real_read(uri)

    monkeypatch.setattr(training_product, "read_bytes", fake_read_v5)
    with pytest.raises(ValueError, match="Feature Release v6"):
        training_product.publish_eligibility(
            result=_eligibility_result(),
            grid_100m=_small_grid(),
            output_root=str(tmp_path / "mixed"),
            run_id="r",
            policy_hash="p",
            v3_config_hash=v6_hash,
            feature_valid_uri="f",
            feature_provenance_uri="prov",
            landsat_cog_uri="l",
            landsat_flag_uri="lf",
            geometry_id="g",
            schema=schema,
        )


def test_readback_accepts_the_v6_channel_count(tmp_path, monkeypatch) -> None:
    """The publisher readback expects 22 channels on the v6 path (Bugbot)."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )
    from berlin_lst_downscaling.data.training import release as training_release
    from berlin_lst_downscaling.data.training.paths import (
        cells_parquet,
        manifest_parquet,
        scaler_json,
    )
    from berlin_lst_downscaling.data.training.report import TrainingRunReport

    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    root = str(tmp_path)
    blob = pa.BufferOutputStream()
    pq.write_table(pa.table({"a": [1]}), blob)
    parquet_bytes = blob.getvalue().to_pybytes()
    scaler = {"policy_hash": "p", "channels": [{"name": n} for n in schema.names]}
    payloads = {
        manifest_parquet(root): parquet_bytes,
        cells_parquet(root): parquet_bytes,
        scaler_json(root): json.dumps(scaler).encode(),
    }
    import berlin_lst_downscaling.data.io as io_module

    monkeypatch.setattr(io_module, "exists", lambda uri: True)
    monkeypatch.setattr(training_release, "exists", lambda uri: True)
    monkeypatch.setattr(training_release, "read_bytes", lambda uri: payloads[str(uri)])
    report = TrainingRunReport(
        run_id="r",
        timestamp="t",
        inputs={},
        fingerprints={},
        grid={},
        policy_hash="p",
        v3_config_hash="h",
        total_pairings=0,
        assessed=0,
        processed=0,
        failed=0,
        excluded=0,
        exclusion_reasons={},
        aggregate={},
        scenes=[],
    )
    ok_v6 = training_release._readback_release(
        report=report, output_root=root, policy_hash="p", marker_expected=False, schema=schema
    )
    assert ok_v6["ok"], ok_v6["errors"]
    stale = training_release._readback_release(
        report=report, output_root=root, policy_hash="p", marker_expected=False
    )
    assert not stale["ok"]
    assert any("channel count" in e for e in stale["errors"])


def test_writers_refuse_published_roots_before_any_work() -> None:
    """Guards run inside the writers, not just in tests (Bugbot)."""
    from berlin_lst_downscaling.data.ard.publish_guard import FrozenPrefixError
    from berlin_lst_downscaling.data.selection import pipeline as selection_pipeline
    from berlin_lst_downscaling.data.training import patch_index as patch_module
    from berlin_lst_downscaling.data.training import pipeline as training_pipeline

    with pytest.raises(FrozenPrefixError):
        selection_pipeline.run_couple(
            OmegaConf.create({"output_root": "gs://berlin-lst-training-data/manifests/v3/x/"})
        )
    with pytest.raises(FrozenPrefixError):
        training_pipeline.run_training_data(
            OmegaConf.create(
                {
                    "features_root": "gs://berlin-lst-training-data/features/v5",
                    "output_root": "gs://berlin-lst-training-data/training/v1",
                }
            )
        )
    with pytest.raises(FrozenPrefixError):
        patch_module.publish_patch_index(
            patch_module.PatchIndexBuild(
                rows=[], qa={}, source_policy_hash="", index_policy_hash=""
            ),
            output_root="gs://berlin-lst-training-data/training/patch-index/v2",
            run_id="r",
        )


def test_eda_v3_records_its_own_roots_and_policy(monkeypatch) -> None:
    """EDA v3 names features/v6 and verifies the v6 policy (Bugbot)."""
    import importlib.util
    import sys

    from berlin_lst_downscaling.data.eda.rules import FEATURES_ROOT, RELEASE_ID
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )
    from berlin_lst_downscaling.data.training.contracts import training_policy_hash

    path = _REPO / "scripts/operators/run_training_eda.py"
    spec = importlib.util.spec_from_file_location("run_training_eda", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", ["run_training_eda.py", "--output-dir", "x"])
    assert module.parse_args().features_root == FEATURES_ROOT
    assert module.parse_args().release_id == RELEASE_ID
    assert "--features-root" in open(path).read()
    assert "--release-id" in open(path).read()

    v6_hash = "shared-v6-hash"
    v6_policy = training_policy_hash(
        v3_config_hash=v6_hash,
        feature_schema_version=FEATURE_SCHEMA_VERSION_V6,
        schema=feature_schema(FEATURE_SCHEMA_VERSION_V6),
    )
    scaler = {"policy_hash": v6_policy, "v3_config_hash": v6_hash}
    assert module._expected_policy_for(TRAINING_V3, scaler) == v6_policy
    assert module._expected_feature_hash_for(TRAINING_V3) is None
    assert (
        module._expected_policy_for("gs://berlin-lst-training-data/training/v2", {})
        == training_policy_hash()
    )

    launcher = (
        _REPO / ".opencode/skills/google-access/scripts/run-training-eda-v3-vm.sh"
    ).read_text()
    assert "--features-root" in launcher
    assert "features/v6" in launcher
    assert "--release-id eda-v3" in launcher
