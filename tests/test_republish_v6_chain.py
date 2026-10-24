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
        "configs/lod_vintages/v6.yaml": ["raw_root", "source_root", "derived_root"],
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
        "gs://berlin-lst-training-data/lod_vintages/2017/LoD1_2017.zip",
        "gs://berlin-lst-training-data/lod_vintages/2021/LoD2_BE_1_33_2021.zip",
        "gs://berlin-lst-training-data/lod_vintages/2022/LoD2_2022.zip",
    }
    present |= {uri for uri, _ in preflight.v6_vintage_horizon_uris()}
    present |= {uri for uri, _ in preflight.v6_vintage_product_uris()}
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
        ("lod_vintages_v6", "configs/lod_vintages", "v6"),
        ("dynamic_v6", "configs/dynamic", "v6"),
        ("features_v6", "configs/features", "v6"),
        ("training_v3", "configs/training", "v3"),
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

    assert config_hash_for_svf(10, 16, "upstream") != config_hash_for_svf(5, 16, "upstream")
    derived = OmegaConf.load(_REPO / "configs/static_derived/v6.yaml")
    assert int(derived["svf_max_radius"]) == 10
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


# ── 6. ARD v10 reuse seeding ──────────────────────────────────────────


def _seed_row(source, scene_id, *, schema_hash, schema_version=7, status="done"):
    from berlin_lst_downscaling.data.ard.ledger import LedgerRow

    return LedgerRow(
        scene_id=scene_id,
        source=source,
        year=2024,
        path_cog=f"cog-{scene_id}",
        path_flag=f"flag-{scene_id}",
        path_stac=None,
        status=status,
        schema_hash=schema_hash,
        schema_version=schema_version,
    )


def _good_hashes():
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.landsat_target import landsat_schema_hash

    landsat = landsat_schema_hash(
        contract_for_source("landsat-c2-l2").schema_version, "average", "native_bit_or"
    )
    ecostress = contract_for_source("ecostress").schema_version_str()
    return landsat, ecostress


def _load_seed_module():
    import importlib.util

    path = _REPO / "scripts/operators/seed_ard_v10_reuse.py"
    spec = importlib.util.spec_from_file_location("seed_ard_v10_reuse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_fixture(tmp_path, monkeypatch, rows, manifest_scenes):
    """Write COGs + source ledger + manifest under tmp_path.

    ``manifest_scenes`` lists the ``(scene_id, source)`` keys present in
    the v4 bundle; rows outside it model v4-dropped scenes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    monkeypatch.chdir(tmp_path)
    for row in rows:
        if row.status == "done" and row.path_cog and row.path_flag:
            Path(row.path_cog).write_bytes(b"x")
            Path(row.path_flag).write_bytes(b"x")
    ledger = Ledger.open("source/ledger.parquet")
    for row in rows:
        ledger.upsert(row)
    Path("bundle").mkdir(parents=True, exist_ok=True)
    manifest = pa.table(
        {
            "scene_id": [sid for sid, _ in manifest_scenes],
            "source": [src for _, src in manifest_scenes],
        }
    )
    pq.write_table(manifest, "bundle/manifest.parquet")
    return "source/ledger.parquet", "bundle/manifest.parquet", "dest"


def test_seed_copies_done_rows_and_is_idempotent(tmp_path, monkeypatch) -> None:
    import os

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    landsat_hash, eco_hash = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [
            _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash),
            _seed_row("ecostress", "E1", schema_hash=eco_hash),
            _seed_row("landsat-c2-l2", "LDROP", schema_hash=landsat_hash),
            _seed_row("sentinel-2-l2a", "S1", schema_hash="9", schema_version=9),
        ],
        [("L1", "landsat-c2-l2"), ("E1", "ecostress")],
    )
    module = _load_seed_module()

    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 2
    assert summary["skipped_not_in_manifest"] == 1
    dest = Ledger.open(os.path.join(dest_root, "ledger.parquet"))
    assert {(r.scene_id, r.source) for r in dest.scenes_for_source("landsat-c2-l2")} == {
        ("L1", "landsat-c2-l2")
    }
    assert {(r.scene_id, r.source) for r in dest.scenes_for_source("ecostress")} == {
        ("E1", "ecostress")
    }
    assert dest.scenes_for_source("sentinel-2-l2a") == []

    again = module.seed(source_uri, dest_root, manifest_uri)
    assert again["staged"] == 0
    assert again["skipped_identical"] == 2
    assert again["skipped_not_in_manifest"] == 1


def test_seed_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    import os

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)],
        [("L1", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    summary = module.seed(source_uri, dest_root, manifest_uri, dry_run=True)
    assert summary["staged"] == 1
    assert summary["copied"] == 0
    assert not os.path.exists(os.path.join(dest_root, "ledger.parquet"))


def test_seed_refuses_contract_mismatch(tmp_path, monkeypatch) -> None:
    import os

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [
            _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash),
            _seed_row(
                "landsat-c2-l2",
                "LBAD",
                schema_hash="6|st=average|qa=native_bit_or",
                schema_version=6,
            ),
        ],
        [("L1", "landsat-c2-l2"), ("LBAD", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    with pytest.raises(RuntimeError, match="nothing written"):
        module.seed(source_uri, dest_root, manifest_uri)
    assert not os.path.exists(os.path.join(dest_root, "ledger.parquet"))


def test_seed_gaps_missing_cog_but_writes_verified_rows(tmp_path, monkeypatch) -> None:
    """A done row with a missing COG is a gap; verified rows still seed."""
    import os

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    landsat_hash, _ = _good_hashes()
    monkeypatch.chdir(tmp_path)
    Path("flag-L1").write_bytes(b"x")
    ledger = Ledger.open("source/ledger.parquet")
    ledger.upsert(_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash))
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path("bundle").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"scene_id": ["L1"], "source": ["landsat-c2-l2"]}),
        "bundle/manifest.parquet",
    )
    module = _load_seed_module()
    summary = module.seed("source/ledger.parquet", "dest", "bundle/manifest.parquet")
    # Missing COG (cog-L1 was never written) is a gap, not an abort.
    assert summary["staged"] == 0
    assert [(g["scene_id"], g["source"], g["reason"]) for g in summary["gaps"]] == [
        ("L1", "landsat-c2-l2", "missing_cog")
    ]
    # The verified-rows write still happened (ledger exists, empty of L1).
    dest = Ledger.open(os.path.join("dest", "ledger.parquet"))
    assert dest.get("L1", "landsat-c2-l2") is None


def test_seed_gaps_new_manifest_scene_without_reuse_row(tmp_path, monkeypatch) -> None:
    """A manifest key with no reuse row is a gap for the gap pass."""
    import os

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)],
        [("L1", "landsat-c2-l2"), ("LNEW", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 1
    assert [(g["scene_id"], g["source"], g["reason"]) for g in summary["gaps"]] == [
        ("LNEW", "landsat-c2-l2", "missing_row")
    ]
    assert os.path.exists(os.path.join(dest_root, "ledger.parquet"))


def test_seed_gaps_not_done_manifest_row(tmp_path, monkeypatch) -> None:
    """A failed reuse row is a gap, not cover.

    The S2-only run_ard never writes that key, so the gap pass must
    process it; validate_ard would otherwise fail later as a key mismatch.
    """
    import os

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [
            _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash),
            _seed_row("landsat-c2-l2", "L2", schema_hash=landsat_hash, status="failed"),
        ],
        [("L1", "landsat-c2-l2"), ("L2", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 1
    assert [(g["scene_id"], g["source"], g["reason"]) for g in summary["gaps"]] == [
        ("L2", "landsat-c2-l2", "not_done")
    ]
    assert os.path.exists(os.path.join(dest_root, "ledger.parquet"))


def test_seed_gap_list_is_complete_and_written_to_gap_out(tmp_path, monkeypatch) -> None:
    """The machine-readable gap list is never truncated; the log line is."""
    import json
    import os

    landsat_hash, eco_hash = _good_hashes()
    rows = [_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)]
    manifest = [("L1", "landsat-c2-l2")]
    # 12 further gap keys of mixed kinds (missing rows, non-done, and a
    # done row with a missing COG).
    for i in range(12):
        manifest.append((f"M{i:02d}", "ecostress"))
    rows.append(_seed_row("ecostress", "EFAIL", schema_hash=eco_hash, status="failed"))
    manifest.append(("EFAIL", "ecostress"))
    rows.append(_seed_row("ecostress", "ENOCG", schema_hash=eco_hash))
    manifest.append(("ENOCG", "ecostress"))
    source_uri, manifest_uri, dest_root = _seed_fixture(tmp_path, monkeypatch, rows, manifest)
    # ENOCG's COG must be absent while its flag exists.
    import pathlib

    (pathlib.Path("cog-ENOCG")).unlink(missing_ok=True)
    module = _load_seed_module()
    gap_out = os.path.join(dest_root, "seed_gaps.json")
    summary = module.seed(source_uri, dest_root, manifest_uri, gap_out=gap_out)
    assert summary["gap_count"] == 14
    assert len(summary["gaps"]) == 14
    on_disk = json.loads(pathlib.Path(gap_out).read_bytes())
    assert len(on_disk["gaps"]) == 14
    reasons = {(g["scene_id"], g["reason"]) for g in on_disk["gaps"]}
    assert ("M00", "missing_row") in reasons
    assert ("EFAIL", "not_done") in reasons
    assert ("ENOCG", "missing_cog") in reasons
    # Verified rows still seeded despite the gaps.
    assert summary["staged"] == 1


def test_seed_gap_rerun_is_idempotent(tmp_path, monkeypatch) -> None:
    """A rerun re-reports the same gaps and skips seeded rows."""
    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [
            _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash),
            _seed_row("landsat-c2-l2", "L2", schema_hash=landsat_hash, status="failed"),
        ],
        [("L1", "landsat-c2-l2"), ("L2", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    first = module.seed(source_uri, dest_root, manifest_uri)
    assert first["staged"] == 1
    second = module.seed(source_uri, dest_root, manifest_uri)
    assert second["staged"] == 0
    assert second["skipped_identical"] == 1
    assert second["gaps"] == first["gaps"]
    assert second["gap_count"] == 1


def _seed_gs_fixture(tmp_path, monkeypatch, rows, manifest_scenes, existing_uris):
    """Write a source ledger + manifest whose rows may carry gs:// URIs.

    No local COG files are written; ``module.exists`` is faked so exactly
    the URIs in ``existing_uris`` exist (local paths delegate to the real
    ``exists``). Returns ``(module, source_uri, manifest_uri,
    dest_root)``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    monkeypatch.chdir(tmp_path)
    ledger = Ledger.open("source/ledger.parquet")
    for row in rows:
        ledger.upsert(row)
    Path("bundle").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "scene_id": [sid for sid, _ in manifest_scenes],
                "source": [src for _, src in manifest_scenes],
            }
        ),
        "bundle/manifest.parquet",
    )
    module = _load_seed_module()
    real_exists = module.exists

    def _fake_exists(uri) -> bool:
        s = str(uri)
        if s.startswith("gs://"):
            return s in existing_uris
        return bool(real_exists(uri))

    monkeypatch.setattr(module, "exists", _fake_exists)
    return module, "source/ledger.parquet", "bundle/manifest.parquet", "dest"


def _legacy_row(source, scene_id, *, schema_hash, schema_version=7, status="done"):
    """A reuse row whose stored provenance points at the retired bucket."""
    row = _seed_row(
        source, scene_id, schema_hash=schema_hash, schema_version=schema_version, status=status
    )
    stem = f"gs://berlin-lst-data/ard/full/x/{source}/2024/{scene_id}"
    row.path_cog = f"{stem}/cog.tif"
    row.path_flag = f"{stem}/flag.tif"
    return row


def _canonical_copy(uri: str) -> str:
    from berlin_lst_downscaling.data.io.uri_mapping import resolve_canonical_uri

    return resolve_canonical_uri(uri)


def test_seed_legacy_bucket_row_seeds_when_canonical_copy_exists(tmp_path, monkeypatch) -> None:
    """Legacy-bucket rows whose canonical copy exists seed unchanged.

    Stored URIs stay the legacy ones (provenance); downstream readers
    resolve at read time. A rerun is a no-op for the seeded row.
    """
    import os

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    landsat_hash, _ = _good_hashes()
    row = _legacy_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)
    existing = {_canonical_copy(row.path_cog), _canonical_copy(row.path_flag)}
    module, source_uri, manifest_uri, dest_root = _seed_gs_fixture(
        tmp_path,
        monkeypatch,
        [row],
        [("L1", "landsat-c2-l2")],
        existing,
    )
    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 1
    assert summary["gaps"] == []
    assert summary["replaced_nondone"] == 0
    dest = Ledger.open(os.path.join(dest_root, "ledger.parquet"))
    seeded = dest.get("L1", "landsat-c2-l2")
    assert seeded is not None
    assert seeded.status == "done"
    # Stored URIs are kept as provenance, not rewritten.
    assert seeded.path_cog == row.path_cog
    assert seeded.path_flag == row.path_flag

    again = module.seed(source_uri, dest_root, manifest_uri)
    assert again["staged"] == 0
    assert again["skipped_identical"] == 1
    assert again["gaps"] == []


def test_seed_legacy_bucket_row_gaps_when_canonical_copy_missing(tmp_path, monkeypatch) -> None:
    """Legacy-bucket rows with no canonical copy stay gaps.

    The gap detail carries both the stored and the resolved URI.
    """
    landsat_hash, _ = _good_hashes()
    row = _legacy_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)
    module, source_uri, manifest_uri, dest_root = _seed_gs_fixture(
        tmp_path,
        monkeypatch,
        [row],
        [("L1", "landsat-c2-l2")],
        set(),
    )
    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 0
    assert len(summary["gaps"]) == 1
    gap = summary["gaps"][0]
    assert (gap["scene_id"], gap["source"], gap["reason"]) == (
        "L1",
        "landsat-c2-l2",
        "missing_cog",
    )
    assert row.path_cog in gap["detail"]
    assert _canonical_copy(row.path_cog) in gap["detail"]


def test_seed_foreign_bucket_row_is_hard_error(tmp_path, monkeypatch) -> None:
    """A gs:// URI outside the canonical/legacy buckets aborts the seed.

    Fail-closed per ``uri_mapping``: a foreign bucket is never a gap and
    nothing (neither ledger nor gap file) is written.
    """
    import os

    landsat_hash, _ = _good_hashes()
    row = _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)
    row.path_cog = "gs://other-bucket/ard/cog.tif"
    row.path_flag = "gs://other-bucket/ard/flag.tif"
    module, source_uri, manifest_uri, dest_root = _seed_gs_fixture(
        tmp_path,
        monkeypatch,
        [row],
        [("L1", "landsat-c2-l2")],
        {row.path_cog, row.path_flag},
    )
    with pytest.raises(RuntimeError, match="nothing written"):
        module.seed(source_uri, dest_root, manifest_uri)
    assert not os.path.exists(os.path.join(dest_root, "ledger.parquet"))


def test_seed_replaces_nondone_dest_row(tmp_path, monkeypatch) -> None:
    """A non-done dest row for a now-verified key is replaced (logged).

    Models the VM rerun where the dest ledger holds a failed/pending row
    for a key the reuse ledger verifies: the verified reuse row wins. A
    rerun afterwards is a no-op.
    """
    import os

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)],
        [("L1", "landsat-c2-l2")],
    )
    dest = Ledger.open(os.path.join(dest_root, "ledger.parquet"))
    failed = _seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash, status="failed")
    dest.upsert(failed)
    assert dest.get("L1", "landsat-c2-l2").status == "failed"

    module = _load_seed_module()
    summary = module.seed(source_uri, dest_root, manifest_uri)
    assert summary["staged"] == 1
    assert summary["replaced_nondone"] == 1
    assert summary["gaps"] == []
    dest = Ledger.open(os.path.join(dest_root, "ledger.parquet"))
    assert dest.get("L1", "landsat-c2-l2").status == "done"

    again = module.seed(source_uri, dest_root, manifest_uri)
    assert again["staged"] == 0
    assert again["skipped_identical"] == 1
    assert again["replaced_nondone"] == 0


def test_features_inventory_resolves_legacy_ard_uris() -> None:
    """The features ARD reader resolves stored URIs onto the canonical bucket.

    ``build_inventory`` (via ``_resolve_scene``) is the reader the
    features/v6 stack build uses for ARD COGs/flags; a legacy-bucket
    stored URI must resolve to its canonical copy at read time.
    """
    from berlin_lst_downscaling.data.qa.inventory import _resolve_scene

    legacy_ls_cog = "gs://berlin-lst-data/ard/full/x/landsat-c2-l2/2024/L1/cog.tif"
    legacy_ls_flag = "gs://berlin-lst-data/ard/full/x/landsat-c2-l2/2024/L1/flag.tif"
    legacy_s2_cog = "gs://berlin-lst-data/ard/full/x/sentinel-2-l2a/2024/S1/cog.tif"
    legacy_s2_flag = "gs://berlin-lst-data/ard/full/x/sentinel-2-l2a/2024/S1/flag.tif"
    ard_rows = {
        ("landsat-c2-l2", "L1"): {
            "source": "landsat-c2-l2",
            "scene_id": "L1",
            "status": "done",
            "path_cog": legacy_ls_cog,
            "path_flag": legacy_ls_flag,
        },
        ("sentinel-2-l2a", "S1"): {
            "source": "sentinel-2-l2a",
            "scene_id": "S1",
            "status": "done",
            "path_cog": legacy_s2_cog,
            "path_flag": legacy_s2_flag,
        },
    }
    resolved = _resolve_scene(
        ls_id="L1",
        year=2024,
        pairings={"L1": {"sentinel2_scene_id": "S1"}},
        manifest_rows={"L1": {"scene_id": "L1", "year": 2024}},
        ard_rows=ard_rows,
        derived_rows={},
        source_rows={},
        dynamic_rows={},
        mapping=None,
        ard_root="gs://berlin-lst-training-data/ard/full/v10",
        dynamic_root="gs://berlin-lst-training-data/dynamic/v6",
    )
    assert resolved.landsat_cog == (
        "gs://berlin-lst-training-data/ard/full/x/landsat-c2-l2/2024/L1/cog.tif"
    )
    assert resolved.landsat_flag == (
        "gs://berlin-lst-training-data/ard/full/x/landsat-c2-l2/2024/L1/flag.tif"
    )
    assert resolved.s2_cog == (
        "gs://berlin-lst-training-data/ard/full/x/sentinel-2-l2a/2024/S1/cog.tif"
    )
    assert resolved.s2_flag == (
        "gs://berlin-lst-training-data/ard/full/x/sentinel-2-l2a/2024/S1/flag.tif"
    )


def test_seed_refuses_overwrite_and_missing_source(tmp_path, monkeypatch) -> None:
    import os

    from berlin_lst_downscaling.data.ard.ledger import Ledger

    landsat_hash, _ = _good_hashes()
    source_uri, manifest_uri, dest_root = _seed_fixture(
        tmp_path,
        monkeypatch,
        [_seed_row("landsat-c2-l2", "L1", schema_hash=landsat_hash)],
        [("L1", "landsat-c2-l2")],
    )
    module = _load_seed_module()
    assert module.seed(source_uri, dest_root, manifest_uri)["staged"] == 1

    dest = Ledger.open(os.path.join(dest_root, "ledger.parquet"))
    row = dest.get("L1", "landsat-c2-l2")
    assert row is not None
    row.path_cog = "cog-elsewhere"
    dest.upsert(row)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module.seed(source_uri, dest_root, manifest_uri)
    assert dest.get("L1", "landsat-c2-l2").path_cog == "cog-elsewhere"

    with pytest.raises(RuntimeError, match="source ledger missing"):
        module.seed("nope/ledger.parquet", "dest2", manifest_uri)


def test_ard_launcher_seeds_before_processing() -> None:
    text = (_REPO / ".opencode/skills/google-access/scripts/run-ard-v10-vm.sh").read_text()
    seed_pos = text.index("seed_ard_v10_reuse.py")
    run_pos = text.index("scripts/runners/run_ard.py")
    assert seed_pos < run_pos
    assert "--manifest-uri ${MANIFEST_URI}" in text
    assert "--dest-root ${OUTPUT_ROOT}" in text
    assert "--gap-out ${OUTPUT_ROOT}/seed_gaps.json" in text
    assert "${REUSE_ROOT}/ledger.parquet" in text
    # S2-only mosaic pass first, Landsat/ECOSTRESS gap pass second.
    assert text.index("--config-name ${CONFIG_NAME}") < text.index(
        "--config-name ${GAP_CONFIG_NAME}"
    )
    assert "landsat_ecostress_v10" in text


def test_ard_v10_gap_config_matches_seed_contracts() -> None:
    """The gap pass writes into the v10 root with the seeded contracts."""
    gap = OmegaConf.load(_REPO / "configs/ard/landsat_ecostress_v10.yaml")
    assert sorted(gap["sources"]) == ["ecostress", "landsat-c2-l2"]
    assert str(gap["manifest_uri"]).rstrip("/") == MANIFEST_V4_URI
    assert str(gap["output_root"]).rstrip("/") == ARD_V10_ROOT
    # Same Landsat policy the seeder verifies (area-mean + native bit-OR).
    base = OmegaConf.load(_REPO / "configs/ard/_base.yaml")
    assert str(base["landsat_st_resampling"]) == "average"
    assert str(base["landsat_qa_aggregation"]) == "native_bit_or"

    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.landsat_target import landsat_schema_hash

    assert contract_for_source("landsat-c2-l2").schema_version == 7
    assert contract_for_source("ecostress").schema_version == 7
    assert landsat_schema_hash(7, "average", "native_bit_or") == "7|st=average|qa=native_bit_or"


def test_validate_ard_fails_when_a_manifest_source_has_no_done_row(
    tmp_path, monkeypatch, capsys
) -> None:
    """validate_ard fails per source, not just on key mismatch.

    The ledger carries every manifest key but none is done (e.g. a
    skipped gap pass): key-set equality passes, yet each manifest source
    must still report its missing done rows.
    """
    import importlib.util
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest_path = tmp_path / "manifest.parquet"
    ledger_path = tmp_path / "ledger.parquet"
    pq.write_table(
        pa.table({"scene_id": ["L1", "E1"], "source": ["landsat-c2-l2", "ecostress"]}),
        manifest_path,
    )
    pq.write_table(
        pa.table(
            {
                "scene_id": ["L1", "E1"],
                "source": ["landsat-c2-l2", "ecostress"],
                "status": ["failed", "failed"],
            }
        ),
        ledger_path,
    )
    path = _REPO / "scripts/validators/validate_ard.py"
    spec = importlib.util.spec_from_file_location("validate_ard_gap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_ard_gap"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_ard.py", "--ledger", str(ledger_path), "--manifest", str(manifest_path)],
    )
    assert module.main() == 1
    out = capsys.readouterr().out
    assert "landsat-c2-l2: 0/1 done" in out
    assert "ecostress: 0/1 done" in out


# ── 7. Expected counts derived from the v4 bundle ─────────────────────


def _bundle_tables():
    import pyarrow as pa

    manifest = pa.table(
        {
            "scene_id": ["A17", "A24", "A25", "A26", "S1", "S2", "E1"],
            "source": [
                "landsat-c2-l2",
                "landsat-c2-l2",
                "landsat-c2-l2",
                "landsat-c2-l2",
                "sentinel-2-l2a",
                "sentinel-2-l2a",
                "ecostress",
            ],
            "role": [
                "anchor",
                "anchor",
                "anchor",
                "anchor",
                "predictor",
                "predictor",
                "validation",
            ],
            "year": [2017, 2024, 2025, 2026, 2017, 2024, 2018],
        }
    )
    pairings = pa.table(
        {
            "landsat_scene_id": ["A17", "A24", "A25", "A26"],
            "sentinel2_scene_id": ["S1", "S2", "S2", "S2"],
        }
    )
    return manifest, pairings


def test_bundle_counts_derive_both_universes() -> None:
    from berlin_lst_downscaling.data.selection.bundle_counts import expected_counts

    manifest, pairings = _bundle_tables()
    assessable, total = expected_counts(manifest, pairings)
    assert assessable == 3
    assert total == 4


def test_bundle_counts_reject_anchor_pairing_mismatch() -> None:
    import pyarrow as pa

    from berlin_lst_downscaling.data.selection.bundle_counts import expected_counts

    manifest, _ = _bundle_tables()
    short = pa.table(
        {"landsat_scene_id": ["A17"], "sentinel2_scene_id": ["S1"]},
    )
    with pytest.raises(ValueError, match="disagree"):
        expected_counts(manifest, short)


def test_validators_derive_counts_without_hard_coding(monkeypatch) -> None:
    import sys

    import berlin_lst_downscaling.data.selection.bundle_counts as counts_module

    class DerivationUsed(Exception):
        pass

    seen: list[str] = []

    def fake_derive(uri: str):
        seen.append(uri)
        raise DerivationUsed(uri)

    monkeypatch.setattr(counts_module, "expected_counts_for_bundle", fake_derive)
    for script, argv in (
        ("validate_feature_stacks.py", ["--root", "somewhere"]),
        ("validate_training_data.py", ["--release-root", "somewhere"]),
        (
            "validate_dynamic.py",
            ["--output-root", "somewhere", "--expected-role", "anchor"],
        ),
    ):
        import importlib.util

        path = _REPO / "scripts/validators" / script
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                *argv,
                "--expected-scenes-from-manifest",
                "bundle/manifest.parquet",
            ],
        )
        with pytest.raises(DerivationUsed):
            module.main()
    assert seen == [
        "bundle/manifest.parquet",
        "bundle/manifest.parquet",
        "bundle/manifest.parquet",
    ]

    # The two flags are mutually exclusive.
    import importlib.util

    for script, argv in (
        ("validate_feature_stacks.py", ["--root", "x"]),
        ("validate_training_data.py", ["--release-root", "x"]),
        (
            "validate_dynamic.py",
            ["--output-root", "x", "--expected-role", "anchor"],
        ),
    ):
        path = _REPO / "scripts/validators" / script
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                *argv,
                "--expected-scenes",
                "5",
                "--expected-scenes-from-manifest",
                "y",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            module.main()
        assert exc.value.code == 2


def test_dynamic_validator_uses_assessable_anchor_count(monkeypatch) -> None:
    """The dynamic step validates the anchor role, so the manifest-derived
    count must be the assessable 2017-2025 universe (as for features/v6),
    not the every-anchor total that training/v3 uses."""
    import importlib.util
    import sys

    import berlin_lst_downscaling.data.selection.bundle_counts as counts_module

    monkeypatch.setattr(counts_module, "expected_counts_for_bundle", lambda uri: (7, 11))
    path = _REPO / "scripts/validators/validate_dynamic.py"
    spec = importlib.util.spec_from_file_location("validate_dynamic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = {(src, "done"): 7 for src in ("era5_land", "shadow_building", "shadow_vegetation")}
    ledger = {
        "total_rows": 21,
        "counts": counts,
        "non_done": [],
        "role_counts": {"anchor": 21},
        "missing_role": 0,
    }
    monkeypatch.setattr(module, "_read_ledger", lambda root: ledger)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-root",
            "somewhere",
            "--expected-role",
            "anchor",
            "--expected-scenes-from-manifest",
            "bundle/manifest.parquet",
        ],
    )
    # Exit 0 proves the assessable count (7) was used: the total (11)
    # would fail every per-source check.
    assert module.main() == 0


def _compose_dynamic(name: str):
    """Replicate the Hydra composition ``_base <- full <- <name>``."""
    parts = []
    for fname in ("_base.yaml", "full.yaml", f"{name}.yaml"):
        cfg = OmegaConf.load(_REPO / "configs/dynamic" / fname)
        if "defaults" in cfg:
            del cfg["defaults"]
        parts.append(cfg)
    return OmegaConf.merge(*parts)


def test_dynamic_v6_config_uses_manifest_as_universe() -> None:
    """v6 nulls the inherited 324 count; legacy configs keep their gates."""
    v6 = _compose_dynamic("v6")
    assert v6.get("expected_scene_count") is None
    full = _compose_dynamic("full")
    assert int(full["expected_scene_count"]) == 324
    inference = OmegaConf.load(_REPO / "configs/dynamic/inference_2026.yaml")
    assert int(inference["expected_scene_count"]) == 21


def test_dynamic_gate_manifest_universe_and_legacy_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v6 run processes fewer-than-324 anchors; legacy 324 still fails.

    The manifest loader is faked with 7 anchor scenes and the geometry
    mapping loader raises past the gate, so reaching it proves the gate
    passed (legacy configs must return 1 before it).
    """
    from types import SimpleNamespace

    from berlin_lst_downscaling.data.dynamic import pipeline as dynamic_pipeline

    class _PastGate(Exception):
        pass

    def _fake_anchors(*args, **kwargs):
        return SimpleNamespace(
            ok=True,
            scenes=[f"S{i}" for i in range(7)],
            errors=[],
            total_rows=7,
            manifest_hash="fakehash",
        )

    def _raise_past_gate(*args, **kwargs):
        raise _PastGate()

    monkeypatch.setattr(dynamic_pipeline, "load_landsat_anchors", _fake_anchors)
    monkeypatch.setattr(dynamic_pipeline, "load_geometry_mapping", _raise_past_gate)

    with pytest.raises(_PastGate):
        dynamic_pipeline.run_dynamic(_compose_dynamic("v6"), run_id="testv6")

    legacy = _compose_dynamic("full")
    assert legacy.get("expected_scene_count") == 324
    assert dynamic_pipeline.run_dynamic(legacy, run_id="testlegacy") == 1

    matching = _compose_dynamic("full")
    monkeypatch.setattr(
        dynamic_pipeline,
        "load_landsat_anchors",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            scenes=[f"S{i}" for i in range(324)],
            errors=[],
            total_rows=324,
            manifest_hash="fakehash",
        ),
    )
    with pytest.raises(_PastGate):
        dynamic_pipeline.run_dynamic(matching, run_id="testmatch")


def test_v6_launchers_derive_counts_from_the_v4_bundle() -> None:
    scripts = _REPO / ".opencode/skills/google-access/scripts"
    features = (scripts / "run-features-v6-vm.sh").read_text()
    assert "--expected-scenes-from-manifest" in features
    assert "--expected-scenes 324" not in features
    training = (scripts / "run-training-v3-vm.sh").read_text()
    assert "--expected-scenes-from-manifest" in training
    assert "manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet" in training
    dynamic = (scripts / "run-dynamic-v6-vm.sh").read_text()
    assert '--expected-scenes-from-manifest "$MANIFEST_URI"' in dynamic
    assert "--expected-scenes 324" not in dynamic
    assert "manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet" in dynamic

    # No v6/v4 launcher hard-codes a scene count (legacy v5 launchers
    # run-dynamic-vm.sh, run-training-data-vm.sh, run-features-vm.sh stay
    # untouched and are excluded here).
    import re

    for name in (
        "run-selection-v4-vm.sh",
        "run-ard-v10-vm.sh",
        "run-static-sources-v6-vm.sh",
        "run-static-derived-v6-vm.sh",
        "run-lod-vintages-v6-vm.sh",
        "run-dynamic-v6-vm.sh",
        "run-features-v6-vm.sh",
        "run-training-v3-vm.sh",
        "run-patch-index-v3-vm.sh",
        "run-training-eda-v3-vm.sh",
    ):
        text = (scripts / name).read_text()
        assert re.search(r"--expected-scenes(?!-from-manifest)\b", text) is None, (
            f"{name} hard-codes a scene count"
        )
