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
