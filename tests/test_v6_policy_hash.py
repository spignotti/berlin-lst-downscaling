"""training/v3 policy hash: publishers and modeling readers agree (v6 ledger hash)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.training import patch_index as patch_index_module
from berlin_lst_downscaling.data.training.contracts import (
    expected_training_policy_hash,
    scaler_transform_for,
    training_policy_hash,
    v6_training_policy_hash,
)
from berlin_lst_downscaling.modeling.patches import RealSourceConfig, load_patch_refs, load_scaler
from berlin_lst_downscaling.modeling.run_identity import _training_policy_from_root

_REPO = Path(__file__).resolve().parents[1]
_V3_ROOT = "gs://berlin-lst-training-data/training/v3"
# Published values recorded in docs/results/eda-v3/summary.json.
_V6_LEDGER_HASH = "278427dea6ea41f0"
_V3_POLICY_HASH = "193fa9b23943a839"
_V1_POLICY_HASH = "08c2ec4d5a257c7e"
_V6_NAMES = feature_schema(FEATURE_SCHEMA_VERSION_V6).names


def _scaler(names: tuple[str, ...], *, policy_hash: str, v3_config_hash: str | None) -> dict:
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6) if names == _V6_NAMES else None
    payload: dict = {
        "channel_order": list(names),
        "policy_hash": policy_hash,
        "channels": [
            {
                "channel_index": i,
                "channel_name": name,
                "transform": scaler_transform_for(name, schema),
                "mean": 0.0,
                "std": 1.0,
            }
            for i, name in enumerate(names)
        ],
    }
    if v3_config_hash is not None:
        payload["v3_config_hash"] = v3_config_hash
    return payload


def _v3_scaler(**overrides) -> dict:
    kwargs = {"policy_hash": _V3_POLICY_HASH, "v3_config_hash": _V6_LEDGER_HASH, **overrides}
    return _scaler(_V6_NAMES, **kwargs)


def _v1_scaler() -> dict:
    return _scaler(FEATURE_CHANNEL_NAMES, policy_hash=_V1_POLICY_HASH, v3_config_hash=None)


def _write_root(root: Path, scaler: dict, *, marker_policy: str | None = None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    if marker_policy is not None:
        (root / "complete.json").write_text(
            json.dumps({"policy_hash": marker_policy}), encoding="utf-8"
        )
    return str(root)


# ── shared helper ─────────────────────────────────────────────────────


def test_v6_helper_reproduces_published_v3_hash() -> None:
    assert v6_training_policy_hash(_V6_LEDGER_HASH) == _V3_POLICY_HASH
    assert expected_training_policy_hash(_v3_scaler()) == _V3_POLICY_HASH


def test_v1_v2_hash_unchanged() -> None:
    assert training_policy_hash() == _V1_POLICY_HASH
    assert expected_training_policy_hash(_v1_scaler()) == _V1_POLICY_HASH
    # A v1/v2 scaler that records the pinned feature hash keeps the pinned policy hash.
    assert expected_training_policy_hash({**_v1_scaler(), "v3_config_hash": "x"}) == (
        _V1_POLICY_HASH
    )


def test_v6_helper_without_ledger_hash_fails() -> None:
    with pytest.raises(RuntimeError, match="v3_config_hash"):
        expected_training_policy_hash(_v3_scaler(v3_config_hash=None))
    with pytest.raises(RuntimeError, match="v3_config_hash"):
        expected_training_policy_hash(_v3_scaler(v3_config_hash=""))


def test_helper_rejects_channel_order_drift() -> None:
    with pytest.raises(RuntimeError, match="channel order drift"):
        expected_training_policy_hash({"channel_order": list(_V6_NAMES[:-1])})


def test_publisher_and_reader_computations_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = expected_training_policy_hash(_v3_scaler())
    legacy_publisher = training_policy_hash(
        v3_config_hash=_V6_LEDGER_HASH,
        feature_schema_version=FEATURE_SCHEMA_VERSION_V6,
        schema=feature_schema(FEATURE_SCHEMA_VERSION_V6),
    )
    assert reader == legacy_publisher == _V3_POLICY_HASH

    path = _REPO / "scripts/operators/run_training_eda.py"
    spec = importlib.util.spec_from_file_location("run_training_eda_policy", path)
    assert spec is not None and spec.loader is not None
    eda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda)
    assert eda._expected_policy_for(_V3_ROOT, _v3_scaler()) == reader

    files = {
        f"{_V3_ROOT}/complete.json": json.dumps({"policy_hash": _V3_POLICY_HASH}).encode(),
        f"{_V3_ROOT}/scaler.json": json.dumps(_v3_scaler()).encode(),
        f"{_V3_ROOT}/manifest.parquet": b"",
    }
    monkeypatch.setattr(patch_index_module, "exists", lambda uri: uri in files)
    monkeypatch.setattr(patch_index_module, "read_bytes", lambda uri: files[uri])
    assert patch_index_module._verify_source_release(_V3_ROOT) == reader


# ── load_scaler ───────────────────────────────────────────────────────


def test_load_scaler_accepts_published_v3_hash(tmp_path: Path) -> None:
    spec = load_scaler(_write_root(tmp_path / "v3", _v3_scaler()))
    assert spec.policy_hash == _V3_POLICY_HASH
    assert spec.channel_order == _V6_NAMES


def test_load_scaler_v1_path_unchanged(tmp_path: Path) -> None:
    spec = load_scaler(_write_root(tmp_path / "v1", _v1_scaler()))
    assert spec.policy_hash == _V1_POLICY_HASH
    assert len(spec.channels) == len(FEATURE_CHANNEL_NAMES)


def test_load_scaler_rejects_mismatching_ledger_hash(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "v3", _v3_scaler(v3_config_hash="0000000000000000"))
    with pytest.raises(RuntimeError, match="stale"):
        load_scaler(root)


def test_load_scaler_rejects_missing_ledger_hash(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "v3", _v3_scaler(v3_config_hash=None))
    with pytest.raises(RuntimeError, match="v3_config_hash"):
        load_scaler(root)


def test_load_scaler_rejects_pre_fix_v6_hash(tmp_path: Path) -> None:
    pre_fix = training_policy_hash(schema=feature_schema(FEATURE_SCHEMA_VERSION_V6))
    root = _write_root(tmp_path / "v3", _v3_scaler(policy_hash=pre_fix))
    with pytest.raises(RuntimeError, match="stale"):
        load_scaler(root)


# ── load_patch_refs ───────────────────────────────────────────────────


def _patch_index(root: Path, *, source_policy_hash: str) -> str:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "patch_id": f"S{year}_0_0",
            "scene_id": f"S{year}",
            "s2_scene_id": f"T{year}",
            "split": split,
            "year": year,
            "row": 0,
            "col": 0,
            "n_eligible": 100,
            "eligibility_mask": f"mask/S{year}.tif",
        }
        for year, split in ((2020, "train"), (2024, "validation"), (2025, "test"))
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "patch_index.parquet")
    (root / "complete.json").write_text(
        json.dumps(
            {
                "index_policy_hash": "idx",
                "source_policy_hash": source_policy_hash,
                "patches_total": len(rows),
                "published_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return str(root)


def _training_root(root: Path, scaler: dict) -> str:
    _write_root(root, scaler)
    manifest = [
        {"scene_id": f"S{y}", "year": y, "split": s, "eligibility_mask": f"mask/S{y}.tif"}
        for y, s in ((2020, "train"), (2024, "validation"), (2025, "test"))
    ]
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest.parquet")
    return str(root)


def _source(tmp_path: Path, scaler: dict, source_policy_hash: str) -> RealSourceConfig:
    return RealSourceConfig(
        patch_index_root=_patch_index(tmp_path / "pi", source_policy_hash=source_policy_hash),
        training_root=_training_root(tmp_path / "tr", scaler),
        features_root=str(tmp_path / "features"),
        ard_root=str(tmp_path / "ard"),
    )


def test_load_patch_refs_accepts_v3_chain(tmp_path: Path) -> None:
    refs = load_patch_refs(
        _source(tmp_path, _v3_scaler(), _V3_POLICY_HASH), splits=("train", "validation")
    )
    assert [r.split for r in refs] == ["train", "validation"]


def test_load_patch_refs_v1_path_unchanged(tmp_path: Path) -> None:
    refs = load_patch_refs(_source(tmp_path, _v1_scaler(), _V1_POLICY_HASH), splits=("train",))
    assert [r.scene_id for r in refs] == ["S2020"]


def test_load_patch_refs_rejects_v1_hash_on_v3_chain(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="the index is stale"):
        load_patch_refs(_source(tmp_path, _v3_scaler(), _V1_POLICY_HASH), splits=("train",))


def test_load_patch_refs_rejects_v3_index_on_v1_root(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="the index is stale"):
        load_patch_refs(_source(tmp_path, _v1_scaler(), _V3_POLICY_HASH), splits=("train",))


# ── run identity ──────────────────────────────────────────────────────


def test_run_identity_records_v3_policy_hash(tmp_path: Path) -> None:
    with_marker = _write_root(tmp_path / "a", _v3_scaler(), marker_policy=_V3_POLICY_HASH)
    without_marker = _write_root(tmp_path / "b", _v3_scaler())
    assert _training_policy_from_root(with_marker) == _V3_POLICY_HASH
    assert _training_policy_from_root(without_marker) == _V3_POLICY_HASH


def test_run_identity_v1_unchanged(tmp_path: Path) -> None:
    assert _training_policy_from_root(None) == _V1_POLICY_HASH
    root = _write_root(tmp_path / "v1", _v1_scaler(), marker_policy=_V1_POLICY_HASH)
    assert _training_policy_from_root(root) == _V1_POLICY_HASH


def test_run_identity_rejects_marker_scaler_disagreement(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "v3", _v3_scaler(), marker_policy=_V1_POLICY_HASH)
    with pytest.raises(ValueError, match="recomputed from its scaler"):
        _training_policy_from_root(root)


# ── preflight ─────────────────────────────────────────────────────────


def _load_preflight():
    path = _REPO / "scripts/validators/preflight_v6_reads.py"
    spec = importlib.util.spec_from_file_location("preflight_v6_reads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "config_name", ["v6_l4_locked", "random_forest_v6_smoke", "baseline_v6_full"]
)
def test_preflight_reads_train_and_validation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_name: str
) -> None:
    preflight = _load_preflight()
    local_root = _write_root(tmp_path / "v3", _v3_scaler())
    requested: list[tuple[str, ...]] = []

    def fake_refs(source, *, splits, scene_ids=None, limit=None):
        assert source.training_root == _V3_ROOT
        requested.append(tuple(splits))
        return [object()] * int(limit)

    monkeypatch.setattr(preflight, "load_scaler", lambda root: load_scaler(local_root))
    monkeypatch.setattr(
        preflight, "patch_index_fingerprints", lambda root: {"source_policy_hash": _V3_POLICY_HASH}
    )
    monkeypatch.setattr(preflight, "load_patch_refs", fake_refs)
    assert preflight.main(["--config-name", config_name, "--limit", "2"]) == 0
    assert requested == [("train",), ("validation",)]
    assert all("test" not in splits for splits in requested)


def test_preflight_rejects_v1_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = _load_preflight()
    local_root = _write_root(tmp_path / "v1", _v1_scaler())
    monkeypatch.setattr(preflight, "load_scaler", lambda root: load_scaler(local_root))
    with pytest.raises(RuntimeError, match="not a v6 release"):
        preflight.main(["--config-name", "v6_l4_locked"])
