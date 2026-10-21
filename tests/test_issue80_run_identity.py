"""Issue #80: run identity resolution and run_lock enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from berlin_lst_downscaling.modeling.run_identity import (
    RunIdentity,
    enforce_run_lock,
    persist_run_identity,
    resolve_run_identity,
    training_content_fingerprint,
)


def _identity(**overrides: object) -> RunIdentity:
    base = {
        "git_commit": "abc",
        "vertex_source_sha": None,
        "image_digest": None,
        "data_release_id": "training/v1",
        "training_root": None,
        "patch_index_root": None,
        "training_policy_hash": "deadbeef",
        "patch_index_fingerprints": {},
        "training_content_fingerprint": None,
        "patch_index_content_fingerprint": None,
        "patch_index_source_release_root": None,
        "aggregation": "arithmetic",
    }
    base.update(overrides)
    return RunIdentity(**base)  # type: ignore[arg-type]


def test_enforce_run_lock_warns_on_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "aggregation": "arithmetic",
            "run_lock": {"mode": "warn", "expect": {"aggregation": "t4"}},
        }
    )
    notes = enforce_run_lock(cfg, _identity())
    assert notes


def test_expect_without_mode_defaults_to_error() -> None:
    cfg = OmegaConf.create(
        {"aggregation": "arithmetic", "run_lock": {"mode": "off", "expect": {"aggregation": "t4"}}}
    )
    with pytest.raises(ValueError, match="run_lock"):
        enforce_run_lock(cfg, _identity())


def test_missing_actual_for_expected_field_is_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "aggregation": "arithmetic",
            "run_lock": {"expect": {"image_digest": "sha256:" + "a" * 64}},
        }
    )
    with pytest.raises(ValueError, match="unavailable"):
        enforce_run_lock(cfg, _identity(image_digest=None))


def test_training_content_fingerprint_differs_for_distinct_bytes(tmp_path: Path) -> None:
    root_a = tmp_path / "release_a"
    root_b = tmp_path / "release_b"
    for root in (root_a, root_b):
        root.mkdir()
        (root / "complete.json").write_text('{"policy_hash":"same"}', encoding="utf-8")
        (root / "manifest.parquet").write_bytes(b"a")
        (root / "scaler.json").write_text("{}", encoding="utf-8")
    (root_b / "manifest.parquet").write_bytes(b"b")
    fp_a = training_content_fingerprint(str(root_a))
    fp_b = training_content_fingerprint(str(root_b))
    assert fp_a.composite_sha256 != fp_b.composite_sha256


def test_persist_run_identity_roundtrip(tmp_path: Path) -> None:
    identity = _identity()
    path = persist_run_identity(tmp_path, identity)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["aggregation"] == "arithmetic"


def test_resolve_run_identity_without_expectations(tmp_path: Path) -> None:
    identity = resolve_run_identity(
        OmegaConf.create({"aggregation": "arithmetic", "data_release_id": "x"}),
        run_id="r1",
        output_root=str(tmp_path),
    )
    assert identity.aggregation == "arithmetic"
