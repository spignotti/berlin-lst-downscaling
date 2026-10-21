"""Issue #80: run identity resolution and run_lock enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from berlin_lst_downscaling.modeling.run_identity import (
    RunIdentity,
    enforce_run_lock,
    persist_run_identity,
    resolve_run_identity,
)


def test_enforce_run_lock_warns_on_mismatch(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "aggregation": "arithmetic",
            "run_lock": {
                "mode": "warn",
                "expect": {"aggregation": "t4"},
            },
        }
    )
    identity = RunIdentity(
        git_commit="abc",
        vertex_source_sha=None,
        image_digest=None,
        data_release_id="training/v1",
        training_policy_hash="deadbeef",
        patch_index_fingerprints={},
        aggregation="arithmetic",
    )
    notes = enforce_run_lock(cfg, identity)
    assert notes


def test_enforce_run_lock_errors(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "aggregation": "arithmetic",
            "run_lock": {"mode": "error", "expect": {"aggregation": "t4"}},
        }
    )
    identity = resolve_run_identity(
        OmegaConf.create({"aggregation": "arithmetic", "data_release_id": "x"}),
        run_id="r1",
        output_root=str(tmp_path),
    )
    with pytest.raises(ValueError, match="run_lock"):
        enforce_run_lock(cfg, identity)


def test_persist_run_identity_roundtrip(tmp_path: Path) -> None:
    identity = RunIdentity(
        git_commit="abc",
        vertex_source_sha="abc",
        image_digest="sha256:img",
        data_release_id="training/v1",
        training_policy_hash="policy",
        patch_index_fingerprints={"index_policy_hash": "idx"},
        aggregation="arithmetic",
    )
    path = persist_run_identity(tmp_path, identity)
    assert path.is_file()
    assert "aggregation" in path.read_text(encoding="utf-8")
