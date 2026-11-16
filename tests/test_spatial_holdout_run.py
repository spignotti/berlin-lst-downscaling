"""Spatial holdout through the launcher, the evidence upload and one tiny fit (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from test_launch_vertex_v6 import _argv, launcher  # noqa: F401  (``launcher`` is a fixture)
from test_parallel_patch_loading import _build_world, _patch_sources
from test_spatial_holdout_training import _held_out, _module
from test_vertex_evidence_v6 import evidence  # noqa: F401  (``evidence`` is a fixture)

from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling import spatial_holdout as sh
from berlin_lst_downscaling.modeling.patches import RealSourceConfig

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_SEEDED = ("--seed", "0", "--max-epochs", "40", "--lr-schedule", "cosine")

# ---------------------------------------------------------------- launcher


@pytest.mark.parametrize("arm", ["l1", "l4"])
def test_holdout_preflight_is_validation_only_and_bounded(
    launcher: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arm: str,
) -> None:
    mode = f"v6-{arm}-spatial-holdout"
    monkeypatch.setattr(sys, "argv", _argv(mode, *_SEEDED, "--skip-test"))
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert f"{mode} / v6_{arm}_spatial_holdout" in out
    assert "test pass      : skipped" in out and "evaluate_test=false" in out
    assert "patch cache    : job-local build" in out
    assert "timeout        : 10800s" in out and "~$3.17" in out
    label = "unit-seed0-ep40-cos-notest"
    assert f"/{label}/eval_validation_patches.parquet" in out
    assert "preflight only: no job submitted" in out


def test_holdout_modes_refuse_test_scoring_prebuilt_cache_and_higher_bounds(
    launcher: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode = "v6-l4-spatial-holdout"
    for extra, match in (
        ((), "requires --skip-test"),
        (("--skip-test", "--patch-cache", "prebuilt"), "builds its patch cache in the job"),
        (("--skip-test", "--timeout-seconds", "10801"), "timeout cannot exceed"),
        (("--skip-test", "--max-exposure-usd", "4"), "exposure cannot exceed"),
    ):
        monkeypatch.setattr(sys, "argv", _argv(mode, *_SEEDED, *extra))
        with pytest.raises(SystemExit, match=match):
            launcher.main()
    monkeypatch.setattr(sys, "argv", _argv(mode, *_SEEDED, "--skip-test", rate=False))
    with pytest.raises(SystemExit, match="requires the verified regional rate"):
        launcher.main()
    monkeypatch.setattr(launcher, "_evidence_exists", lambda uri: uri.endswith(".parquet"))
    monkeypatch.setattr(sys, "argv", _argv(mode, *_SEEDED, "--skip-test"))
    with pytest.raises(SystemExit, match="validation export already exists"):
        launcher.main()


def test_matrix_modes_get_no_validation_export(
    launcher: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", *_SEEDED, "--skip-test"))
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert "val export" not in out and "spatial holdout" not in out


# ---------------------------------------------------------------- evidence


def test_validation_export_is_uploaded_before_the_evidence(
    evidence: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    (tmp_path / "eval_validation_patches.parquet").write_bytes(b"rows")
    monkeypatch.setattr(
        evidence,
        "_upload_checkpoint_create_only",
        lambda path, uri: order.append(uri) or {"uri": uri, "sha256": "0" * 64, "bytes": 4},
    )
    uploaded: dict = {}

    def _upload(uri: str, data: bytes) -> None:
        order.append(uri)
        uploaded.update(json.loads(data))

    monkeypatch.setattr(evidence, "_upload_create_only", _upload)
    argv = [
        "vertex_evidence.py",
        "--run-root",
        str(tmp_path),
        "--evidence-uri",
        "gs://b/p/evidence.json",
        "--run-label",
        "l",
        "--source-sha",
        "s",
        "--image-digest",
        "d",
        "--profile",
        "smoke",
        "--validation-export-uri",
        "gs://b/p/eval_validation_patches.parquet",
    ]
    monkeypatch.setattr("sys.argv", argv)
    evidence.main()
    assert order == ["gs://b/p/eval_validation_patches.parquet", "gs://b/p/evidence.json"]
    assert uploaded["validation_export"]["uri"].endswith("eval_validation_patches.parquet")
    assert "validation_export_upload" in uploaded["phase_timings"]["uploader"]

    (tmp_path / "eval_validation_patches.parquet").unlink()
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="validation export not found"):
        evidence.main()


# ------------------------------------------------------------- run wiring


def test_assignment_from_another_patch_index_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="v6_l1_spatial_holdout")
    GlobalHydra.instance().clear()
    source = RealSourceConfig("idx", "trn", "feat", "ard")
    committed = sh.load_assignment(sh.ASSIGNMENT_DIR / "v1.json").patch_index
    monkeypatch.setattr(run_module, "patch_index_fingerprints", lambda root: dict(committed))
    resolved = run_module.resolve_spatial_holdout(cfg, source)
    assert resolved is not None
    assignment, info = resolved
    assert info["holdout_anchors"] == len(assignment.holdout) == 54
    assert info["assignment_sha256"] == sh.assignment_sha256(sh.ASSIGNMENT_DIR / "v1.json")
    drifted = {**committed, "patches_total": "1"}
    monkeypatch.setattr(run_module, "patch_index_fingerprints", lambda root: drifted)
    with pytest.raises(RuntimeError, match="another patch index"):
        run_module.resolve_spatial_holdout(cfg, source)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        locked = compose(config_name="v6_l1_locked")
    GlobalHydra.instance().clear()
    assert run_module.resolve_spatial_holdout(locked, source) is None


@pytest.fixture
def lifecycle_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    identity = SimpleNamespace(to_payload=lambda: {"git_commit": "test"})
    monkeypatch.setattr(run_module, "resolve_run_identity", lambda *a, **k: identity)
    monkeypatch.setattr(run_module, "enforce_run_lock", lambda cfg, ident: None)
    monkeypatch.setattr(run_module, "persist_run_identity", lambda root, ident: None)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        composed = compose(
            config_name="contract_smoke",
            overrides=["data.n_active_channels=22", "trainer.max_epochs=1", "evaluate_test=false"],
        )
    GlobalHydra.instance().clear()
    cfg = OmegaConf.create(OmegaConf.to_container(composed, resolve=True))
    cfg.output_root = str(tmp_path / "data" / "runs" / "holdout")
    cfg.stage1_full = True  # retain full_summary.json, as a matrix run does
    cfg.stage1_residual_prior = True  # as on the v6 lock; predictions stay inside the eval clamp
    cfg.trainer.accelerator = "cpu"
    assert isinstance(cfg, DictConfig)
    return cfg


def test_fit_selects_on_seen_anchors_and_exports_all_validation_patches(
    lifecycle_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    world = _build_world(tmp_path_factory.mktemp("holdout-fit"), 22)
    _patch_sources(monkeypatch, world)
    module = _module(world, holdout=_held_out(world), batch_size=1, mode="eager")
    info = {"assignment": "unit", "holdout_anchors": 2}
    result = run_module._fit_contract_lifecycle(
        lifecycle_cfg,
        "holdout",
        data_module=module,
        source_metadata={"spatial_holdout": info},
        reproducibility="test",
    )
    assert result.run_ok
    root = Path(str(lifecycle_cfg.output_root))
    summary = json.loads((root / "full_summary.json").read_text())
    block = summary["spatial_holdout"]
    assert block["assignment"] == "unit"
    assert (block["validation_patches_seen"], block["validation_patches_held_out"]) == (1, 1)
    # Selection and its report cover the seen validation patch only.
    rows = {
        row["patch_id"]: row
        for row in pq.read_table(root / "eval_validation_patches.parquet").to_pylist()
    }
    assert set(rows) == {"p3", "p4"}
    assert summary["validation_report"]["valid_cells"] == rows["p3"]["valid_cells"]
    assert block["validation_report_held_out"]["valid_cells"] == rows["p4"]["valid_cells"]
    seen_mae = rows["p3"]["abs_error_sum"] / rows["p3"]["valid_cells"]
    assert summary["best_metric"] == pytest.approx(seen_mae, rel=1e-3)
    assert summary["test"] == {"skipped": True}
    scope = json.loads((root / "data_scope.json").read_text())
    assert scope["spatial_holdout"]["train_removed"] == 1
    assert scope["patch_ids"]["train"] == ["p0", "p2"]
