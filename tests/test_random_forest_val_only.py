"""Offline tests for the validation-only (no test 2025) RF/baseline mode."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling.contracts import REAL_PATCH_CELLS, REAL_PATCH_PX
from berlin_lst_downscaling.modeling.patches import PatchRef, RealSample, RealSampleMeta

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "modeling"
N_V6 = 22
V3_CHAIN_KEYS = (
    "data_release_id",
    "features_root",
    "training_root",
    "patch_index_root",
    "ard_root",
)


def _load_script(relative: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_script("scripts/runners/run_random_forest.py", "run_rf_under_test")


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    return _load_script("scripts/validators/validate_random_forest.py", "validate_rf_val_only")


def _compose(name: str, overrides: list[str] | None = None, hydra_config: bool = False):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name=name, overrides=overrides or [], return_hydra_config=hydra_config
        )


# ── a. split resolution ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ("validation", "test")),
        (["validation", "test"], ("validation", "test")),
        (["validation"], ("validation",)),
    ],
)
def test_resolve_splits_accepts(runner: ModuleType, raw, expected) -> None:
    assert runner._resolve_splits(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [["test"], ["test", "validation"], [], ["validation", "validation"], ["train", "validation"]],
)
def test_resolve_splits_rejects(runner: ModuleType, raw) -> None:
    with pytest.raises(ValueError, match="splits must be exactly"):
        runner._resolve_splits(raw)


# ── b. configs ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "parent", "expected_splits"),
    [
        ("random_forest_v6_val_full", "random_forest_v6_full", ["validation"]),
        ("random_forest_v6_val_smoke", "random_forest_v6_smoke", ["validation"]),
        ("baseline_v6_val", "baseline_v6_full", ["train", "validation"]),
    ],
)
def test_val_only_configs_match_parent(name: str, parent: str, expected_splits: list[str]) -> None:
    cfg, base = _compose(name), _compose(parent)
    assert list(cfg.splits) == expected_splits
    assert "test" not in cfg.splits
    assert str(cfg.output_root).endswith("-notest")
    for key in V3_CHAIN_KEYS:
        assert cfg[key] == base[key]
    assert cfg.get("profile") == base.get("profile")
    if "forest" in base:
        assert OmegaConf.to_container(cfg.forest) == OmegaConf.to_container(base.forest)
        assert cfg.seed == base.seed
        assert cfg.max_cells == base.max_cells


@pytest.mark.parametrize("name", ["random_forest_v6_full", "random_forest_v6_smoke"])
def test_parent_rf_configs_keep_test(name: str) -> None:
    assert "test" in _compose(name).splits


# ── c. end-to-end runner with fakes ───────────────────────────────────


def _meta(ref: PatchRef) -> RealSampleMeta:
    return RealSampleMeta(
        patch_id=ref.patch_id,
        scene_id=ref.scene_id,
        split=ref.split,
        year=ref.year,
        row=ref.row,
        col=ref.col,
        filled_feature_pixels=0,
    )


def _make_sample(ref: PatchRef, seed: int) -> RealSample:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N_V6, REAL_PATCH_PX, REAL_PATCH_PX)).astype(np.float32)
    prior = np.full((1, REAL_PATCH_PX, REAL_PATCH_PX), 290.0, dtype=np.float32)
    target = (290.0 + rng.normal(size=(1, REAL_PATCH_CELLS, REAL_PATCH_CELLS))).astype(np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    return RealSample(_meta(ref), features, prior, target, mask)


def _pool() -> list[PatchRef]:
    refs = []
    for split, year, count in (("train", 2018, 3), ("validation", 2024, 2), ("test", 2025, 2)):
        for i in range(count):
            refs.append(
                PatchRef(
                    f"{split}-{i}",
                    f"LC08_{split}",
                    "S2_X",
                    split,
                    year,
                    i * REAL_PATCH_CELLS,
                    0,
                    240,
                    "m",
                )
            )
    return refs


class _Harness:
    def __init__(self) -> None:
        self.requested_splits: tuple[str, ...] | None = None
        self.read_refs: list[PatchRef] = []
        self.wandb_name: str | None = None


def _install_fakes(monkeypatch: pytest.MonkeyPatch, runner: ModuleType) -> _Harness:
    harness = _Harness()
    pool = _pool()

    def fake_load_patch_refs(source, *, splits, **_kwargs):
        harness.requested_splits = tuple(splits)
        return [ref for ref in pool if ref.split in splits]

    class FakeReader:
        def __init__(self, _source) -> None:
            self.exclusions: Counter[str] = Counter()
            self.scaler = object()

        def read_patch(self, ref: PatchRef) -> RealSample:
            harness.read_refs.append(ref)
            return _make_sample(ref, seed=len(harness.read_refs))

    class FakeRun:
        id = "fake-run"

        def log(self, _payload) -> None:
            return None

    @contextmanager
    def fake_wandb_init(**kwargs):
        harness.wandb_name = kwargs["name"]
        yield FakeRun()

    monkeypatch.setattr(runner, "load_patch_refs", fake_load_patch_refs)
    monkeypatch.setattr(runner, "RealPatchReader", FakeReader)
    monkeypatch.setattr(runner, "assert_scaler_spec_matches", lambda *a, **k: None)
    monkeypatch.setattr(runner, "git_revision_for", lambda *a, **k: "test-rev")
    monkeypatch.setattr(runner.wandb, "init", fake_wandb_init)
    monkeypatch.setattr(
        "berlin_lst_downscaling.modeling.random_forest.patch_index_fingerprints",
        lambda _root: {"index_policy_hash": "x", "source_policy_hash": "y", "patches_total": 7},
    )
    return harness


def _run(
    runner: ModuleType,
    name: str,
    tmp_path: Path,
    extra: list[str] | None = None,
    *,
    small: bool = True,
) -> dict:
    overrides = [f"output_root={tmp_path / 'out'}", *(extra or [])]
    if small:
        overrides += ["forest.n_estimators=2", "forest.max_depth=3", "max_cells=200"]
    cfg: DictConfig = _compose(name, overrides, hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    assert runner.main.__wrapped__(cfg) == 0
    return json.loads((tmp_path / "out" / "random_forest_report.json").read_text())


def test_val_only_run_never_touches_test(
    monkeypatch: pytest.MonkeyPatch, runner: ModuleType, tmp_path: Path
) -> None:
    harness = _install_fakes(monkeypatch, runner)
    report = _run(runner, "random_forest_v6_val_smoke", tmp_path)
    assert harness.requested_splits == ("train", "validation")
    assert harness.read_refs
    assert all(ref.split != "test" for ref in harness.read_refs)
    assert harness.wandb_name is not None and "-notest-" in harness.wandb_name
    scope = report["scope"]
    assert scope["test_split"] == "skipped"
    assert set(scope["eval_splits"]) == {"validation"}
    for stage in report["stages"].values():
        assert set(stage["splits"]) == {"validation"}
        assert {record["split"] for record in stage["patches"]} == {"validation"}
    assert {entry["split"] for entry in report["probes"]["patches"]} == {"validation"}


def test_val_and_test_run_unchanged(
    monkeypatch: pytest.MonkeyPatch, runner: ModuleType, tmp_path: Path
) -> None:
    harness = _install_fakes(monkeypatch, runner)
    report = _run(runner, "random_forest_v6_smoke", tmp_path)
    assert harness.requested_splits == ("train", "validation", "test")
    assert any(ref.split == "test" for ref in harness.read_refs)
    assert harness.wandb_name is not None and "-notest" not in harness.wandb_name
    assert "test_split" not in report["scope"]
    assert set(report["scope"]["eval_splits"]) == {"validation", "test"}


def test_full_val_only_still_requires_all_patches(
    monkeypatch: pytest.MonkeyPatch, runner: ModuleType, tmp_path: Path
) -> None:
    harness = _install_fakes(monkeypatch, runner)
    with pytest.raises(ValueError, match="full RF requires all patches"):
        _run(
            runner, "random_forest_v6_val_full", tmp_path, ["max_patches_per_split=1"], small=False
        )
    assert harness.requested_splits is None


# ── d. validator protocol ─────────────────────────────────────────────


def _index_rows() -> list[dict]:
    return [
        {"split": "train", "year": 2018, "patch_id": "t0"},
        {"split": "validation", "year": 2024, "patch_id": "v0"},
        {"split": "validation", "year": 2024, "patch_id": "v1"},
        {"split": "test", "year": 2025, "patch_id": "x0"},
    ]


def _full_report(*, test: bool, marker: str | None = None) -> dict:
    counts = {"validation": 2, "test": 1}
    splits = ("validation", "test") if test else ("validation",)
    scope: dict = {
        "profile": "full",
        "train_patches": 1,
        "max_train_patches": None,
        "max_patches_per_split": None,
        "eval_splits": {s: counts[s] for s in splits},
    }
    if marker is not None:
        scope["test_split"] = marker
    stage = {
        "splits": {s: {"requested_patches": counts[s]} for s in splits},
        "patches": [{"patch_id": f"{s}-0", "split": s} for s in splits],
    }
    return {
        "scope": scope,
        "forest": {
            "min_samples_leaf": 5,
            "max_features": 1.0,
            "criterion": "squared_error",
            "bootstrap": True,
            "random_state": 0,
            "n_estimators": 200,
            "max_depth": 20,
            "n_jobs": 1,
        },
        "train_pool": {"seed": 0, "max_cells": 500_000, "requested_patches": 1},
        "stages": {"1": stage, "2": json.loads(json.dumps(stage))},
        "probes": {"patches": [{"patch_id": f"{s}-0", "split": s} for s in splits]},
    }


def _errors(validator: ModuleType, report: dict) -> list[str]:
    errors: list[str] = []
    validator._check_protocol(report, _index_rows(), errors)
    return errors


def test_validator_val_only_report_passes(validator: ModuleType) -> None:
    assert _errors(validator, _full_report(test=False, marker="skipped")) == []


@pytest.mark.parametrize("where", ["scope", "stage_splits", "stage_patches", "probes"])
def test_validator_val_only_rejects_test_leak(validator: ModuleType, where: str) -> None:
    report = _full_report(test=False, marker="skipped")
    if where == "scope":
        report["scope"]["eval_splits"]["test"] = 1
    elif where == "stage_splits":
        report["stages"]["1"]["splits"]["test"] = {"requested_patches": 1}
    elif where == "stage_patches":
        report["stages"]["1"]["patches"].append({"patch_id": "x0", "split": "test"})
    else:
        report["probes"]["patches"].append({"patch_id": "x0", "split": "test"})
    assert any("must not contain the test split" in e for e in _errors(validator, report))


def test_validator_val_and_test_unchanged(validator: ModuleType) -> None:
    assert _errors(validator, _full_report(test=True)) == []
    errors = _errors(validator, _full_report(test=False))
    assert any("must evaluate validation and test" in e for e in errors)


def test_validator_rejects_invalid_test_split_marker(validator: ModuleType) -> None:
    errors = _errors(validator, _full_report(test=True, marker="maybe"))
    assert any("scope.test_split" in e for e in errors)


# ── e. baseline ───────────────────────────────────────────────────────


class _Stop(Exception):
    pass


def test_baseline_val_config_never_requests_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _load_script("scripts/runners/run_baseline.py", "run_baseline_under_test")
    seen: dict = {}

    def fake_load_refs(_source, *, splits, **_kwargs):
        seen["splits"] = tuple(splits)
        raise _Stop

    monkeypatch.setattr(baseline, "load_refs", fake_load_refs)
    cfg = _compose("baseline_v6_val", [f"output_root={tmp_path / 'out'}"])
    with pytest.raises(_Stop):
        baseline.main.__wrapped__(cfg)
    assert seen["splits"] == ("train", "validation")
