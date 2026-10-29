"""Offline tests for the validation-only (evaluate_test=false) v6 baselines."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
)
from berlin_lst_downscaling.modeling import baseline as baseline_mod
from berlin_lst_downscaling.modeling.baseline import BaselineReport
from berlin_lst_downscaling.modeling.patches import RealSourceConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "modeling"


def _load(relative: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rf_runner() -> ModuleType:
    return _load("scripts/runners/run_random_forest.py", "run_rf_under_test")


@pytest.fixture(scope="module")
def baseline_runner() -> ModuleType:
    return _load("scripts/runners/run_baseline.py", "run_baseline_under_test")


@pytest.fixture(scope="module")
def rf_validator() -> ModuleType:
    return _load("scripts/validators/validate_random_forest.py", "val_rf_under_test")


@pytest.fixture(scope="module")
def baseline_validator() -> ModuleType:
    return _load("scripts/validators/validate_baseline.py", "val_baseline_under_test")


def _compose(name: str, *overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _source() -> RealSourceConfig:
    return RealSourceConfig(
        patch_index_root="i", training_root="t", features_root="f", ard_root="a"
    )


# ── RF runner ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["random_forest_v6_full", "random_forest_v6_smoke"])
def test_rf_resolve_eval_splits(rf_runner: ModuleType, name: str) -> None:
    v6 = FEATURE_SCHEMA_VERSION_V6
    assert rf_runner._resolve_eval_splits(_compose(name), v6) == ("validation", "test")
    off = _compose(name, "splits=[validation]", "evaluate_test=false")
    assert rf_runner._resolve_eval_splits(off, v6) == ("validation",)
    with pytest.raises(ValueError):
        rf_runner._resolve_eval_splits(_compose(name, "splits=[validation]"), v6)
    for splits in ("[validation,test]", "[test]"):
        with pytest.raises(ValueError):
            rf_runner._resolve_eval_splits(
                _compose(name, f"splits={splits}", "evaluate_test=false"), v6
            )


def test_rf_v1_rejects_val_only(rf_runner: ModuleType) -> None:
    cfg = _compose("random_forest_full", "splits=[validation]", "evaluate_test=false")
    with pytest.raises(ValueError):
        rf_runner._resolve_eval_splits(cfg, FEATURE_SCHEMA_VERSION)


def test_rf_scoped_refs_never_request_test(
    rf_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake(_cfg: RealSourceConfig, *, splits: tuple[str, ...]) -> list:
        seen.append(splits)
        return []

    monkeypatch.setattr(rf_runner, "load_patch_refs", fake)
    cfg = _compose("random_forest_v6_full", "splits=[validation]", "evaluate_test=false")
    splits = rf_runner._resolve_eval_splits(cfg, FEATURE_SCHEMA_VERSION_V6)
    rf_runner._load_scoped_refs(_source(), splits)
    assert seen == [("train", "validation")]


# ── naive baseline ───────────────────────────────────────────────────────────


def test_baseline_resolve_splits(baseline_runner: ModuleType) -> None:
    v6 = FEATURE_SCHEMA_VERSION_V6
    default = _compose("baseline_v6_full")
    assert baseline_runner._resolve_splits(default, v6) == ("train", "validation", "test")
    off = _compose("baseline_v6_full", "splits=[validation]", "evaluate_test=false")
    assert baseline_runner._resolve_splits(off, v6) == ("validation",)
    for splits in ("[validation,test]", "[train]"):
        with pytest.raises(ValueError):
            baseline_runner._resolve_splits(
                _compose("baseline_v6_full", f"splits={splits}", "evaluate_test=false"), v6
            )
    v1 = _compose("baseline_full", "splits=[validation]", "evaluate_test=false")
    with pytest.raises(ValueError):
        baseline_runner._resolve_splits(v1, FEATURE_SCHEMA_VERSION)


def test_baseline_refs_never_request_test(
    baseline_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake(_cfg: RealSourceConfig, *, splits: tuple[str, ...], scene_ids: object) -> list:
        seen.append(splits)
        return []

    monkeypatch.setattr(baseline_mod, "load_patch_refs", fake)
    cfg = _compose("baseline_v6_full", "splits=[validation]", "evaluate_test=false")
    splits = baseline_runner._resolve_splits(cfg, FEATURE_SCHEMA_VERSION_V6)
    baseline_mod.load_refs(_source(), splits=splits)
    assert seen == [("validation",)]


def test_baseline_report_payload_has_evaluate_test() -> None:
    kwargs = {
        "method": "m",
        "prior_rule": "p",
        "primary_metric": "a",
        "secondary_metric": "b",
        "splits": {},
        "patches": [],
        "exclusions": {},
        "source": {},
        "fingerprints": {},
        "git_revision": "x",
        "written_at": "t",
    }
    assert BaselineReport(**kwargs).to_payload()["evaluate_test"] is True
    assert BaselineReport(**kwargs, evaluate_test=False).to_payload()["evaluate_test"] is False


# ── RF validator ─────────────────────────────────────────────────────────────


def _rows() -> list[dict]:
    rows = [{"patch_id": f"tr{i}", "split": "train", "year": 2020} for i in range(2)]
    rows += [{"patch_id": f"va{i}", "split": "validation", "year": 2024} for i in range(3)]
    rows += [{"patch_id": f"te{i}", "split": "test", "year": 2025} for i in range(2)]
    return rows


def _rf_report(splits: dict[str, int], evaluate_test: object) -> dict:
    scope: dict = {
        "profile": "full",
        "eval_splits": dict(splits),
        "train_patches": 2,
        "max_train_patches": None,
        "max_patches_per_split": None,
    }
    if evaluate_test is not None:
        scope["evaluate_test"] = evaluate_test
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
        "train_pool": {"seed": 0, "max_cells": 500_000, "requested_patches": 2},
        "stages": {
            "1": {
                "splits": {s: {"requested_patches": n} for s, n in splits.items()},
                "patches": [{"split": s, "patch_id": f"{s}0"} for s in splits],
            }
        },
    }


def _protocol_errors(mod: ModuleType, report: dict) -> list[str]:
    errors: list[str] = []
    mod._check_protocol(report, _rows(), errors)
    return errors


def test_rf_validator_val_only(rf_validator: ModuleType) -> None:
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3}, False)) == []
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3, "test": 2}, True)) == []
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3}, True))
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3}, None))
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3, "test": 2}, False))
    assert _protocol_errors(rf_validator, _rf_report({"validation": 2}, False))
    assert _protocol_errors(rf_validator, _rf_report({"validation": 3}, "no"))


# ── baseline validator ───────────────────────────────────────────────────────


def test_baseline_validator_eval_scope(baseline_validator: ModuleType) -> None:
    def run(report: dict) -> list[str]:
        errors: list[str] = []
        baseline_validator._check_eval_scope(report, errors)
        return errors

    ok = {
        "evaluate_test": False,
        "splits": {"validation": {}},
        "patches": [{"split": "validation"}],
    }
    assert run(ok) == []
    assert run({**ok, "splits": {"validation": {}, "test": {}}})
    assert run({**ok, "patches": [{"split": "test"}]})
    assert run({**ok, "splits": {"train": {}}})
    assert run({**ok, "evaluate_test": "false"})
    legacy = {"splits": {"validation": {}, "test": {}}, "patches": [{"split": "test"}]}
    assert run(legacy) == []


# ── VM launchers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", ["run-random-forest-vm.sh", "run-baseline-vm.sh"])
def test_launchers_have_val_only_switch(script: str) -> None:
    path = ROOT / "scripts" / "runners" / "vm" / script
    text = path.read_text(encoding="utf-8")
    assert "EVAL_SPLITS" in text
    assert "evaluate_test=false" in text
    subprocess.run(["bash", "-n", str(path)], check=True)  # noqa: S603, S607
