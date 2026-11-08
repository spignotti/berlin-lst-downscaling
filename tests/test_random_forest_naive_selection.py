"""Naive evidence selection rules of scripts/validators/validate_random_forest.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6

ROOT = Path(__file__).resolve().parents[1]
VM = ROOT / "scripts" / "runners" / "vm"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_random_forest_sel", ROOT / "scripts" / "validators" / "validate_random_forest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_random_forest_sel"] = module
    spec.loader.exec_module(module)
    return module


vrf = _load()
LEGACY_GLOB = "baseline-full-*/baseline_report.json"


def _resolve(report: dict, *, naive: str | None = None, skip: bool = False):
    errors: list[str] = []
    path = vrf._resolve_naive_path(report, naive_report=naive, skip_naive=skip, errors=errors)
    return path, errors


def test_v6_report_never_auto_selects() -> None:
    assert list((ROOT / "docs" / "results").glob(LEGACY_GLOB))
    assert vrf._find_naive_report(None, require_explicit=True) is None
    path, errors = _resolve(
        {"feature_schema_version": FEATURE_SCHEMA_VERSION_V6, "scope": {"profile": "smoke"}}
    )
    assert path is None
    assert any("explicit --naive-report" in e for e in errors)


def test_val_only_report_never_auto_selects() -> None:
    path, errors = _resolve({"scope": {"profile": "smoke", "test_split": "skipped"}})
    assert path is None
    assert any("explicit --naive-report" in e for e in errors)


def test_legacy_report_keeps_auto_selection() -> None:
    path, errors = _resolve({"scope": {"profile": "smoke"}})
    assert errors == []
    assert path is not None and "docs/results/baseline-full-" in path


def test_explicit_path_wins() -> None:
    path, errors = _resolve(
        {"feature_schema_version": FEATURE_SCHEMA_VERSION_V6, "scope": {"test_split": "skipped"}},
        naive="/x/naive.json",
    )
    assert path == "/x/naive.json"
    assert errors == []


def test_full_profile_cannot_skip_naive() -> None:
    for report in (
        {"feature_schema_version": FEATURE_SCHEMA_VERSION_V6, "scope": {"profile": "full"}},
        {"scope": {"profile": "full"}},
    ):
        path, errors = _resolve(report, skip=True)
        assert path is None
        assert any("cannot skip" in e for e in errors)


def test_launchers_never_skip_naive() -> None:
    for name in ("run-random-forest-vm.sh", "naive-evidence-lib.sh", "val-baselines-vm.sh"):
        assert "--skip-naive" not in (VM / name).read_text()


@pytest.mark.parametrize("value", ["six", 6.5, True, [6]])
def test_non_integer_schema_version_layout_raises(value: object) -> None:
    with pytest.raises(ValueError, match="feature_schema_version must be an integer"):
        vrf._layout_for({"feature_schema_version": value})


def test_non_integer_schema_version_requires_explicit_naive() -> None:
    path, errors = _resolve({"feature_schema_version": "six", "scope": {"profile": "smoke"}})
    assert path is None
    assert any("explicit --naive-report" in e for e in errors)


def test_main_rejects_non_integer_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "random_forest_report.json"
    report.write_text(json.dumps({"feature_schema_version": "six"}))
    monkeypatch.setattr(sys, "argv", ["validate_random_forest.py", "--report", str(report)])
    assert vrf.main() == 1
    out = capsys.readouterr().out
    assert "feature_schema_version must be an integer" in out
    assert "FAIL" in out
