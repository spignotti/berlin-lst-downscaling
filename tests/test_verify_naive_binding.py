"""Offline tests for scripts/validators/verify_naive_binding.py (synthetic fixture only)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_naive_v6_val"
CONFIG = "random_forest_v6_val_smoke"
PIN_FPS = {
    "index_policy_hash": "79add7e6a0736fb9",
    "source_policy_hash": "193fa9b23943a839",
    "patches_total": "8078",
}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_naive_binding", ROOT / "scripts" / "validators" / "verify_naive_binding.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_naive_binding"] = module
    spec.loader.exec_module(module)
    return module


vnb = _load()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(report: Path, sha: str, splits: str = "validation") -> int:
    return int(
        vnb.main(
            [
                "--report",
                str(report),
                "--sha256",
                sha,
                "--config-name",
                CONFIG,
                "--expected-splits",
                splits,
            ]
        )
    )


@pytest.fixture(autouse=True)
def _real_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    # tests/conftest.py stubs the pin; the binding must be checked against the committed one.
    monkeypatch.setattr(
        vnb.ssim_range,
        "SSIM_DATA_RANGE_ARTIFACT",
        ROOT / "configs" / "modeling" / "ssim_data_range" / "v1.yaml",
    )


@pytest.fixture
def box(tmp_path: Path) -> Path:
    dest = tmp_path / "run1"
    shutil.copytree(FIXTURE, dest)
    return dest


def _mutate_report(box: Path, fn: Callable[[dict], None]) -> Path:
    path = box / "baseline_report.json"
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("config", ["random_forest_v6_val_smoke", "random_forest_v6_val_full"])
def test_expected_fingerprints_from_pin(config: str) -> None:
    assert vnb.expected_fingerprints(config) == PIN_FPS


def test_pin_root_mismatch_raises(tmp_path: Path) -> None:
    pin = tmp_path / "pin.yaml"
    pin.write_text(
        "patch_index_root: gs://x/patch-index/v9\n"
        "patch_index_fingerprints:\n  index_policy_hash: a\n  source_policy_hash: b\n"
        "  patches_total: '1'\n"
    )
    with pytest.raises(vnb.BindingError, match="pin root"):
        vnb.expected_fingerprints(CONFIG, pin_path=pin)


def test_valid_binding_passes(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 0
    assert "OK: naive evidence bound" in capsys.readouterr().out


def _set_fp(key: str, value: object) -> Callable[[dict], None]:
    def fn(d: dict) -> None:
        d["fingerprints"][key] = value

    return fn


def _add_test_split(d: dict) -> None:
    d["splits"]["test"] = dict(d["splits"]["validation"])


def _stale_v3(d: dict) -> None:
    d["fingerprints"]["source_policy_hash"] = "9f9287255803d759"
    d["fingerprints"]["patches_total"] = "10018"
    _add_test_split(d)


def _no_rev(d: dict) -> None:
    d["git_revision"] = "unknown"


def _zero_cells(d: dict) -> None:
    d["splits"]["validation"]["valid_cells"] = 0


def _set(key: str, value: object) -> Callable[[dict], None]:
    def fn(d: dict) -> None:
        d[key] = value

    return fn


def _set_context(box: Path, payload: dict) -> None:
    (box / "logs" / "baseline" / "synthetic.context.json").write_text(json.dumps(payload))


REV = "1234567890abcdef1234567890abcdef12345678"

REPORT_CASES: dict[str, Callable[[dict], None]] = {
    "index_policy_hash": _set_fp("index_policy_hash", "deadbeef"),
    "source_policy_hash": _set_fp("source_policy_hash", "deadbeef"),
    "patches_total": _set_fp("patches_total", "9999"),
    "test_split": _add_test_split,
    "wrong_method": _set("method", "random_forest"),
    "stale_v3": _stale_v3,
    "no_revision": _no_rev,
    "zero_cells": _zero_cells,
}


@pytest.mark.parametrize("case", sorted(REPORT_CASES))
def test_report_mutation_rejected(box: Path, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    path = _mutate_report(box, REPORT_CASES[case])
    if case == "stale_v3":
        path = path.rename(box / "old_report.json")
    assert _run(path, _sha(path), "validation") == 1
    assert "FAIL" in capsys.readouterr().out


def test_stale_v3_with_both_splits_also_rejected(
    box: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _mutate_report(box, _stale_v3)
    assert _run(path, _sha(path), "test,validation") == 1
    assert "fingerprint" in capsys.readouterr().out


def test_missing_file(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(box / "nope.json", "a" * 64) == 1
    assert "FAIL" in capsys.readouterr().out


def test_wrong_sha(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(box / "baseline_report.json", "a" * 64) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("sha", ["ABC", "A" * 64, "g" * 64, ""])
def test_malformed_sha(box: Path, sha: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(box / "baseline_report.json", sha) == 1
    assert "FAIL" in capsys.readouterr().out


def test_corrupt_json(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = box / "baseline_report.json"
    path.write_bytes(b"\xff\x00garbage{")
    assert _run(path, _sha(path)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_bounded_coverage(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (box / "validation.txt").write_text("  selection validation: bounded, 2 scored\n")
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_fail_line_in_validation_txt(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with (box / "validation.txt").open("a") as fh:
        fh.write("FAIL: something\n")
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_validation_txt(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (box / "validation.txt").unlink()
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_dirty_context(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _set_context(box, {"git_commit": REV, "git_dirty": True})
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_context(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (box / "logs" / "baseline" / "synthetic.context.json").unlink()
    report = box / "baseline_report.json"
    assert _run(report, _sha(report)) == 1
    assert "FAIL" in capsys.readouterr().out
