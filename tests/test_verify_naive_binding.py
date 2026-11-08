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


RELS = ["baseline_report.json", "validation.txt", "logs/baseline/synthetic.context.json"]


@pytest.fixture
def box(tmp_path: Path) -> Path:
    dest = tmp_path / "run1"
    shutil.copytree(FIXTURE, dest)
    return dest


def _write_manifest(box: Path, lines: list[str] | None = None) -> str:
    manifest = box / vnb.MANIFEST_NAME
    if lines is None:
        manifest.write_text("".join(f"{_sha(box / r)}  {r}\n" for r in RELS))
    else:
        manifest.write_text("\n".join(lines))
    return _sha(manifest)


def _bound(box: Path) -> int:
    return _run(box / "baseline_report.json", _write_manifest(box))


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
    _mutate_report(box, REPORT_CASES[case])
    assert _bound(box) == 1
    assert "FAIL" in capsys.readouterr().out


def test_valid_binding_passes(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = box / "baseline_report.json"
    assert _bound(box) == 0
    out = capsys.readouterr().out
    assert "OK: naive evidence bound" in out
    assert f"report_sha256={_sha(report)}" in out


def test_stale_v3_with_both_splits_also_rejected(
    box: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _mutate_report(box, _stale_v3)
    assert _run(path, _write_manifest(box), "test,validation") == 1
    assert "fingerprint" in capsys.readouterr().out


def test_missing_file(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(box)
    assert _run(box / "nope.json", "a" * 64) == 1
    assert "FAIL" in capsys.readouterr().out


def test_wrong_sha(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(box)
    assert _run(box / "baseline_report.json", "a" * 64) == 1
    assert "manifest sha256" in capsys.readouterr().out


@pytest.mark.parametrize("sha", ["ABC", "A" * 64, "g" * 64, ""])
def test_malformed_sha(box: Path, sha: str, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(box)
    assert _run(box / "baseline_report.json", sha) == 1
    assert "FAIL" in capsys.readouterr().out


def test_corrupt_json(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (box / "baseline_report.json").write_bytes(b"\xff\x00garbage{")
    assert _bound(box) == 1
    assert "FAIL" in capsys.readouterr().out


def test_bounded_coverage(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (box / "validation.txt").write_text("  selection validation: bounded, 2 scored\n")
    assert _bound(box) == 1
    assert "FAIL" in capsys.readouterr().out


def test_fail_line_in_validation_txt(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with (box / "validation.txt").open("a") as fh:
        fh.write("FAIL: something\n")
    assert _bound(box) == 1
    assert "FAIL" in capsys.readouterr().out


def test_dirty_context(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _set_context(box, {"git_commit": REV, "git_dirty": True})
    assert _bound(box) == 1
    assert "FAIL" in capsys.readouterr().out


def test_manifest_missing(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(box / "baseline_report.json", "a" * 64) == 1
    assert "manifest missing" in capsys.readouterr().out


def test_report_sha_as_binding_rejected(box: Path) -> None:
    report = box / "baseline_report.json"
    _write_manifest(box)
    assert _run(report, _sha(report)) == 1


@pytest.mark.parametrize("rel", RELS)
def test_covered_file_changed(box: Path, rel: str, capsys: pytest.CaptureFixture[str]) -> None:
    sha = _write_manifest(box)
    with (box / rel).open("ab") as fh:
        fh.write(b" ")
    assert _run(box / "baseline_report.json", sha) == 1
    assert "covered file changed" in capsys.readouterr().out


@pytest.mark.parametrize("rel", RELS)
def test_covered_file_missing(box: Path, rel: str, capsys: pytest.CaptureFixture[str]) -> None:
    sha = _write_manifest(box)
    (box / rel).unlink()
    assert _run(box / "baseline_report.json", sha) == 1
    out = capsys.readouterr().out
    # The report itself is rejected earlier by the explicit existence check.
    assert ("naive evidence not found" if rel == RELS[0] else "covered file missing") in out


def test_unlisted_context_rejected(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sha = _write_manifest(box)
    ctx = box / "logs" / "baseline"
    shutil.copy(ctx / "synthetic.context.json", ctx / "other.context.json")
    assert _run(box / "baseline_report.json", sha) == 1
    assert "unlisted" in capsys.readouterr().out


def test_two_listed_contexts_rejected(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctx = box / "logs" / "baseline"
    shutil.copy(ctx / "synthetic.context.json", ctx / "other.context.json")
    lines = [f"{_sha(box / r)}  {r}" for r in [*RELS, "logs/baseline/other.context.json"]]
    sha = _write_manifest(box, [*lines, ""])
    assert _run(box / "baseline_report.json", sha) == 1
    assert "exactly one" in capsys.readouterr().out


def test_manifest_unknown_entry(box: Path) -> None:
    (box / "notes.txt").write_text("x")
    lines = [f"{_sha(box / r)}  {r}" for r in [*RELS, "notes.txt"]]
    sha = _write_manifest(box, [*lines, ""])
    assert _run(box / "baseline_report.json", sha) == 1


def _entry_lines(box: Path, replace_validation: tuple[str, str]) -> list[str]:
    name, sha = replace_validation
    return [
        f"{sha}  {name}" if r == "validation.txt" else f"{_sha(box / r)}  {r}" for r in RELS
    ] + [""]


@pytest.mark.parametrize("case", ["parent", "deep_parent", "absolute", "symlink"])
def test_manifest_entry_outside(
    box: Path, tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    if case == "symlink":
        (box / "validation.txt").unlink()
        (box / "validation.txt").symlink_to(outside)
        sha = _write_manifest(box)
    else:
        name = {
            "parent": "../outside.json",
            "deep_parent": "logs/baseline/../../../outside.context.json",
            "absolute": str(outside),
        }[case]
        sha = _write_manifest(box, _entry_lines(box, (name, _sha(outside))))
    assert _run(box / "baseline_report.json", sha) == 1
    assert "outside the evidence dir" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["garbage", "", "upper", "single_space"])
def test_manifest_malformed_line(box: Path, bad: str) -> None:
    good = [f"{_sha(box / r)}  {r}" for r in RELS]
    line = {
        "garbage": "garbage",
        "": "",
        "upper": f"{_sha(box / RELS[1]).upper()}  {RELS[1]}",
        "single_space": f"{_sha(box / RELS[1])} {RELS[1]}",
    }[bad]
    sha = _write_manifest(box, [good[0], line, good[2], ""])
    assert _run(box / "baseline_report.json", sha) == 1


def test_duplicate_manifest_entry(box: Path) -> None:
    lines = [f"{_sha(box / r)}  {r}" for r in [*RELS, RELS[0]]]
    sha = _write_manifest(box, [*lines, ""])
    assert _run(box / "baseline_report.json", sha) == 1


def test_report_name_required(box: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = box / "other.json"
    shutil.copy(box / "baseline_report.json", other)
    sha = _write_manifest(box)
    assert _run(other, sha) == 1
    assert "must be named" in capsys.readouterr().out


def test_manifest_name_matches_lib() -> None:
    for rel in ("naive-evidence-lib.sh", "run-random-forest-vm.sh"):
        assert vnb.MANIFEST_NAME in (ROOT / "scripts" / "runners" / "vm" / rel).read_text()
