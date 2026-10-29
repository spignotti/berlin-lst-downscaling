"""Epoch-curve helper: reads only train/validation MAE, never test metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "operators" / "read_epoch_curve.py"

_EPOCHS = [
    {"epoch": 1, "train_mae_100m": 3.0, "validation_mae_100m": 2.5, "test_mae_100m": 9.87654},
    {"epoch": 2, "train_mae_100m": 2.5, "validation_mae_100m": 2.1},
    {"epoch": 3, "train_mae_100m": 2.0, "validation_mae_100m": 2.1},
    {"epoch": 4, "train_mae_100m": None, "validation_mae_100m": None},
]


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("read_epoch_curve", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, payload: object) -> str:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_evidence_record_marks_best_and_hides_test(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = {
        "full": {"epoch_metrics": _EPOCHS},
        "summary": {"test": {"mae_100m": 9.87654}, "test_scope": "full"},
    }
    assert tool.main([_write(tmp_path, record)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "epoch  train_mae_100m  validation_mae_100m"
    best_lines = [line for line in out.splitlines() if line.endswith("*best")]
    assert len(best_lines) == 1 and best_lines[0].split()[0] == "2"
    assert "best_epoch=2 validation_mae_100m=2.1000" in out
    assert "9.87654" not in out
    assert "test" not in out
    assert "-" in out.splitlines()[4]


def test_plain_list_and_top_level_and_probe(tool: ModuleType, tmp_path: Path) -> None:
    for payload in (_EPOCHS, {"epoch_metrics": _EPOCHS}, {"probe": {"epoch_metrics": _EPOCHS}}):
        assert tool.main([_write(tmp_path, payload)]) == 0


def test_extract_curve_ignores_extra_keys(tool: ModuleType) -> None:
    curve = tool.extract_curve(_EPOCHS)
    assert curve[0] == (1, 3.0, 2.5)
    assert tool.best_epoch(curve) == (2, 2.1)


def test_empty_curve_exits_1(tool: ModuleType, tmp_path: Path) -> None:
    assert tool.main([_write(tmp_path, {"full": {"epoch_metrics": []}})]) == 1


def test_no_finite_validation_exits_1(tool: ModuleType, tmp_path: Path) -> None:
    payload = [{"epoch": 1, "train_mae_100m": 1.0, "validation_mae_100m": None}]
    assert tool.main([_write(tmp_path, payload)]) == 1


def test_unreadable_input_exits_1(tool: ModuleType, tmp_path: Path) -> None:
    assert tool.main([str(tmp_path / "missing.json")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert tool.main([str(bad)]) == 1
    assert tool.main([_write(tmp_path, {"unrelated": 1})]) == 1
