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


# ---------------------------------------------------- checkpoint-selection optimism


def _run(best_epoch: int, best: float, tail: list[float], *, test: float = 9.87654) -> list[dict]:
    """40 epochs: validation 3.0 everywhere, ``best`` at ``best_epoch``, ``tail`` = epochs 36-40."""
    entries = [
        {"epoch": e, "train_mae_100m": 3.0, "validation_mae_100m": 3.0, "test_mae_100m": test}
        for e in range(1, 41)
    ]
    entries[best_epoch - 1]["validation_mae_100m"] = best
    for epoch, value in zip(range(36, 41), tail, strict=True):
        entries[epoch - 1]["validation_mae_100m"] = value
    return entries


def _files(tmp_path: Path, tag: str, runs: list[list[dict]]) -> list[str]:
    out = []
    for i, entries in enumerate(runs):
        path = tmp_path / f"{tag}{i}.epoch_metrics.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        out.append(str(path))
    return out


def test_run_optimism_is_best_last_and_the_mean_of_the_last_five(tool: ModuleType) -> None:
    curve = tool.extract_curve(_run(30, 0.40, [0.50, 0.50, 0.50, 0.50, 0.60]))
    out = tool.run_optimism(curve)
    assert out["best_epoch"] == 30 and out["best"] == 0.40 and out["last"] == 0.60
    assert out["mean_last"] == pytest.approx(0.52)
    assert out["best_minus_last"] == pytest.approx(-0.20)
    assert out["best_minus_mean_last"] == pytest.approx(-0.12)


def test_run_optimism_refuses_short_or_incomplete_curves(tool: ModuleType) -> None:
    short = tool.extract_curve(_run(30, 0.4, [0.5] * 5)[:39])
    with pytest.raises(ValueError, match="ends at epoch 39"):
        tool.run_optimism(short)
    gap = _run(30, 0.4, [0.5] * 5)
    gap[37]["validation_mae_100m"] = None
    with pytest.raises(ValueError, match="36-40"):
        tool.run_optimism(tool.extract_curve(gap))


def test_arm_report_gives_mean_and_sd_over_seeds_and_names_the_holdout_metric(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matrix = _files(
        tmp_path,
        "m",
        [
            _run(30, 0.40, [0.5] * 5),
            _run(33, 0.42, [0.5] * 5),
            _run(40, 0.44, [0.46, 0.46, 0.46, 0.46, 0.44]),
        ],
    )
    holdout = _files(tmp_path, "h", [_run(35, 0.30, [0.4] * 5)])
    rc = tool.main(["--arm", "L1", *matrix, "--holdout-arm", "L1-holdout", *holdout])
    assert rc == 0
    out = capsys.readouterr().out
    assert "L1: 3 seed(s), best epochs 30,33,40" in out
    best_line = next(line for line in out.splitlines() if "best epoch (selected)" in line)
    assert "0.4200 (SD 0.0200)" in best_line  # mean 0.42, SD (ddof 1) of 0.40, 0.42, 0.44
    gap_line = next(line for line in out.splitlines() if line.strip().startswith("best minus last"))
    assert "-0.0600" in gap_line  # (-0.10 - 0.08 + 0.0) / 3
    assert "L1-holdout: 1 seed(s)" in out and "(SD -)" in out
    assert out.count("seen-anchor metric the checkpoint was selected on") == 1
    assert "9.87654" not in out and "test" not in out


def test_arm_report_reads_local_files_only_and_exits_1_on_bad_input(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["--arm", "L1", "gs://bucket/run/epoch_metrics.json"]) == 1
    assert "local files only" in capsys.readouterr().err
    assert tool.main(["--arm", "L1", str(tmp_path / "missing.json")]) == 1
    with pytest.raises(SystemExit):
        tool.main([])
    with pytest.raises(SystemExit):
        tool.main(["evidence.json", "--arm", "L1", "a.json"])
