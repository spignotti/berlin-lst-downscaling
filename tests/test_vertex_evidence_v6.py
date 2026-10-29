"""Evidence record rules for early-stopped and test-skipped v6 runs (offline)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "operators" / "vertex_evidence.py"


@pytest.fixture(scope="module")
def evidence() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vertex_evidence_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(**extra: object) -> dict:
    return {"max_epochs": 20, "resolved_config": {}, **extra}


def test_v1_shaped_summary_unchanged(evidence: ModuleType) -> None:
    assert evidence.full_run_status([{}] * 20, _summary(test={"x": 1})) == (True, False)
    assert evidence.full_run_status([{}] * 19, _summary()) == (False, False)
    assert evidence.full_run_status([], None) == (False, False)


_ES_ON = {"enabled": True, "patience": 5, "min_delta": 0.0}


def test_early_stopped_run_is_complete(evidence: ModuleType) -> None:
    stopped = _summary(stopped_epoch=12, early_stopping=_ES_ON)
    assert evidence.full_run_status([{}] * 12, stopped) == (True, False)
    assert evidence.full_run_status([{}] * 20, stopped)[0] is False
    assert evidence.full_run_status([{}] * 20, _summary(stopped_epoch=None))[0] is True


@pytest.mark.parametrize(
    "extra",
    [
        {"stopped_epoch": 12},
        {"stopped_epoch": 12, "early_stopping": {**_ES_ON, "enabled": False}},
        {"stopped_epoch": 0, "early_stopping": _ES_ON},
        {"stopped_epoch": 21, "early_stopping": _ES_ON},
        {"stopped_epoch": True, "early_stopping": _ES_ON},
    ],
)
def test_inconsistent_stopped_epoch_rejected(evidence: ModuleType, extra: dict) -> None:
    with pytest.raises(ValueError, match="stopped_epoch"):
        evidence.full_run_status([{}] * 12, _summary(**extra))


def test_skipped_test_accepted_when_recorded(evidence: ModuleType) -> None:
    summary = _summary(test={"skipped": True})
    summary["resolved_config"] = {"evaluate_test": False}
    assert evidence.full_run_status([{}] * 20, summary) == (True, True)


def test_skipped_test_rejected_when_not_recorded(evidence: ModuleType) -> None:
    with pytest.raises(ValueError, match="not recorded"):
        evidence.full_run_status([{}] * 20, _summary(test={"skipped": True}))
    with pytest.raises(ValueError, match="not recorded"):
        evidence.full_run_status(
            [{}] * 20,
            {
                "max_epochs": 20,
                "test": {"skipped": True},
                "resolved_config": {"evaluate_test": True},
            },
        )


def test_evaluate_test_false_without_skipped_test_rejected(evidence: ModuleType) -> None:
    summary = _summary(test={"x": 1})
    summary["resolved_config"] = {"evaluate_test": False}
    with pytest.raises(ValueError, match="not marked skipped"):
        evidence.full_run_status([{}] * 20, summary)


def test_build_record_full_reports_test_skipped(evidence: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "epoch_metrics.json").write_text(json.dumps([{}] * 3))
    (tmp_path / "full_summary.json").write_text(
        json.dumps(
            {
                "max_epochs": 20,
                "stopped_epoch": 3,
                "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0},
                "test": {"skipped": True},
                "evaluate_test": False,
                "resolved_config": {"evaluate_test": False},
            }
        )
    )
    record = evidence.build_record(
        tmp_path,
        run_label="l",
        source_sha="s",
        image_digest="d",
        result_file=None,
        profile="full",
    )
    assert record["full"]["epochs_complete"] is True
    assert record["full"]["test_skipped"] is True


def _run_main(evidence: ModuleType, monkeypatch: pytest.MonkeyPatch, run_root: Path) -> list:
    uploads: list[tuple[str, bytes]] = []
    monkeypatch.setattr(
        evidence, "_upload_create_only", lambda uri, data: uploads.append((uri, data))
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "vertex_evidence.py",
            "--run-root",
            str(run_root),
            "--evidence-uri",
            "gs://b/p/evidence.json",
            "--run-label",
            "l",
            "--source-sha",
            "s",
            "--image-digest",
            "d",
            "--profile",
            "full",
            "--epoch-metrics-uri",
            "gs://b/p/epoch_metrics.json",
            "--epoch-metrics-only",
        ],
    )
    evidence.main()
    return uploads


def test_epoch_metrics_only_uploads_only_that_file(
    evidence: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "epoch_metrics.json").write_text("[]")
    uploads = _run_main(evidence, monkeypatch, tmp_path)
    assert uploads == [("gs://b/p/epoch_metrics.json", b"[]")]


def test_epoch_metrics_only_handles_missing_file(
    evidence: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_main(evidence, monkeypatch, tmp_path) == []
    assert "not found" in capsys.readouterr().out


def test_v1_full_record_has_no_test_skipped_key(evidence: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "epoch_metrics.json").write_text(json.dumps([{}] * 20))
    (tmp_path / "full_summary.json").write_text(
        json.dumps({"max_epochs": 20, "test": {"mae_100m": 1.0}})
    )
    record = evidence.build_record(
        tmp_path,
        run_label="l",
        source_sha="s",
        image_digest="d",
        result_file=None,
        profile="full",
    )
    assert record["full"]["epochs_complete"] is True
    assert "test_skipped" not in record["full"]
