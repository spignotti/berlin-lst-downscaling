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


def _write_modeling_log(run_root: Path, records: list[dict]) -> None:
    logs = run_root / "logs" / "modeling"
    logs.mkdir(parents=True)
    (logs / "abc.context.json").write_text(json.dumps({"pipeline": "modeling", "run_id": "abc"}))
    (logs / "abc.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _record(evidence: ModuleType, run_root: Path, **kwargs: object) -> dict:
    return evidence.build_record(
        run_root,
        run_label="l",
        source_sha="s",
        image_digest="d",
        result_file=None,
        profile="smoke",
        **kwargs,
    )


def test_evidence_retains_phase_and_epoch_timings(evidence: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "data_scope.json").write_text(
        json.dumps({"mode": "eager", "cache_build_seconds": 4.5, "cache_reused": False})
    )
    _write_modeling_log(
        tmp_path,
        [
            {"event": "phase_start", "phase": "fit"},
            {
                "event": "phase_end",
                "phase": "fit",
                "started_at": "a",
                "ended_at": "b",
                "duration_seconds": 12.5,
                "status": "ok",
            },
            {
                "event": "epoch_timing",
                "epoch": 0,
                "train_seconds": 5.0,
                "validation_seconds": 2.0,
                "epoch_seconds": 7.0,
            },
            {"event": "phase_unavailable", "phase": "container_bootstrap", "reason": "unset"},
        ],
    )
    timings = _record(evidence, tmp_path, uploader_timings={"checkpoint_hash_upload": 1.5})[
        "phase_timings"
    ]
    assert timings["available"] is True
    assert timings["phases"]["fit"]["duration_seconds"] == 12.5
    assert timings["epochs"] == [
        {"epoch": 0, "train_seconds": 5.0, "validation_seconds": 2.0, "epoch_seconds": 7.0}
    ]
    assert timings["data_prep"] == {"cache_build_seconds": 4.5, "cache_reused": False}
    assert timings["uploader"] == {"checkpoint_hash_upload": 1.5}
    assert timings["unavailable"]["container_bootstrap"] == {
        "duration_seconds": None,
        "reason": "unset",
    }


def test_missing_timing_logs_stay_backward_compatible(evidence: ModuleType, tmp_path: Path) -> None:
    record = _record(evidence, tmp_path)
    assert record["profile"] == "smoke"
    assert record["result_tail"] == []
    timings = record["phase_timings"]
    assert timings["available"] is False
    assert timings["reason"]
    assert timings["phases"] == {} and timings["epochs"] == []


def test_unobservable_intervals_are_explicitly_unavailable(
    evidence: ModuleType, tmp_path: Path
) -> None:
    unavailable = _record(evidence, tmp_path)["phase_timings"]["unavailable"]
    for name in ("provision_to_container", "platform_teardown", "evidence_upload"):
        assert unavailable[name]["duration_seconds"] is None
        assert unavailable[name]["reason"]


def test_artifact_uploads_are_timed_before_the_evidence_upload(
    evidence: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"ckpt")
    (tmp_path / "epoch_metrics.json").write_text("[]")
    monkeypatch.setattr(
        evidence,
        "_upload_checkpoint_create_only",
        lambda path, uri: (
            order.append("checkpoint") or {"uri": uri, "sha256": "0" * 64, "bytes": 4}
        ),
    )

    def _upload(uri: str, data: bytes) -> None:
        order.append(uri)
        if uri.endswith("evidence.json"):
            uploaded.update(json.loads(data))

    uploaded: dict = {}
    monkeypatch.setattr(evidence, "_upload_create_only", _upload)
    monkeypatch.setattr(
        "sys.argv",
        [
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
            "--epoch-metrics-uri",
            "gs://b/p/epoch_metrics.json",
            "--checkpoint-path",
            str(checkpoint),
            "--checkpoint-uri",
            "gs://b/p/best.ckpt",
        ],
    )
    evidence.main()
    assert order == ["gs://b/p/epoch_metrics.json", "checkpoint", "gs://b/p/evidence.json"]
    uploader = uploaded["phase_timings"]["uploader"]
    assert set(uploader) == {"epoch_curve_upload", "checkpoint_hash_upload"}
    assert all(seconds >= 0 for seconds in uploader.values())
    assert "evidence_upload" not in uploader
