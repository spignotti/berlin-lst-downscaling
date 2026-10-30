"""Baseline and RF report structure and the shared evaluation clamp (offline)."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from berlin_lst_downscaling.modeling import random_forest as rf
from berlin_lst_downscaling.modeling.baseline import (
    PatchRecord,
    SplitSummary,
    _score_patch,
    record_payload,
    splits_payload,
)
from berlin_lst_downscaling.modeling.evaluation_exports import REPORT_SSIM_KEYS, report_metrics
from berlin_lst_downscaling.modeling.ssim_range import test_override_ssim_data_range


def _record(patch_id: str, split: str, **kw: object) -> PatchRecord:
    base: dict[str, object] = {
        "patch_id": patch_id,
        "split": split,
        "valid_cells": 10,
        "abs_error_sum": 20.0,
        "ssim_sum": 3.0,
        "ssim_windows": 4,
        "filled_feature_pixels": 0,
        "scene_id": "S",
        "ssim_100m_L250": 0.5,
        "ssim_100m_Lpatch": 0.25,
        "ssim_windows_Lpatch": 4,
        "ssim_Lpatch_zero_range_patches": 1,
    }
    base.update(kw)
    return PatchRecord(**base)  # type: ignore[arg-type]


def _summary(records: list[PatchRecord]) -> SplitSummary:
    return rf.summarize_records(records, requested=len(records), exclusions={})


def test_splits_payload_has_report_keys_and_pooled_values() -> None:
    val = [
        _record("a", "validation"),
        _record(
            "b",
            "validation",
            ssim_sum=1.0,
            ssim_windows=4,
            ssim_100m_L250=None,
            ssim_100m_Lpatch=None,
            ssim_windows_Lpatch=0,
            ssim_Lpatch_zero_range_patches=2,
        ),
    ]
    test = [_record("c", "test")]
    payload = splits_payload({"validation": _summary(val), "test": _summary(test)}, [*val, *test])
    assert set(payload) == {"validation", "test"}
    for split in payload.values():
        assert set(REPORT_SSIM_KEYS) <= set(split)
    v = payload["validation"]
    assert v["ssim_100m"] == pytest.approx(4.0 / 8)
    assert v["ssim_100m_L250"] == pytest.approx(0.5 * 4 / 8)
    assert v["ssim_100m_Lpatch"] == pytest.approx(0.25)
    assert v["ssim_windows_Lpatch"] == 4
    assert v["ssim_Lpatch_zero_range_patches"] == 3
    assert v["mae"] == pytest.approx(2.0)
    assert payload["test"]["ssim_100m"] == pytest.approx(0.75)
    assert payload["test"]["ssim_windows_Lpatch"] == 4


def test_baseline_ssim_keys_match_report_metrics_keys() -> None:
    records = [_record("a", "validation"), _record("b", "validation")]
    payload = splits_payload({"validation": _summary(records)}, records)["validation"]
    rows = [dict(r.__dict__) for r in records]
    expected = [k for k in report_metrics(rows) if k in REPORT_SSIM_KEYS]
    assert [k for k in payload if k in REPORT_SSIM_KEYS] == expected
    assert set(expected) == set(REPORT_SSIM_KEYS)
    for key in REPORT_SSIM_KEYS:
        assert payload[key] == pytest.approx(report_metrics(rows)[key])


def test_record_payload_adds_ssim_100m() -> None:
    assert record_payload(_record("a", "test"))["ssim_100m"] == pytest.approx(0.75)
    assert record_payload(_record("a", "test", ssim_windows=0, ssim_sum=0.0))["ssim_100m"] is None


def _fake_sample() -> SimpleNamespace:
    meta = SimpleNamespace(
        patch_id="p1",
        split="test",
        filled_feature_pixels=0,
        scene_id="S1",
        scene_date="2025-06-01",
    )
    return SimpleNamespace(
        meta=meta,
        lst_prior_k=np.full((1, 160, 160), 500.0, np.float32),
        target_100m=np.full((1, 16, 16), 300.0, np.float32),
        mask_100m=np.ones((1, 16, 16), bool),
    )


def test_baseline_score_patch_clamps_prediction() -> None:
    record = _score_patch(_fake_sample(), aggregation="arithmetic", data_range_k=40.0)  # type: ignore[arg-type]
    assert record.valid_cells == 256
    assert record.abs_error_sum == pytest.approx(100.0 * 256)
    assert record.scene_id == "S1"


def test_rf_score_stage_clamps_prediction(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rf, "predict_10m", lambda *a, **k: np.zeros((160, 160), np.float32))
    monkeypatch.setattr(
        rf,
        "reconstruct_10m_prediction",
        lambda *a, **k: np.full((1, 1, 160, 160), 500.0, np.float32),
    )
    feature = tmp_path / "f.npy"
    np.save(feature, np.zeros((1, 160, 160), np.float32))
    cache = rf.EvalSplit(
        split="validation",
        requested_patches=1,
        patch_ids=["a"],
        feature_files=[str(feature)],
        prior_100m=np.full((1, 16, 16), 300.0, np.float32),
        target_100m=np.full((1, 16, 16), 300.0, np.float32),
        mask_100m=np.ones((1, 16, 16), bool),
        filled_feature_pixels=[0],
        exclusions=Counter(),  # type: ignore[arg-type]
        seconds=0.0,
        scene_ids=["s1"],
        scene_dates=[None],
    )
    stage = rf.StageSelection(stage=1, names=(), indices=(), feature_order="x")
    records = rf.score_stage(
        cache,
        None,  # type: ignore[arg-type]
        stage,
        ssim_range=test_override_ssim_data_range(23.0),
    )
    assert records[0].valid_cells == 256
    assert records[0].abs_error_sum == pytest.approx(100.0 * 256)
