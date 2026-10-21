"""Issue #80: aggregation analysis helpers and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from berlin_lst_downscaling.analysis.training_aggregation import (
    BlockRecord,
    SceneMeta,
    block_aggregation_pair,
    largest_remainder_allocation,
    run_synthetic_analysis,
    select_train_scenes,
    summarize_records,
    write_outputs,
)


def test_largest_remainder_allocation_replays_seed_zero() -> None:
    weights = [1.0, 1.0, 1.0]
    assert largest_remainder_allocation(weights, 30) == [10, 10, 10]


def test_block_aggregation_pair_sigma_and_ratio_unavailable_for_single_cell() -> None:
    stats = block_aggregation_pair([300.0])
    assert stats["approx_ratio"] is None
    assert stats["signed_diff_k"] == pytest.approx(0.0, abs=1e-6)


def test_select_train_scenes_sample30_policy_persisted() -> None:
    scenes = [
        SceneMeta("A", "train", 2017, __import__("datetime").date(2017, 1, 1)),
        SceneMeta("B", "train", 2018, __import__("datetime").date(2018, 2, 1)),
    ]
    ids, policy = select_train_scenes(scenes, scope="sample30", seed=0)
    assert policy["seed"] == 0
    assert "scene_ids" in policy
    assert ids


def test_synthetic_analysis_writes_outputs(tmp_path: Path) -> None:
    result = run_synthetic_analysis(train_scope="full", runtime_seconds=1.0)
    paths = write_outputs(result, tmp_path / "out")
    payload = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert payload["train_scope"] == "full"
    assert payload["summaries"]


def test_scene_and_block_weighted_summaries_differ_with_unequal_blocks() -> None:
    records = [
        BlockRecord(
            scene_id="A",
            split="train",
            domain="primary_native_valid",
            block_id="A:0",
            support_count=2,
            arithmetic_k=300.0,
            t4_k=301.0,
            signed_diff_k=1.0,
            sigma_k=1.0,
            reference_t_k=300.0,
            approx_diff_k=0.0,
            approx_ratio=None,
            strict_patch_block=False,
        ),
        BlockRecord(
            scene_id="A",
            split="train",
            domain="primary_native_valid",
            block_id="A:1",
            support_count=2,
            arithmetic_k=300.0,
            t4_k=303.0,
            signed_diff_k=3.0,
            sigma_k=1.0,
            reference_t_k=300.0,
            approx_diff_k=0.0,
            approx_ratio=None,
            strict_patch_block=False,
        ),
        BlockRecord(
            scene_id="B",
            split="train",
            domain="primary_native_valid",
            block_id="B:0",
            support_count=2,
            arithmetic_k=300.0,
            t4_k=305.0,
            signed_diff_k=5.0,
            sigma_k=1.0,
            reference_t_k=300.0,
            approx_diff_k=0.0,
            approx_ratio=None,
            strict_patch_block=False,
        ),
    ]
    block = summarize_records(
        records, split="train", domain="primary_native_valid", weighting="block_weighted"
    )
    scene = summarize_records(
        records, split="train", domain="primary_native_valid", weighting="scene_weighted"
    )
    assert block.signed_mean_diff_k != scene.signed_mean_diff_k
