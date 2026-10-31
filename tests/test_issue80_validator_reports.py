"""Issue #80: validator report aggregation resolution and sample30 strata."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from berlin_lst_downscaling.analysis.training_aggregation import SceneMeta, select_train_scenes
from berlin_lst_downscaling.modeling import ssim_range
from berlin_lst_downscaling.modeling.aggregation import resolve_report_aggregation
from berlin_lst_downscaling.modeling.baseline import build_report as build_baseline_report
from berlin_lst_downscaling.modeling.patches import RealSourceConfig
from berlin_lst_downscaling.modeling.random_forest import (
    ForestSpec,
    TrainPool,
)
from berlin_lst_downscaling.modeling.random_forest import (
    build_report as build_rf_report,
)

_STUB_FINGERPRINTS = {
    "index_policy_hash": "idx",
    "source_policy_hash": "src",
    "patches_total": "0",
    "published_at": "2020-01-01T00:00:00Z",
}


@pytest.fixture(autouse=True)
def _stub_patch_index_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    write_ssim_artifact,  # noqa: ANN001
) -> None:
    artifact = write_ssim_artifact(
        tmp_path_factory.mktemp("ssim_stub") / "v1.yaml", fingerprints=_STUB_FINGERPRINTS
    )
    monkeypatch.setattr(ssim_range, "SSIM_DATA_RANGE_ARTIFACT", artifact)

    def stub(_root: str) -> dict[str, str]:
        return dict(_STUB_FINGERPRINTS)

    monkeypatch.setattr(
        "berlin_lst_downscaling.modeling.baseline.patch_index_fingerprints",
        stub,
    )
    monkeypatch.setattr(
        "berlin_lst_downscaling.modeling.random_forest.patch_index_fingerprints",
        stub,
    )


def _t4_source_cfg() -> RealSourceConfig:
    return RealSourceConfig(
        patch_index_root="gs://example/patch-index",
        training_root="gs://example/training/v1",
        features_root="gs://example/features",
        ard_root="gs://example/ard",
        aggregation="t4",
    )


def test_baseline_report_builder_stores_aggregation_under_source() -> None:
    payload = build_baseline_report(
        cfg=_t4_source_cfg(),
        records=[],
        summaries={},
        exclusions={},
        git_revision="abc",
        ssim_range=ssim_range.load_ssim_data_range(),
    ).to_payload()
    assert "aggregation" not in payload
    assert payload["source"]["aggregation"] == "t4"
    assert resolve_report_aggregation(payload) == "t4"


def test_rf_report_builder_stores_aggregation_under_source() -> None:
    pool = TrainPool(
        keys=[],
        features=np.zeros((0, 28), dtype=np.float32),
        prior_100m=np.zeros(0, dtype=np.float32),
        residual_k=np.zeros(0, dtype=np.float32),
        candidate_cells=0,
        duplicate_cells=0,
        requested_patches=0,
        read_patches=0,
        exclusions={},
        anchor_stride_cells=1,
        duplicate_anchors=0,
        overlapping_windows=0,
        seconds=0.0,
    )
    payload = build_rf_report(
        cfg=_t4_source_cfg(),
        spec=ForestSpec(),
        scope={"profile": "smoke"},
        pool=pool,
        max_cells=1,
        seed=0,
        rows_file="rows.parquet",
        rows_sha256="0" * 64,
        stage_results={},
        probes={},
        resolved_config="aggregation: t4",
        runtime_seconds=1.0,
        cache_bytes=0,
        git_revision="abc",
        ssim_range=ssim_range.load_ssim_data_range(),
    ).to_payload()
    assert resolve_report_aggregation(payload) == "t4"


def test_resolve_report_aggregation_defaults_to_arithmetic_without_fields() -> None:
    historical = {"source": {"patch_index_root": "x", "training_root": "y"}}
    assert resolve_report_aggregation(historical) == "arithmetic"


def test_resolve_report_aggregation_rejects_conflicting_top_level_and_source() -> None:
    report = {"aggregation": "arithmetic", "source": {"aggregation": "t4"}}
    with pytest.raises(ValueError, match="mismatch"):
        resolve_report_aggregation(report)


def test_rf_vs_naive_comparison_rejects_mismatched_methods() -> None:
    rf_payload = build_rf_report(
        cfg=_t4_source_cfg(),
        spec=ForestSpec(),
        scope={},
        pool=TrainPool(
            keys=[],
            features=np.zeros((0, 28), dtype=np.float32),
            prior_100m=np.zeros(0, dtype=np.float32),
            residual_k=np.zeros(0, dtype=np.float32),
            candidate_cells=0,
            duplicate_cells=0,
            requested_patches=0,
            read_patches=0,
            exclusions={},
            anchor_stride_cells=1,
            duplicate_anchors=0,
            overlapping_windows=0,
            seconds=0.0,
        ),
        max_cells=1,
        seed=0,
        rows_file="rows.parquet",
        rows_sha256="0" * 64,
        stage_results={},
        probes={},
        resolved_config="",
        runtime_seconds=0.0,
        cache_bytes=0,
        git_revision="abc",
        ssim_range=ssim_range.load_ssim_data_range(),
    ).to_payload()
    naive_payload = build_baseline_report(
        cfg=RealSourceConfig(
            patch_index_root="gs://example/patch-index",
            training_root="gs://example/training/v1",
            features_root="gs://example/features",
            ard_root="gs://example/ard",
            aggregation="arithmetic",
        ),
        records=[],
        summaries={},
        exclusions={},
        git_revision="abc",
        ssim_range=ssim_range.load_ssim_data_range(),
    ).to_payload()
    assert resolve_report_aggregation(rf_payload) == "t4"
    assert resolve_report_aggregation(naive_payload) == "arithmetic"
    assert resolve_report_aggregation(rf_payload) != resolve_report_aggregation(naive_payload)


def test_sample30_gt_30_strata_uses_seeded_draw_and_includes_later_years() -> None:
    scenes: list[SceneMeta] = []
    for index in range(36):
        year = 2000 + index // 12
        month = (index % 12) + 1
        scenes.append(SceneMeta(f"S{year}{month:02d}", "train", year, date(year, month, 15)))
    ids_a, policy_a = select_train_scenes(scenes, scope="sample30", seed=0)
    ids_b, policy_b = select_train_scenes(scenes, scope="sample30", seed=0)
    assert ids_a == ids_b
    assert policy_a == policy_b
    assert policy_a["rule"] == "seeded_sample_of_30_strata_one_scene_each_when_gt_30_strata"
    assert len(policy_a["selected_strata"]) == 30
    assert len(ids_a) == 30
    late_strata = {(2002, month) for month in range(7, 13)}
    chosen = {(s["year"], s["month"]) for s in policy_a["selected_strata"]}
    assert chosen & late_strata
