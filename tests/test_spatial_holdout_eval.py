"""Spatial holdout evaluation: strata, scene-equal MAE, G and K on synthetic exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from test_scene_bootstrap import SCHEMA, build_arm, effect_fn, make_world, write_export

from berlin_lst_downscaling.analysis import scene_bootstrap as sb
from berlin_lst_downscaling.analysis import spatial_holdout_eval as ev
from berlin_lst_downscaling.data.training.contracts import cell_id
from berlin_lst_downscaling.modeling import spatial_holdout as sh

CELLS = 100
# Per-stratum patch MAE of the synthetic matrix model, and the planted effects.
MATRIX_MAE = {ev.SEEN: 1.0, ev.UNSEEN_EDGE: 2.0, ev.UNSEEN_INTERIOR: 3.0}
G_TRUE = 0.5
K_TRUE = -0.2


@pytest.fixture(scope="module")
def assignment() -> sh.SpatialHoldoutAssignment:
    # 9 x 10 anchors: nine complete blocks of 90 anchors, two drawn = 20 %.
    anchors = {(r * 16, c * 16): sh.AnchorCounts(3, 1) for r in range(9) for c in range(10)}
    # Two side-by-side blocks would give interior anchors only by luck, so pick a
    # seed whose draw has both positions.
    for seed in range(50):
        drawn = sh.build_assignment(anchors, seed=seed)
        if {h.position for h in drawn.holdout} == {sh.POSITION_EDGE, sh.POSITION_INTERIOR}:
            return drawn
    raise AssertionError("no seed draws both positions")


def _stratum(anchor: sh.Anchor, assignment: sh.SpatialHoldoutAssignment) -> str:
    return ev.strata_of(anchor, assignment)[-1]


def _scene_anchors(assignment: sh.SpatialHoldoutAssignment) -> dict[str, list[sh.Anchor]]:
    """Six scenes: four cover every anchor, one only seen anchors, one only edge anchors."""
    every = sorted(assignment.anchors)
    seen = [a for a in every if _stratum(a, assignment) == ev.SEEN]
    edge = [a for a in every if _stratum(a, assignment) == ev.UNSEEN_EDGE]
    scenes = {f"scene{i}": every for i in range(4)}
    scenes["scene4"] = seen[:7]
    scenes["scene5"] = edge[:3]
    return scenes


def _export(
    path: Path,
    assignment: sh.SpatialHoldoutAssignment,
    *,
    arm: str,
    seed: int,
    unseen_shift: float = 0.0,
    offset: float = 0.0,
    drop: str | None = None,
) -> Path:
    cols: dict[str, list] = {name: [] for name in SCHEMA.names}
    for scene_index, (scene, anchors) in enumerate(_scene_anchors(assignment).items()):
        for row, col in anchors:
            patch_id = f"{scene}:{cell_id(row, col)}"
            if patch_id == drop:
                continue
            stratum = _stratum((row, col), assignment)
            mae = MATRIX_MAE[stratum] + 0.1 * scene_index + 0.01 * seed + offset
            if stratum != ev.SEEN:
                mae += unseen_shift
            values = {
                "schema_version": 2,
                "arm": arm,
                "seed": seed,
                "run_id": f"{path.stem}",
                "checkpoint": "best",
                "patch_id": patch_id,
                "scene_id": scene,
                "scene_date": None,
                "split": "validation",
                "abs_error_sum": mae * CELLS,
                "valid_cells": CELLS,
                "ssim_sum": 5.0,
                "ssim_windows": 10,
                "filled_feature_pixels": None,
            }
            for key, value in values.items():
                cols[key].append(value)
    return write_export(path, pa.table(cols, schema=SCHEMA))


@pytest.fixture
def exports(tmp_path: Path, assignment: sh.SpatialHoldoutAssignment) -> dict[str, list[Path]]:
    def many(tag: str, arm: str, seeds: tuple[int, ...], **kw: float) -> list[Path]:
        return [
            _export(tmp_path / f"{tag}-{arm}-s{s}.parquet", assignment, arm=arm, seed=s, **kw)
            for s in seeds
        ]

    return {
        "matrix_l1": many("matrix", "L1", (0, 1, 2)),
        "matrix_l4": many("matrix", "L4", (0, 1, 2), offset=-0.05),
        "holdout_l1": many("holdout", "L1", (0,), unseen_shift=G_TRUE),
        "holdout_l4": many(
            "holdout", "L4", (0,), unseen_shift=G_TRUE + K_TRUE + 0.05, offset=-0.05
        ),
    }


def test_anchor_round_trips_through_the_patch_id() -> None:
    for anchor in ((0, 0), (144, 96), (336, 416)):
        assert ev.anchor_from_patch_id(f"LC08_X_20240601:{cell_id(*anchor)}") == anchor
    with pytest.raises(ev.SpatialHoldoutEvalError, match="cell id"):
        ev.anchor_from_patch_id("LC08_X_20240601")
    with pytest.raises(ev.SpatialHoldoutEvalError, match="off the canonical"):
        ev.anchor_from_patch_id("s:E369240N5838410")


def test_bootstrap_is_the_matrix_bootstrap() -> None:
    world = make_world()
    effects = {i: 0.3 * np.sin(i) for i in range(20)}
    result = sb.compare_arms(
        build_arm("A", world, effect_fn(effects)), build_arm("B", world, effect_fn(effects))
    )
    diffs = [scene.diff for scene in result.scenes]
    assert ev.scene_bootstrap_mean(diffs) == (result.point_estimate, result.ci_low, result.ci_high)
    assert ev.scene_bootstrap_mean(diffs, n_bootstrap=500, rng_seed=3) != ev.scene_bootstrap_mean(
        diffs, n_bootstrap=500, rng_seed=4
    )
    with pytest.raises(ev.SpatialHoldoutEvalError):
        ev.scene_bootstrap_mean([])


def test_strata_mae_is_scene_equal_and_counts_its_support(
    exports: dict[str, list[Path]], assignment: sh.SpatialHoldoutAssignment
) -> None:
    runs = ev.load_runs(exports["matrix_l1"])
    seen = ev.stratum_mae(runs, assignment, ev.SEEN, label="m")
    edge = ev.stratum_mae(runs, assignment, ev.UNSEEN_EDGE, label="m")
    interior = ev.stratum_mae(runs, assignment, ev.UNSEEN_INTERIOR, label="m")
    unseen = ev.stratum_mae(runs, assignment, ev.UNSEEN, label="m")
    # scenes 0-4 have seen patches, 0-3 and 5 edge patches, 0-3 interior patches
    assert (seen.n_scenes, edge.n_scenes, interior.n_scenes, unseen.n_scenes) == (5, 5, 4, 5)
    assert seen.seeds == (0, 1, 2)
    # scene MAE = stratum level + 0.1 * scene index + mean seed term 0.01
    assert seen.point == pytest.approx(1.0 + 0.1 * np.mean([0, 1, 2, 3, 4]) + 0.01)
    assert edge.point == pytest.approx(2.0 + 0.1 * np.mean([0, 1, 2, 3, 5]) + 0.01)
    assert interior.point == pytest.approx(3.0 + 0.1 * np.mean([0, 1, 2, 3]) + 0.01)
    assert seen.ci_low <= seen.point <= seen.ci_high
    n_interior = sum(h.position == sh.POSITION_INTERIOR for h in assignment.holdout)
    assert interior.n_patches == 4 * n_interior
    assert unseen.n_patches == edge.n_patches + interior.n_patches
    gap = ev.stratum_gap(runs, assignment, label="m")
    assert gap.n_scenes == 4  # scenes 4 and 5 have one side only
    assert gap.point > 0


def test_g_and_k_recover_the_planted_effects(
    exports: dict[str, list[Path]], assignment: sh.SpatialHoldoutAssignment
) -> None:
    load = ev.load_runs
    result = ev.evaluate(
        assignment,
        matrix={"L1": load(exports["matrix_l1"]), "L4": load(exports["matrix_l4"])},
        holdout={"L1": load(exports["holdout_l1"]), "L4": load(exports["holdout_l4"])},
        n_bootstrap=400,
    )
    for stratum in (ev.UNSEEN, ev.UNSEEN_INTERIOR, ev.UNSEEN_EDGE):
        # G pairs holdout seed 0 with matrix seed 0 only, so the seed term cancels.
        assert result["G"]["L1"][stratum]["point"] == pytest.approx(G_TRUE)
        assert result["G"]["L1"][stratum]["seeds"] == [0]
        assert result["G"]["L4"][stratum]["point"] == pytest.approx(G_TRUE + K_TRUE + 0.05)
        assert result["K"][stratum]["point"] == pytest.approx(K_TRUE)
    assert result["G"]["L1"][ev.UNSEEN]["n_scenes"] == 5
    assert result["G"]["L1"][ev.UNSEEN]["ci_95"] == pytest.approx([G_TRUE, G_TRUE])
    assert result["L4_minus_L1"]["matrix"][ev.SEEN]["point"] == pytest.approx(-0.05)
    assert result["mae"]["holdout/L1"][ev.SEEN]["point"] == pytest.approx(
        result["mae"]["matrix/L1"][ev.SEEN]["point"] - 0.01
    )
    assert result["stage2"]["triggered"] is True
    assert set(result["stage2"]["checks"]) == {"G_L1", "G_L4", "K"}
    text = ev.render_markdown(result)
    assert "## G: holdout model minus matrix model" in text and "Triggered: True." in text


def test_matrix_only_evaluation_has_no_g_or_k(
    exports: dict[str, list[Path]], assignment: sh.SpatialHoldoutAssignment, tmp_path: Path
) -> None:
    result = ev.evaluate(
        assignment, matrix={"L1": ev.load_runs(exports["matrix_l1"])}, n_bootstrap=200
    )
    assert "G" not in result and "K" not in result and "stage2" not in result
    assert set(result["mae"]) == {"matrix/L1"}
    json_path, md_path = ev.write_outputs(result, tmp_path / "out" / "matrix")
    assert json_path.is_file() and md_path.read_text().startswith("# Spatial holdout evaluation")


def test_stage2_trigger_follows_the_rule() -> None:
    def est(point: float, low: float, high: float) -> ev.Estimate:
        return ev.Estimate("d", "x", ev.UNSEEN, (0,), 3, 3, point, low, high, ())

    assert ev._stage2_check({"G_L1": est(0.01, -0.02, 0.02)})["triggered"] is False
    assert ev._stage2_check({"G_L1": est(0.01, 0.001, 0.02)})["triggered"] is True
    assert ev._stage2_check({"K": est(-0.03, -0.08, 0.02)})["triggered"] is True


def test_mismatched_universes_raise(
    exports: dict[str, list[Path]], assignment: sh.SpatialHoldoutAssignment, tmp_path: Path
) -> None:
    matrix = ev.load_runs(exports["matrix_l1"])
    holdout = ev.load_runs(exports["holdout_l1"])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="seeds differ"):
        ev.paired_difference(holdout, matrix, assignment, ev.UNSEEN, label="G")
    held = sorted(assignment.holdout_anchors)[0]
    short = _export(
        tmp_path / "short.parquet",
        assignment,
        arm="L1",
        seed=0,
        drop=f"scene0:{cell_id(*held)}",
    )
    with pytest.raises(ev.SpatialHoldoutEvalError, match="different patches"):
        ev.paired_difference(ev.load_runs([short]), matrix[:1], assignment, ev.UNSEEN, label="G")
    with pytest.raises(ev.SpatialHoldoutEvalError, match="different patch set"):
        ev.stratum_mae([*ev.load_runs([short]), matrix[1]], assignment, ev.UNSEEN, label="m")
    other = sh.build_assignment(
        {(r * 16, c * 16): sh.AnchorCounts(1, 1) for r in range(3) for c in range(5)},
        min_fraction=0.5,
        max_fraction=0.7,
    )
    with pytest.raises(ev.SpatialHoldoutEvalError, match="different indexes"):
        ev.stratum_mae(matrix, other, ev.SEEN, label="m")
    with pytest.raises(ev.SpatialHoldoutEvalError, match="no matrix run for seed"):
        ev.evaluate(assignment, matrix={"L1": matrix[1:]}, holdout={"L1": holdout}, n_bootstrap=50)
