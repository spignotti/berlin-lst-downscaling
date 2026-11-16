"""Spatial holdout evaluation: strata, scene-equal MAE, G and K on synthetic exports."""

from __future__ import annotations

import importlib.util
import json
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


# ------------------------------------------------------------- all nine arms

GOLDEN = Path(__file__).parent / "fixtures" / "spatial_holdout_stage1_golden.json"


def _assert_close(actual: object, expected: object, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, value in enumerate(expected):
            _assert_close(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), path
    else:
        assert actual == expected, path


def test_stage1_output_is_unchanged_by_the_generic_evaluation(
    exports: dict[str, list[Path]], assignment: sh.SpatialHoldoutAssignment
) -> None:
    """L1 and L4 (matrix seeds 0-2, holdout seed 0) keep every figure of the two-arm version.

    The golden file was written by the evaluation before it became generic over arms;
    keys added since (``pairings``, ``notes``) are not part of it.
    """
    load = ev.load_runs
    result = ev.evaluate(
        assignment,
        matrix={"L1": load(exports["matrix_l1"]), "L4": load(exports["matrix_l4"])},
        holdout={"L1": load(exports["holdout_l1"]), "L4": load(exports["holdout_l4"])},
        n_bootstrap=400,
    )
    golden = json.loads(GOLDEN.read_text())
    assert {"mae", "unseen_minus_seen", "L4_minus_L1", "G", "K", "stage2"} <= set(golden)
    current = json.loads(json.dumps(result))
    assert set(current) - set(golden) == {"pairings", "notes"}
    _assert_close({key: current[key] for key in golden}, golden)


# Planted level of each arm's MAE (K), the G shift of its holdout model on the unseen
# anchors, and the per-seed term 0.01 * seed that every arm shares. The seed SD of one
# arm is then 0.01, so criterion (c) asks for |difference| >= 0.02.
LEVEL = {
    "L1": 0.0,
    "L2": -0.03,
    "L3": -0.06,
    "L4": -0.07,
    "L5": -0.02,
    "LOO-M": 0.0,
    "LOO-S": -0.045,
    "NO-NDVI-NDWI": -0.07,
    "NO-RED-NIR-IDX": -0.055,
}
SHIFT = {arm: 0.5 + 0.05 * i for i, arm in enumerate(LEVEL)}


@pytest.fixture
def nine_arms(
    tmp_path: Path, assignment: sh.SpatialHoldoutAssignment
) -> tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]]:
    def runs(tag: str, arm: str, shift: float) -> list[ev.Run]:
        paths = [
            _export(
                tmp_path / f"{tag}-{arm}-s{s}.parquet",
                assignment,
                arm=arm,
                seed=s,
                offset=LEVEL[arm],
                unseen_shift=shift,
            )
            for s in (0, 1, 2)
        ]
        return ev.load_runs(paths)

    matrix = {arm: runs("matrix", arm, 0.0) for arm in LEVEL}
    holdout = {arm: runs("holdout", arm, SHIFT[arm]) for arm in LEVEL}
    return matrix, holdout


def _pairing(result: dict, model: str, name: str) -> dict:
    (found,) = [p for p in result["pairings"][model] if p["name"] == name]
    return found


def test_g_and_the_matrix_families_cover_all_nine_arms(
    nine_arms: tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]],
    assignment: sh.SpatialHoldoutAssignment,
) -> None:
    matrix, holdout = nine_arms
    result = ev.evaluate(assignment, matrix=matrix, holdout=holdout, n_bootstrap=200)
    assert set(result["G"]) == set(LEVEL) and result["notes"] == []
    assert set(result["mae"]) == {f"{m}/{a}" for m in ("matrix", "holdout") for a in LEVEL}
    for arm in LEVEL:
        for stratum in (ev.UNSEEN, ev.UNSEEN_INTERIOR, ev.UNSEEN_EDGE):
            assert result["G"][arm][stratum]["point"] == pytest.approx(SHIFT[arm])
            assert result["G"][arm][stratum]["seeds"] == [0, 1, 2]
    # the stage-2 trigger stays the pre-registered one: G of L1 and L4, and K
    assert set(result["stage2"]["checks"]) == {"G_L1", "G_L4", "K"}
    expected = [f"{a}-{b}" for _, pairs in ev.PAIRING_FAMILIES for a, b in pairs]
    assert len(expected) == 10
    holdout_names = [p["name"] for p in result["pairings"]["holdout"]]
    matrix_names = [p["name"] for p in result["pairings"]["matrix"]]
    assert holdout_names == expected
    assert matrix_names == [name for name in expected if name != "L4-L1"]  # K: holdout only
    assert {p["family"] for p in result["pairings"]["holdout"]} == {"A", "B", "K"}
    for model, shifts in (("matrix", {a: 0.0 for a in LEVEL}), ("holdout", SHIFT)):
        for pairing in result["pairings"][model]:
            a, b = pairing["a"], pairing["b"]
            assert set(pairing["strata"]) == set(ev.STRATA)
            unseen = pairing["strata"][ev.UNSEEN]
            assert unseen["point"] == pytest.approx(LEVEL[a] - LEVEL[b] + shifts[a] - shifts[b])
            seen = pairing["strata"][ev.SEEN]
            assert seen["point"] == pytest.approx(LEVEL[a] - LEVEL[b])
    text = ev.render_markdown(result)
    assert "## Paired differences A - B, holdout models" in text and "## Skipped" not in text


@pytest.mark.parametrize(
    ("name", "a", "b", "c"),
    [
        ("L2-L1", True, True, True),  # -0.03: beyond 2 x 0.01
        ("L4-L3", True, True, False),  # -0.01: consistent but within 2 seed SD
        ("NO-NDVI-NDWI-L4", False, False, False),  # no difference at all
        ("LOO-S-L4", True, True, True),
    ],
)
def test_rule_criteria_per_pairing(
    nine_arms: tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]],
    assignment: sh.SpatialHoldoutAssignment,
    name: str,
    a: bool,
    b: bool,
    c: bool,
) -> None:
    matrix, _ = nine_arms
    result = ev.evaluate(assignment, matrix=matrix, n_bootstrap=200)
    criteria = _pairing(result, "matrix", name)["strata"][ev.SEEN]["criteria"]
    assert criteria["available"] is True
    assert criteria["a_interval_excludes_zero"] is a
    assert criteria["b_same_sign_all_seeds"] is b
    assert criteria["c_abs_difference_at_least_2_seed_sd"] is c
    assert criteria["all_met"] is (a and b and c)
    assert criteria["larger_seed_sd"] == pytest.approx(0.01)


def test_criteria_come_from_the_matrix_bootstrap_on_the_stratum(
    nine_arms: tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]],
    assignment: sh.SpatialHoldoutAssignment,
) -> None:
    matrix, _ = nine_arms
    for stratum in ev.STRATA:
        direct = sb.compare_arms(
            ev._stratum_table(matrix["L2"], assignment, stratum),
            ev._stratum_table(matrix["L1"], assignment, stratum),
            n_bootstrap=200,
        )
        ours = ev.paired_difference(
            matrix["L2"], matrix["L1"], assignment, stratum, label="x", n_bootstrap=200
        )
        assert (ours.point, ours.ci_low, ours.ci_high) == pytest.approx(
            (direct.point_estimate, direct.ci_low, direct.ci_high), abs=1e-12
        )
        assert ours.n_scenes == len(direct.scenes)
    criteria = ev.seed_rule_criteria(
        matrix["L2"], matrix["L1"], assignment, ev.UNSEEN_EDGE, n_bootstrap=200
    )
    assert criteria["seed_differences"] == pytest.approx([-0.03, -0.03, -0.03])


def test_missing_arms_and_seeds_are_skipped_with_a_note_not_an_error(
    exports: dict[str, list[Path]],
    nine_arms: tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]],
    assignment: sh.SpatialHoldoutAssignment,
) -> None:
    matrix, holdout = nine_arms
    load = ev.load_runs
    # stage-1 shape: two arms, one holdout seed each
    result = ev.evaluate(
        assignment,
        matrix={"L1": load(exports["matrix_l1"]), "L4": load(exports["matrix_l4"])},
        holdout={"L1": load(exports["holdout_l1"]), "L4": load(exports["holdout_l4"])},
        n_bootstrap=100,
    )
    assert [p["name"] for p in result["pairings"]["holdout"]] == ["L4-L1"]
    assert result["pairings"]["matrix"] == []
    assert "holdout L2-L1 (A): skipped, no export of L2" in result["notes"]
    # one holdout seed: the pairing exists, the criteria need seeds 0, 1, 2
    criteria = result["pairings"]["holdout"][0]["strata"][ev.UNSEEN]["criteria"]
    assert criteria["available"] is False and "seeds [0, 1, 2]" in criteria["reason"]
    text = ev.render_markdown(result)
    assert "## Skipped" in text and "n/a" in text
    # a holdout arm without a matrix export, a matrix arm without a holdout export
    result = ev.evaluate(
        assignment,
        matrix={arm: matrix[arm] for arm in ("L1", "L2")},
        holdout={arm: holdout[arm] for arm in ("L1", "L3")},
        n_bootstrap=100,
    )
    assert set(result["G"]) == {"L1"}
    assert "G L3: skipped, no matrix export" in result["notes"]
    assert "G L2: skipped, no holdout export" in result["notes"]
    # seeds that differ between the two sides of a pairing
    result = ev.evaluate(
        assignment, matrix={"L1": matrix["L1"], "L2": matrix["L2"][:2]}, n_bootstrap=100
    )
    assert any(
        note.startswith("matrix L2-L1 (A): skipped, seeds differ") for note in result["notes"]
    )


def test_cli_evaluates_whatever_arms_have_exports(
    nine_arms: tuple[dict[str, list[ev.Run]], dict[str, list[ev.Run]]],
    assignment: sh.SpatialHoldoutAssignment,
    exports: dict[str, list[Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "analyze_spatial_holdout", root / "scripts" / "analyze_spatial_holdout.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_assignment", lambda path: assignment)
    prefix = tmp_path / "out" / "all"
    argv = [
        "--matrix-export",
        *map(str, exports["matrix_l1"] + exports["matrix_l4"]),
        "--holdout-export",
        *map(str, exports["holdout_l1"] + exports["holdout_l4"]),
        "--n-bootstrap",
        "50",
        "--output-prefix",
        str(prefix),
    ]
    assert cli.main(argv) == 0
    result = json.loads((tmp_path / "out" / "all.json").read_text())
    assert set(result["G"]) == {"L1", "L4"}
    assert [p["name"] for p in result["pairings"]["holdout"]] == ["L4-L1"]
    assert any("skipped" in note for note in result["notes"])
    assert "## Skipped" in (tmp_path / "out" / "all.md").read_text()
