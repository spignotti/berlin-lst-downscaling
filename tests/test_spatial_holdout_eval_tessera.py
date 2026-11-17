"""Spatial holdout evaluation of the TESSERA arm: comparisons on patch-set intersections."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from test_scene_bootstrap import write_export
from test_spatial_holdout_eval import (  # noqa: F401  (``assignment`` is a fixture)
    _export,
    assignment,
)

from berlin_lst_downscaling.analysis import spatial_holdout_eval as ev
from berlin_lst_downscaling.data.training.contracts import cell_id
from berlin_lst_downscaling.modeling import spatial_holdout as sh

SEEDS = (0, 1, 2)
TESSERA_OFFSET = -0.1
UNSEEN_L4 = 0.4
UNSEEN_TESSERA = 0.3


def _without(path: Path, lost: set[str]) -> Path:
    """Rewrite an export without the patches ``lost`` (the TESSERA arm drops invalid windows)."""
    table = pq.read_table(path)
    keep = pc.invert(pc.is_in(table.column("patch_id"), pa.array(sorted(lost))))
    return write_export(path, table.filter(keep))


@pytest.fixture
def world(tmp_path: Path, assignment: sh.SpatialHoldoutAssignment) -> dict:  # noqa: F811
    every = sorted(assignment.anchors)
    held = sorted(assignment.holdout_anchors)
    seen = [a for a in every if a not in assignment.holdout_anchors]
    # scene5 holds only edge anchors (all lost: the scene disappears from the TESSERA arm), plus
    # one unseen patch of scene0 and one seen patch of scene1.
    lost = {f"scene5:{cell_id(*a)}" for a in held} | {
        f"scene0:{cell_id(*held[0])}",
        f"scene1:{cell_id(*seen[0])}",
    }

    def many(tag: str, arm: str, **kw: float) -> list[Path]:
        return [
            _export(tmp_path / f"{tag}-{s}.parquet", assignment, arm=arm, seed=s, **kw)
            for s in SEEDS
        ]

    runs = {
        "matrix_l4": many("m-l4", "L4"),
        "holdout_l4": many("h-l4", "L4", unseen_shift=UNSEEN_L4),
        "matrix_t": many("m-t", "tessera_arm", offset=TESSERA_OFFSET),
        "holdout_t": many(
            "h-t", "tessera_arm_spatial_holdout", offset=TESSERA_OFFSET, unseen_shift=UNSEEN_TESSERA
        ),
    }
    existing = {
        pid
        for path in runs["matrix_t"]
        for pid in pq.read_table(path).column("patch_id").to_pylist()
    }
    lost &= existing
    for key in ("matrix_t", "holdout_t"):
        for path in runs[key]:
            _without(path, lost)
    return {"paths": runs, "lost": lost, "assignment": assignment}


def _load(paths: list[Path]) -> list[ev.Run]:
    return ev.load_runs(paths)


def test_tessera_labels_map_to_the_arm() -> None:
    assert {"tessera", "tessera_arm", "tessera_arm_spatial_holdout", "TESSERA"} == set(
        ev.TESSERA_ARM_LABELS
    )


def test_comparisons_run_on_the_intersection_and_report_what_each_side_lost(world: dict) -> None:
    paths = world["paths"]
    result = ev.evaluate(
        world["assignment"],
        matrix={"L4": _load(paths["matrix_l4"]), "TESSERA": _load(paths["matrix_t"])},
        holdout={"L4": _load(paths["holdout_l4"]), "TESSERA": _load(paths["holdout_t"])},
        n_bootstrap=200,
    )
    section = result["tessera"]
    lost_unseen = {
        p
        for p in world["lost"]
        if ev.anchor_from_patch_id(p) in world["assignment"].holdout_anchors
    }
    assert lost_unseen and any(p.startswith("scene5:") for p in lost_unseen)

    # G_TESSERA: both sides miss the same patches, so neither loses any in the intersection.
    g = section["G_TESSERA"][ev.UNSEEN]
    assert g["point"] == pytest.approx(UNSEEN_TESSERA)
    assert g["intersection"]["a"]["patches_lost"] == g["intersection"]["b"]["patches_lost"] == 0
    assert g["criteria"]["available"] is True and g["seeds"] == [0, 1, 2]

    # Holdout TESSERA - L4 on the unseen anchors: L4 loses the patches TESSERA lacks.
    holdout = section["TESSERA_minus_L4_holdout"][ev.UNSEEN]
    assert holdout["point"] == pytest.approx(TESSERA_OFFSET + UNSEEN_TESSERA - UNSEEN_L4)
    side = holdout["intersection"]
    assert side["a"]["patches_lost"] == 0 and side["a"]["scenes_lost"] == 0
    assert side["b"]["patches_lost"] == len(lost_unseen)
    assert side["b"]["scenes_lost"] == 1  # scene5 holds only unseen anchors
    assert holdout["n_patches"] == side["patches"] == side["b"]["patches"] - len(lost_unseen)
    assert holdout["criteria"]["available"] is True

    # Temporal models on all validation patches.
    matrix = section["TESSERA_minus_L4_matrix"][ev.ALL]
    assert matrix["point"] == pytest.approx(TESSERA_OFFSET)
    assert matrix["intersection"]["b"]["patches_lost"] == len(world["lost"])
    assert matrix["criteria"]["available"] is True
    text = ev.render_markdown(result)
    assert "## TESSERA, on patch-set intersections" in text and "G_TESSERA" in text
    ev.write_outputs(result, world["paths"]["matrix_l4"][0].parent / "out")


def test_other_comparisons_keep_the_strict_identical_patch_set_check(world: dict) -> None:
    paths = world["paths"]
    short = _without(paths["holdout_l4"][0], {next(iter(world["lost"]))})
    holdout = _load([short, *paths["holdout_l4"][1:]])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="different patch set"):
        ev.evaluate(
            world["assignment"],
            matrix={"L4": _load(paths["matrix_l4"])},
            holdout={"L4": holdout},
            n_bootstrap=50,
        )


def test_without_both_sides_a_quantity_is_skipped_with_a_note(world: dict) -> None:
    paths = world["paths"]
    result = ev.evaluate(
        world["assignment"],
        matrix={"L4": _load(paths["matrix_l4"]), "TESSERA": _load(paths["matrix_t"])},
        n_bootstrap=50,
    )
    assert set(result["tessera"]) == {"rule", "TESSERA_minus_L4_matrix"}
    assert any(note.startswith("G_TESSERA: skipped") for note in result["notes"])
    assert "G" not in result  # the matrix-only output keeps its shape
