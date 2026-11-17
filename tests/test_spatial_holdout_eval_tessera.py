"""Spatial holdout evaluation of the TESSERA arm: comparisons on patch-set intersections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _run_eval(world: dict, **overrides: Any) -> dict:
    paths = {**world["paths"], **overrides}
    return ev.evaluate(
        world["assignment"],
        matrix={"L4": _load(paths["matrix_l4"]), "TESSERA": _load(paths["matrix_t"])},
        holdout={"L4": _load(paths["holdout_l4"]), "TESSERA": _load(paths["holdout_t"])},
        n_bootstrap=50,
    )


def _held_patch(world: dict) -> str:
    held = sorted(world["assignment"].holdout_anchors)
    return f"scene2:{cell_id(*held[0])}"


def test_g_tessera_raises_when_either_side_loses_a_patch(world: dict) -> None:
    for path in world["paths"]["holdout_t"]:
        _without(path, {_held_patch(world)})
    with pytest.raises(ev.SpatialHoldoutEvalError, match=r"G TESSERA.*side B lost 1") as err:
        _run_eval(world)
    assert _held_patch(world) in str(err.value)


def test_tessera_minus_l4_raises_when_the_tessera_side_loses_a_patch(world: dict) -> None:
    for path in world["paths"]["holdout_l4"]:
        _without(path, {_held_patch(world)})
    with pytest.raises(ev.SpatialHoldoutEvalError, match=r"TESSERA side lost 1 patch") as err:
        _run_eval(world)
    assert _held_patch(world) in str(err.value)


def test_l4_may_lose_only_patches_the_tessera_export_lacks_and_the_scope_is_checked(
    world: dict,
) -> None:
    held = world["assignment"].holdout_anchors
    dropped = sorted(p for p in world["lost"] if ev.anchor_from_patch_id(p) in held)
    scope = frozenset(world["lost"])
    paths = world["paths"]
    result = ev.evaluate(
        world["assignment"],
        matrix={"L4": _load(paths["matrix_l4"]), "TESSERA": _load(paths["matrix_t"])},
        holdout={"L4": _load(paths["holdout_l4"]), "TESSERA": _load(paths["holdout_t"])},
        n_bootstrap=50,
        tessera_invalid=scope,
    )
    est = result["tessera"]["TESSERA_minus_L4_holdout"][ev.UNSEEN]
    assert est["intersection"]["b"]["patches_lost"] == len(dropped)
    assert est["intersection"]["scope_verified"] is True
    assert not any("unverified" in note for note in result["notes"])
    # without the scope file the loss is reported as unverified
    unverified = _run_eval(world)
    assert any("tessera_invalid" in n and "unverified" in n for n in unverified["notes"])
    assert (
        unverified["tessera"]["TESSERA_minus_L4_holdout"][ev.UNSEEN]["intersection"][
            "scope_verified"
        ]
        is False
    )
    # a scope that records fewer invalid patches than L4 lost is refused
    with pytest.raises(ev.SpatialHoldoutEvalError, match="tessera_invalid"):
        ev.evaluate(
            world["assignment"],
            matrix={"L4": _load(paths["matrix_l4"]), "TESSERA": _load(paths["matrix_t"])},
            holdout={"L4": _load(paths["holdout_l4"]), "TESSERA": _load(paths["holdout_t"])},
            n_bootstrap=50,
            tessera_invalid=frozenset(sorted(scope)[1:]),
        )


def test_tessera_scope_file_lists_the_invalid_validation_patches(tmp_path: Path) -> None:
    scope = tmp_path / "data_scope.json"
    scope.write_text(
        json.dumps(
            {
                "skipped_refs": {
                    "train": [{"patch_id": "a", "reason": "tessera_invalid"}],
                    "validation": [
                        {"patch_id": "b", "reason": "tessera_invalid"},
                        {"patch_id": "c", "reason": "no_landsat"},
                    ],
                }
            }
        )
    )
    assert ev.read_tessera_invalid(scope) == frozenset({"b"})
    # The uploaded evidence record keeps the same list under ``data_scope``.
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"run_label": "x", "data_scope": json.loads(scope.read_text())}))
    assert ev.read_tessera_invalid(evidence) == frozenset({"b"})
    scope.write_text("{}")
    with pytest.raises(ev.SpatialHoldoutEvalError, match="skipped_refs"):
        ev.read_tessera_invalid(scope)
