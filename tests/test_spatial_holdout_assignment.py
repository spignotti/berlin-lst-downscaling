"""Spatial holdout assignment: determinism, bounds, block and edge rules, config guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling import spatial_holdout as sh
from berlin_lst_downscaling.modeling.guards import contract_invariants, guard_modeling_config

_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO / "configs" / "modeling"
_COMMITTED = sh.ASSIGNMENT_DIR / "v1.json"
_RUN_OVERRIDES = [
    "seed=0",
    "trainer.max_epochs=40",
    "trainer.lr_schedule.name=cosine",
    "evaluate_test=false",
]


def _lattice(rows: int, cols: int, *, train: int = 3, validation: int = 1) -> dict:
    return {
        (r * 16, c * 16): sh.AnchorCounts(train=train, validation=validation)
        for r in range(rows)
        for c in range(cols)
    }


# ------------------------------------------------------------------ committed file


def test_committed_assignment_is_reproduced_from_its_own_anchor_table() -> None:
    committed = sh.load_assignment(_COMMITTED)
    rebuilt = sh.build_assignment(
        committed.anchors,
        seed=committed.seed,
        neighbourhood=committed.neighbourhood,
        patch_index=committed.patch_index,
    )
    assert sh.assignment_bytes(rebuilt) == _COMMITTED.read_bytes()
    again = sh.build_assignment(
        committed.anchors, seed=committed.seed, patch_index=committed.patch_index
    )
    assert sh.assignment_bytes(again) == sh.assignment_bytes(rebuilt)


def test_committed_assignment_holds_18_to_22_percent_in_whole_complete_blocks() -> None:
    a = sh.load_assignment(_COMMITTED)
    assert len(a.anchors) == 263
    assert a.seed == 0 and a.neighbourhood == 8
    assert sh.MIN_FRACTION <= a.fraction <= sh.MAX_FRACTION
    assert len(a.holdout) == 9 * len(a.blocks)
    assert set(a.blocks) <= set(a.eligible_blocks)
    for block in a.blocks:
        members = [h for h in a.holdout if h.block == block]
        assert len(members) == 9
        assert {(h.row // 16 % 3, h.col // 16 % 3) for h in members} == {
            (i, j) for i in range(3) for j in range(3)
        }
    assert {h.position for h in a.holdout} == {sh.POSITION_EDGE, sh.POSITION_INTERIOR}
    assert a.patch_index["root"].endswith("/training/patch-index/v3")


def test_a_tampered_file_is_refused(tmp_path: Path) -> None:
    payload = json.loads(_COMMITTED.read_text())
    payload["holdout"][0]["position"] = (
        "interior" if payload["holdout"][0]["position"] == "edge" else "edge"
    )
    with pytest.raises(sh.SpatialHoldoutError, match="does not match"):
        sh.parse_assignment(payload)
    payload = json.loads(_COMMITTED.read_text())
    held = payload["holdout"][0]
    payload["anchors"] = [
        row for row in payload["anchors"] if row[:2] != [held["row"], held["col"]]
    ]  # the block is no longer complete, so the rule cannot have drawn it
    path = tmp_path / "v9.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(sh.SpatialHoldoutError, match="does not match"):
        sh.load_assignment(path)


# ------------------------------------------------------------------------- rule


def test_draw_is_deterministic_and_seed_dependent() -> None:
    anchors = _lattice(9, 10)  # 9 complete blocks of 90 anchors; 2 blocks = 20 %
    first = sh.build_assignment(anchors, seed=0)
    assert sh.build_assignment(anchors, seed=0).to_payload() == first.to_payload()
    assert len(first.blocks) == 2 and first.fraction == pytest.approx(0.2)
    draws = {sh.build_assignment(anchors, seed=s).blocks for s in range(8)}
    assert len(draws) > 1


def test_only_complete_blocks_are_eligible() -> None:
    anchors = _lattice(9, 10)
    del anchors[(0, 0)]  # block (0, 0) loses one anchor
    assert (0, 0) not in sh.complete_blocks(anchors)
    assert len(sh.complete_blocks(anchors)) == 8
    for seed in range(8):
        assert (0, 0) not in sh.build_assignment(anchors, seed=seed).blocks
    # column 9 belongs to the incomplete blocks (r, 3) and is never held out
    assert all(col < 9 * 16 for _row, col in sh.build_assignment(anchors, seed=0).holdout_anchors)


def test_bounds_fail_closed() -> None:
    with pytest.raises(sh.SpatialHoldoutError, match="exceeds"):
        sh.build_assignment(_lattice(9, 9), seed=0)  # 2 of 9 blocks = 22.2 %
    sparse = _lattice(3, 3)  # one complete block of 100 anchors = 9 %
    sparse.update({(48, c * 16): sh.AnchorCounts(1, 1) for c in range(91)})
    with pytest.raises(sh.SpatialHoldoutError, match="cannot reach"):
        sh.build_assignment(sparse, seed=0)


def test_edge_and_interior_labels() -> None:
    anchors = _lattice(9, 9)
    block = [(r * 16, c * 16) for r in range(3, 6) for c in range(3, 6)]
    labels = sh.label_positions(block, anchors)
    assert labels[(64, 64)] == sh.POSITION_INTERIOR
    assert sum(v == sh.POSITION_EDGE for v in labels.values()) == 8

    # two blocks side by side: the facing middle anchors have no training neighbour
    pair = block + [(r * 16, c * 16) for r in range(3, 6) for c in range(6, 9)]
    labels = sh.label_positions(pair, _lattice(9, 12))
    assert labels[(64, 80)] == labels[(64, 96)] == sh.POSITION_INTERIOR
    assert labels[(48, 80)] == sh.POSITION_EDGE

    # a neighbour without train patches is not a training anchor
    no_train = {a: sh.AnchorCounts(train=0, validation=1) for a in anchors}
    assert set(sh.label_positions(block, no_train).values()) == {sh.POSITION_INTERIOR}

    # corner-only contact counts for the 8-neighbourhood, not for the 4-neighbourhood
    plus = {(16, 16), (0, 16), (32, 16), (16, 0), (16, 32)}
    table = {a: sh.AnchorCounts(1, 1) for a in plus | {(0, 0)}}
    assert sh.label_positions(plus, table, neighbourhood=8)[(16, 16)] == sh.POSITION_EDGE
    assert sh.label_positions(plus, table, neighbourhood=4)[(16, 16)] == sh.POSITION_INTERIOR


def test_anchor_counts_keep_anchors_of_other_splits() -> None:
    rows = [
        {"row": 0, "col": 0, "split": "train"},
        {"row": 0, "col": 0, "split": "validation"},
        {"row": 0, "col": 16, "split": "test"},
    ]
    counts = sh.anchor_counts(rows)
    assert counts[(0, 0)] == sh.AnchorCounts(train=1, validation=1)
    assert counts[(0, 16)] == sh.AnchorCounts(train=0, validation=0)
    with pytest.raises(sh.SpatialHoldoutError, match="lattice"):
        sh.build_assignment({(8, 0): sh.AnchorCounts(1, 1)})


# ----------------------------------------------------------------------- configs


def _compose(name: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name=name, overrides=overrides)


@pytest.mark.parametrize("arm", ["l1", "l4"])
def test_holdout_config_equals_the_locked_arm_except_for_the_option(arm: str) -> None:
    name = f"v6_{arm}_spatial_holdout"
    holdout = _compose(name, _RUN_OVERRIDES)
    guard_modeling_config(holdout, name)
    contract_invariants(holdout)
    locked = _compose(f"v6_{arm}_locked", _RUN_OVERRIDES)
    a = OmegaConf.to_container(holdout, resolve=True)
    b = OmegaConf.to_container(locked, resolve=True)
    assert isinstance(a, dict) and isinstance(b, dict)
    differing = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    assert differing == {"output_root", "spatial_holdout"}
    assert a["spatial_holdout"] == {"enabled": True, "assignment": "v1"}
    assert a["ablation_arm"] == sh.SPATIAL_HOLDOUT_CONFIG_ARMS[name] == arm.upper()


def test_option_is_absent_from_every_other_config() -> None:
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if path.stem in sh.SPATIAL_HOLDOUT_CONFIG_ARMS:
            continue
        assert "spatial_holdout" not in path.read_text(), path.name
    locked = _compose("v6_l4_locked", _RUN_OVERRIDES)
    assert "spatial_holdout" not in locked
    assert sh.spatial_holdout_from_config(locked) is None


def test_guard_rejects_the_option_elsewhere_and_off_contract_holdout_configs() -> None:
    on = ["+spatial_holdout.enabled=true", "+spatial_holdout.assignment=v1"]
    with pytest.raises(ValueError, match="only admitted on"):
        guard_modeling_config(_compose("v6_l4_locked", [*_RUN_OVERRIDES, *on]), "v6_l4_locked")
    name = "v6_l1_spatial_holdout"
    for overrides, match in (
        (["evaluate_test=true"], "test 2025 stays sealed"),
        (["spatial_holdout.enabled=false"], "enabled must be true"),
        (["spatial_holdout.assignment=v999"], "does not load"),
        (["+data.patch_cache.source=prebuilt"], "job_local"),
        (["data.batch_size=8"], "v6 ablation lock is off-contract"),
    ):
        with pytest.raises(ValueError, match=match):
            guard_modeling_config(_compose(name, [*_RUN_OVERRIDES, *overrides]), name)
    with pytest.raises(ValueError, match="unknown spatial_holdout key"):
        sh.spatial_holdout_from_config(_compose(name, ["+spatial_holdout.buffer=1"]))
