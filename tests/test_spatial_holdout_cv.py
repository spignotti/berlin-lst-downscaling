"""Spatial holdout cross-validation ``cv5``: partition rule, files, guard, launcher, evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from test_launch_vertex_v6 import _PATH, _argv
from test_scene_bootstrap import SCHEMA, write_export

from berlin_lst_downscaling.analysis import spatial_holdout_eval as ev
from berlin_lst_downscaling.data.training.contracts import cell_id
from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling import spatial_holdout as sh
from berlin_lst_downscaling.modeling.channels import V6_ABLATION_ARMS
from berlin_lst_downscaling.modeling.guards import assert_spatial_holdout, guard_modeling_config
from berlin_lst_downscaling.modeling.patches import RealSourceConfig

_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO / "configs" / "modeling"
_V1 = sh.ASSIGNMENT_DIR / "v1.json"
FOLDS = sh.SPATIAL_HOLDOUT_CV_ASSIGNMENTS
_RUN_OVERRIDES = [
    "seed=0",
    "trainer.max_epochs=40",
    "trainer.lr_schedule.name=cosine",
    "evaluate_test=false",
]
_CV_CONFIGS = ("v6_l1_spatial_holdout", "v6_l4_spatial_holdout")
_OTHER_CONFIGS = sorted(set(sh.SPATIAL_HOLDOUT_CONFIG_ARMS) - set(_CV_CONFIGS))
# (blocks, anchors, train patches, validation patches) per fold: the owner's expected result
_EXPECTED = [(9, 49, 1100, 199), (9, 54, 1107, 201), (9, 54, 1100, 204), (9, 54, 1111, 213)] + [
    (9, 52, 1105, 230)
]


def _load_folds() -> list[sh.SpatialHoldoutAssignment]:
    return [sh.load_assignment(sh.resolve_assignment_path(name)) for name in FOLDS]


def _builder() -> ModuleType:
    path = _REPO / "scripts" / "operators" / "build_spatial_holdout.py"
    spec = importlib.util.spec_from_file_location("build_spatial_holdout_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The Python launcher, with the Vertex SDK stubbed when absent; nothing is sent."""
    if importlib.util.find_spec("google.cloud.aiplatform_v1") is None:
        stub = MagicMock()
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1", stub)
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1.types", stub.types)
    spec = importlib.util.spec_from_file_location("launch_vertex_modeling_cv_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _no_client(region: str) -> None:
        raise AssertionError("the Vertex client must not be created in these tests")

    monkeypatch.setattr(module, "_evidence_exists", lambda uri: False)
    monkeypatch.setattr(module, "_client", _no_client)
    return module


# ------------------------------------------------------------- the rule and the files


def test_partition_rule_on_a_small_table() -> None:
    # Six blocks of one anchor: train 5, 5, 3, 3, 2, 1. Sorted by (-train, row, col) they go to the
    # fold with the fewest train patches so far, ties to the lowest fold; the last one joins fold 4.
    table = {
        (0, 0): 5,
        (0, 48): 5,
        (0, 96): 3,
        (48, 0): 3,
        (48, 48): 2,
        (48, 96): 1,
    }
    anchors = {a: sh.AnchorCounts(train=n, validation=1) for a, n in table.items()}
    assert sh.partition_blocks(anchors) == (
        ((0, 0),),
        ((0, 1),),
        ((0, 2),),
        ((1, 0),),
        ((1, 1), (1, 2)),
    )
    # equal loads: the blocks are placed in (row, col) order, the lowest fold first
    even = {a: sh.AnchorCounts(train=4, validation=1) for a in table}
    assert [len(b) for b in sh.partition_blocks(even)] == [2, 1, 1, 1, 1]
    assert sh.partition_blocks(even)[0] == ((0, 0), (1, 2))
    with pytest.raises(sh.SpatialHoldoutError, match="cannot fill"):
        sh.partition_blocks({a: anchors[a] for a in list(anchors)[:4]})


def test_incomplete_blocks_take_part_in_the_partition() -> None:
    anchors = {(0, 0): sh.AnchorCounts(2, 1), (0, 16): sh.AnchorCounts(2, 1)}  # block (0, 0)
    anchors.update({(0, 48 * k): sh.AnchorCounts(1, 1) for k in range(1, 5)})
    folds = sh.build_fold_assignments(anchors)
    assert sum(len(f.holdout) for f in folds) == len(anchors)
    assert {len(f.holdout) for f in folds} == {1, 2}
    assert all(
        f.eligible_blocks == tuple(sorted({sh.block_of(*a) for a in anchors})) for f in folds
    )


def test_folds_are_disjoint_cover_all_anchors_and_match_the_expected_counts() -> None:
    folds = _load_folds()
    v1 = sh.load_assignment(_V1)
    every = [a for fold in folds for a in fold.holdout_anchors]
    assert len(every) == len(set(every)) == len(v1.anchors) == 263
    assert set(every) == set(v1.anchors)
    assert all(f.anchors == v1.anchors and f.patch_index == v1.patch_index for f in folds)
    got = [
        (
            len(f.blocks),
            len(f.holdout),
            sum(f.anchors[a].train for a in f.holdout_anchors),
            sum(f.anchors[a].validation for a in f.holdout_anchors),
        )
        for f in folds
    ]
    assert got == _EXPECTED
    assert sum(len(f.blocks) for f in folds) == len(folds[0].eligible_blocks) == 45
    assert [f.fold for f in folds] == [0, 1, 2, 3, 4]
    assert all(f.seed is None and f.min_fraction is None for f in folds)
    assert all(a.block == sh.block_of(a.row, a.col) for f in folds for a in f.holdout)
    # an anchor with validation patches is unseen in exactly one fold
    assert all(
        sum(a in f.holdout_anchors for f in folds) == 1
        for a, c in v1.anchors.items()
        if c.validation > 0
    )
    # positions: the same 8-neighbour rule, a neighbour in another fold is a training anchor
    for fold in folds:
        assert {a.position for a in fold.holdout} == {sh.POSITION_EDGE, sh.POSITION_INTERIOR}
        assert fold.positions() == sh.label_positions(fold.holdout_anchors, fold.anchors)


def test_folds_are_reproduced_bit_for_bit_by_the_builder(tmp_path: Path) -> None:
    v1 = sh.load_assignment(_V1)
    rebuilt = sh.build_fold_assignments(v1.anchors, patch_index=v1.patch_index)
    for name, fold in zip(FOLDS, rebuilt, strict=True):
        assert sh.assignment_bytes(fold) == sh.resolve_assignment_path(name).read_bytes()
    builder = _builder()
    argv = ["--partition", "cv5", "--anchors-from", str(_V1)]
    assert builder.main([*argv, "--output-dir", str(tmp_path)]) == 0
    for name in FOLDS:
        assert (tmp_path / f"{name}.json").read_bytes() == sh.resolve_assignment_path(
            name
        ).read_bytes()
    assert builder.main([*argv, "--check-dir", str(sh.ASSIGNMENT_DIR)]) == 0
    assert builder.main([*argv, "--check-dir", str(tmp_path)]) == 0
    (tmp_path / f"{FOLDS[0]}.json").write_text("{}")
    assert builder.main([*argv, "--check-dir", str(tmp_path)]) == 1
    with pytest.raises(SystemExit, match="never overwritten"):
        builder.main([*argv, "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="need --partition cv5"):
        builder.main(["--output-dir", str(tmp_path)])


def test_v1_is_unchanged_and_not_a_fold() -> None:
    v1 = sh.load_assignment(_V1)
    assert v1.fold is None and v1.seed == 0 and len(v1.holdout) == 54
    assert "partition" not in v1.to_payload()["rule"]
    assert sh.assignment_bytes(v1) == _V1.read_bytes()


def test_a_fold_file_that_disagrees_with_its_rule_is_refused(tmp_path: Path) -> None:
    good = json.loads(sh.resolve_assignment_path("cv5-fold2").read_text())

    def refused(payload: dict, match: str = "does not match|unknown partition") -> None:
        with pytest.raises(sh.SpatialHoldoutError, match=match):
            sh.parse_assignment(payload)

    moved = json.loads(json.dumps(good))
    moved["holdout"] = moved["holdout"][1:]  # one anchor dropped from the fold
    refused(moved)
    other_fold = json.loads(json.dumps(good))
    other_fold["rule"]["fold"] = 3  # the content of fold 2 under the name of fold 3
    refused(other_fold)
    for key, value in (("fold", 5), ("fold", True), ("partition", "cv4")):
        bad = json.loads(json.dumps(good))
        bad["rule"][key] = value
        refused(bad, "unknown partition")
    counts = json.loads(json.dumps(good))
    first = counts["holdout"][0]
    row = next(r for r in counts["anchors"] if r[:2] == [first["row"], first["col"]])
    row[2] += (
        1000  # the block of a fold-2 anchor becomes the heaviest: the table yields another fold
    )
    refused(counts)
    relabelled = json.loads(json.dumps(good))
    relabelled["holdout"][0]["position"] = (
        "interior" if relabelled["holdout"][0]["position"] == "edge" else "edge"
    )
    refused(relabelled)
    path = tmp_path / "cv5-fold2.json"
    path.write_text(json.dumps(moved))
    with pytest.raises(sh.SpatialHoldoutError, match="does not match"):
        sh.load_assignment(path)


def test_a_fold_drawn_from_another_patch_index_is_refused_at_run_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="v6_l4_spatial_holdout",
            overrides=[*_RUN_OVERRIDES, "spatial_holdout.assignment=cv5-fold3"],
        )
    GlobalHydra.instance().clear()
    source = RealSourceConfig("idx", "trn", "feat", "ard")
    committed = sh.load_assignment(sh.resolve_assignment_path("cv5-fold3")).patch_index
    monkeypatch.setattr(run_module, "patch_index_fingerprints", lambda root: dict(committed))
    resolved = run_module.resolve_spatial_holdout(cfg, source)
    assert resolved is not None
    assignment, info = resolved
    assert info["assignment"] == "cv5-fold3" and info["holdout_anchors"] == 54
    assert len(assignment.holdout) == 54 and assignment.fold == 3 and info["seed"] is None
    assert info["assignment_sha256"] == sh.assignment_sha256(
        sh.resolve_assignment_path("cv5-fold3")
    )
    monkeypatch.setattr(
        run_module, "patch_index_fingerprints", lambda root: {**committed, "patches_total": "1"}
    )
    with pytest.raises(RuntimeError, match="another patch index"):
        run_module.resolve_spatial_holdout(cfg, source)


# ---------------------------------------------------------------------------- guard


def _compose(name: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("name", _CV_CONFIGS)
def test_guard_admits_the_folds_on_l1_and_l4_at_seed_zero(name: str, fold: str) -> None:
    on = [*_RUN_OVERRIDES, f"spatial_holdout.assignment={fold}"]
    guard_modeling_config(_compose(name, on), name)
    for overrides, match in (
        (["seed=1"], "seed 0 only"),
        (["evaluate_test=true"], "test 2025 stays sealed"),
        (["+data.patch_cache.source=prebuilt"], "job_local"),
        (["spatial_holdout.enabled=false"], "enabled must be true"),
    ):
        with pytest.raises(ValueError, match=match):
            guard_modeling_config(_compose(name, [*on, *overrides]), name)
    # v1 stays admitted at any seed, as before
    guard_modeling_config(_compose(name, [*_RUN_OVERRIDES, "seed=2"]), name)


@pytest.mark.parametrize("name", _OTHER_CONFIGS)
def test_guard_refuses_the_folds_on_every_other_arm(name: str) -> None:
    assert len(_OTHER_CONFIGS) == 7
    for fold in FOLDS:
        cfg = _compose(name, [*_RUN_OVERRIDES, f"spatial_holdout.assignment={fold}"])
        with pytest.raises(ValueError, match="only 'v1' is admitted"):
            guard_modeling_config(cfg, name)


def test_guard_refuses_unknown_assignments_and_the_tessera_arm() -> None:
    for name in _CV_CONFIGS:
        for assignment in ("cv5-fold5", "cv5-fold-1", "cv5", "cv5-fold0.json", "v2"):
            cfg = _compose(name, [*_RUN_OVERRIDES, f"spatial_holdout.assignment={assignment}"])
            with pytest.raises(ValueError, match="only 'v1' is admitted"):
                guard_modeling_config(cfg, name)
    on = ["+spatial_holdout.enabled=true", "+spatial_holdout.assignment=cv5-fold0"]
    with pytest.raises(ValueError, match="only admitted on"):
        assert_spatial_holdout(_compose("tessera_arm", on), "tessera_arm")
    with pytest.raises(ValueError, match="only admitted on"):
        assert_spatial_holdout(_compose("v6_l1_locked", [*_RUN_OVERRIDES, *on]), "v6_l1_locked")


def test_the_arm_list_is_l1_and_l4() -> None:
    arms = {V6_ABLATION_ARMS[a].config_name for a in sh.SPATIAL_HOLDOUT_CV_ARMS}
    assert {sh.SPATIAL_HOLDOUT_CONFIG_ARMS[c] for c in _CV_CONFIGS} == set(
        sh.SPATIAL_HOLDOUT_CV_ARMS
    )
    assert arms == {"v6_l1_locked", "v6_l4_locked"}
    matrix = yaml.safe_load((_REPO / "configs" / "ablation" / "v6_matrix.yaml").read_text())
    assert set(sh.SPATIAL_HOLDOUT_CV_ARMS) <= set(matrix["arms"])


# -------------------------------------------------------------------------- launcher


def _launch(mode: str, *extra: str) -> list[str]:
    return _argv(
        mode, "--seed", "0", "--max-epochs", "40", "--lr-schedule", "cosine", "--skip-test", *extra
    )


def _submitted(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
    captured: dict[str, object] = {}

    def _fake_submit(client: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "projects/p/locations/r/customJobs/1"

    monkeypatch.setattr(launcher, "_client", lambda region: object())
    monkeypatch.setattr(launcher, "_submit", _fake_submit)
    monkeypatch.setattr(
        launcher,
        "_poll_until_terminal",
        lambda *a, **k: (launcher.JobState.JOB_STATE_SUCCEEDED, False),
    )
    argv = [a for a in argv if a != "--preflight"]
    monkeypatch.setattr(sys, "argv", argv)
    assert launcher.main() == 0
    return captured


@pytest.mark.parametrize("mode", ["v6-l1-spatial-holdout", "v6-l4-spatial-holdout"])
def test_a_fold_run_has_its_own_label_display_name_output_root_and_env(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = load_launcher(monkeypatch)
    seen: dict[str, dict] = {}
    for fold in ("v1", *FOLDS):
        extra = () if fold == "v1" else ("--assignment", fold)
        got = _submitted(launcher, monkeypatch, _launch(mode, *extra))
        seen[fold] = {"display_name": got["display_name"], **dict(got["env"])}  # type: ignore[arg-type]
    v1 = seen["v1"]
    assert v1["display_name"] == f"{mode}-unit-seed0-ep40-cos-notest"
    assert v1["VERTEX_RUN_LABEL"] == "unit-seed0-ep40-cos-notest"
    assert v1["VERTEX_OUTPUT_ROOT"] == f"data/runs/{mode}/unit-seed0-ep40-cos-notest/seed-0"
    assert v1["VERTEX_EVIDENCE_URI"].endswith("/unit-seed0-ep40-cos-notest/evidence.json")
    assert "VERTEX_SPATIAL_HOLDOUT_ASSIGNMENT" not in v1
    for fold in FOLDS:
        run = seen[fold]
        label = f"unit-{fold}-seed0-ep40-cos-notest"
        assert run["display_name"] == f"{mode}-{label}" and run["VERTEX_RUN_LABEL"] == label
        assert run["VERTEX_OUTPUT_ROOT"] == f"data/runs/{mode}/{label}/seed-0"
        assert run["VERTEX_EVIDENCE_URI"].endswith(f"/{label}/evidence.json")
        assert run["VERTEX_VALIDATION_EXPORT_URI"].endswith(
            f"/{label}/eval_validation_patches.parquet"
        )
        assert run["VERTEX_SPATIAL_HOLDOUT_ASSIGNMENT"] == fold
        assert run["VERTEX_SKIP_TEST"] == "1"
    assert len({run["display_name"] for run in seen.values()}) == 6
    assert len({run["VERTEX_OUTPUT_ROOT"] for run in seen.values()}) == 6
    assert len({run["VERTEX_EVIDENCE_URI"] for run in seen.values()}) == 6


def test_v1_output_is_identical_with_and_without_the_explicit_assignment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = load_launcher(monkeypatch)
    outputs = []
    for extra in ((), ("--assignment", "v1")):
        monkeypatch.setattr(sys, "argv", _launch("v6-l1-spatial-holdout", *extra))
        assert launcher.main() == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    assert "spatial_holdout.assignment" not in outputs[0] and "assignment     :" not in outputs[0]
    monkeypatch.setattr(sys, "argv", _launch("v6-l1-spatial-holdout", "--assignment", "cv5-fold0"))
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert "spatial_holdout.assignment=cv5-fold0" in out
    assert "unit-cv5-fold0-seed0-ep40-cos-notest" in out and "assignment     : cv5-fold0" in out


@pytest.mark.parametrize(
    "mode", sorted(m for m in sh.SPATIAL_HOLDOUT_CONFIG_ARMS if m.endswith("_spatial_holdout"))
)
def test_the_launcher_refuses_a_fold_for_every_arm_but_l1_and_l4(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = load_launcher(monkeypatch)
    dashed = mode.replace("_", "-")
    arm = sh.SPATIAL_HOLDOUT_CONFIG_ARMS[mode]
    monkeypatch.setattr(sys, "argv", _launch(dashed, "--assignment", "cv5-fold0"))
    if arm in sh.SPATIAL_HOLDOUT_CV_ARMS:
        assert launcher.main() == 0
    else:
        with pytest.raises(SystemExit, match="admitted for the arms"):
            launcher.main()


def test_the_launcher_refuses_a_fold_run_that_breaks_the_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = load_launcher(monkeypatch)
    mode, fold = "v6-l1-spatial-holdout", ("--assignment", "cv5-fold1")
    base = _launch(mode, *fold)
    without_skip = [a for a in base if a != "--skip-test"]
    cases = (
        (without_skip, "requires --skip-test"),
        ([*base, "--patch-cache", "prebuilt"], "builds its patch cache in the job"),
        ([*base, "--seed", "1"], "seed 0 only"),
        (_launch("v6-l1", "--assignment", "cv5-fold0"), "only accepted by the spatial holdout"),
        (_launch("v6-l4", "--assignment", "v1"), "only accepted by the spatial holdout"),
        (_launch("tessera", "--assignment", "cv5-fold0"), "only accepted by the spatial holdout"),
    )
    for argv, match in cases:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit, match=match):
            launcher.main()
    for unknown in ("cv5-fold5", "cv6-fold0", "fold0", "v2"):  # argparse refuses: exit code 2
        monkeypatch.setattr(sys, "argv", _launch(mode, "--assignment", unknown))
        with pytest.raises(SystemExit) as raised:
            launcher.main()
        assert raised.value.code == 2
        assert "invalid choice" in capsys.readouterr().err
    monkeypatch.setattr(launcher, "_evidence_exists", lambda uri: uri.endswith(".parquet"))
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="validation export already exists"):
        launcher.main()


# ------------------------------------------------------------------------ evaluation

CELLS = 100
BASE = {"L1": 1.0, "L4": 0.95}
SEEN_SHIFT = -0.02


def _shift(arm: str, fold: int) -> float:
    """What the fold model of ``arm`` adds to its error on the fold's own (unseen) anchors."""
    return 0.5 + 0.01 * fold if arm == "L1" else 0.45 + 0.02 * fold


@pytest.fixture(scope="module")
def folds() -> dict[str, sh.SpatialHoldoutAssignment]:
    # 6 x 9 anchors = six complete blocks: fold 0 holds two of them, folds 1-4 one each
    anchors = {(r * 16, c * 16): sh.AnchorCounts(3, 1) for r in range(6) for c in range(9)}
    built = sh.build_fold_assignments(anchors, patch_index={"root": "synthetic"})
    return dict(zip(FOLDS, built, strict=True))


def _fold_of(folds: dict[str, sh.SpatialHoldoutAssignment], anchor: sh.Anchor) -> int:
    (owner,) = (f.fold for f in folds.values() if anchor in f.holdout_anchors)
    assert owner is not None
    return owner


def _scenes(folds: dict[str, sh.SpatialHoldoutAssignment]) -> dict[str, list[sh.Anchor]]:
    every = sorted(folds[FOLDS[0]].anchors)
    return {**{f"scene{i}": every for i in range(4)}, "scene4": every[:20]}


def _sha(fold: sh.SpatialHoldoutAssignment) -> str:
    return hashlib.sha256(sh.assignment_bytes(fold)).hexdigest()


def _write_record(path: Path, name: str, sha256: str, *, evidence: bool = True) -> Path:
    """The run record next to an export: ``<stem>.evidence.json`` (evidence) or the raw summary."""
    block = {"assignment": name, "assignment_sha256": sha256, "holdout_anchors": 54}
    body = (
        {"full": {"summary": {"spatial_holdout": block}}}
        if evidence
        else {"spatial_holdout": block}
    )
    target = path.with_suffix(".evidence.json")
    target.write_text(json.dumps(body))
    return target


def _export(
    path: Path,
    folds: dict[str, sh.SpatialHoldoutAssignment],
    *,
    arm: str,
    fold: int | None = None,
    seed: int = 0,
    drop: str | None = None,
    recorded: str | None = None,
) -> Path:
    cols: dict[str, list] = {name: [] for name in SCHEMA.names}
    for index, (scene, anchors) in enumerate(_scenes(folds).items()):
        for row, col in anchors:
            patch_id = f"{scene}:{cell_id(row, col)}"
            if patch_id == drop:
                continue
            mae = BASE[arm] + 0.1 * index + 0.01 * seed
            if fold is not None:
                own = _fold_of(folds, (row, col)) == fold
                mae += _shift(arm, fold) if own else SEEN_SHIFT
            values = {
                "schema_version": 2,
                "arm": arm,
                "seed": seed,
                "run_id": path.stem,
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
    write_export(path, pa.table(cols, schema=SCHEMA))
    if fold is not None:  # the run's own record names the fold it trained without
        named = recorded or f"cv5-fold{fold}"
        _write_record(path, named, _sha(folds[named]))
    return path


def _runs(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment], *, skip: tuple[str, ...] = ()
) -> tuple[dict[str, list[ev.Run]], dict[str, dict[str, list[ev.Run]]]]:
    matrix = {
        arm: ev.load_runs(
            [
                _export(tmp_path / f"matrix-{arm}-s{s}.parquet", folds, arm=arm, seed=s)
                for s in (0, 1)
            ]
        )
        for arm in ("L1", "L4")
    }
    holdout: dict[str, dict[str, list[ev.Run]]] = {}
    for name, fold in folds.items():
        for arm in ("L1", "L4"):
            if f"{name}/{arm}" in skip:
                continue
            path = tmp_path / f"{name}-{arm}.parquet"
            _export(path, folds, arm=arm, fold=fold.fold)
            holdout.setdefault(name, {})[arm] = ev.load_fold_runs([path])
    return matrix, holdout


def _expected_oof(folds: dict[str, sh.SpatialHoldoutAssignment], arm: str) -> float:
    """Mean over scenes of the scene's mean error when every patch comes from its own-fold model."""
    values = []
    for index, anchors in enumerate(_scenes(folds).values()):
        values.append(
            np.mean([BASE[arm] + 0.1 * index + _shift(arm, _fold_of(folds, a)) for a in anchors])
        )
    return float(np.mean(values))


def _expected_matrix(folds: dict[str, sh.SpatialHoldoutAssignment], arm: str) -> float:
    return float(np.mean([BASE[arm] + 0.1 * i for i in range(5)]))


def test_cv_evaluation_recovers_the_planted_folds_and_out_of_fold_effects(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    result = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=300)
    assert result["kind"] == "spatial_holdout_cv" and result["missing"] == {}
    assert result["bootstrap"]["n_bootstrap"] == 300 and result["bootstrap"]["rng_seed"] == 0
    for k, name in enumerate(FOLDS):
        fold = result["folds"][name]
        assert fold["anchors"] == len(folds[name].holdout) and fold["evaluation"] is not None
        evaluation = fold["evaluation"]
        for stratum in (ev.UNSEEN, ev.UNSEEN_INTERIOR, ev.UNSEEN_EDGE):
            assert evaluation["G"]["L1"][stratum]["point"] == pytest.approx(_shift("L1", k))
            assert evaluation["G"]["L4"][stratum]["point"] == pytest.approx(_shift("L4", k))
            assert evaluation["K"][stratum]["point"] == pytest.approx(
                BASE["L4"] + _shift("L4", k) - BASE["L1"] - _shift("L1", k)
            )
        # the matrix side is the seed-0 run (the seed-1 export would add 0.01)
        assert evaluation["G"]["L1"][ev.UNSEEN]["seeds"] == [0]
        assert evaluation["mae"]["holdout/L1"][ev.SEEN]["point"] == pytest.approx(
            evaluation["mae"]["matrix/L1"][ev.SEEN]["point"] + SEEN_SHIFT
        )
    k_points = [BASE["L4"] + _shift("L4", k) - BASE["L1"] - _shift("L1", k) for k in range(5)]
    summary = result["fold_summary"]
    assert summary["K"]["n_folds"] == 5 and summary["K"]["folds"] == list(FOLDS)
    assert summary["K"]["mean"] == pytest.approx(np.mean(k_points))
    assert (summary["K"]["min"], summary["K"]["max"]) == pytest.approx(
        (min(k_points), max(k_points))
    )
    g_l1 = [_shift("L1", k) for k in range(5)]
    assert summary["G"]["L1"]["mean"] == pytest.approx(np.mean(g_l1))
    assert (summary["G"]["L1"]["min"], summary["G"]["L1"]["max"]) == pytest.approx((0.5, 0.54))

    oof = result["out_of_fold"]
    assert oof["complete"] is True and oof["patches"] == 54 * 4 + 20
    for arm in ("L1", "L4"):
        assert oof["mae"][arm]["point"] == pytest.approx(_expected_oof(folds, arm))
        assert oof["G_cv"][arm]["point"] == pytest.approx(
            _expected_oof(folds, arm) - _expected_matrix(folds, arm)
        )
        assert oof["G_cv"][arm]["n_scenes"] == 5
    assert oof["K_cv"]["point"] == pytest.approx(
        _expected_oof(folds, "L4") - _expected_oof(folds, "L1")
    )
    low, high = oof["K_cv"]["ci_95"]
    assert low <= oof["K_cv"]["point"] <= high
    text = ev.render_cv_markdown(result)
    assert "## Out of fold" in text and "K_cv = L4 - L1" in text and "# Fold cv5-fold4" in text
    json_path, md_path = ev.write_outputs(result, tmp_path / "out" / "cv5")
    assert json.loads(json_path.read_text())["kind"] == "spatial_holdout_cv"
    assert md_path.read_text() == text


def test_cv_out_of_fold_uses_the_model_that_did_not_train_on_the_patch(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    # A fold model that is bad on its unseen anchors and perfect elsewhere: if the summary used
    # any other fold's model for a patch, the out-of-fold error would fall below the planted one.
    matrix, holdout = _runs(tmp_path, folds)
    result = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=100)
    seen_only = BASE["L1"] + SEEN_SHIFT + np.mean([0.1 * i for i in range(5)])
    assert result["out_of_fold"]["mae"]["L1"]["point"] > seen_only + 0.4


def test_missing_folds_are_listed_and_the_out_of_fold_summary_is_incomplete(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    skip = ("cv5-fold3/L4", "cv5-fold1/L1", "cv5-fold1/L4")
    matrix, holdout = _runs(tmp_path, folds, skip=skip)
    result = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=100)
    assert result["missing"] == {"cv5-fold1": ["L1", "L4"], "cv5-fold3": ["L4"]}
    assert result["folds"]["cv5-fold1"]["evaluation"] is None
    assert result["folds"]["cv5-fold3"]["evaluation"] is not None  # L1 alone is still evaluated
    oof = result["out_of_fold"]
    assert oof["complete"] is False and set(oof) == {"complete", "reason"}
    assert "cv5-fold1 lacks L1, L4" in oof["reason"] and "cv5-fold3 lacks L4" in oof["reason"]
    # the spread names the folds it covers instead of pretending to cover five
    assert result["fold_summary"]["K"]["folds"] == ["cv5-fold0", "cv5-fold2", "cv5-fold4"]
    assert result["fold_summary"]["G"]["L1"]["n_folds"] == 4
    assert result["fold_summary"]["G"]["L4"]["n_folds"] == 3
    assert any("cv5-fold1" in note for note in result["notes"])
    text = ev.render_cv_markdown(result)
    assert "Incomplete, not computed" in text and "Missing:" in text
    nothing = ev.evaluate_cv(folds, matrix=matrix, holdout={}, n_bootstrap=100)
    assert nothing["fold_summary"] == {} and nothing["out_of_fold"]["complete"] is False


def test_cv_refuses_inconsistent_inputs(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    kwargs = {"n_bootstrap": 50}
    # an anchor with validation patches must be unseen in exactly one fold
    stolen = sorted(folds["cv5-fold2"].holdout_anchors)[0]
    twice = dataclasses.replace(
        folds["cv5-fold1"],
        holdout=(
            *folds["cv5-fold1"].holdout,
            next(h for h in folds["cv5-fold2"].holdout if (h.row, h.col) == stolen),
        ),
    )
    with pytest.raises(ev.SpatialHoldoutEvalError, match="exactly one fold"):
        ev.evaluate_cv({**folds, "cv5-fold1": twice}, matrix=matrix, holdout=holdout, **kwargs)
    never = dataclasses.replace(folds["cv5-fold2"], holdout=folds["cv5-fold2"].holdout[1:])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="exactly one fold"):
        ev.evaluate_cv({**folds, "cv5-fold2": never}, matrix=matrix, holdout=holdout, **kwargs)
    other_index = dataclasses.replace(folds["cv5-fold4"], patch_index={"root": "elsewhere"})
    with pytest.raises(ev.SpatialHoldoutEvalError, match="another anchor table or patch index"):
        ev.evaluate_cv(
            {**folds, "cv5-fold4": other_index}, matrix=matrix, holdout=holdout, **kwargs
        )
    with pytest.raises(ev.SpatialHoldoutEvalError, match="needs the assignments"):
        ev.evaluate_cv(
            {k: v for k, v in folds.items() if k != "cv5-fold4"},
            matrix=matrix,
            holdout=holdout,
            **kwargs,
        )
    with pytest.raises(ev.SpatialHoldoutEvalError, match="unknown fold"):
        ev.evaluate_cv(folds, matrix=matrix, holdout={"cv5-fold9": holdout["cv5-fold0"]}, **kwargs)
    with pytest.raises(ev.SpatialHoldoutEvalError, match="exactly one matrix seed-0 run of L4"):
        ev.evaluate_cv(folds, matrix={"L1": matrix["L1"]}, holdout=holdout, **kwargs)
    doubled = {
        **holdout,
        "cv5-fold0": {**holdout["cv5-fold0"], "L1": holdout["cv5-fold0"]["L1"] * 2},
    }
    with pytest.raises(ev.SpatialHoldoutEvalError, match="one seed-0 run of each"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=doubled, **kwargs)
    seed_one = ev.load_fold_runs(
        [_export(tmp_path / "s1.parquet", folds, arm="L1", fold=0, seed=1)]
    )
    wrong_seed = {**holdout, "cv5-fold0": {**holdout["cv5-fold0"], "L1": seed_one}}
    with pytest.raises(ev.SpatialHoldoutEvalError, match="one seed-0 run of each"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=wrong_seed, **kwargs)
    extra_arm = {**holdout, "cv5-fold0": {**holdout["cv5-fold0"], "L2": holdout["cv5-fold0"]["L1"]}}
    with pytest.raises(ev.SpatialHoldoutEvalError, match="one seed-0 run of each"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=extra_arm, **kwargs)


def test_cv_out_of_fold_needs_the_same_patches_as_the_matrix_model(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    held = sorted(folds["cv5-fold2"].holdout_anchors)[0]
    short = _export(
        tmp_path / "short.parquet",
        folds,
        arm="L1",
        fold=2,
        drop=f"scene0:{cell_id(*held)}",
    )
    holdout["cv5-fold2"]["L1"] = ev.load_fold_runs([short])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="different patches"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=50)


def test_cv_command_line(
    tmp_path: Path,
    folds: dict[str, sh.SpatialHoldoutAssignment],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _REPO / "scripts" / "analyze_spatial_holdout.py"
    spec = importlib.util.spec_from_file_location("analyze_spatial_holdout_under_test", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_assignment", lambda p: folds[Path(p).stem])
    monkeypatch.setattr(cli, "assignment_sha256", lambda p: _sha(folds[Path(p).stem]))
    for s in (0, 1):
        for arm in ("L1", "L4"):
            _export(tmp_path / f"m-{arm}-{s}.parquet", folds, arm=arm, seed=s)
    fold_args: list[str] = []
    for name, fold in folds.items():
        if name == "cv5-fold4":
            continue  # one fold is missing
        files = [
            str(_export(tmp_path / f"{name}-{arm}.parquet", folds, arm=arm, fold=fold.fold))
            for arm in ("L1", "L4")
        ]
        fold_args += ["--fold-export", name, *files]
    argv = [
        "--matrix-export",
        *map(str, sorted(tmp_path.glob("m-*.parquet"))),
        *fold_args,
        "--n-bootstrap",
        "100",
        "--output-prefix",
        str(tmp_path / "res" / "cv"),
    ]
    assert cli.main(argv) == 0
    result = json.loads((tmp_path / "res" / "cv.json").read_text())
    assert result["missing"] == {"cv5-fold4": ["L1", "L4"]}
    assert result["out_of_fold"]["complete"] is False
    assert "Incomplete, not computed" in capsys.readouterr().out
    # an unknown or repeated fold, or the plain holdout option next to the folds, is refused
    bad = ["--fold-export", "cv5-fold9", str(tmp_path / "cv5-fold0-L1.parquet")]
    assert (
        cli.main(["--matrix-export", *argv[1:3], *bad, "--output-prefix", str(tmp_path / "x")]) == 2
    )
    swapped = [
        "--fold-export",
        "cv5-fold0",
        *(str(tmp_path / f"cv5-fold1-{a}.parquet") for a in ("L1", "L4")),
    ]
    assert (
        cli.main(
            [
                "--matrix-export",
                *map(str, sorted(tmp_path.glob("m-*.parquet"))),
                *swapped,
                "--output-prefix",
                str(tmp_path / "x"),
            ]
        )
        == 2
    )
    assert "run record names 'cv5-fold1'" in capsys.readouterr().err
    twice = [*fold_args[:4], *fold_args[:4]]
    assert (
        cli.main(["--matrix-export", *argv[1:3], *twice, "--output-prefix", str(tmp_path / "x")])
        == 2
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--matrix-export",
                *argv[1:3],
                "--holdout-export",
                *fold_args[2:3],
                *fold_args[:4],
                "--output-prefix",
                str(tmp_path / "x"),
            ]
        )


# ----------------------------------------------- the run's own record names the fold


def test_a_fold_export_filed_under_another_fold_is_refused(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    swapped = _export(tmp_path / "swap.parquet", folds, arm="L1", fold=1)
    holdout["cv5-fold0"]["L1"] = ev.load_fold_runs([swapped])  # a fold-1 run filed as fold 0
    with pytest.raises(ev.SpatialHoldoutEvalError, match="swap.parquet.*cv5-fold0.*cv5-fold1"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=50)


def test_a_fold_export_whose_record_has_another_hash_is_refused(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    # the expected hash is the committed file's: the command line passes it in
    with pytest.raises(ev.SpatialHoldoutEvalError, match="sha256"):
        ev.evaluate_cv(
            folds,
            matrix=matrix,
            holdout=holdout,
            fold_sha256={name: "1" * 64 for name in folds},
            n_bootstrap=50,
        )
    forged = _export(tmp_path / "forged.parquet", folds, arm="L4", fold=2)
    _write_record(forged, "cv5-fold2", "0" * 64)  # the name matches, the file the run read does not
    holdout["cv5-fold2"]["L4"] = ev.load_fold_runs([forged])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="forged.parquet.*sha256"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=50)


def test_a_fold_export_without_a_record_or_with_a_v1_record_is_refused(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    bare = _export(tmp_path / "bare.parquet", folds, arm="L1", fold=3)
    bare.with_suffix(".evidence.json").unlink()
    with pytest.raises(ev.SpatialHoldoutEvalError, match="bare.parquet.*run record"):
        ev.load_fold_runs([bare])
    # a job that ran on an old image trained v1 under the fold label: its record says v1
    old = _export(tmp_path / "old.parquet", folds, arm="L1", fold=3)
    _write_record(old, "v1", "2" * 64)
    holdout["cv5-fold3"]["L1"] = ev.load_fold_runs([old])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="old.parquet.*'v1'.*cv5-fold3"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=50)
    # runs read without a record (plain load_runs) are not accepted in cross-validation
    holdout["cv5-fold3"]["L1"] = ev.load_runs([bare])
    with pytest.raises(ev.SpatialHoldoutEvalError, match="no recorded assignment"):
        ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=50)


def test_the_record_may_be_the_raw_run_summary(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    path = _export(tmp_path / "raw.parquet", folds, arm="L1", fold=4)
    _write_record(path, "cv5-fold4", _sha(folds["cv5-fold4"]), evidence=False)
    (run,) = ev.load_fold_runs([path])
    assert (run.assignment, run.assignment_sha256) == ("cv5-fold4", _sha(folds["cv5-fold4"]))
    broken = _export(tmp_path / "broken.parquet", folds, arm="L1", fold=4)
    broken.with_suffix(".evidence.json").write_text(json.dumps({"full": {"summary": {}}}))
    with pytest.raises(ev.SpatialHoldoutEvalError, match="broken.parquet.*spatial_holdout"):
        ev.load_fold_runs([broken])


def test_guard_refuses_an_assignment_that_differs_from_the_one_in_the_output_root() -> None:
    # the launcher puts the fold in the label and so in the output root: a wrong override fails
    name = "v6_l1_spatial_holdout"
    root = "output_root=data/runs/v6-l1-spatial-holdout/unit-cv5-fold1-seed0-ep40-cos-notest/seed-0"
    guard_modeling_config(
        _compose(name, [*_RUN_OVERRIDES, root, "spatial_holdout.assignment=cv5-fold1"]), name
    )
    for assignment in ("cv5-fold0", "v1"):
        cfg = _compose(name, [*_RUN_OVERRIDES, root, f"spatial_holdout.assignment={assignment}"])
        with pytest.raises(ValueError, match="output_root names.*cv5-fold1"):
            guard_modeling_config(cfg, name)
    # a v1 root carries no fold, and a run without a label in its root is not refused
    v1 = "output_root=data/runs/v6-l1-spatial-holdout/unit-seed0-ep40-cos-notest/seed-0"
    guard_modeling_config(_compose(name, [*_RUN_OVERRIDES, v1]), name)
    cfg = _compose(name, [*_RUN_OVERRIDES, v1, "spatial_holdout.assignment=cv5-fold2"])
    guard_modeling_config(cfg, name)


def test_cv_overpass_sensitivity_is_opt_in(
    tmp_path: Path,
    folds: dict[str, sh.SpatialHoldoutAssignment],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from berlin_lst_downscaling.analysis import spatial_error

    # the synthetic scenes are no Landsat products: let scene0 and scene1 share an overpass
    overpass = {"scene0": "o0", "scene1": "o0", "scene2": "o2", "scene3": "o3", "scene4": "o4"}
    monkeypatch.setattr(spatial_error, "overpass_id", lambda scene_id: overpass[scene_id])
    matrix, holdout = _runs(tmp_path, folds)
    plain = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=100)
    asked = ev.evaluate_cv(
        folds, matrix=matrix, holdout=holdout, n_bootstrap=100, overpass_sensitivity=True
    )
    assert "overpass_sensitivity" not in json.dumps(plain)

    def strip(node: object) -> object:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "overpass_sensitivity"}
        return [strip(v) for v in node] if isinstance(node, list) else node

    assert strip(asked) == plain
    fold = asked["folds"]["cv5-fold1"]["evaluation"]
    assert fold["G"]["L1"][ev.UNSEEN]["overpass_sensitivity"]["n_overpasses"] == 4
    assert asked[ev.OUT_OF_FOLD]["K_cv"]["overpass_sensitivity"]["available"] is True
    text = ev.render_cv_markdown(asked)
    # the out-of-fold table and the table of each of the five folds
    assert text.count("Sensitivity: overpass as resampling unit") == 6
    assert "Sensitivity: overpass" not in ev.render_cv_markdown(plain)


def test_cv_g_decomposition_fails_soft_for_one_fold(
    tmp_path: Path, folds: dict[str, sh.SpatialHoldoutAssignment]
) -> None:
    matrix, holdout = _runs(tmp_path, folds)
    fold = folds["cv5-fold1"]
    seen = next(a for a in _scenes(folds)["scene0"] if a not in fold.holdout_anchors)
    # both fold models miss one seen patch that the matrix models scored (L4 - L1 stays valid)
    reference = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=100)
    for arm in ("L1", "L4"):
        path = tmp_path / f"short-fold1-{arm}.parquet"
        _export(path, folds, arm=arm, fold=1, drop=f"scene0:{cell_id(*seen)}")
        holdout["cv5-fold1"][arm] = ev.load_fold_runs([path])
    result = ev.evaluate_cv(folds, matrix=matrix, holdout=holdout, n_bootstrap=100)
    decomposition = result["folds"]["cv5-fold1"]["evaluation"]["G_decomposition"]
    assert decomposition["L1"]["available"] is False
    assert decomposition["L1"]["seen_patches"]["only_in_matrix"] == 1
    assert decomposition["L4"]["available"] is False
    for name in ("cv5-fold0", "cv5-fold2"):
        assert result["folds"][name]["evaluation"]["G_decomposition"]["L1"]["available"] is True
    # everything but that decomposition is as with the complete export
    assert result["fold_summary"] == reference["fold_summary"]
    assert result[ev.OUT_OF_FOLD] == reference[ev.OUT_OF_FOLD]
    assert (
        result["folds"]["cv5-fold1"]["evaluation"]["G"]
        == reference["folds"]["cv5-fold1"]["evaluation"]["G"]
    )
