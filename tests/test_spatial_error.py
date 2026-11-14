"""Spatial error analysis: synthetic, offline tests."""

from __future__ import annotations

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_scene_bootstrap import SCHEMA as PATCH_SCHEMA
from test_scene_bootstrap import build_arm, effect_fn, make_world, write_export

from berlin_lst_downscaling.analysis import scene_bootstrap as sb
from berlin_lst_downscaling.analysis import spatial_error as se
from berlin_lst_downscaling.modeling import cell_exports
from berlin_lst_downscaling.modeling.evaluation_exports import EvaluationIdentity

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "analyze_spatial_errors.py"
SIZE = 6  # cells per patch side
FOREST, BLOCK, WATER = 55, 1, 100  # typ codes of three proposal-B groups
GROUP_OF = {FOREST: "forest", BLOCK: "dense_closed_block", WATER: "water"}


# ── synthetic world ───────────────────────────────────────────────────


def scene_id(index: int) -> str:
    day = date(2024, 5, 1) + timedelta(days=7 * index)
    return f"LC08_L2SP_193023_{day:%Y%m%d}_02_T1"


def make_cells(n_scenes: int = 12, patches: int = 2, seed: int = 0) -> pd.DataFrame:
    """Cells of ``patches`` adjacent SIZE x SIZE patches per scene, a few masked out."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        for p in range(patches):
            for r in range(SIZE):
                for c in range(SIZE):
                    if rng.random() < 0.05:
                        continue
                    rows.append(
                        {
                            "scene_id": scene_id(s),
                            "patch_id": f"{scene_id(s)}:p{p}",
                            "row": 100 + r,
                            "col": 200 + p * SIZE + c,
                            "target_k": 300.0 + s + 0.1 * r,
                        }
                    )
    return pd.DataFrame(rows)


def cell_group(cells: pd.DataFrame) -> np.ndarray:
    """Left patch column block is forest, the right one dense block (pure cells)."""
    return np.where(cells["col"].to_numpy() < 200 + SIZE, FOREST, BLOCK)


def make_errors(cells: pd.DataFrame, arm: str, seed: int) -> np.ndarray:
    """Scene bias + a group effect that only arm A removes + seed noise."""
    rng = np.random.default_rng(1000 + 10 * seed + (0 if arm == "A" else 1))
    scene_index = pd.factorize(cells["scene_id"], sort=True)[0]
    bias = 0.2 * np.sin(scene_index) + (0.05 if arm == "B" else 0.0)
    group_effect = np.where(cell_group(cells) == BLOCK, 0.6 if arm == "B" else 0.2, 0.0)
    sign = np.where((cells["row"].to_numpy() + cells["col"].to_numpy()) % 2 == 0, 1.0, -1.0)
    return bias + sign * (0.3 + group_effect) + rng.normal(0.0, 0.02, len(cells))


def write_cells(root: Path, cells: pd.DataFrame, arm: str, seed: int) -> Path:
    label = f"{arm.lower()}-seed{seed}"
    columns = {
        "scene_id": cells["scene_id"].to_numpy(dtype=object),
        "patch_id": cells["patch_id"].to_numpy(dtype=object),
        "row": cells["row"].to_numpy(dtype=np.int32),
        "col": cells["col"].to_numpy(dtype=np.int32),
        "target_k": cells["target_k"].to_numpy(dtype=np.float32),
        "prediction_k": (cells["target_k"].to_numpy() + make_errors(cells, arm, seed)).astype(
            np.float32
        ),
    }
    return cell_exports.write_cell_export(
        columns,
        root / label / se.CELL_EXPORT_FILENAME,
        identity=EvaluationIdentity(arm=arm, seed=seed, run_id=label, checkpoint="best"),
        split="validation",
        aggregation="arithmetic",
    )


def write_patches(root: Path, cell_path: Path, *, scale: float = 1.0) -> Path:
    """Derive the per-patch export of a run from its cell export."""
    footer = json.loads(pq.read_schema(cell_path).metadata[se.CELL_EXPORT_METADATA_KEY])
    frame = pq.read_table(cell_path).to_pandas()
    frame["abs"] = (
        frame["prediction_k"].astype(np.float64) - frame["target_k"].astype(np.float64)
    ).abs()
    cols: dict[str, list] = {name: [] for name in PATCH_SCHEMA.names}
    for (scene, patch), part in frame.groupby(["scene_id", "patch_id"], sort=True):
        row = {
            "schema_version": 2,
            "arm": footer["arm"],
            "seed": footer["seed"],
            "run_id": footer["run_id"],
            "checkpoint": "best",
            "patch_id": patch,
            "scene_id": scene,
            "scene_date": None,
            "split": "validation",
            "abs_error_sum": float(part["abs"].sum()) * scale,
            "valid_cells": int(len(part)),
            "ssim_sum": 0.5,
            "ssim_windows": 1,
            "filled_feature_pixels": 0,
        }
        for key, value in row.items():
            cols[key].append(value)
    path = root / footer["run_id"] / se.PATCH_EXPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_export(path, pa.table(cols, schema=PATCH_SCHEMA))


def write_shares(path: Path, cells: pd.DataFrame, *, vintage: int = 2024) -> Path:
    unique = cells[["row", "col"]].drop_duplicates().reset_index(drop=True)
    unique["typ"] = cell_group(unique)
    unique["vintage"] = vintage
    unique["share_of_cell"] = 0.8
    unique["share_of_covered"] = 1.0
    pq.write_table(pa.Table.from_pandas(unique, preserve_index=False), path)
    return path


LABELS = tuple(f"{arm.lower()}-seed{seed}" for arm in ("A", "B") for seed in (0, 1, 2))


@pytest.fixture
def exports(tmp_path: Path) -> dict[str, Path]:
    cells = make_cells()
    cell_root, patch_root = tmp_path / "cells", tmp_path / "patches"
    for arm in ("A", "B"):
        for seed in (0, 1, 2):
            write_patches(patch_root, write_cells(cell_root, cells, arm, seed))
    return {
        "cells": cell_root,
        "patches": patch_root,
        "shares": write_shares(tmp_path / "shares.parquet", cells),
    }


def load_all(exports: dict[str, Path]) -> se.CellErrors:
    return se.load_cell_exports(
        [exports["cells"] / label / se.CELL_EXPORT_FILENAME for label in LABELS]
    )


# ── constants and bootstrap ───────────────────────────────────────────


def test_literal_constants_match_their_sources() -> None:
    assert se._BATCH_INDICES == sb._BATCH_INDICES
    assert se.CELL_EXPORT_SCHEMA_VERSION == cell_exports.CELL_EXPORT_SCHEMA_VERSION
    assert se.CELL_EXPORT_METADATA_KEY == cell_exports.CELL_EXPORT_METADATA_KEY
    assert se.CELL_EXPORT_COLUMNS == cell_exports.CELL_EXPORT_COLUMNS
    assert se.CELL_EXPORT_FILENAME == cell_exports.CELL_EXPORT_FILENAME


@pytest.mark.parametrize("n_bootstrap", [200, 10_000])
def test_bootstrap_reproduces_compare_arms(n_bootstrap: int) -> None:
    world = make_world()
    effects = {i: 0.3 * np.sin(i) - 0.05 for i in range(20)}
    fn = effect_fn(effects)
    result = sb.compare_arms(
        build_arm("A", world, fn), build_arm("B", world, fn), n_bootstrap=n_bootstrap, rng_seed=7
    )
    ci = se.scene_bootstrap_ci(
        [scene.diff for scene in result.scenes], n_bootstrap=n_bootstrap, rng_seed=7
    )
    assert ci.point == pytest.approx(result.point_estimate, rel=1e-12)
    assert (ci.ci_low, ci.ci_high) == (result.ci_low, result.ci_high)
    assert ci.n_units == len(result.scenes)


def test_weighted_bootstrap_is_the_ratio_estimator() -> None:
    values = np.array([0.1, -0.4, 0.3, 0.0, -0.2])
    weights = np.array([10.0, 1000.0, 5.0, 20.0, 50.0])
    equal = se.scene_bootstrap_ci(values, n_bootstrap=500)
    flat = se.scene_bootstrap_ci(values, weights=np.full(5, 3.0), n_bootstrap=500)
    assert (flat.ci_low, flat.ci_high) == pytest.approx((equal.ci_low, equal.ci_high))
    pooled = se.scene_bootstrap_ci(values, weights=weights, n_bootstrap=500)
    assert pooled.point == pytest.approx((values * weights).sum() / weights.sum())
    assert pooled.ci_low <= pooled.point <= pooled.ci_high
    assert pooled.point < equal.point  # the heavy scene carries the negative difference


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": [1.0]},
        {"values": [1.0, float("nan")]},
        {"values": [1.0, 2.0], "weights": [1.0, 0.0]},
        {"values": [1.0, 2.0], "n_bootstrap": 0},
        {"values": [1.0, 2.0], "rng_seed": -1},
        {"values": [1.0, 2.0], "level": 1.0},
    ],
)
def test_bootstrap_rejects_bad_input(kwargs: dict) -> None:
    with pytest.raises(se.SpatialErrorError):
        se.scene_bootstrap_ci(**kwargs)


# ── scene helpers and step 4 ──────────────────────────────────────────


def test_scene_date_and_season() -> None:
    assert se.scene_date("LC08_L2SP_192023_20240513_02_T1") == date(2024, 5, 13)
    assert se.season_of(date(2024, 5, 13)) == "MAM"
    assert se.season_of(date(2024, 12, 1)) == "DJF"
    assert [se.season_of(date(2024, m, 1)) for m in (6, 9)] == ["JJA", "SON"]
    with pytest.raises(se.SpatialErrorError):
        se.scene_date("scene00")
    with pytest.raises(se.SpatialErrorError):
        se.scene_date("LC08_L2SP_192023_20241340_02_T1")


def test_season_agrees_with_the_eda_rule() -> None:
    from berlin_lst_downscaling.data.eda.rules import season_of_month

    for month in range(1, 13):
        assert se.season_of(date(2024, month, 15)) == season_of_month(month)


def _comparison(exports: dict[str, Path]) -> sb.ComparisonResult:
    tables = {
        arm: sb.load_patch_exports(
            [
                exports["patches"] / f"{arm.lower()}-seed{s}" / se.PATCH_EXPORT_FILENAME
                for s in (0, 1, 2)
            ],
            split="validation",
        )
        for arm in ("A", "B")
    }
    return sb.compare_arms(tables["A"], tables["B"], n_bootstrap=300)


def test_scene_table_splits_the_gap_exactly(exports: dict[str, Path]) -> None:
    result = _comparison(exports)
    properties = pd.DataFrame(
        {
            "scene_id": [scene_id(i) for i in range(12)],
            "scene_mean_lst_k": np.linspace(295.0, 310.0, 12),
            "scene_std_lst_k": 3.0,
            "scene_eligible_cells": 1000,
        }
    )
    table = se.scene_difference_table(result, properties)
    assert list(table["date"]) == sorted(table["date"])
    assert table["cell_share"].sum() == pytest.approx(1.0)
    assert set(table["season"]) <= {"MAM", "JJA"}
    block = se.pooled_vs_scene_equal(table, n_bootstrap=300)
    equal, pooled = block["scene_equal"], block["cell_pooled"]
    assert equal["point_estimate"] == pytest.approx(result.point_estimate)
    assert equal["ci"] == [result.ci_low, result.ci_high]
    cells = table["n_cells"].to_numpy(dtype=float)
    assert pooled["point_estimate"] == pytest.approx((cells * table["diff"]).sum() / cells.sum())
    assert pooled["mae_a"] - pooled["mae_b"] == pytest.approx(pooled["point_estimate"])
    gap = block["gap_pooled_minus_scene_equal"]
    assert gap == pytest.approx(table["gap_contribution"].sum())
    assert gap == pytest.approx(sum(s["gap_contribution"] for s in block["by_season"]))
    assert sum(s["n_scenes"] for s in block["by_season"]) == 12
    corr = block["correlation_of_diff_with"]
    assert set(corr) == {"n_cells", "mae_b", "scene_mean_lst_k", "scene_std_lst_k"}
    assert corr["scene_std_lst_k"]["pearson"] is None  # constant column
    assert -1.0 <= corr["scene_mean_lst_k"]["spearman"] <= 1.0


def test_scene_properties_keep_validation_rows_only(tmp_path: Path) -> None:
    rows = []
    for split, scene in (
        ("validation", scene_id(0)),
        ("test", scene_id(1)),
        ("train", scene_id(2)),
    ):
        for weighting in ("observation", "equal_scene"):
            rows.append(
                {
                    "value": "lst_k",
                    "weighting": weighting,
                    "grain": "scene",
                    "scene_id": scene,
                    "split": split,
                    "n": 10,
                    "mean": 300.0,
                    "std": 2.0,
                }
            )
    path = tmp_path / "lst_distribution.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), path)
    out = se.load_scene_properties(path)
    assert list(out["scene_id"]) == [scene_id(0)]
    assert list(out.columns) == [
        "scene_id",
        "scene_mean_lst_k",
        "scene_std_lst_k",
        "scene_eligible_cells",
    ]


# ── step 5: single-run pair ───────────────────────────────────────────


def _single(arm: str, mae_of_scene, world) -> pa.Table:
    cols: dict[str, list] = {name: [] for name in PATCH_SCHEMA.names}
    for pid, scene, _date, cells, wins in world:
        row = {
            "schema_version": 2,
            "arm": arm,
            "seed": 0,
            "run_id": f"{arm}-run",
            "checkpoint": None,
            "patch_id": pid,
            "scene_id": scene,
            "scene_date": None,
            "split": "validation",
            "abs_error_sum": mae_of_scene(int(scene[5:])) * cells,
            "valid_cells": cells,
            "ssim_sum": 0.5 * wins,
            "ssim_windows": wins,
            "filled_feature_pixels": None,
        }
        for key, value in row.items():
            cols[key].append(value)
    return pa.table(cols, schema=PATCH_SCHEMA)


def test_compare_single_runs_matches_a_manual_bootstrap() -> None:
    world = make_world(15)
    stage4 = _single("random_forest_stage4", lambda s: 1.0 + 0.01 * s, world)
    stage1 = _single("random_forest_stage1", lambda s: 1.1 + 0.02 * s, world)
    out = se.compare_single_runs(stage4, stage1, n_bootstrap=400, rng_seed=3)
    diffs = np.array([-0.1 - 0.01 * s for s in range(15)])
    manual = se.scene_bootstrap_ci(diffs, n_bootstrap=400, rng_seed=3)
    assert out["scene_equal"]["point_estimate"] == pytest.approx(diffs.mean())
    assert out["scene_equal"]["ci"] == pytest.approx([manual.ci_low, manual.ci_high])
    assert out["scene_equal"]["n_scenes"] == 15
    assert out["fractions"] == {"a_better": 1.0, "tie": 0.0, "b_better": 0.0}
    assert out["arm_a"]["arm"] == "random_forest_stage4"
    cells = np.array([s["n_cells"] for s in out["scenes"]], dtype=float)
    assert out["cell_pooled"]["point_estimate"] == pytest.approx(
        (cells * diffs).sum() / cells.sum()
    )
    assert out["n_patches"] == len(world)


def test_compare_single_runs_rejects_mismatch() -> None:
    world = make_world(6)
    a = _single("random_forest_stage4", lambda s: 1.0, world)
    with pytest.raises(se.SpatialErrorError, match="differ in patches"):
        se.compare_single_runs(a, _single("random_forest_stage1", lambda s: 1.0, world[:-1]))
    with pytest.raises(se.SpatialErrorError, match="same run"):
        se.compare_single_runs(a, a)
    both = pa.concat_tables([a, _single("random_forest_stage1", lambda s: 1.0, world)])
    with pytest.raises(se.SpatialErrorError, match="exactly one arm"):
        se.compare_single_runs(both, a)


# ── cell exports ──────────────────────────────────────────────────────


def test_load_cell_exports_aligns_runs(exports: dict[str, Path]) -> None:
    loaded = load_all(exports)
    assert loaded.arms() == ("A", "B")
    assert loaded.run_indices("B") == [3, 4, 5]
    assert loaded.errors.shape == (6, len(loaded.cells))
    cells = make_cells()
    order = cells.sort_values(["scene_id", "row", "col"]).index.to_numpy()
    expected = make_errors(cells, "A", 0)[order]
    assert np.allclose(loaded.errors[0], expected, atol=1e-4)  # float32 storage


def test_load_cell_exports_rejections(tmp_path: Path) -> None:
    cells = make_cells(3)
    a0 = write_cells(tmp_path / "x", cells, "A", 0)
    with pytest.raises(se.SpatialErrorError, match="given twice"):
        se.load_cell_exports([a0, a0])
    fewer = write_cells(tmp_path / "y", cells.iloc[:-1], "B", 0)
    with pytest.raises(se.SpatialErrorError, match="different cells"):
        se.load_cell_exports([a0, fewer])
    shifted = cells.assign(target_k=cells["target_k"] + 1.0)
    with pytest.raises(se.SpatialErrorError, match="different targets"):
        se.load_cell_exports([a0, write_cells(tmp_path / "z", shifted, "B", 1)])
    with pytest.raises(FileNotFoundError):
        se.load_cell_exports([tmp_path / "missing.parquet"])
    with pytest.raises(se.SpatialErrorError, match="no cell export"):
        se.load_cell_exports([])


def _rewrite_footer(path: Path, out: Path, **changes: object) -> Path:
    table = pq.read_table(path)
    footer = json.loads(table.schema.metadata[se.CELL_EXPORT_METADATA_KEY])
    footer.update(changes)
    pq.write_table(
        table.replace_schema_metadata({se.CELL_EXPORT_METADATA_KEY: json.dumps(footer).encode()}),
        out,
    )
    return out


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"split": "test"}, "validation-only"),
        ({"schema_version": 99}, "schema_version"),
        ({"prediction_clamp_k": [0.0, 1000.0]}, "clamp"),
        ({"arm": ""}, "arm or run_id"),
    ],
)
def test_cell_export_footer_is_enforced(tmp_path: Path, changes: dict, match: str) -> None:
    good = write_cells(tmp_path / "x", make_cells(2), "A", 0)
    bad = _rewrite_footer(good, tmp_path / "bad.parquet", **changes)
    with pytest.raises(se.SpatialErrorError, match=match):
        se.load_cell_exports([bad])
    plain = tmp_path / "plain.parquet"
    pq.write_table(pq.read_table(good).replace_schema_metadata({}), plain)
    with pytest.raises(se.SpatialErrorError, match="footer key"):
        se.load_cell_exports([plain])


def test_cells_reproduce_the_patch_export(exports: dict[str, Path], tmp_path: Path) -> None:
    loaded = load_all(exports)
    patch_path = exports["patches"] / "a-seed0" / se.PATCH_EXPORT_FILENAME
    table = sb.load_patch_exports([patch_path], split="validation")
    se.check_cells_against_patch_export(loaded, "a-seed0", table)
    with pytest.raises(se.SpatialErrorError, match="does not belong"):
        se.check_cells_against_patch_export(loaded, "a-seed1", table)
    cell_path = exports["cells"] / "a-seed0" / se.CELL_EXPORT_FILENAME
    off = sb.load_patch_exports(
        [write_patches(tmp_path / "off", cell_path, scale=1.01)], split="validation"
    )
    with pytest.raises(se.SpatialErrorError, match="do not reproduce"):
        se.check_cells_against_patch_export(loaded, "a-seed0", off)


# ── step 1: decomposition ─────────────────────────────────────────────


def _cell_errors(cells: pd.DataFrame, errors: dict[tuple[str, int], np.ndarray]) -> se.CellErrors:
    order = cells.sort_values(["scene_id", "row", "col"]).index.to_numpy()
    runs = tuple(se.RunIdentity(arm, seed, f"{arm}-{seed}") for arm, seed in errors)
    stacked = np.vstack([values[order] for values in errors.values()])
    return se.CellErrors(cells.loc[order].reset_index(drop=True), runs, stacked)


def test_decomposition_recovers_bias_and_centred_parts() -> None:
    cells = make_cells(5, seed=3)
    rng = np.random.default_rng(0)
    scene_index = pd.factorize(cells["scene_id"], sort=True)[0]
    true_bias = np.array([0.5, -0.3, 0.0, 1.0, -1.2])
    noise = rng.normal(0.0, 0.4, len(cells))
    for s in range(5):  # centre the noise per scene, so the bias is exact
        noise[scene_index == s] -= noise[scene_index == s].mean()
    loaded = _cell_errors(cells, {("A", 0): true_bias[scene_index] + noise})
    out = se.scene_decomposition(loaded)
    assert np.allclose(out["bias_k"], true_bias)
    assert np.allclose(out["rmse_k"] ** 2, out["bias_k"] ** 2 + out["centred_rmse_k"] ** 2)
    assert np.allclose(out["bias_share_of_mse"], out["bias_k"] ** 2 / out["rmse_k"] ** 2)
    centred = se.centred_errors(loaded)
    assert np.allclose(np.bincount(scene_index, weights=centred[0]), 0.0, atol=1e-9)
    for s in range(5):
        part = noise[scene_index == s]
        assert out["centred_mae_k"].iloc[s] == pytest.approx(np.abs(part).mean())
        assert out["n_cells"].iloc[s] == part.size
    # the unbiased scene keeps its full error in the centred part
    assert out["bias_share_of_mse"].iloc[2] == pytest.approx(0.0, abs=1e-12)
    assert out["mae_k"].iloc[2] == pytest.approx(out["centred_mae_k"].iloc[2])


def test_decomposition_summary_averages_seeds_and_splits_the_pooled_mse() -> None:
    cells = make_cells(4, seed=1)
    scene_index = pd.factorize(cells["scene_id"], sort=True)[0]
    sign = np.where((cells["row"] + cells["col"]) % 2 == 0, 1.0, -1.0).astype(float)
    errors = {
        ("A", 0): 0.2 + 0.5 * sign,
        ("A", 1): 0.4 + 0.5 * sign,
        ("B", 0): np.zeros(len(cells)) + 0.1 * scene_index,
    }
    loaded = _cell_errors(cells, errors)
    per_run = se.scene_decomposition(loaded)
    arms = se.arm_scene_decomposition(per_run)
    a = arms[arms["arm"] == "A"]
    assert np.allclose(a["bias_k"], 0.3 + 0.5 * _mean_sign(cells), atol=1e-12)
    assert set(a["n_seeds"]) == {2} and set(arms[arms["arm"] == "B"]["n_seeds"]) == {1}
    summary = {s["arm"]: s for s in se.decomposition_summary(per_run)}
    b = summary["B"]
    assert b["scene_equal"]["bias_k"] == pytest.approx(0.15)
    assert b["scene_equal"]["centred_rmse_k"] == pytest.approx(0.0, abs=1e-12)
    assert b["cell_pooled_mse"]["scene_bias_share"] == pytest.approx(1.0)
    pooled = summary["A"]["cell_pooled_mse"]
    mse = np.mean([np.mean(errors[("A", s)] ** 2) for s in (0, 1)])
    assert pooled["total_k2"] == pytest.approx(mse)
    assert 0.0 < pooled["scene_bias_share"] < 1.0


def _mean_sign(cells: pd.DataFrame) -> np.ndarray:
    sign = np.where((cells["row"] + cells["col"]) % 2 == 0, 1.0, -1.0)
    return pd.Series(sign).groupby(cells["scene_id"].to_numpy()).mean().sort_index().to_numpy()


# ── step 2: Stadtstruktur groups ──────────────────────────────────────


def _shares(rows: list[tuple[int, int, int, float, float]], vintage: int = 2024) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["row", "col", "typ", "share_of_cell", "share_of_covered"])
    frame["vintage"] = vintage
    return frame


def test_group_weights_sum_types_and_apply_min_share() -> None:
    shares = _shares(
        [
            (0, 0, 1, 0.3, 0.5),  # dense_closed_block
            (0, 0, 2, 0.2, 0.3),  # dense_closed_block (same group, summed)
            (0, 0, FOREST, 0.1, 0.2),
            (0, 1, FOREST, 0.9, 1.0),
            (0, 2, WATER, 0.0, 0.0),  # zero share is dropped
        ]
    )
    mapping = {1: "dense_closed_block", 2: "dense_closed_block", FOREST: "forest", WATER: "water"}
    weights = se.group_weights(shares, mapping)
    got = {(r.row, r.col, r.group): r.weight for r in weights.itertuples()}
    assert got == pytest.approx(
        {(0, 0, "dense_closed_block"): 0.5, (0, 0, "forest"): 0.1, (0, 1, "forest"): 0.9}
    )
    covered = se.group_weights(shares, mapping, share_column="share_of_covered")
    assert covered["weight"].sum() == pytest.approx(0.8 + 0.2 + 1.0)
    dominant = se.group_weights(shares, mapping, min_share=0.8)
    assert [(r.row, r.col, r.group) for r in dominant.itertuples()] == [(0, 1, "forest")]


def test_group_weights_rejections() -> None:
    shares = _shares([(0, 0, FOREST, 0.5, 0.5)])
    with pytest.raises(se.SpatialErrorError, match="without a group"):
        se.group_weights(shares, {1: "dense_closed_block"})
    with pytest.raises(se.SpatialErrorError, match="share_column"):
        se.group_weights(shares, GROUP_OF, share_column="majority")
    with pytest.raises(se.SpatialErrorError, match="min_share"):
        se.group_weights(shares, GROUP_OF, min_share=1.5)
    two = pd.concat([shares, _shares([(0, 1, FOREST, 0.5, 0.5)], vintage=2023)])
    with pytest.raises(se.SpatialErrorError, match="one vintage"):
        se.group_weights(two, GROUP_OF)
    over = _shares([(0, 0, FOREST, 0.7, 0.7), (0, 0, BLOCK, 0.6, 0.6)])
    with pytest.raises(se.SpatialErrorError, match="more than 1"):
        se.group_weights(over, GROUP_OF)
    with pytest.raises(se.SpatialErrorError, match="lacks columns"):
        se.group_weights(shares.drop(columns=["typ"]), GROUP_OF)


def test_group_metrics_are_area_weighted() -> None:
    scene = scene_id(0)
    cells = pd.DataFrame(
        {
            "scene_id": scene,
            "patch_id": "p",
            "row": [0, 0, 0, 0],
            "col": [0, 1, 2, 3],
            "target_k": 300.0,
        }
    )
    error = np.array([1.0, -1.0, 3.0, 5.0])  # scene bias 2.0
    loaded = _cell_errors(cells, {("A", 0): error})
    weights = pd.DataFrame(
        {
            "row": [0, 0, 0, 0],
            "col": [0, 1, 2, 2],
            "group": ["forest", "forest", "forest", "water"],
            "weight": [1.0, 0.5, 0.5, 0.25],
        }
    )
    out = se.group_scene_metrics(loaded, weights, min_effective_cells=1.0).set_index("group")
    forest = out.loc["forest"]
    assert forest["effective_cells"] == pytest.approx(2.0)
    assert forest["cells_touched"] == 3
    assert forest["bias_k"] == pytest.approx((1.0 - 0.5 + 1.5) / 2.0)
    assert forest["centred_bias_k"] == pytest.approx(forest["bias_k"] - 2.0)
    assert forest["mae_k"] == pytest.approx((1.0 + 0.5 + 1.5) / 2.0)
    assert forest["rmse_k"] == pytest.approx(np.sqrt((1.0 + 0.5 + 4.5) / 2.0))
    assert bool(forest["supported"]) is True
    water = out.loc["water"]
    assert water["mae_k"] == pytest.approx(3.0) and bool(water["supported"]) is False
    coverage = se.mapped_coverage(loaded, weights)
    assert coverage == {
        "evaluated_cells": 4,
        "cells_with_a_group": 3,
        "assigned_area_share": pytest.approx(2.25 / 4),
    }
    with pytest.raises(se.SpatialErrorError, match="no evaluated cell"):
        se.group_scene_metrics(loaded, weights.assign(row=99))


def test_group_summary_and_paired_differences(exports: dict[str, Path]) -> None:
    loaded = load_all(exports)
    shares = pq.read_table(exports["shares"]).to_pandas()
    weights = se.group_weights(shares, GROUP_OF)
    per_run = se.group_scene_metrics(loaded, weights, min_effective_cells=20.0)
    assert per_run["supported"].all()  # ~34 cells x 0.8 per scene and group
    summary = se.group_summary(per_run).set_index(["arm", "group"])
    assert set(summary["n_scenes"]) == {12}
    # arm B carries the larger block effect; forest is the same for both arms
    assert summary.loc[("B", "dense_closed_block"), "scene_equal_mae_k"] > 0.8
    assert summary.loc[("A", "dense_closed_block"), "scene_equal_mae_k"] < 0.6
    assert summary.loc[("A", "forest"), "scene_equal_mae_k"] == pytest.approx(
        summary.loc[("B", "forest"), "scene_equal_mae_k"], abs=0.06
    )
    diffs = {d["group"]: d for d in se.group_paired_differences(per_run, "A", "B", n_bootstrap=300)}
    block = diffs["dense_closed_block"]
    assert block["family"] == "exploratory"
    assert block["diff"] == pytest.approx(-0.4, abs=0.05)
    assert block["ci"][0] < block["diff"] < block["ci"][1] < 0
    assert block["fraction_scenes_a_better"] == 1.0
    assert block["diff"] == pytest.approx(block["mae_a"] - block["mae_b"])

    sparse = se.group_paired_differences(per_run, "A", "B", min_scenes=13, n_bootstrap=300)
    assert all(d["ci"] is None and "fewer than 13" in d["interval_note"] for d in sparse)
    with pytest.raises(se.SpatialErrorError, match="two loaded arms"):
        se.group_paired_differences(per_run, "A", "C")
    with pytest.raises(se.SpatialErrorError, match="min_scenes"):
        se.group_paired_differences(per_run, "A", "B", min_scenes=1)


def test_minimum_cell_rule_drops_thin_scene_groups(exports: dict[str, Path]) -> None:
    loaded = load_all(exports)
    weights = se.group_weights(pq.read_table(exports["shares"]).to_pandas(), GROUP_OF)
    per_run = se.group_scene_metrics(loaded, weights, min_effective_cells=27.5)
    support = per_run.drop_duplicates(["scene_id", "group"])
    assert 0 < support["supported"].sum() < len(support)
    summary = se.group_summary(per_run)
    assert summary["n_scenes"].max() < 12
    expected = support[support["supported"]].groupby("group").size()
    for row in summary[summary["arm"] == "A"].itertuples():
        assert row.n_scenes == expected[row.group]


# ── step 3: Moran's I ─────────────────────────────────────────────────


def _dense_moran(rows, cols, values, neighbourhood: str, weights: str) -> float:
    n = len(values)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dr, dc = abs(rows[i] - rows[j]), abs(cols[i] - cols[j])
            if i != j and max(dr, dc) == 1 and (neighbourhood == "queen" or dr + dc == 1):
                w[i, j] = 1.0
    if weights == "row":
        sums = w.sum(axis=1, keepdims=True)
        w = np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)
    z = values - values.mean()
    return float(n / w.sum() * (z @ w @ z) / (z @ z))


@pytest.mark.parametrize("neighbourhood", ["rook", "queen"])
@pytest.mark.parametrize("weights", ["row", "binary"])
def test_morans_i_matches_the_dense_definition(neighbourhood: str, weights: str) -> None:
    rng = np.random.default_rng(5)
    rr, cc = np.meshgrid(np.arange(9), np.arange(11), indexing="ij")
    keep = rng.random(rr.shape) < 0.7
    keep[0, :] = False
    keep[0, 0] = True  # an island for rook and queen (row 1 col 0/1 removed below)
    keep[1, 0] = keep[1, 1] = False
    rows, cols = rr[keep] + 40, cc[keep] - 3
    values = rng.normal(size=rows.size) + 0.3 * rows
    got = se.morans_i(rows, cols, values, neighbourhood=neighbourhood, weights=weights)  # type: ignore[arg-type]
    assert got.i == pytest.approx(_dense_moran(rows, cols, values, neighbourhood, weights))
    assert got.n_islands >= 1 and got.n_cells == rows.size
    assert got.expected == pytest.approx(-1.0 / (rows.size - 1))


def test_morans_i_known_patterns() -> None:
    rr, cc = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    checker = np.where((rows + cols) % 2 == 0, 1.0, -1.0)
    assert se.morans_i(rows, cols, checker, neighbourhood="rook").i == pytest.approx(-1.0)
    halves = np.where(cols < 4, 1.0, -1.0)
    assert se.morans_i(rows, cols, halves, neighbourhood="queen").i > 0.7
    flat = se.morans_i(rows, cols, np.ones(64))
    assert flat.i is None and flat.expected is None
    lonely = se.morans_i(np.array([0, 5]), np.array([0, 5]), np.array([1.0, 2.0]))
    assert lonely.i is None and lonely.n_islands == 2


def test_morans_i_rejections() -> None:
    with pytest.raises(se.SpatialErrorError, match="more than once"):
        se.morans_i(np.array([0, 0]), np.array([1, 1]), np.array([1.0, 2.0]))
    with pytest.raises(se.SpatialErrorError, match="neighbourhood"):
        se.morans_i(np.array([0]), np.array([0]), np.array([1.0]), neighbourhood="bishop")  # type: ignore[arg-type]
    with pytest.raises(se.SpatialErrorError, match="weights"):
        se.morans_i(np.array([0]), np.array([0]), np.array([1.0]), weights="idw")  # type: ignore[arg-type]
    with pytest.raises(se.SpatialErrorError, match="finite"):
        se.morans_i(np.array([0]), np.array([0]), np.array([np.nan]))


def test_scene_morans_uses_centred_residuals_per_scene(exports: dict[str, Path]) -> None:
    loaded = load_all(exports)
    per_run = se.scene_morans(loaded, neighbourhood="rook")
    assert len(per_run) == 6 * 12
    # the synthetic error alternates in a checkerboard: strongly negative for rook
    assert (per_run["moran_i"] < -0.8).all()
    shifted = se.CellErrors(loaded.cells, loaded.runs, loaded.errors + 7.0)
    assert np.allclose(
        se.scene_morans(shifted, neighbourhood="rook")["moran_i"], per_run["moran_i"]
    )
    scenes, summary = se.moran_summary(per_run)
    assert len(scenes) == 2 * 12
    assert {s["arm"] for s in summary} == {"A", "B"}
    assert all(s["n_scenes_defined"] == 12 and s["min"] <= s["median"] <= s["max"] for s in summary)


# ── report and CLI ────────────────────────────────────────────────────


def _config(exports: dict[str, Path], **over: object) -> se.AnalysisConfig:
    base: dict[str, object] = {
        "run_labels": LABELS,
        "arm_a": "A",
        "arm_b": "B",
        "patch_export_root": exports["patches"],
        "cell_export_root": exports["cells"],
        "stadtstruktur_shares": exports["shares"],
        "n_bootstrap": 200,
        "min_effective_cells": 20.0,
    }
    base.update(over)
    return se.AnalysisConfig(**base)  # type: ignore[arg-type]


def test_run_analysis_end_to_end(exports: dict[str, Path], tmp_path: Path) -> None:
    report = se.run_analysis(_config(exports))
    assert report["exploratory"] is True and report["split"] == "validation"
    assert report["skipped"] == {"reference_pair": "no pair of single-run reference exports given"}
    assert report["cells"]["checked_against_patch_exports"] == sorted(LABELS)
    assert report["scene_table"]["estimators"]["n_scenes"] == 12
    assert len(report["scene_decomposition"]["scenes"]) == 2 * 12
    groups = report["stadtstruktur_groups"]
    assert groups["family"] == "exploratory" and groups["vintage"] == 2024
    assert {d["group"] for d in groups["paired_differences"]} == {"forest", "dense_closed_block"}
    assert {d["role"] for d in groups["paired_differences"]} == {"comparison"}
    assert groups["coverage"]["assigned_area_share"] == pytest.approx(0.8)
    assert report["parameters"]["min_effective_cells"] == 20.0
    json_path, md_path = se.write_outputs(report, tmp_path / "out")
    assert json.loads(json_path.read_text())["kind"] == "spatial_error_analysis"
    text = md_path.read_text()
    for heading in (
        "## Pooled against scene-equal",
        "## Scene bias and centred residuals",
        "## Stadtstruktur groups (proposal B), exploratory",
        "## Moran's I of the centred residuals",
        "## Skipped steps",
    ):
        assert heading in text
    assert "Exploratory family" in text


def test_run_analysis_skips_what_it_cannot_compute(exports: dict[str, Path]) -> None:
    report = se.run_analysis(_config(exports, cell_export_root=None, stadtstruktur_shares=None))
    assert set(report["skipped"]) == {
        "reference_pair",
        "scene_decomposition",
        "stadtstruktur_groups",
        "morans_i",
    }
    assert "scene_table" in report and "cells" not in report
    no_shares = se.run_analysis(_config(exports, stadtstruktur_shares=None))
    assert "morans_i" in no_shares and "stadtstruktur_groups" in no_shares["skipped"]
    cells_only = se.run_analysis(_config(exports, patch_export_root=None))
    assert "scene_table" in cells_only["skipped"] and "scene_decomposition" in cells_only
    assert cells_only["cells"]["checked_against_patch_exports"] == []


def test_run_analysis_rejects_inconsistent_inputs(exports: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(se.SpatialErrorError, match="not among the loaded arms"):
        se.run_analysis(_config(exports, arm_a="C"))
    with pytest.raises(se.SpatialErrorError, match="duplicates"):
        se.run_analysis(_config(exports, run_labels=(*LABELS, LABELS[0])))
    cell_path = exports["cells"] / "a-seed0" / se.CELL_EXPORT_FILENAME
    write_patches(exports["patches"], cell_path, scale=1.05)  # overwrite with drifted sums
    with pytest.raises(se.SpatialErrorError, match="do not reproduce"):
        se.run_analysis(_config(exports))


def test_reference_pair_in_the_report(exports: dict[str, Path], tmp_path: Path) -> None:
    world = [(pid, scene_id(int(scene[5:])), d, c, w) for pid, scene, d, c, w in make_world(8)]
    paths = {}
    for stage, level in ((1, 1.2), (4, 1.1)):
        arm = f"random_forest_stage{stage}"
        table = _single(
            arm,
            lambda s, level=level: level,
            [(p, f"scene{i:02d}", d, c, w) for i, (p, _s, d, c, w) in enumerate(world)],
        )
        paths[stage] = write_export(tmp_path / f"rf{stage}.parquet", table)
    report = se.run_analysis(
        _config(
            exports,
            cell_export_root=None,
            stadtstruktur_shares=None,
            reference_a=paths[4],
            reference_b=paths[1],
        )
    )
    pair = report["reference_pair"]
    assert pair["scene_equal"]["point_estimate"] == pytest.approx(-0.1)
    assert pair["arm_a"]["arm"] == "random_forest_stage4"
    assert "Single-run reference pair" in se.render_markdown(report)


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_spatial_errors", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_outputs_and_reports_errors(
    cli: ModuleType, exports: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--run-labels",
        *LABELS,
        "--arm-a",
        "A",
        "--arm-b",
        "B",
        "--patch-export-root",
        str(exports["patches"]),
        "--cell-export-root",
        str(exports["cells"]),
        "--stadtstruktur-shares",
        str(exports["shares"]),
        "--min-effective-cells",
        "20",
        "--n-bootstrap",
        "100",
        "--moran-neighbourhood",
        "rook",
        "--moran-weights",
        "binary",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert cli.main(argv) == 0
    report = json.loads((tmp_path / "out" / "spatial_error_analysis.json").read_text())
    assert report["parameters"]["moran_neighbourhood"] == "rook"
    assert report["parameters"]["moran_weights"] == "binary"
    assert "Exploratory family" in capsys.readouterr().out
    assert cli.main([*argv[:-1], str(tmp_path / "out2"), "--arm-a", "Z"]) == 1
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([*argv, "--reference-a", str(tmp_path / "x.parquet")])
