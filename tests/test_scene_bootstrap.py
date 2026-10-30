"""Paired scene bootstrap: synthetic, offline tests."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from berlin_lst_downscaling.analysis import scene_bootstrap as sb

SCHEMA = pa.schema(
    [
        ("schema_version", pa.int32()),
        ("arm", pa.string()),
        ("seed", pa.int64()),
        ("run_id", pa.string()),
        ("checkpoint", pa.string()),
        ("patch_id", pa.string()),
        ("scene_id", pa.string()),
        ("scene_date", pa.string()),
        ("split", pa.string()),
        ("abs_error_sum", pa.float64()),
        ("valid_cells", pa.int64()),
        ("ssim_sum", pa.float64()),
        ("ssim_windows", pa.int64()),
        ("filled_feature_pixels", pa.int64()),
    ]
)

# patch spec: (patch_id, scene_id, date, cells, windows)
Patch = tuple[str, str, str | None, int, int]
# value fn: (arm, seed, scene_index, patch) -> (mae, ssim)
ValueFn = Callable[[str, int, int, Patch], tuple[float, float]]


def make_world(n_scenes: int = 20, seed: int = 123) -> list[Patch]:
    rng = np.random.default_rng(seed)
    out: list[Patch] = []
    for s in range(n_scenes):
        for p in range(int(rng.integers(1, 6))):
            out.append(
                (
                    f"s{s:02d}_p{p}",
                    f"scene{s:02d}",
                    f"2024-06-{s + 1:02d}",
                    int(rng.integers(50, 400)),
                    int(rng.integers(5, 30)),
                )
            )
    return out


def export_table(
    arm: str, seed: int, split: str, patches: list[Patch], fn: ValueFn, *, run_id: str | None = None
) -> pa.Table:
    rid = run_id or f"{arm}-run{seed}"
    cols: dict[str, list] = {n: [] for n in SCHEMA.names}
    for pid, scene, date, cells, wins in patches:
        mae, ssim = fn(arm, seed, int(scene[5:]), (pid, scene, date, cells, wins))
        row = {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "run_id": rid,
            "checkpoint": "best",
            "patch_id": pid,
            "scene_id": scene,
            "scene_date": date,
            "split": split,
            "abs_error_sum": mae * cells,
            "valid_cells": cells,
            "ssim_sum": ssim * wins,
            "ssim_windows": wins,
            "filled_feature_pixels": None,
        }
        for k, v in row.items():
            cols[k].append(v)
    return pa.table(cols, schema=SCHEMA)


def write_export(path: Path, table: pa.Table, *, footer_split: str | None = None) -> Path:
    first = {
        n: table.column(n)[0].as_py() for n in ("arm", "seed", "run_id", "checkpoint", "split")
    }
    footer = {"schema_version": 1, **first}
    if footer_split is not None:
        footer["split"] = footer_split
    pq.write_table(
        table.replace_schema_metadata({b"patch_export": json.dumps(footer).encode()}), path
    )
    return path


def build_arm(arm: str, patches: list[Patch], fn: ValueFn, split: str = "validation") -> pa.Table:
    return pa.concat_tables([export_table(arm, s, split, patches, fn) for s in (0, 1, 2)])


def flat(base: float) -> ValueFn:
    return lambda arm, seed, si, patch: (base, 0.5)


def effect_fn(effects: dict[int, float], noise: float = 0.0) -> ValueFn:
    def fn(arm: str, seed: int, si: int, patch: Patch) -> tuple[float, float]:
        mae = 5.0 + 0.1 * si + (effects[si] if arm == "A" else 0.0) + noise * seed
        return mae, 0.5 + (0.01 * si if arm == "A" else 0.0)

    return fn


@pytest.fixture
def world() -> list[Patch]:
    return make_world()


# ------------------------------------------------------------------- pooling


def test_unequal_support_pooling() -> None:
    patches: list[Patch] = [("p1", "scene00", None, 10, 4), ("p2", "scene00", None, 90, 6)]
    # patch1 MAE 10/10 cells = mean 1.0 ; patch 2 MAE 2.0 => (10 + 180)/100 = 1.9
    rows = pa.table(
        {
            "arm": ["A", "A"],
            "seed": [0, 0],
            "run_id": ["r", "r"],
            "checkpoint": ["c", "c"],
            "split": ["validation"] * 2,
            "scene_id": ["scene00"] * 2,
            "patch_id": ["p1", "p2"],
            "scene_date": [None, None],
            "abs_error_sum": [10.0, 180.0],
            "valid_cells": [10, 90],
            "ssim_sum": [4 * 0.9, 6 * 0.3],
            "ssim_windows": [4, 6],
        }
    )
    agg = sb.aggregate_scenes(rows).to_pylist()[0]
    assert agg["mae_100m"] == pytest.approx(1.9)
    assert agg["mae_100m"] != pytest.approx(1.5)
    assert agg["ssim_100m"] == pytest.approx((3.6 + 1.8) / 10)
    assert agg["n_patches"] == 2
    assert agg["n_cells"] == 100
    assert patches[0][3] + patches[1][3] == 100


def test_null_ssim_without_support() -> None:
    rows = pa.table(
        {
            "arm": ["A"],
            "seed": [0],
            "run_id": ["r"],
            "checkpoint": [None],
            "split": ["validation"],
            "scene_id": ["s"],
            "patch_id": ["p"],
            "scene_date": ["2024-01-01"],
            "abs_error_sum": [3.0],
            "valid_cells": [3],
            "ssim_sum": [0.0],
            "ssim_windows": [0],
        }  # fmt: skip
    )
    agg = sb.aggregate_scenes(rows).to_pylist()[0]
    assert agg["ssim_100m"] is None
    assert agg["scene_date"] == "2024-01-01"


def test_conflicting_dates_raise() -> None:
    rows = pa.table(
        {
            "arm": ["A", "A"],
            "seed": [0, 0],
            "run_id": ["r"] * 2,
            "checkpoint": [None] * 2,
            "split": ["validation"] * 2,
            "scene_id": ["s"] * 2,
            "patch_id": ["p", "q"],
            "scene_date": ["2024-01-01", "2024-01-02"],
            "abs_error_sum": [1.0, 1.0],
            "valid_cells": [1, 1],
            "ssim_sum": [0.0, 0.0],
            "ssim_windows": [0, 0],
        }  # fmt: skip
    )
    with pytest.raises(sb.SceneBootstrapError, match="conflicting dates"):
        sb.aggregate_scenes(rows)


# ----------------------------------------------------------------- bootstrap


def test_bootstrap_reproducible_and_order_independent(world: list[Patch]) -> None:
    def fn(arm: str, seed: int, si: int, patch: Patch) -> tuple[float, float]:
        k = int(patch[0][-1])
        mae = 1.0 + 0.0713 * si + 0.0317 * k + 0.0123 * seed + (0.3171 if arm == "A" else 0.0)
        return mae, 0.41 + 0.0137 * k + 0.0051 * seed + (0.0219 * si if arm == "A" else 0.0)

    a, b = build_arm("A", world, fn), build_arm("B", world, fn)
    r1 = sb.compare_arms(a, b, n_bootstrap=500, rng_seed=7)
    r2 = sb.compare_arms(a, b, n_bootstrap=500, rng_seed=7)
    assert r1 == r2
    perm_a = np.random.default_rng(0).permutation(a.num_rows)
    perm_b = np.random.default_rng(1).permutation(b.num_rows)
    r3 = sb.compare_arms(a.take(perm_a), b.take(perm_b), n_bootstrap=500, rng_seed=7)
    assert r3.point_estimate == r1.point_estimate
    assert (r3.ci_low, r3.ci_high) == (r1.ci_low, r1.ci_high)
    assert r3.scenes == r1.scenes
    assert r3.seed_diagnostics == r1.seed_diagnostics
    r4 = sb.compare_arms(a, b, n_bootstrap=500, rng_seed=8)
    assert (r4.ci_low, r4.ci_high) != (r1.ci_low, r1.ci_high)


def test_known_effect_ci_contains_truth_and_excludes_zero(world: list[Patch]) -> None:
    eff = {i: -2.0 + (0.4 if i % 2 else -0.4) for i in range(20)}
    fn = effect_fn(eff)
    res = sb.compare_arms(build_arm("A", world, fn), build_arm("B", world, fn), n_bootstrap=2000)
    assert res.ci_low < -2.0 < res.ci_high
    assert res.ci_high < 0
    assert res.frac_a_better == 1.0
    assert res.direction == "lower_is_better"


def test_null_effect_ci_includes_zero(world: list[Patch]) -> None:
    eff = {i: (0.5 if i % 2 else -0.5) for i in range(20)}
    fn = effect_fn(eff)
    res = sb.compare_arms(build_arm("A", world, fn), build_arm("B", world, fn), n_bootstrap=2000)
    assert res.ci_low < 0 < res.ci_high


def test_constant_effect_degenerate_ci(world: list[Patch]) -> None:
    fn = effect_fn(dict.fromkeys(range(20), -1.0))
    res = sb.compare_arms(build_arm("A", world, fn), build_arm("B", world, fn), n_bootstrap=300)
    assert res.point_estimate == pytest.approx(-1.0)
    assert res.ci_low == pytest.approx(-1.0)
    assert res.ci_high == pytest.approx(-1.0)


def test_patch_multiplicity_leaves_scene_estimates_unchanged(world: list[Patch]) -> None:
    eff = {i: -1.0 + 0.3 * math.sin(i) for i in range(20)}
    fn = effect_fn(eff)
    base = sb.compare_arms(build_arm("A", world, fn), build_arm("B", world, fn), n_bootstrap=400)
    doubled = world + [(f"{pid}_dup", sc, d, c, w) for pid, sc, d, c, w in world]
    dup = sb.compare_arms(build_arm("A", doubled, fn), build_arm("B", doubled, fn), n_bootstrap=400)
    assert dup.point_estimate == pytest.approx(base.point_estimate)
    assert dup.ci_low == pytest.approx(base.ci_low)
    assert dup.ci_high == pytest.approx(base.ci_high)
    for x, y in zip(base.scenes, dup.scenes, strict=True):
        assert x.metric_a == pytest.approx(y.metric_a)
        assert y.n_patches == 2 * x.n_patches


def test_seed_diagnostics_hand_computed() -> None:
    patches: list[Patch] = [("p1", "scene00", None, 10, 1), ("p2", "scene01", None, 30, 1)]
    seed_mae_a = {0: (1.0, 2.0), 1: (2.0, 3.0), 2: (3.0, 5.0)}  # per patch
    seed_mae_b = {0: (1.5, 2.0), 1: (2.0, 3.0), 2: (2.0, 4.0)}

    def fn(arm: str, seed: int, si: int, patch: Patch) -> tuple[float, float]:
        src = seed_mae_a if arm == "A" else seed_mae_b
        return src[seed][int(patch[0][1]) - 1], 0.5

    res = sb.compare_arms(build_arm("A", patches, fn), build_arm("B", patches, fn), n_bootstrap=50)
    pooled_a = [(m1 * 10 + m2 * 30) / 40 for m1, m2 in seed_mae_a.values()]
    pooled_b = [(m1 * 10 + m2 * 30) / 40 for m1, m2 in seed_mae_b.values()]
    d = res.seed_diagnostics
    assert d.metric_a == pytest.approx(pooled_a)
    assert d.metric_b == pytest.approx(pooled_b)
    assert d.sd_a == pytest.approx(float(np.std(pooled_a, ddof=1)))
    assert d.mean_b == pytest.approx(float(np.mean(pooled_b)))
    diffs = [x - y for x, y in zip(pooled_a, pooled_b, strict=True)]
    assert d.diffs == pytest.approx(diffs)
    assert (d.n_positive, d.n_negative, d.n_zero) == (1, 1, 1)
    assert d.sign_consistency.startswith("1/3 same sign")


def test_sign_consistency_and_metric_direction(world: list[Patch]) -> None:
    fn = effect_fn(dict.fromkeys(range(20), -1.0))
    a, b = build_arm("A", world, fn), build_arm("B", world, fn)
    mae = sb.compare_arms(a, b, n_bootstrap=50)
    assert mae.seed_diagnostics.sign_consistency.startswith("3/3 same sign")
    assert mae.seed_diagnostics.n_negative == 3
    ssim = sb.compare_arms(a, b, metric="ssim_100m", n_bootstrap=50)
    assert ssim.direction == "higher_is_better"
    # A has higher SSIM for si > 0, equal at si == 0 (tie)
    assert ssim.frac_a_better == pytest.approx(19 / 20)
    assert ssim.frac_tie == pytest.approx(1 / 20)
    assert ssim.frac_b_better == 0.0
    assert mae.frac_a_better == 1.0


# ---------------------------------------------------------------- mismatches


def _drop_scene(table: pa.Table, scene: str) -> pa.Table:
    import pyarrow.compute as pc

    return table.filter(pc.not_equal(table.column("scene_id"), scene))


def test_scene_mismatch_raises(world: list[Patch]) -> None:
    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = _drop_scene(build_arm("B", world, fn), "scene03")
    with pytest.raises(sb.SceneBootstrapError, match="scene03"):
        sb.compare_arms(a, b, n_bootstrap=10)


def test_patch_mismatch_raises(world: list[Patch]) -> None:
    import pyarrow.compute as pc

    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = build_arm("B", world, fn)
    multi = [p for p in world if p[1] == "scene05"]
    assert len(multi) >= 1
    b = b.filter(pc.not_equal(b.column("patch_id"), multi[0][0]))
    with pytest.raises(sb.SceneBootstrapError, match="differ"):
        sb.compare_arms(a, b, n_bootstrap=10)


def test_within_arm_seed_mismatch_raises(world: list[Patch]) -> None:
    import pyarrow.compute as pc

    fn = flat(1.0)
    a = build_arm("A", world, fn)
    mask = pc.and_(pc.equal(a.column("seed"), 1), pc.equal(a.column("scene_id"), "scene02"))
    a = a.filter(pc.invert(mask))
    with pytest.raises(sb.SceneBootstrapError, match="A/seed1"):
        sb.compare_arms(a, build_arm("B", world, fn), n_bootstrap=10)


def test_allow_mismatch_drops_whole_scenes(world: list[Patch]) -> None:
    import pyarrow.compute as pc

    fn = flat(1.0)
    a = _drop_scene(build_arm("A", world, fn), "scene03")
    b = build_arm("B", world, fn)
    multi = next(p for p in world if p[1] == "scene07" and p[0].endswith("p0"))
    b = b.filter(
        pc.invert(pc.and_(pc.equal(b.column("patch_id"), multi[0]), pc.equal(b.column("seed"), 2)))
    )
    res = sb.compare_arms(a, b, n_bootstrap=20, allow_scene_mismatch=True)
    dropped = {d.scene_id: d for d in res.dropped_scenes}
    assert list(dropped) == ["scene03", "scene07"]
    assert dropped["scene03"].reason == "scene_missing_in_some_runs"
    assert dropped["scene07"].reason == "patch_set_differs_across_runs"
    assert dropped["scene07"].affected_runs
    kept = {s.scene_id for s in res.scenes}
    assert "scene03" not in kept and "scene07" not in kept
    assert len(kept) == 18


def test_allow_mismatch_empty_intersection_raises(world: list[Patch]) -> None:
    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = build_arm(
        "B", [(f"x{p}", f"scene{99 - i}", d, c, w) for i, (p, _s, d, c, w) in enumerate(world)], fn
    )
    with pytest.raises(sb.SceneBootstrapError, match="no scenes remain"):
        sb.compare_arms(a, b, n_bootstrap=10, allow_scene_mismatch=True)


@pytest.mark.parametrize("field", ["valid_cells", "ssim_windows"])
@pytest.mark.parametrize("allow", [False, True])
def test_patch_support_mismatch_raises_even_with_flag(
    world: list[Patch], field: str, allow: bool
) -> None:
    a = build_arm("A", world, flat(1.0))
    if field == "valid_cells":
        changed = [(p, s, d, c + 1, w) for p, s, d, c, w in world]
    else:
        changed = [(p, s, d, c, w + 1) for p, s, d, c, w in world]
    b = build_arm("B", changed, flat(1.0))
    with pytest.raises(sb.SceneBootstrapError, match=field):
        sb.compare_arms(a, b, n_bootstrap=10, allow_scene_mismatch=allow)


def _set_cells(table: pa.Table, row: int, **values: object) -> pa.Table:
    for col, value in values.items():
        field = table.schema.field(col)
        vals = table.column(col).to_pylist()
        vals[row] = value
        table = table.set_column(
            table.schema.get_field_index(col), field, pa.array(vals, type=field.type)
        )
    return table


def _retype(table: pa.Table, col: str, typ: pa.DataType) -> pa.Table:
    return table.set_column(
        table.schema.get_field_index(col),
        pa.field(col, typ),
        table.column(col).cast(typ),
    )


@pytest.mark.parametrize("allow", [False, True])
def test_patch_scene_conflict_raises_even_with_mismatch_allowed(
    world: list[Patch], allow: bool
) -> None:
    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = build_arm("B", world, fn)
    pid, _scene, _d, _c, _w = world[0]
    other = next(p for p in world if p[1] != world[0][1])
    row = len(world)  # first patch of seed 1
    a = _set_cells(a, row, scene_id=other[1], scene_date=other[2])
    with pytest.raises(sb.SceneBootstrapError, match=pid):
        sb.compare_arms(a, b, n_bootstrap=10, allow_scene_mismatch=allow)


def test_cross_arm_scene_date_conflict_raises_even_with_flag(world: list[Patch]) -> None:
    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = build_arm("B", world, fn)
    dates = [
        "2030-01-01" if s == "scene00" else d
        for s, d in zip(
            b.column("scene_id").to_pylist(), b.column("scene_date").to_pylist(), strict=True
        )
    ]
    b = b.set_column(
        b.schema.get_field_index("scene_date"), SCHEMA.field("scene_date"), pa.array(dates)
    )
    with pytest.raises(sb.SceneBootstrapError, match="conflicting dates"):
        sb.compare_arms(a, b, n_bootstrap=10, allow_scene_mismatch=True)


def test_missing_seed_and_duplicate_run_raise(world: list[Patch]) -> None:
    import pyarrow.compute as pc

    fn = flat(1.0)
    a = build_arm("A", world, fn)
    b = build_arm("B", world, fn)
    with pytest.raises(sb.SceneBootstrapError, match="seeds"):
        sb.compare_arms(a.filter(pc.not_equal(a.column("seed"), 2)), b, n_bootstrap=10)
    extra = export_table("A", 0, "validation", world, fn, run_id="other-run")
    with pytest.raises(sb.SceneBootstrapError, match="multiple run_ids"):
        sb.compare_arms(pa.concat_tables([a, extra]), b, n_bootstrap=10)
    same = build_arm("A", world, fn)
    with pytest.raises(sb.SceneBootstrapError, match="different arm labels"):
        sb.compare_arms(a, same, n_bootstrap=10)


def test_invalid_bootstrap_args(world: list[Patch]) -> None:
    a, b = build_arm("A", world, flat(1.0)), build_arm("B", world, flat(1.0))
    for n in (0, -1, True, 1.5):
        with pytest.raises(sb.SceneBootstrapError):
            sb.compare_arms(a, b, n_bootstrap=n)  # type: ignore[arg-type]
    with pytest.raises(sb.SceneBootstrapError):
        sb.compare_arms(a, b, n_bootstrap=10, rng_seed=-1)


def test_selected_metric_without_support_raises() -> None:
    patches: list[Patch] = [("p1", "scene00", None, 10, 0), ("p2", "scene01", None, 10, 3)]
    fn: ValueFn = lambda arm, seed, si, patch: (1.0, 0.0 if patch[4] == 0 else 0.5)  # noqa: E731
    a, b = build_arm("A", patches, fn), build_arm("B", patches, fn)
    assert sb.compare_arms(a, b, n_bootstrap=10).metric == "mae_100m"
    with pytest.raises(sb.SceneBootstrapError, match="support"):
        sb.compare_arms(a, b, metric="ssim_100m", n_bootstrap=10)


# ------------------------------------------------------------------- loading


def test_load_rejects_invalid_rows(tmp_path: Path, world: list[Patch]) -> None:
    good = export_table("A", 0, "validation", world, flat(1.0))
    assert sb.load_patch_exports(
        [write_export(tmp_path / "ok.parquet", good)], split="validation"
    ).num_rows
    with pytest.raises(sb.SceneBootstrapError, match="no patch export"):
        sb.load_patch_exports([], split="validation")
    p = tmp_path / "ok.parquet"
    with pytest.raises(sb.SceneBootstrapError, match="duplicate"):
        sb.load_patch_exports([p, tmp_path / "." / "ok.parquet"], split="validation")

    def bad(col: str, value: object, name: str) -> Path:
        t = good
        idx = t.schema.get_field_index(col)
        vals = t.column(col).to_pylist()
        vals[0] = value
        t = t.set_column(idx, t.schema.field(col), pa.array(vals, type=t.schema.field(col).type))
        return write_export(tmp_path / name, t)

    cases = [
        ("valid_cells", 0, "finite"),
        ("abs_error_sum", float("nan"), "non-finite"),
        ("ssim_windows", -1, "negative"),
        ("split", "test", "split"),
        ("scene_id", None, "nulls"),
    ]
    for i, (col, val, msg) in enumerate(cases):
        with pytest.raises(sb.SceneBootstrapError):
            sb.load_patch_exports([bad(col, val, f"bad{i}.parquet")], split="validation")
        del msg
    # windows == 0 with nonzero ssim_sum
    t = good
    wins = t.column("ssim_windows").to_pylist()
    wins[0] = 0
    t = t.set_column(
        t.schema.get_field_index("ssim_windows"), "ssim_windows", pa.array(wins, type=pa.int64())
    )
    with pytest.raises(sb.SceneBootstrapError, match="ssim_windows == 0"):
        sb.load_patch_exports([write_export(tmp_path / "w.parquet", t)], split="validation")
    # duplicate patch ids
    dup = pa.concat_tables([good, good.slice(0, 1)])
    with pytest.raises(sb.SceneBootstrapError, match="unique"):
        sb.load_patch_exports([write_export(tmp_path / "d.parquet", dup)], split="validation")
    with pytest.raises(FileNotFoundError):
        sb.load_patch_exports([tmp_path / "missing.parquet"], split="validation")
    with pytest.raises(sb.SceneBootstrapError):
        sb.load_patch_exports([Path("gs://bucket/x.parquet")], split="validation")
    with pytest.raises(sb.SceneBootstrapError):
        sb.load_patch_exports([tmp_path / "*.parquet"], split="validation")


def test_footer_split_mismatch_rejected_before_reading_table(
    tmp_path: Path, world: list[Patch], monkeypatch: pytest.MonkeyPatch
) -> None:
    t = export_table("A", 0, "test", world, flat(1.0))
    path = write_export(tmp_path / "t.parquet", t)

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("table must not be read")

    monkeypatch.setattr(sb.pq, "read_table", boom)
    with pytest.raises(sb.SceneBootstrapError, match="footer split"):
        sb.load_patch_exports([path], split="validation")


def test_footer_row_disagreement_rejected(tmp_path: Path, world: list[Patch]) -> None:
    t = export_table("A", 0, "validation", world, flat(1.0))
    arm_col = pa.array(["A"] * (t.num_rows - 1) + ["Z"])
    t = t.set_column(t.schema.get_field_index("arm"), "arm", arm_col)
    with pytest.raises(sb.SceneBootstrapError, match="arm"):
        sb.load_patch_exports([write_export(tmp_path / "x.parquet", t)], split="validation")


# ---------------------------------------------------------------------- holm


def test_holm() -> None:
    out = sb.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert list(out) == ["a", "b", "c"]
    tied = sb.holm_adjust({"a": 0.02, "b": 0.02})
    assert tied == pytest.approx({"a": 0.04, "b": 0.04})
    assert sb.holm_adjust({"a": 0.9, "b": 0.8}) == pytest.approx({"a": 1.0, "b": 1.0})
    assert sb.holm_adjust({}) == {}
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(sb.SceneBootstrapError):
            sb.holm_adjust({"a": bad})


# -------------------------------------------------------------------- output


def test_json_and_markdown_outputs(tmp_path: Path, world: list[Patch]) -> None:
    fn = effect_fn(dict.fromkeys(range(20), -1.0), noise=0.02)
    a = _drop_scene(build_arm("A", world, fn), "scene03")
    res = sb.compare_arms(a, build_arm("B", world, fn), n_bootstrap=100, allow_scene_mismatch=True)
    jp, mp = sb.write_outputs(res, tmp_path / "nested" / "out")
    assert jp.name == "out.json" and mp.name == "out.md"
    data = json.loads(jp.read_text())
    assert data["schema_version"] == 1
    assert data["split"] == "validation"
    assert data["direction"] == "lower_is_better"
    assert data["retained_scenes"] == 19
    assert data["bootstrap"] == {
        "n_bootstrap": 100, "rng_seed": 0, "method": "percentile", "unit": "scene"
    }  # fmt: skip
    assert data["arm_a"]["seeds"] == [0, 1, 2]
    assert data["dropped_scenes"][0]["scene_id"] == "scene03"
    assert len(data["ci_95"]) == 2
    assert set(data["scenes"][0]) >= {
        "scene_id",
        "date",
        "metric_a",
        "metric_b",
        "diff",
        "n_patches",
        "n_cells",
    }
    assert data["seed_diagnostics"]["kind"] == "pooled_overall_metrics"
    md = mp.read_text()
    for token in (
        "95% percentile CI",
        "Sign consistency",
        "Dropped scenes",
        "scene03",
        "seed 0",
        "SD (ddof=1)",
    ):
        assert token in md


# ------------------------------------------------------- validation of inputs


@pytest.mark.parametrize(
    ("col", "value", "typ", "field"),
    [
        ("abs_error_sum", float("nan"), None, "abs_error_sum"),
        ("abs_error_sum", float("inf"), None, "abs_error_sum"),
        ("ssim_sum", float("nan"), None, "ssim_sum"),
        ("ssim_sum", float("-inf"), None, "ssim_sum"),
        ("abs_error_sum", -0.5, None, "abs_error_sum"),
        ("valid_cells", 0, None, "valid_cells"),
        ("valid_cells", -1, None, "valid_cells"),
        ("valid_cells", 3.5, pa.float64(), "valid_cells"),
        ("valid_cells", 3.0, pa.float64(), "valid_cells"),
        ("valid_cells", None, None, "valid_cells"),
        ("ssim_windows", -1, None, "ssim_windows"),
        ("ssim_windows", 2.5, pa.float64(), "ssim_windows"),
        ("ssim_windows", None, None, "ssim_windows"),
        ("abs_error_sum", None, None, "abs_error_sum"),
    ],
)
@pytest.mark.parametrize("which", ["A", "B"])
def test_compare_rejects_invalid_direct_table_values(
    world: list[Patch], col: str, value: object, typ: pa.DataType | None, field: str, which: str
) -> None:
    tables = {"A": build_arm("A", world, flat(1.0)), "B": build_arm("B", world, flat(1.0))}
    t = tables[which]
    if typ is not None:
        t = _retype(t, col, typ)
    tables[which] = _set_cells(t, 0, **{col: value})
    with pytest.raises(sb.SceneBootstrapError, match=field):
        sb.compare_arms(tables["A"], tables["B"], n_bootstrap=10)


def test_compare_rejects_nonzero_ssim_with_zero_windows(world: list[Patch]) -> None:
    a = _set_cells(build_arm("A", world, flat(1.0)), 0, ssim_windows=0)
    with pytest.raises(sb.SceneBootstrapError, match="ssim_windows == 0"):
        sb.compare_arms(a, build_arm("B", world, flat(1.0)), n_bootstrap=10)


@pytest.mark.parametrize("col", ["patch_id", "scene_id", "arm", "run_id"])
@pytest.mark.parametrize("bad", ["", "   "])
def test_compare_rejects_empty_identifiers(world: list[Patch], col: str, bad: str) -> None:
    a = build_arm("A", world, flat(1.0))
    if col in ("arm", "run_id"):
        n = a.num_rows
        a = a.set_column(
            a.schema.get_field_index(col), SCHEMA.field(col), pa.array([bad] * n, pa.string())
        )
    else:
        a = _set_cells(a, 0, **{col: bad})
    with pytest.raises(sb.SceneBootstrapError, match=col):
        sb.compare_arms(a, build_arm("B", world, flat(1.0)), n_bootstrap=10)


def test_compare_rejects_test_tables(world: list[Patch]) -> None:
    a = build_arm("A", world, flat(1.0), split="test")
    b = build_arm("B", world, flat(1.0), split="test")
    with pytest.raises(sb.SceneBootstrapError, match="validation"):
        sb.compare_arms(a, b, n_bootstrap=10)


def test_nonvalidation_split_rejected_before_path_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("path must not be accessed")

    for target in ("_check_path",):
        monkeypatch.setattr(sb, target, boom)
    monkeypatch.setattr(sb.pq, "read_schema", boom)
    monkeypatch.setattr(sb.pq, "read_table", boom)
    with pytest.raises(sb.SceneBootstrapError, match="validation"):
        sb.load_patch_exports([Path("x")], split="test")


@pytest.mark.parametrize("empty_arm", ["A", "B"])
def test_compare_rejects_empty_arm(world: list[Patch], empty_arm: str) -> None:
    tables = {"A": build_arm("A", world, flat(1.0)), "B": build_arm("B", world, flat(1.0))}
    tables[empty_arm] = tables[empty_arm].slice(0, 0)
    with pytest.raises(sb.SceneBootstrapError, match="no rows"):
        sb.compare_arms(tables["A"], tables["B"], n_bootstrap=10)


def test_single_scene_single_patch_degenerate_interval() -> None:
    patches: list[Patch] = [("p1", "scene00", "2024-06-01", 50, 10)]

    def fn(arm: str, seed: int, si: int, patch: Patch) -> tuple[float, float]:
        return (3.0, 0.8) if arm == "A" else (5.0, 0.5)

    a, b = build_arm("A", patches, fn), build_arm("B", patches, fn)
    for metric, known in (("mae_100m", -2.0), ("ssim_100m", 0.3)):
        res = sb.compare_arms(a, b, metric=metric, n_bootstrap=100)  # type: ignore[arg-type]
        assert len(res.scenes) == 1 and res.scenes[0].n_patches == 1
        for v in (res.point_estimate, res.ci_low, res.ci_high):
            assert math.isfinite(v)
            assert v == pytest.approx(known)


def _agg_rows(**overrides: list) -> pa.Table:
    cols: dict[str, list] = {
        "arm": ["A"],
        "seed": [0],
        "run_id": ["r"],
        "checkpoint": [None],
        "split": ["validation"],
        "scene_id": ["s"],
        "patch_id": ["p"],
        "scene_date": [None],
        "abs_error_sum": [3.0],
        "valid_cells": [3],
        "ssim_sum": [1.0],
        "ssim_windows": [2],
    }
    cols.update(overrides)
    return pa.table(cols)


def test_aggregate_empty_table_preserves_schema() -> None:
    empty = _agg_rows().slice(0, 0)
    out = sb.aggregate_scenes(empty)
    assert out.num_rows == 0
    assert dict(zip(out.schema.names, map(str, out.schema.types), strict=True)) == {
        "arm": "string",
        "seed": "int64",
        "run_id": "string",
        "checkpoint": "string",
        "split": "string",
        "scene_id": "string",
        "scene_date": "string",
        "abs_error_sum": "double",
        "n_cells": "int64",
        "mae_100m": "double",
        "ssim_sum": "double",
        "ssim_windows": "int64",
        "ssim_100m": "double",
        "n_patches": "int64",
    }


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"abs_error_sum": [float("nan")]}, "abs_error_sum"),
        ({"abs_error_sum": [float("inf")]}, "abs_error_sum"),
        ({"abs_error_sum": [-1.0]}, "abs_error_sum"),
        ({"ssim_sum": [float("nan")]}, "ssim_sum"),
        ({"abs_error_sum": [None]}, "abs_error_sum"),
        ({"valid_cells": [0]}, "valid_cells"),
        ({"valid_cells": [3.0]}, "valid_cells"),
        ({"valid_cells": [None]}, "valid_cells"),
        ({"ssim_windows": [-1]}, "ssim_windows"),
        ({"ssim_windows": [1.5]}, "ssim_windows"),
        ({"ssim_windows": [0]}, "ssim_windows == 0"),
        ({"patch_id": [" "]}, "patch_id"),
        ({"scene_id": [""]}, "scene_id"),
        ({"arm": [None]}, "arm"),
    ],
)
def test_aggregate_rejects_invalid_metric_values(overrides: dict, field: str) -> None:
    with pytest.raises(sb.SceneBootstrapError, match=field):
        sb.aggregate_scenes(_agg_rows(**overrides))


def test_nonfinite_aggregates_raise() -> None:
    # two finite patch sums whose pooled scene sum overflows
    big = _agg_rows(
        patch_id=["p", "q"],
        arm=["A", "A"],
        seed=[0, 0],
        run_id=["r", "r"],
        checkpoint=[None, None],
        split=["validation"] * 2,
        scene_id=["s", "s"],
        scene_date=[None, None],
        abs_error_sum=[1e308, 1e308],
        valid_cells=[1, 1],
        ssim_sum=[0.0, 0.0],
        ssim_windows=[0, 0],
    )
    with pytest.raises(sb.SceneBootstrapError, match="not finite"):
        sb.aggregate_scenes(big)

    two: list[Patch] = [("p1", "scene00", None, 1, 1), ("p2", "scene00", None, 1, 1)]
    a = build_arm("A", two, lambda *_: (1e308, 0.5))
    b = build_arm("B", two, flat(1.0))
    with pytest.raises(sb.SceneBootstrapError, match="not finite"):
        sb.compare_arms(a, b, n_bootstrap=10)
    # per-run scene metrics are finite but their sum over three seeds overflows
    one: list[Patch] = [("p1", "scene00", None, 1, 1)]
    a = build_arm("A", one, lambda *_: (1.7e308, 0.5))
    b = build_arm("B", one, flat(1.0))
    with pytest.raises(sb.SceneBootstrapError, match="not finite"):
        sb.compare_arms(a, b, n_bootstrap=10)


@pytest.mark.parametrize("small_batches", [False, True])
def test_bootstrap_matches_explicit_linear_percentile_reference(
    monkeypatch: pytest.MonkeyPatch, small_batches: bool
) -> None:
    n, n_boot, seed = 4, 1000, 11
    patches: list[Patch] = [(f"p{i}", f"scene{i:02d}", None, 20, 5) for i in range(n)]
    known = {0: -1.0, 1: -2.5, 2: 0.5, 3: -4.0}

    def fn(arm: str, s: int, si: int, patch: Patch) -> tuple[float, float]:
        return (10.0 + known[si] if arm == "A" else 10.0), 0.5

    a, b = build_arm("A", patches, fn), build_arm("B", patches, fn)
    if small_batches:
        monkeypatch.setattr(sb, "_BATCH_INDICES", n * 3)
    res = sb.compare_arms(a, b, n_bootstrap=n_boot, rng_seed=seed)
    diff = np.array([sc.diff for sc in res.scenes])
    assert diff == pytest.approx([known[i] for i in range(n)])
    rng = np.random.default_rng(seed)
    step = 3 if small_batches else n_boot
    reps = np.concatenate(
        [
            diff[rng.integers(0, n, size=(min(step, n_boot - k), n))].mean(axis=1)
            for k in range(0, n_boot, step)
        ]
    )
    lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
    assert res.point_estimate == pytest.approx(float(diff.mean()))
    assert res.ci_low == lo
    assert res.ci_high == hi
    if small_batches:
        full = diff[np.random.default_rng(seed).integers(0, n, size=(n_boot, n))].mean(axis=1)
        assert np.array_equal(full, reps)


# ------------------------------------------------- writer exports with SSIM variants


def _writer_arm(
    tmp_path: Path,
    arm: str,
    patches: list[Patch],
    *,
    main: float,
    l250: float,
    lpatch: float | None,
) -> list[Path]:
    from berlin_lst_downscaling.modeling.evaluation_exports import (
        EvaluationIdentity,
        write_patch_export,
    )
    from berlin_lst_downscaling.modeling.ssim_range import test_override_ssim_data_range

    paths: list[Path] = []
    for seed in (0, 1, 2):
        records = [
            {
                "patch_id": pid,
                "scene_id": scene,
                "scene_date": date,
                "split": "validation",
                "abs_error_sum": float(cells),
                "valid_cells": cells,
                "ssim_sum": main * wins,
                "ssim_windows": wins,
                "ssim_100m_L250": l250,
                "ssim_100m_Lpatch": lpatch,
                "ssim_windows_Lpatch": 0 if lpatch is None else wins,
                "ssim_Lpatch_zero_range_patches": 1 if lpatch is None else 0,
                "filled_feature_pixels": None,
            }
            for pid, scene, date, cells, wins in patches
        ]
        identity = EvaluationIdentity(arm=arm, seed=seed, run_id=f"{arm}-run{seed}", checkpoint="b")
        paths.append(
            write_patch_export(
                records,
                tmp_path / f"{arm}{seed}.parquet",
                identity=identity,
                split="validation",
                ssim_range=test_override_ssim_data_range(40.0),
            )
        )
    return paths


def test_ssim_bootstrap_uses_main_fixed_range_sums(tmp_path: Path, world: list[Patch]) -> None:
    a = sb.load_patch_exports(
        _writer_arm(tmp_path, "A", world, main=0.6, l250=0.1, lpatch=0.05), split="validation"
    )
    b = sb.load_patch_exports(
        _writer_arm(tmp_path, "B", world, main=0.4, l250=0.9, lpatch=0.95), split="validation"
    )
    res = sb.compare_arms(a, b, metric="ssim_100m", n_bootstrap=200)
    assert res.metric == "ssim_100m"
    assert res.point_estimate == pytest.approx(0.2)
    assert res.ci_low == pytest.approx(0.2) and res.ci_high == pytest.approx(0.2)
    assert res.frac_a_better == 1.0


def test_ssim_bootstrap_ignores_null_lpatch_diagnostic(tmp_path: Path, world: list[Patch]) -> None:
    a = sb.load_patch_exports(
        _writer_arm(tmp_path, "A", world, main=0.6, l250=0.3, lpatch=None), split="validation"
    )
    b = sb.load_patch_exports(
        _writer_arm(tmp_path, "B", world, main=0.4, l250=0.3, lpatch=0.2), split="validation"
    )
    res = sb.compare_arms(a, b, metric="ssim_100m", n_bootstrap=200)
    assert res.point_estimate == pytest.approx(0.2)
