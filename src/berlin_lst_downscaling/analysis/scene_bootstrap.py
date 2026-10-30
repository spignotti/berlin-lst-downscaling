"""Paired scene cluster bootstrap for ablation arm comparisons.

Inputs are per-patch export Parquet files (schema_version 1), one file per run
(arm, seed, run_id) and split. Patch sums are pooled within scenes, averaged
over three seeds per arm, differenced per scene (A - B), and the scene is the
only resampling unit. Only NumPy and PyArrow are used; nothing here evaluates
a model or reads data products.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
FOOTER_KEY = b"patch_export"
SEEDS: tuple[int, ...] = (0, 1, 2)
_BATCH_INDICES = 1_000_000

Metric = Literal["mae_100m", "ssim_100m"]

_COLUMN_CHECKS: dict[str, Callable[[pa.DataType], bool]] = {
    "schema_version": pa.types.is_int32,
    "arm": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "seed": pa.types.is_int64,
    "run_id": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "checkpoint": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "patch_id": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "scene_id": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "scene_date": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "split": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "abs_error_sum": pa.types.is_float64,
    "valid_cells": pa.types.is_int64,
    "ssim_sum": pa.types.is_float64,
    "ssim_windows": pa.types.is_int64,
    "filled_feature_pixels": pa.types.is_int64,
}
_NON_NULL = (
    "schema_version",
    "arm",
    "run_id",
    "patch_id",
    "scene_id",
    "split",
    "abs_error_sum",
    "valid_cells",
    "ssim_sum",
    "ssim_windows",
)
_FOOTER_FIELDS = ("arm", "seed", "run_id", "checkpoint")


class SceneBootstrapError(ValueError):
    """Raised for any invalid input or contract violation in this module."""


_ID_FIELDS = ("arm", "run_id", "patch_id", "scene_id", "split")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _validate_rows(name: str, cols: Mapping[str, Sequence[Any]]) -> None:
    """Validate per-row identifiers and metric inputs of a patch table."""
    pids = cols["patch_id"]
    for field in _ID_FIELDS:
        for i, v in enumerate(cols[field]):
            if not isinstance(v, str) or not v.strip():
                raise SceneBootstrapError(
                    f"{name}: {field} must be a nonempty string (patch {pids[i]!r}, got {v!r})"
                )
    for i, pid in enumerate(pids):
        for field in ("abs_error_sum", "ssim_sum"):
            v = cols[field][i]
            if not _is_real(v) or not math.isfinite(v):
                raise SceneBootstrapError(
                    f"{name}: {field} must be a finite real number (patch {pid!r}, got {v!r})"
                )
        if cols["abs_error_sum"][i] < 0:
            raise SceneBootstrapError(f"{name}: abs_error_sum must be non-negative (patch {pid!r})")
        cells = cols["valid_cells"][i]
        if not _is_int(cells) or cells <= 0:
            raise SceneBootstrapError(
                f"{name}: valid_cells must be an int > 0 (patch {pid!r}, got {cells!r})"
            )
        wins = cols["ssim_windows"][i]
        if not _is_int(wins) or wins < 0:
            raise SceneBootstrapError(
                f"{name}: ssim_windows must be an int >= 0 (patch {pid!r}, got {wins!r})"
            )
        if wins == 0 and cols["ssim_sum"][i] != 0:
            raise SceneBootstrapError(
                f"{name}: ssim_windows == 0 requires ssim_sum == 0 (patch {pid!r})"
            )


_ROW_FIELDS = (*_ID_FIELDS, "abs_error_sum", "ssim_sum", "valid_cells", "ssim_windows")


@dataclass(frozen=True)
class ArmIdentity:
    arm: str
    seeds: tuple[int, ...]
    run_ids: tuple[str, ...]
    checkpoints: tuple[str | None, ...]


@dataclass(frozen=True)
class SceneContribution:
    scene_id: str
    scene_date: str | None
    metric_a: float
    metric_b: float
    diff: float
    n_patches: int
    n_cells: int


@dataclass(frozen=True)
class DroppedScene:
    scene_id: str
    reason: str
    affected_runs: tuple[str, ...]


@dataclass(frozen=True)
class SeedDiagnostics:
    metric_a: tuple[float, ...]
    metric_b: tuple[float, ...]
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    diffs: tuple[float, ...]
    n_positive: int
    n_negative: int
    n_zero: int
    sign_consistency: str


@dataclass(frozen=True)
class ComparisonResult:
    schema_version: int
    split: str
    metric: str
    direction: str
    arm_a: ArmIdentity
    arm_b: ArmIdentity
    scenes: tuple[SceneContribution, ...]
    dropped_scenes: tuple[DroppedScene, ...]
    n_bootstrap: int
    rng_seed: int
    point_estimate: float
    ci_low: float
    ci_high: float
    frac_a_better: float
    frac_tie: float
    frac_b_better: float
    seed_diagnostics: SeedDiagnostics
    allow_scene_mismatch: bool


# --------------------------------------------------------------------------- load


def _check_path(path: Path) -> Path:
    text = str(path)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:/", text) or any(ch in path.name for ch in "*?["):
        raise SceneBootstrapError(f"only explicit local files are accepted, got {text!r}")
    if not path.is_file():
        raise FileNotFoundError(f"patch export not found or not a file: {text}")
    return path.resolve()


def _read_footer(path: Path) -> dict[str, Any]:
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(FOOTER_KEY)
    if raw is None:
        raise SceneBootstrapError(f"{path}: missing Parquet footer key {FOOTER_KEY!r}")
    try:
        footer = json.loads(raw)
    except ValueError as exc:
        raise SceneBootstrapError(f"{path}: footer is not valid JSON: {exc}") from exc
    if not isinstance(footer, dict):
        raise SceneBootstrapError(f"{path}: footer must be a JSON object")
    return footer


def _validate_table(path: Path, table: pa.Table, footer: dict[str, Any], split: str) -> pa.Table:
    missing = [c for c in _COLUMN_CHECKS if c not in table.column_names]
    if missing:
        raise SceneBootstrapError(f"{path}: missing columns {missing}")
    table = table.select(list(_COLUMN_CHECKS))
    for name, check in _COLUMN_CHECKS.items():
        if not check(table.schema.field(name).type):
            raise SceneBootstrapError(
                f"{path}: column {name!r} has type {table.schema.field(name).type}, "
                "which violates schema_version 1"
            )
    if table.num_rows == 0:
        raise SceneBootstrapError(f"{path}: export has no rows")
    for name in _NON_NULL:
        if table.column(name).null_count:
            raise SceneBootstrapError(f"{path}: column {name!r} contains nulls")
    if set(table.column("schema_version").to_pylist()) != {SCHEMA_VERSION}:
        raise SceneBootstrapError(f"{path}: row schema_version must equal {SCHEMA_VERSION}")
    if set(table.column("split").to_pylist()) != {split}:
        raise SceneBootstrapError(f"{path}: row split values differ from footer split {split!r}")
    for field in _FOOTER_FIELDS:
        values = set(table.column(field).to_pylist())
        if values != {footer.get(field)}:
            raise SceneBootstrapError(
                f"{path}: column {field!r} values {sorted(values, key=str)} disagree with "
                f"footer value {footer.get(field)!r}"
            )
    for name in ("abs_error_sum", "ssim_sum"):
        arr = np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(arr).all():
            raise SceneBootstrapError(f"{path}: column {name!r} has non-finite values")
        if name == "abs_error_sum" and (arr < 0).any():
            raise SceneBootstrapError(f"{path}: abs_error_sum must be non-negative")
    for name in ("valid_cells", "ssim_windows", "filled_feature_pixels"):
        vals = [v for v in table.column(name).to_pylist() if v is not None]
        if any(v < 0 for v in vals):
            raise SceneBootstrapError(f"{path}: column {name!r} has negative counts")
    if any(v <= 0 for v in table.column("valid_cells").to_pylist()):
        raise SceneBootstrapError(f"{path}: valid_cells must be > 0 for every patch")
    for ssim_sum, windows in zip(
        table.column("ssim_sum").to_pylist(), table.column("ssim_windows").to_pylist(), strict=True
    ):
        if windows == 0 and ssim_sum != 0:
            raise SceneBootstrapError(f"{path}: ssim_windows == 0 requires ssim_sum == 0")
    _validate_rows(str(path), {n: table.column(n).to_pylist() for n in _ROW_FIELDS})
    patch_ids = table.column("patch_id").to_pylist()
    if len(set(patch_ids)) != len(patch_ids):
        raise SceneBootstrapError(f"{path}: patch_id is not unique within the file")
    return table.cast(_normalized_schema())


def _normalized_schema() -> pa.Schema:
    types = {
        "schema_version": pa.int32(),
        "seed": pa.int64(),
        "valid_cells": pa.int64(),
        "ssim_windows": pa.int64(),
        "filled_feature_pixels": pa.int64(),
        "abs_error_sum": pa.float64(),
        "ssim_sum": pa.float64(),
    }
    return pa.schema([(n, types.get(n, pa.string())) for n in _COLUMN_CHECKS])


def load_patch_exports(paths: Sequence[Path], *, split: str) -> pa.Table:
    """Load explicit per-patch export files for the validation split, validating each."""
    if split != "validation":
        raise SceneBootstrapError(
            f"split {split!r} rejected: the analyzer is validation-only (test split is frozen)"
        )
    if not paths:
        raise SceneBootstrapError("no patch export files given")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = _check_path(Path(raw))
        if path in seen:
            raise SceneBootstrapError(f"duplicate patch export path: {path}")
        seen.add(path)
        resolved.append(path)
    tables: list[pa.Table] = []
    for path in resolved:
        footer = _read_footer(path)
        if footer.get("schema_version") != SCHEMA_VERSION:
            raise SceneBootstrapError(
                f"{path}: footer schema_version {footer.get('schema_version')!r} "
                f"!= {SCHEMA_VERSION}"
            )
        if footer.get("split") != split:
            raise SceneBootstrapError(
                f"{path}: footer split {footer.get('split')!r} does not match requested {split!r}"
            )
        tables.append(_validate_table(path, pq.read_table(path), footer, split))
    return pa.concat_tables(tables)


# ----------------------------------------------------------------------- aggregate


def aggregate_scenes(rows: pa.Table) -> pa.Table:
    """Pool patch sums per (arm, seed, run_id, checkpoint, split, scene_id)."""
    keys = ("arm", "seed", "run_id", "checkpoint", "split", "scene_id")
    cols = {n: rows.column(n).to_pylist() for n in (*keys, "patch_id", "scene_date")}
    abs_s = rows.column("abs_error_sum").to_pylist()
    cells = rows.column("valid_cells").to_pylist()
    ssim_s = rows.column("ssim_sum").to_pylist()
    wins = rows.column("ssim_windows").to_pylist()
    _validate_rows(
        "aggregate_scenes",
        {
            **cols,
            "abs_error_sum": abs_s,
            "valid_cells": cells,
            "ssim_sum": ssim_s,
            "ssim_windows": wins,
        },
    )
    groups: dict[tuple, dict[str, Any]] = {}
    for i in range(rows.num_rows):
        key = tuple(cols[k][i] for k in keys)
        g = groups.setdefault(
            key, {"abs": 0.0, "cells": 0, "ssim": 0.0, "wins": 0, "patches": set(), "date": None}
        )
        g["abs"] += abs_s[i]
        g["cells"] += cells[i]
        g["ssim"] += ssim_s[i]
        g["wins"] += wins[i]
        g["patches"].add(cols["patch_id"][i])
        date = cols["scene_date"][i]
        if date is not None:
            if g["date"] is not None and g["date"] != date:
                raise SceneBootstrapError(
                    f"scene {key[-1]!r} has conflicting dates {g['date']!r} and {date!r}"
                )
            g["date"] = date
    out: dict[str, list[Any]] = {
        k: []
        for k in (
            *keys,
            "scene_date",
            "abs_error_sum",
            "n_cells",
            "mae_100m",
            "ssim_sum",
            "ssim_windows",
            "ssim_100m",
            "n_patches",
        )
    }
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        if not (math.isfinite(g["abs"]) and math.isfinite(g["ssim"])):
            raise SceneBootstrapError(f"scene {key[-1]!r}: pooled sums are not finite (overflow)")
        mae = g["abs"] / g["cells"]
        ssim = g["ssim"] / g["wins"] if g["wins"] else None
        for name, value in zip(keys, key, strict=True):
            out[name].append(value)
        out["scene_date"].append(g["date"])
        out["abs_error_sum"].append(g["abs"])
        out["n_cells"].append(g["cells"])
        out["mae_100m"].append(mae)
        out["ssim_sum"].append(g["ssim"])
        out["ssim_windows"].append(g["wins"])
        out["ssim_100m"].append(ssim)
        out["n_patches"].append(len(g["patches"]))
    schema = pa.schema(
        [
            ("arm", pa.string()),
            ("seed", pa.int64()),
            ("run_id", pa.string()),
            ("checkpoint", pa.string()),
            ("split", pa.string()),
            ("scene_id", pa.string()),
            ("scene_date", pa.string()),
            ("abs_error_sum", pa.float64()),
            ("n_cells", pa.int64()),
            ("mae_100m", pa.float64()),
            ("ssim_sum", pa.float64()),
            ("ssim_windows", pa.int64()),
            ("ssim_100m", pa.float64()),
            ("n_patches", pa.int64()),
        ]
    )
    return pa.table(out, schema=schema)


# ------------------------------------------------------------------------- compare


@dataclass
class _Run:
    arm: str
    seed: int
    run_id: str
    checkpoint: str | None
    split: str
    patches: dict[str, tuple[str, int, int, float, float]]  # pid -> scene, cells, wins, abs, ssim
    dates: dict[str, str | None]  # scene -> date

    @property
    def label(self) -> str:
        return f"{self.arm}/seed{self.seed}/{self.run_id}"

    def scene_patches(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for pid, rec in self.patches.items():
            out.setdefault(rec[0], set()).add(pid)
        return {k: frozenset(v) for k, v in out.items()}


def _arm_runs(table: pa.Table, name: str) -> list[_Run]:
    required = (
        "arm",
        "seed",
        "run_id",
        "checkpoint",
        "split",
        "patch_id",
        "scene_id",
        "scene_date",
        "abs_error_sum",
        "valid_cells",
        "ssim_sum",
        "ssim_windows",
    )
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise SceneBootstrapError(f"{name}: missing columns {missing}")
    if table.num_rows == 0:
        raise SceneBootstrapError(f"{name}: no rows")
    c = {n: table.column(n).to_pylist() for n in required}
    arms = set(c["arm"])
    if len(arms) != 1 or None in arms:
        raise SceneBootstrapError(
            f"{name}: expected exactly one arm label, got {sorted(arms, key=str)}"
        )
    splits = set(c["split"])
    if len(splits) != 1:
        raise SceneBootstrapError(
            f"{name}: expected exactly one split, got {sorted(splits, key=str)}"
        )
    if splits != {"validation"}:
        raise SceneBootstrapError(
            f"{name}: only the validation split is accepted, got {sorted(splits, key=str)}"
        )
    if None in set(c["seed"]):
        raise SceneBootstrapError(f"{name}: null seeds are not allowed")
    if set(c["seed"]) != set(SEEDS):
        raise SceneBootstrapError(
            f"{name}: seeds must be exactly {list(SEEDS)}, got {sorted(set(c['seed']))}"
        )
    _validate_rows(name, {n: c[n] for n in _ROW_FIELDS})
    arm = next(iter(arms))
    split = next(iter(splits))
    by_seed: dict[int, dict[tuple, list[int]]] = {}
    for i in range(table.num_rows):
        by_seed.setdefault(c["seed"][i], {}).setdefault(
            (c["run_id"][i], c["checkpoint"][i]), []
        ).append(i)
    runs: list[_Run] = []
    for seed in SEEDS:
        groups = by_seed[seed]
        if len({rid for rid, _ in groups}) != 1:
            raise SceneBootstrapError(
                f"{name}: seed {seed} has multiple run_ids {sorted({r for r, _ in groups})}"
            )
        if len(groups) != 1:
            raise SceneBootstrapError(f"{name}: seed {seed} has multiple checkpoints")
        (rid, ckpt), idx = next(iter(groups.items()))
        patches: dict[str, tuple[str, int, int, float, float]] = {}
        dates: dict[str, str | None] = {}
        for i in idx:
            pid = c["patch_id"][i]
            if pid in patches:
                raise SceneBootstrapError(f"{name}: duplicate patch_id {pid!r} in run seed {seed}")
            scene = c["scene_id"][i]
            patches[pid] = (scene, c["valid_cells"][i], c["ssim_windows"][i],
                            c["abs_error_sum"][i], c["ssim_sum"][i])  # fmt: skip
            date = c["scene_date"][i]
            if date is not None:
                if dates.get(scene) not in (None, date):
                    raise SceneBootstrapError(f"scene {scene!r} has conflicting dates")
                dates[scene] = date
            else:
                dates.setdefault(scene, None)
        runs.append(_Run(arm, seed, rid, ckpt, split, patches, dates))
    return runs


def _fmt(items: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def _validate_n_bootstrap(n: object, seed: object) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise SceneBootstrapError("n_bootstrap must be a positive int")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SceneBootstrapError("rng_seed must be a non-negative int")


def compare_arms(
    arm_a: pa.Table,
    arm_b: pa.Table,
    *,
    metric: Metric = "mae_100m",
    n_bootstrap: int = 10_000,
    rng_seed: int = 0,
    allow_scene_mismatch: bool = False,
) -> ComparisonResult:
    """Paired scene cluster bootstrap of arm A minus arm B on the 100 m metric."""
    if metric not in ("mae_100m", "ssim_100m"):
        raise SceneBootstrapError(f"unsupported metric {metric!r}")
    _validate_n_bootstrap(n_bootstrap, rng_seed)
    runs_a = _arm_runs(arm_a, "arm_a")
    runs_b = _arm_runs(arm_b, "arm_b")
    if runs_a[0].split != runs_b[0].split:
        raise SceneBootstrapError("both arms must come from the same split")
    if runs_a[0].arm == runs_b[0].arm:
        raise SceneBootstrapError("arm A and arm B must have different arm labels")
    split = runs_a[0].split
    runs = runs_a + runs_b

    # Patch-level consistency across all runs (applies even when dropping scenes).
    ref: dict[str, tuple[int, int]] = {}
    ref_scene: dict[str, str] = {}
    for run in runs:
        for pid, (scene, cells, wins, _a, _s) in run.patches.items():
            if pid in ref and ref[pid] != (cells, wins):
                raise SceneBootstrapError(
                    f"patch {pid!r} has differing valid_cells/ssim_windows across runs "
                    f"(run {run.label})"
                )
            ref[pid] = (cells, wins)
            if ref_scene.setdefault(pid, scene) != scene:
                raise SceneBootstrapError(
                    f"patch {pid!r} maps to scenes {ref_scene[pid]!r} and {scene!r} "
                    f"(run {run.label})"
                )
    dates: dict[str, str] = {}
    for run in runs:
        for scene, date in run.dates.items():
            if date is None:
                continue
            if dates.setdefault(scene, date) != date:
                raise SceneBootstrapError(f"scene {scene!r} has conflicting dates across runs")

    scene_maps = {run.label: run.scene_patches() for run in runs}
    all_scenes = sorted({s for m in scene_maps.values() for s in m})
    keep: list[str] = []
    dropped: list[DroppedScene] = []
    for scene in all_scenes:
        present = {lab: m[scene] for lab, m in scene_maps.items() if scene in m}
        if len(present) < len(runs):
            missing = tuple(sorted(set(scene_maps) - set(present)))
            dropped.append(DroppedScene(scene, "scene_missing_in_some_runs", missing))
            continue
        counts = Counter(present.values())
        if len(counts) > 1:
            mode = counts.most_common(1)[0][0]
            odd = tuple(sorted(lab for lab, p in present.items() if p != mode))
            dropped.append(DroppedScene(scene, "patch_set_differs_across_runs", odd))
            continue
        keep.append(scene)
    if dropped and not allow_scene_mismatch:
        affected = sorted({r for d in dropped for r in d.affected_runs})
        raise SceneBootstrapError(
            f"scene/patch sets differ across runs: {len(dropped)} scene(s) "
            f"[{_fmt([d.scene_id for d in dropped])}] affect runs [{_fmt(affected)}]; "
            "fix the exports or pass allow_scene_mismatch=True to drop whole scenes"
        )
    if not keep:
        raise SceneBootstrapError("no scenes remain present in all runs with identical patch sets")

    num_idx, den_idx = (3, 1) if metric == "mae_100m" else (4, 2)
    keep_set = set(keep)

    def run_sums(run: _Run) -> tuple[dict[str, float], dict[str, int]]:
        num: dict[str, float] = dict.fromkeys(keep, 0.0)
        den: dict[str, int] = dict.fromkeys(keep, 0)
        for pid in sorted(run.patches):
            rec = run.patches[pid]
            if rec[0] in keep_set:
                num[rec[0]] += float(rec[num_idx])
                den[rec[0]] += int(rec[den_idx])
        if not all(math.isfinite(v) for v in num.values()):
            raise SceneBootstrapError(
                f"run {run.label}: pooled {metric} sums are not finite (overflow)"
            )
        return num, den

    sums = {run.label: run_sums(run) for run in runs}
    unsupported = sorted({s for _n, d in sums.values() for s, v in d.items() if v == 0})
    if unsupported:
        raise SceneBootstrapError(
            f"retained scenes without {metric} support (zero windows): {_fmt(unsupported)}"
        )

    def arm_scene_mean(rs: list[_Run]) -> np.ndarray:
        acc = np.zeros(len(keep), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for run in rs:
                num, den = sums[run.label]
                acc += np.array([num[s] / den[s] for s in keep], dtype=np.float64)
            return acc / len(rs)

    def require_finite(values: np.ndarray, what: str) -> None:
        if not np.isfinite(values).all():
            raise SceneBootstrapError(f"{what} is not finite (overflow)")

    with np.errstate(over="ignore", invalid="ignore"):
        mean_a = arm_scene_mean(runs_a)
        mean_b = arm_scene_mean(runs_b)
        diff = mean_a - mean_b
    require_finite(mean_a, "scene metric of arm A")
    require_finite(mean_b, "scene metric of arm B")
    require_finite(diff, "scene difference")
    n = len(keep)

    rng = np.random.default_rng(rng_seed)
    batch = max(1, _BATCH_INDICES // n)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        b = min(batch, n_bootstrap - done)
        idx = rng.integers(0, n, size=(b, n))
        with np.errstate(over="ignore", invalid="ignore"):
            reps[done : done + b] = diff[idx].mean(axis=1)
        done += b
    require_finite(reps, "bootstrap replicate")
    with np.errstate(over="ignore", invalid="ignore"):
        lo, hi = np.quantile(reps, [0.025, 0.975], method="linear")
        point = diff.mean()
    require_finite(np.array([lo, hi, point]), "confidence interval or point estimate")

    lower_better = metric == "mae_100m"
    a_better = (diff < 0) if lower_better else (diff > 0)
    b_better = (diff > 0) if lower_better else (diff < 0)
    ties = diff == 0

    patch_count = {s: len(ps) for s, ps in scene_maps[runs[0].label].items()}
    cells_ref = runs[0]
    scene_cells: dict[str, int] = dict.fromkeys(keep, 0)
    for rec in cells_ref.patches.values():
        if rec[0] in keep_set:
            scene_cells[rec[0]] += rec[1]
    contribs = tuple(
        SceneContribution(
            scene_id=s,
            scene_date=dates.get(s),
            metric_a=float(mean_a[i]),
            metric_b=float(mean_b[i]),
            diff=float(diff[i]),
            n_patches=patch_count[s],
            n_cells=scene_cells[s],
        )
        for i, s in enumerate(keep)
    )

    def seed_metrics(rs: list[_Run]) -> list[float]:
        out = []
        for run in rs:
            num, den = sums[run.label]
            tot_n = sum(num[s] for s in keep)
            tot_d = sum(den[s] for s in keep)
            out.append(tot_n / tot_d)
        return out

    sm_a, sm_b = seed_metrics(runs_a), seed_metrics(runs_b)
    sdiffs = [x - y for x, y in zip(sm_a, sm_b, strict=True)]
    with np.errstate(over="ignore", invalid="ignore"):
        stats = np.array(
            [
                np.mean(sm_a),
                np.mean(sm_b),
                np.std(sm_a, ddof=1),
                np.std(sm_b, ddof=1),
            ]
        )
    require_finite(np.array([*sm_a, *sm_b, *sdiffs, *stats]), "seed diagnostic")
    n_pos = sum(d > 0 for d in sdiffs)
    n_neg = sum(d < 0 for d in sdiffs)
    n_zero = sum(d == 0 for d in sdiffs)
    diag = SeedDiagnostics(
        metric_a=tuple(sm_a),
        metric_b=tuple(sm_b),
        mean_a=float(stats[0]),
        mean_b=float(stats[1]),
        sd_a=float(stats[2]),
        sd_b=float(stats[3]),
        diffs=tuple(sdiffs),
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
        sign_consistency=f"{max(n_pos, n_neg)}/{len(SEEDS)} same sign ({n_zero} tied)",
    )

    def ident(rs: list[_Run]) -> ArmIdentity:
        return ArmIdentity(
            rs[0].arm,
            tuple(r.seed for r in rs),
            tuple(r.run_id for r in rs),
            tuple(r.checkpoint for r in rs),
        )

    return ComparisonResult(
        schema_version=SCHEMA_VERSION,
        split=split,
        metric=metric,
        direction="lower_is_better" if lower_better else "higher_is_better",
        arm_a=ident(runs_a),
        arm_b=ident(runs_b),
        scenes=contribs,
        dropped_scenes=tuple(dropped),
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
        point_estimate=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        frac_a_better=float(a_better.mean()),
        frac_tie=float(ties.mean()),
        frac_b_better=float(b_better.mean()),
        seed_diagnostics=diag,
        allow_scene_mismatch=allow_scene_mismatch,
    )


# ---------------------------------------------------------------------------- holm


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment for a declared family of comparisons.

    Intended for a family of comparisons against one reference arm. Bootstrap
    p-values are not produced by this module (secondary, optional, future work).
    """
    for key, p in p_values.items():
        if isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p):
            raise SceneBootstrapError(f"p-value for {key!r} must be a finite number")
        if not 0.0 <= p <= 1.0:
            raise SceneBootstrapError(f"p-value for {key!r} must be in [0, 1]")
    m = len(p_values)
    order = sorted(p_values, key=lambda k: p_values[k])
    out: dict[str, float] = {}
    running = 0.0
    for j, key in enumerate(order):
        running = max(running, min(1.0, (m - j) * float(p_values[key])))
        out[key] = running
    return {k: out[k] for k in p_values}


# --------------------------------------------------------------------------- output


def result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    def arm(a: ArmIdentity) -> dict[str, Any]:
        return {
            "arm": a.arm,
            "seeds": list(a.seeds),
            "run_ids": list(a.run_ids),
            "checkpoints": list(a.checkpoints),
        }

    d = result.seed_diagnostics
    return {
        "schema_version": result.schema_version,
        "split": result.split,
        "metric": result.metric,
        "direction": result.direction,
        "arm_a": arm(result.arm_a),
        "arm_b": arm(result.arm_b),
        "retained_scenes": len(result.scenes),
        "scenes": [
            {
                "scene_id": s.scene_id,
                "date": s.scene_date,
                "metric_a": s.metric_a,
                "metric_b": s.metric_b,
                "diff": s.diff,
                "n_patches": s.n_patches,
                "n_cells": s.n_cells,
            }
            for s in result.scenes
        ],
        "dropped_scenes": [
            {"scene_id": x.scene_id, "reason": x.reason, "affected_runs": list(x.affected_runs)}
            for x in result.dropped_scenes
        ],
        "allow_scene_mismatch": result.allow_scene_mismatch,
        "bootstrap": {
            "n_bootstrap": result.n_bootstrap,
            "rng_seed": result.rng_seed,
            "method": "percentile",
            "unit": "scene",
        },
        "point_estimate": result.point_estimate,
        "ci_95": [result.ci_low, result.ci_high],
        "fractions": {
            "a_better": result.frac_a_better,
            "tie": result.frac_tie,
            "b_better": result.frac_b_better,
        },
        "seed_diagnostics": {
            "kind": "pooled_overall_metrics",
            "metric_a": list(d.metric_a),
            "metric_b": list(d.metric_b),
            "mean_a": d.mean_a,
            "mean_b": d.mean_b,
            "sd_a_ddof1": d.sd_a,
            "sd_b_ddof1": d.sd_b,
            "diffs_a_minus_b": list(d.diffs),
            "n_positive": d.n_positive,
            "n_negative": d.n_negative,
            "n_zero": d.n_zero,
            "sign_consistency": d.sign_consistency,
        },
    }


def render_markdown(result: ComparisonResult) -> str:
    d = result.seed_diagnostics
    a, b = result.arm_a.arm, result.arm_b.arm
    n = len(result.scenes)
    lines = [
        f"# Paired scene bootstrap: {a} vs {b}",
        "",
        f"Split: {result.split}; metric: {result.metric} ({result.direction}); "
        f"retained scenes: {n}.",
        "",
        "## Seed summaries (pooled overall metrics, not the primary estimand)",
        "",
        "| arm | seed 0 | seed 1 | seed 2 | mean | SD (ddof=1) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, vals, mean, sd in (
        (a, d.metric_a, d.mean_a, d.sd_a),
        (b, d.metric_b, d.mean_b, d.sd_b),
    ):
        lines.append(
            f"| {name} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {mean:.4f} | {sd:.4f} |"
        )
    lines += [
        "",
        "## Paired scene difference (A - B)",
        "",
        f"Point estimate: {result.point_estimate:.4f}; 95% percentile CI "
        f"[{result.ci_low:.4f}, {result.ci_high:.4f}] "
        f"(B={result.n_bootstrap}, rng_seed={result.rng_seed}, unit=scene).",
        "",
        f"Scenes where {a} is better: {result.frac_a_better:.1%}; ties: {result.frac_tie:.1%}; "
        f"scenes where {b} is better: {result.frac_b_better:.1%}.",
        "",
        "## Per-seed differences (A - B)",
        "",
    ]
    for s, diff in zip(result.arm_a.seeds, d.diffs, strict=True):
        lines.append(f"- seed {s}: {diff:+.4f}")
    lines += ["", f"Sign consistency: {d.sign_consistency}.", ""]
    lines.append("## Dropped scenes")
    lines.append("")
    if result.dropped_scenes:
        lines.append(f"{len(result.dropped_scenes)} scene(s) dropped as whole scenes:")
        lines.append("")
        for x in result.dropped_scenes:
            lines.append(f"- {x.scene_id}: {x.reason} (runs: {', '.join(x.affected_runs)})")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: ComparisonResult, output_prefix: Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.parent / (prefix.name + ".json")
    md_path = prefix.parent / (prefix.name + ".md")
    json_path.write_text(
        json.dumps(result_to_dict(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path
