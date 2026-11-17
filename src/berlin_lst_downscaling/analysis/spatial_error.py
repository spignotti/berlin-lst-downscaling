"""Spatial error analysis of validation runs (exploratory family).

Five descriptive steps over a list of runs, validation split only:

1. Error decomposition per scene: scene bias plus centred residuals.
2. MAE, bias and RMSE per Stadtstruktur group (proposal B, area-weighted shares
   of the 100 m cell), with a minimum-support rule, and the paired difference
   A - B per group under the scene bootstrap.
3. Moran's I of the centred 100 m residuals per scene.
4. Per-scene table of the paired difference A - B next to scene properties,
   with the cell-pooled and the scene-equal estimator side by side.
5. Paired scene bootstrap between two single-run reference models (the random
   forest stages), which ``scene_bootstrap.compare_arms`` refuses by design.

Steps 4 and 5 read the per-patch exports (``scene_bootstrap`` schema 2). Steps
1-3 need the per-cell export (``modeling.cell_exports``), whose constants are
kept literal here so this module imports neither torch nor the modeling
package. Everything in this module is exploratory: it reports estimates and
intervals and draws no verdict about a model.

The scene is the only resampling unit. ``scene_bootstrap_ci`` draws the same
index matrix as ``scene_bootstrap.compare_arms`` (same generator, seed and
batching), so a mean difference gets the identical percentile interval.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.analysis import scene_bootstrap as sb

SPLIT = "validation"
# modeling.cell_exports constants, kept literal (a test pins them to the source).
CELL_EXPORT_SCHEMA_VERSION = 1
CELL_EXPORT_METADATA_KEY = b"cell_export"
CELL_EXPORT_COLUMNS: tuple[str, ...] = (
    "scene_id",
    "patch_id",
    "row",
    "col",
    "target_k",
    "prediction_k",
)
PATCH_EXPORT_FILENAME = "eval_validation_patches.parquet"
CELL_EXPORT_FILENAME = "eval_validation_cells.parquet"

# Defaults. Each is a reported choice, not a decision of this module.
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_RNG_SEED = 0
DEFAULT_LEVEL = 0.95
DEFAULT_MIN_EFFECTIVE_CELLS = 50.0  # docs/stadtstruktur.md: sparse under 50 effective cells
DEFAULT_MIN_SCENES = 10
DEFAULT_SHARE_COLUMN = "share_of_cell"
DEFAULT_MIN_SHARE = 0.0
DEFAULT_NEIGHBOURHOOD = "queen"
DEFAULT_MORAN_WEIGHTS = "row"

SHARE_COLUMNS: tuple[str, ...] = ("share_of_cell", "share_of_covered")
NEIGHBOURHOODS: tuple[str, ...] = ("rook", "queen")
MORAN_WEIGHTS: tuple[str, ...] = ("row", "binary")
Neighbourhood = Literal["rook", "queen"]
MoranWeights = Literal["row", "binary"]

EXPLORATORY_NOTE = (
    "Exploratory family: not part of the pre-declared comparison families A-C. "
    "Intervals are unadjusted 95 % percentile intervals; no verdict is derived from them."
)

_BATCH_INDICES = 1_000_000  # scene_bootstrap._BATCH_INDICES; a test pins the equivalence
_SCENE_DATE = re.compile(r"^L[A-Z]\d{2}_[A-Z0-9]+_\d{6}_(\d{4})(\d{2})(\d{2})_")
_SCENE_OVERPASS = re.compile(r"^(L[A-Z]\d{2})_[A-Z0-9]+_(\d{3})\d{3}_(\d{8})_")
OVERPASS_NOTE = (
    "Sensitivity, not the frozen rule: scenes of one platform, path and date (adjacent rows of "
    "one overpass) are resampled together as one unit. The frozen rule resamples scenes."
)
_SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


class SpatialErrorError(ValueError):
    """Raised for any invalid input or contract violation in this module."""


# -------------------------------------------------------------------- table helpers


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return a column as a float64 array (``None`` becomes NaN)."""
    return np.asarray(frame[column].to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)


def _labels(frame: pd.DataFrame, column: str) -> list[str]:
    return [str(value) for value in frame[column].tolist()]


def _where(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Rows of ``frame`` where ``mask`` holds, with a fresh index."""
    return frame.take(np.flatnonzero(mask)).reset_index(drop=True)


def _aggregate(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    mean: Sequence[str] = (),
    total: Sequence[str] = (),
    first: Sequence[str] = (),
) -> pd.DataFrame:
    """Group ``frame`` by string ``keys``; rows come back sorted by the keys.

    ``mean`` and ``total`` columns are averaged or summed, ``first`` columns keep
    the first value of the group, and ``n`` counts the rows of the group.
    """
    combined = np.zeros(len(frame), dtype=np.int64)
    names: list[np.ndarray] = []
    for key in keys:
        codes, uniques = pd.factorize(np.asarray(_labels(frame, key), dtype=object), sort=True)
        combined = combined * len(uniques) + np.asarray(codes, dtype=np.int64)
        names.append(np.asarray(uniques, dtype=object))
    groups, first_index, inverse = np.unique(combined, return_index=True, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(groups)).astype(np.float64)
    out: dict[str, Any] = {}
    remainder = groups.copy()
    for key, uniques in zip(reversed(list(keys)), reversed(names), strict=True):
        out[key] = uniques[remainder % len(uniques)]
        remainder = remainder // len(uniques)
    out = {key: out[key] for key in keys}
    for column in mean:
        out[column] = np.bincount(inverse, weights=_values(frame, column)) / counts
    for column in total:
        out[column] = np.bincount(inverse, weights=_values(frame, column))
    for column in first:
        out[column] = frame[column].to_numpy()[first_index]
    out["n"] = counts.astype(np.int64)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    ci_low: float
    ci_high: float
    n_units: int
    n_bootstrap: int
    rng_seed: int
    level: float

    def as_dict(self, unit: str = "scene") -> dict[str, Any]:
        return {
            "point_estimate": self.point,
            "ci": [self.ci_low, self.ci_high],
            "level": self.level,
            "n_units": self.n_units,
            "n_bootstrap": self.n_bootstrap,
            "rng_seed": self.rng_seed,
            "method": "percentile",
            "unit": unit,
        }


def scene_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    level: float = DEFAULT_LEVEL,
) -> BootstrapCI:
    """Percentile interval of a per-scene statistic under scene resampling.

    Without ``weights`` the statistic is the mean of ``values`` (the scene-equal
    estimator); the draws equal those of ``scene_bootstrap.compare_arms``. With
    ``weights`` it is the ratio ``sum(w * v) / sum(w)`` (the cell-pooled
    estimator), recomputed on every resampled set of scenes.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise SpatialErrorError("n_bootstrap must be a positive int")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or rng_seed < 0:
        raise SpatialErrorError("rng_seed must be a non-negative int")
    if not 0.0 < level < 1.0:
        raise SpatialErrorError("level must lie in (0, 1)")
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size < 2:
        raise SpatialErrorError("the scene bootstrap needs at least two scenes")
    if not np.isfinite(v).all():
        raise SpatialErrorError("scene values must be finite")
    w: np.ndarray | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != v.shape or not np.isfinite(w).all() or (w <= 0).any():
            raise SpatialErrorError("weights must be finite, positive and match the values")
    n = int(v.size)
    rng = np.random.default_rng(rng_seed)
    batch = max(1, _BATCH_INDICES // n)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        b = min(batch, n_bootstrap - done)
        idx = rng.integers(0, n, size=(b, n))
        if w is None:
            reps[done : done + b] = v[idx].mean(axis=1)
        else:
            reps[done : done + b] = (w[idx] * v[idx]).sum(axis=1) / w[idx].sum(axis=1)
        done += b
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha], method="linear")
    point = float(v.mean()) if w is None else float((w * v).sum() / w.sum())
    return BootstrapCI(point, float(lo), float(hi), n, n_bootstrap, rng_seed, level)


def unit_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    units: Sequence[str],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    level: float = DEFAULT_LEVEL,
) -> BootstrapCI:
    """Like ``scene_bootstrap_ci``, but scenes that share a unit label are drawn together.

    The estimand stays the same (mean of ``values``, or the ``weights`` ratio);
    only the resampling unit changes. With one scene per unit this is
    ``scene_bootstrap_ci``.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or len(units) != v.size or w.shape != v.shape:
        raise SpatialErrorError("values, units and weights must have one entry per scene")
    codes, names = pd.factorize(np.asarray(units, dtype=object), sort=True)
    denominator = np.bincount(codes, weights=w, minlength=len(names))
    numerator = np.bincount(codes, weights=w * v, minlength=len(names))
    return scene_bootstrap_ci(
        numerator / denominator,
        weights=denominator,
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
        level=level,
    )


def overpass_sensitivity_ci(
    scene_ids: Sequence[str],
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> dict[str, Any]:
    """Scene-equal mean of ``values`` with the overpass as resampling unit (sensitivity).

    The point estimate is the scene-equal mean of the frozen rule; only the unit that
    is resampled changes (``overpass_id``). With fewer than two overpasses there is no
    interval and the entry says so instead of failing.
    """
    units = [overpass_id(scene_id) for scene_id in scene_ids]
    shared: dict[str, list[str]] = {}
    for scene_id, unit in zip(scene_ids, units, strict=True):
        shared.setdefault(unit, []).append(scene_id)
    entry: dict[str, Any] = {
        "note": OVERPASS_NOTE,
        "unit": "overpass",
        "n_scenes": len(units),
        "n_overpasses": len(shared),
        "overpasses_with_several_scenes": {
            name: sorted(ids) for name, ids in sorted(shared.items()) if len(ids) > 1
        },
    }
    if len(shared) < 2:
        return {**entry, "available": False, "reason": "fewer than two overpasses"}
    ci = unit_bootstrap_ci(values, units, n_bootstrap=n_bootstrap, rng_seed=rng_seed)
    return {**entry, "available": True, **ci.as_dict("overpass")}


# -------------------------------------------------------------------- scene helpers


def scene_date(scene_id: str) -> date:
    """Return the acquisition date encoded in a Landsat Collection 2 product id."""
    match = _SCENE_DATE.match(scene_id)
    if match is None:
        raise SpatialErrorError(f"scene id {scene_id!r} carries no Landsat acquisition date")
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError as exc:
        raise SpatialErrorError(f"scene id {scene_id!r} has an invalid date") from exc


def overpass_id(scene_id: str) -> str:
    """Return ``<platform>_<path>_<date>``: equal for adjacent rows of one overpass."""
    match = _SCENE_OVERPASS.match(scene_id)
    if match is None:
        raise SpatialErrorError(f"scene id {scene_id!r} carries no Landsat path and date")
    return f"{match[1]}_{match[2]}_{match[3]}"


def season_of(day: date) -> str:
    """Meteorological season (DJF, MAM, JJA, SON), as in ``data/eda/rules.py``."""
    for name, months in _SEASON_MONTHS.items():
        if day.month in months:
            return name
    raise SpatialErrorError(f"no season for month {day.month}")


def load_scene_properties(path: Path) -> pd.DataFrame:
    """Read per-scene LST statistics from the EDA table ``lst_distribution.parquet``.

    Only validation rows are kept; rows of every other split are dropped before
    anything is computed. Returns ``scene_id``, ``scene_mean_lst_k``,
    ``scene_std_lst_k`` and ``scene_eligible_cells`` (all eligible 100 m cells of
    the scene, not only the cells of the evaluated patches).
    """
    frame = pq.read_table(path).to_pandas()
    need = {"value", "weighting", "grain", "scene_id", "split", "n", "mean", "std"}
    missing = sorted(need - set(frame.columns))
    if missing:
        raise SpatialErrorError(f"{path}: scene properties lack columns {missing}")
    keep = (
        (frame["split"] == SPLIT)
        & (frame["value"] == "lst_k")
        & (frame["weighting"] == "observation")
        & (frame["grain"] == "scene")
    )
    out = frame.loc[keep, ["scene_id", "mean", "std", "n"]].rename(
        columns={"mean": "scene_mean_lst_k", "std": "scene_std_lst_k", "n": "scene_eligible_cells"}
    )
    if out["scene_id"].duplicated().any():
        raise SpatialErrorError(f"{path}: a validation scene occurs more than once")
    return out.reset_index(drop=True)


# ------------------------------------------------------- step 4: pooled vs scene-equal


def scene_difference_table(
    result: sb.ComparisonResult, scene_properties: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per scene: A - B next to the scene's date, support and LST level."""
    if result.split != SPLIT or result.metric != "mae_100m":
        raise SpatialErrorError("the scene table needs a validation mae_100m comparison")
    rows = []
    for scene in result.scenes:
        day = scene_date(scene.scene_id)
        rows.append(
            {
                "scene_id": scene.scene_id,
                "overpass": overpass_id(scene.scene_id),
                "date": day.isoformat(),
                "season": season_of(day),
                "n_patches": scene.n_patches,
                "n_cells": scene.n_cells,
                "mae_a": scene.metric_a,
                "mae_b": scene.metric_b,
                "diff": scene.diff,
            }
        )
    table = pd.DataFrame(rows)
    n_scenes = len(table)
    table["cell_share"] = table["n_cells"] / table["n_cells"].sum()
    # pooled - scene-equal = sum_s (w_s - 1/S) * d_s, exactly.
    table["gap_contribution"] = (table["cell_share"] - 1.0 / n_scenes) * table["diff"]
    if scene_properties is not None:
        table = table.merge(scene_properties, on="scene_id", how="left", validate="one_to_one")
    return table.sort_values("date", kind="stable").reset_index(drop=True)


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"n": n, "pearson": None, "spearman": None}
    rank_x = pd.Series(x[ok]).rank().to_numpy()
    rank_y = pd.Series(y[ok]).rank().to_numpy()
    return {
        "n": n,
        "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
        "spearman": float(np.corrcoef(rank_x, rank_y)[0, 1]),
    }


def pooled_vs_scene_equal(
    table: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Put the cell-pooled and the scene-equal estimator of A - B side by side."""
    diff = table["diff"].to_numpy(dtype=np.float64)
    cells = table["n_cells"].to_numpy(dtype=np.float64)
    mae_a = table["mae_a"].to_numpy(dtype=np.float64)
    mae_b = table["mae_b"].to_numpy(dtype=np.float64)
    equal = scene_bootstrap_ci(diff, n_bootstrap=n_bootstrap, rng_seed=rng_seed, level=level)
    pooled = scene_bootstrap_ci(
        diff, weights=cells, n_bootstrap=n_bootstrap, rng_seed=rng_seed, level=level
    )
    level_equal_b = float(mae_b.mean())
    level_pooled_b = float((cells * mae_b).sum() / cells.sum())
    overpasses = [str(name) for name in table["overpass"]]
    kwargs: dict[str, Any] = {"n_bootstrap": n_bootstrap, "rng_seed": rng_seed, "level": level}
    shared = table.groupby("overpass", sort=True)["scene_id"].apply(list)
    drivers = {"n_cells": cells, "mae_b": mae_b}
    for column in ("scene_mean_lst_k", "scene_std_lst_k"):
        if column in table.columns:
            drivers[column] = table[column].to_numpy(dtype=np.float64)
    seasons = []
    for season, part in table.groupby("season", sort=True):
        part_cells = part["n_cells"].to_numpy(dtype=np.float64)
        part_diff = part["diff"].to_numpy(dtype=np.float64)
        seasons.append(
            {
                "season": str(season),
                "n_scenes": int(len(part)),
                "n_cells": int(part_cells.sum()),
                "cell_share": float(part_cells.sum() / cells.sum()),
                "scene_equal_diff": float(part_diff.mean()),
                "pooled_diff": float((part_cells * part_diff).sum() / part_cells.sum()),
                "gap_contribution": float(_values(part, "gap_contribution").sum()),
            }
        )
    return {
        "n_scenes": int(len(table)),
        "n_cells": int(cells.sum()),
        "scene_equal": {
            **equal.as_dict(),
            "mae_a": float(mae_a.mean()),
            "mae_b": level_equal_b,
            "relative_to_b": equal.point / level_equal_b,
        },
        "cell_pooled": {
            **pooled.as_dict(),
            "mae_a": float((cells * mae_a).sum() / cells.sum()),
            "mae_b": level_pooled_b,
            "relative_to_b": pooled.point / level_pooled_b,
        },
        "gap_pooled_minus_scene_equal": pooled.point - equal.point,
        "fraction_scenes_a_better": float((diff < 0).mean()),
        "correlation_of_diff_with": {k: _correlation(v, diff) for k, v in drivers.items()},
        "by_season": seasons,
        "sensitivity_overpass_units": {
            "note": OVERPASS_NOTE,
            "overpasses_with_several_scenes": {
                str(name): scenes for name, scenes in shared.items() if len(scenes) > 1
            },
            "scene_equal": unit_bootstrap_ci(diff, overpasses, **kwargs).as_dict("overpass"),
            "cell_pooled": unit_bootstrap_ci(diff, overpasses, weights=cells, **kwargs).as_dict(
                "overpass"
            ),
        },
    }


# ------------------------------------------------------- step 5: single-run pair


def _single_run(table: pa.Table, name: str) -> tuple[str, str]:
    arms = set(table.column("arm").to_pylist())
    runs = set(table.column("run_id").to_pylist())
    splits = set(table.column("split").to_pylist())
    if len(arms) != 1 or len(runs) != 1:
        raise SpatialErrorError(f"{name}: expected exactly one arm and one run")
    if splits != {SPLIT}:
        raise SpatialErrorError(f"{name}: only the validation split is accepted")
    return str(next(iter(arms))), str(next(iter(runs)))


def compare_single_runs(
    table_a: pa.Table,
    table_b: pa.Table,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Paired scene bootstrap of ``mae_100m`` between two single runs (A - B).

    Inputs are per-patch export tables as ``scene_bootstrap.load_patch_exports``
    returns them. Both runs must cover identical patches with identical valid
    cells; the per-scene MAE is cell-pooled within the scene.
    """
    arm_a, run_a = _single_run(table_a, "run A")
    arm_b, run_b = _single_run(table_b, "run B")
    # A random forest run shares one run id across its stage exports; the arm tells them apart.
    if (arm_a, run_a) == (arm_b, run_b):
        raise SpatialErrorError("run A and run B are the same run")

    def support(table: pa.Table) -> dict[str, tuple[str, int]]:
        return dict(
            zip(
                table.column("patch_id").to_pylist(),
                zip(
                    table.column("scene_id").to_pylist(),
                    table.column("valid_cells").to_pylist(),
                    strict=True,
                ),
                strict=True,
            )
        )

    if support(table_a) != support(table_b):
        raise SpatialErrorError("the two runs differ in patches, scenes or valid cells")
    scenes_a = sb.aggregate_scenes(table_a).to_pandas().set_index("scene_id").sort_index()
    scenes_b = sb.aggregate_scenes(table_b).to_pandas().set_index("scene_id").sort_index()
    mae_a = scenes_a["mae_100m"].to_numpy(dtype=np.float64)
    mae_b = scenes_b["mae_100m"].to_numpy(dtype=np.float64)
    cells = scenes_a["n_cells"].to_numpy(dtype=np.float64)
    diff = mae_a - mae_b
    equal = scene_bootstrap_ci(diff, n_bootstrap=n_bootstrap, rng_seed=rng_seed, level=level)
    pooled = scene_bootstrap_ci(
        diff, weights=cells, n_bootstrap=n_bootstrap, rng_seed=rng_seed, level=level
    )
    return {
        "metric": "mae_100m",
        "direction": "lower_is_better",
        "split": SPLIT,
        "arm_a": {"arm": arm_a, "run_id": run_a},
        "arm_b": {"arm": arm_b, "run_id": run_b},
        "note": "single run per arm: no seed spread exists, the interval covers scenes only",
        "n_patches": int(table_a.num_rows),
        "n_cells": int(cells.sum()),
        "scene_equal": {
            **equal.as_dict(),
            "mae_a": float(mae_a.mean()),
            "mae_b": float(mae_b.mean()),
            "relative_to_b": equal.point / float(mae_b.mean()),
        },
        "cell_pooled": {
            **pooled.as_dict(),
            "mae_a": float((cells * mae_a).sum() / cells.sum()),
            "mae_b": float((cells * mae_b).sum() / cells.sum()),
            "relative_to_b": pooled.point / float((cells * mae_b).sum() / cells.sum()),
        },
        "fractions": {
            "a_better": float((diff < 0).mean()),
            "tie": float((diff == 0).mean()),
            "b_better": float((diff > 0).mean()),
        },
        "scenes": [
            {
                "scene_id": str(scene),
                "mae_a": float(a),
                "mae_b": float(b),
                "diff": float(d),
                "n_cells": int(c),
            }
            for scene, a, b, d, c in zip(scenes_a.index, mae_a, mae_b, diff, cells, strict=True)
        ],
    }


# ------------------------------------------------------------------- cell exports


@dataclass(frozen=True)
class RunIdentity:
    arm: str
    seed: int | None
    run_id: str


@dataclass(frozen=True)
class CellErrors:
    """Validation cells shared by all runs, with one signed error row per run.

    ``cells`` holds ``scene_id``, ``patch_id``, ``row``, ``col`` and ``target_k``
    sorted by (scene, row, col). ``errors[r, i]`` is prediction minus target of
    run ``r`` on cell ``i`` in Kelvin (positive: the model is too warm).
    """

    cells: pd.DataFrame
    runs: tuple[RunIdentity, ...]
    errors: np.ndarray

    def arms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(run.arm for run in self.runs))

    def run_indices(self, arm: str) -> list[int]:
        return [i for i, run in enumerate(self.runs) if run.arm == arm]


def _read_cell_export(path: Path) -> tuple[RunIdentity, pd.DataFrame]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cell export not found or not a file: {path}")
    raw = (pq.read_schema(path).metadata or {}).get(CELL_EXPORT_METADATA_KEY)
    if raw is None:
        raise SpatialErrorError(f"{path}: missing Parquet footer key {CELL_EXPORT_METADATA_KEY!r}")
    footer = json.loads(raw)
    if footer.get("schema_version") != CELL_EXPORT_SCHEMA_VERSION:
        raise SpatialErrorError(f"{path}: footer schema_version != {CELL_EXPORT_SCHEMA_VERSION}")
    if footer.get("split") != SPLIT:
        raise SpatialErrorError(
            f"{path}: footer split {footer.get('split')!r} rejected: validation-only analysis"
        )
    if footer.get("prediction_clamp_k") != list(sb.PREDICTION_CLAMP_K):
        raise SpatialErrorError(f"{path}: footer prediction_clamp_k differs from the shared clamp")
    arm, run_id, seed = footer.get("arm"), footer.get("run_id"), footer.get("seed")
    if not isinstance(arm, str) or not arm or not isinstance(run_id, str) or not run_id:
        raise SpatialErrorError(f"{path}: footer lacks arm or run_id")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SpatialErrorError(f"{path}: footer seed must be an int or null")
    table = pq.read_table(path)
    missing = [c for c in CELL_EXPORT_COLUMNS if c not in table.column_names]
    if missing:
        raise SpatialErrorError(f"{path}: missing columns {missing}")
    frame = table.select(list(CELL_EXPORT_COLUMNS)).to_pandas()
    if frame.empty:
        raise SpatialErrorError(f"{path}: export has no rows")
    if frame.isna().any().any():
        raise SpatialErrorError(f"{path}: export contains nulls")
    values = frame[["target_k", "prediction_k"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SpatialErrorError(f"{path}: non-finite target or prediction")
    if frame.duplicated(["scene_id", "row", "col"]).any():
        raise SpatialErrorError(f"{path}: a (scene_id, row, col) cell occurs more than once")
    frame = frame.sort_values(["scene_id", "row", "col"], kind="stable").reset_index(drop=True)
    return RunIdentity(arm, seed, run_id), frame


def load_cell_exports(paths: Sequence[Path]) -> CellErrors:
    """Load one cell export per run and align them on identical cells."""
    if not paths:
        raise SpatialErrorError("no cell export files given")
    runs: list[RunIdentity] = []
    errors: list[np.ndarray] = []
    base: pd.DataFrame | None = None
    for path in paths:
        identity, frame = _read_cell_export(Path(path))
        if any(identity.run_id == run.run_id for run in runs):
            raise SpatialErrorError(f"run {identity.run_id!r} was given twice")
        if any((identity.arm, identity.seed) == (run.arm, run.seed) for run in runs):
            raise SpatialErrorError(f"arm {identity.arm!r} seed {identity.seed} was given twice")
        keys = frame[["scene_id", "patch_id", "row", "col"]]
        if base is None:
            base = pd.DataFrame(
                {name: frame[name] for name in ("scene_id", "patch_id", "row", "col", "target_k")}
            )
        else:
            if len(keys) != len(base) or not keys.equals(
                base[["scene_id", "patch_id", "row", "col"]]
            ):
                raise SpatialErrorError(
                    f"run {identity.run_id!r} covers different cells than {runs[0].run_id!r}"
                )
            if not np.array_equal(frame["target_k"].to_numpy(), base["target_k"].to_numpy()):
                raise SpatialErrorError(
                    f"run {identity.run_id!r} carries different targets than {runs[0].run_id!r}"
                )
        runs.append(identity)
        errors.append(
            frame["prediction_k"].to_numpy(dtype=np.float64)
            - frame["target_k"].to_numpy(dtype=np.float64)
        )
    if base is None:
        raise SpatialErrorError("no cell export files given")
    return CellErrors(base, tuple(runs), np.vstack(errors))


def check_cells_against_patch_export(
    cell_errors: CellErrors, run_id: str, patch_table: pa.Table, *, rtol: float = 1e-4
) -> None:
    """Fail unless a run's cell errors reproduce its per-patch export sums."""
    index = [i for i, run in enumerate(cell_errors.runs) if run.run_id == run_id]
    if len(index) != 1:
        raise SpatialErrorError(f"run {run_id!r} is not among the loaded cell exports")
    if set(patch_table.column("run_id").to_pylist()) != {run_id}:
        raise SpatialErrorError(f"the patch export does not belong to run {run_id!r}")
    codes, names = pd.factorize(
        np.asarray(_labels(cell_errors.cells, "patch_id"), dtype=object), sort=True
    )
    absolute = np.bincount(codes, weights=np.abs(cell_errors.errors[index[0]]))
    counts = np.bincount(codes)
    patch_ids = [str(value) for value in patch_table.column("patch_id").to_pylist()]
    order = np.argsort(np.asarray(patch_ids, dtype=object))
    if [patch_ids[i] for i in order] != [str(name) for name in names]:
        raise SpatialErrorError(f"{run_id}: cell and patch exports cover different patches")
    valid = np.asarray(patch_table.column("valid_cells").to_pylist(), dtype=np.int64)[order]
    if not np.array_equal(counts, valid):
        raise SpatialErrorError(f"{run_id}: valid cells differ between cell and patch export")
    expected = np.asarray(patch_table.column("abs_error_sum").to_pylist(), dtype=np.float64)[order]
    if not np.allclose(absolute, expected, rtol=rtol, atol=1e-3):
        raise SpatialErrorError(
            f"{run_id}: absolute cell errors do not reproduce the per-patch abs_error_sum"
        )


# --------------------------------------------------- step 1: scene bias + centred


def _scene_codes(cell_errors: CellErrors) -> tuple[np.ndarray, list[str]]:
    codes, names = pd.factorize(cell_errors.cells["scene_id"], sort=True)
    return np.asarray(codes), [str(name) for name in names]


def _scene_bias(cell_errors: CellErrors) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return ``(bias[r, s], counts[s], scene names, scene code per cell)``."""
    codes, names = _scene_codes(cell_errors)
    counts = np.bincount(codes, minlength=len(names)).astype(np.float64)
    bias = np.vstack(
        [
            np.bincount(codes, weights=row, minlength=len(names)) / counts
            for row in cell_errors.errors
        ]
    )
    return bias, counts, names, codes


def centred_errors(cell_errors: CellErrors) -> np.ndarray:
    """Return the errors minus the scene bias of the same run, shape ``(runs, cells)``."""
    bias, _counts, _names, codes = _scene_bias(cell_errors)
    return cell_errors.errors - bias[:, codes]


def scene_decomposition(cell_errors: CellErrors) -> pd.DataFrame:
    """Split every (run, scene) error into the scene bias and the centred residual.

    ``mse = bias**2 + centred_rmse**2`` holds exactly per (run, scene);
    ``bias_share_of_mse`` is ``bias**2 / mse``.
    """
    bias, counts, names, codes = _scene_bias(cell_errors)
    centred = cell_errors.errors - bias[:, codes]
    n = len(names)
    rows = []
    for r, run in enumerate(cell_errors.runs):
        error = cell_errors.errors[r]
        mae = np.bincount(codes, weights=np.abs(error), minlength=n) / counts
        mse = np.bincount(codes, weights=error**2, minlength=n) / counts
        centred_mae = np.bincount(codes, weights=np.abs(centred[r]), minlength=n) / counts
        centred_mse = np.bincount(codes, weights=centred[r] ** 2, minlength=n) / counts
        for s, scene in enumerate(names):
            rows.append(
                {
                    "arm": run.arm,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "scene_id": scene,
                    "n_cells": int(counts[s]),
                    "bias_k": float(bias[r, s]),
                    "mae_k": float(mae[s]),
                    "rmse_k": float(math.sqrt(mse[s])),
                    "centred_mae_k": float(centred_mae[s]),
                    "centred_rmse_k": float(math.sqrt(centred_mse[s])),
                    "bias_share_of_mse": float(bias[r, s] ** 2 / mse[s]) if mse[s] > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


_DECOMPOSITION_METRICS = (
    "bias_k",
    "mae_k",
    "rmse_k",
    "centred_mae_k",
    "centred_rmse_k",
    "bias_share_of_mse",
)


def arm_scene_decomposition(per_run: pd.DataFrame) -> pd.DataFrame:
    """Average the per-run scene metrics over the seeds of each arm."""
    out = _aggregate(per_run, ("arm", "scene_id"), mean=_DECOMPOSITION_METRICS, first=("n_cells",))
    return out.rename(columns={"n": "n_seeds"})


def decomposition_summary(per_run: pd.DataFrame) -> list[dict[str, Any]]:
    """Per arm: scene-equal means and the cell-pooled split of the MSE."""
    out = []
    for arm, part in per_run.groupby("arm", sort=True):
        scenes = _aggregate(part, ("scene_id",), mean=_DECOMPOSITION_METRICS)
        bias = _values(scenes, "bias_k")
        between, within = [], []
        for _run, run_part in part.groupby("run_id", sort=True):
            cells = _values(run_part, "n_cells")
            between.append(float((cells * _values(run_part, "bias_k") ** 2).sum() / cells.sum()))
            within.append(
                float((cells * _values(run_part, "centred_rmse_k") ** 2).sum() / cells.sum())
            )
        between_mean = float(np.mean(between))
        within_mean = float(np.mean(within))
        total = between_mean + within_mean
        out.append(
            {
                "arm": str(arm),
                "n_seeds": len(between),
                "n_scenes": int(len(scenes)),
                "scene_equal": {
                    "bias_k": float(bias.mean()),
                    "abs_bias_k": float(np.abs(bias).mean()),
                    "bias_sd_over_scenes_k": float(bias.std(ddof=1)) if bias.size > 1 else None,
                    "mae_k": float(_values(scenes, "mae_k").mean()),
                    "rmse_k": float(_values(scenes, "rmse_k").mean()),
                    "centred_mae_k": float(_values(scenes, "centred_mae_k").mean()),
                    "centred_rmse_k": float(_values(scenes, "centred_rmse_k").mean()),
                    "bias_share_of_mse": float(_values(scenes, "bias_share_of_mse").mean()),
                },
                "cell_pooled_mse": {
                    "scene_bias_k2": between_mean,
                    "centred_k2": within_mean,
                    "total_k2": total,
                    "scene_bias_share": between_mean / total if total > 0 else None,
                },
            }
        )
    return out


# ------------------------------------------------------ step 2: Stadtstruktur groups


def group_weights(
    shares: pd.DataFrame,
    typ_to_group: Mapping[int, str],
    *,
    share_column: str = DEFAULT_SHARE_COLUMN,
    min_share: float = DEFAULT_MIN_SHARE,
) -> pd.DataFrame:
    """Turn the per-``typ`` share table of one vintage into cell weights per group.

    The weight of a group in a cell is the sum of ``share_column`` over the
    group's types. ``share_of_cell`` is the area share of the whole 100 m cell
    (street land and other unmapped area stay unassigned); ``share_of_covered``
    rescales to the mapped area of the cell. A group whose weight in a cell is
    below ``min_share`` is dropped there; ``min_share = 0`` keeps every mixed
    cell, a value near 1 keeps only cells the group dominates.
    """
    if share_column not in SHARE_COLUMNS:
        raise SpatialErrorError(f"share_column must be one of {SHARE_COLUMNS}")
    if not 0.0 <= min_share <= 1.0:
        raise SpatialErrorError("min_share must lie in [0, 1]")
    missing = sorted({"vintage", "row", "col", "typ", share_column} - set(shares.columns))
    if missing:
        raise SpatialErrorError(f"share table lacks columns {missing}")
    if shares["vintage"].nunique() != 1:
        raise SpatialErrorError("the share table must hold exactly one vintage")
    unknown = sorted(set(shares["typ"].astype(int)) - set(typ_to_group))
    if unknown:
        raise SpatialErrorError(f"share table holds typ codes without a group: {unknown}")
    frame = pd.DataFrame(
        {
            "row": shares["row"].astype(np.int64),
            "col": shares["col"].astype(np.int64),
            "group": [typ_to_group[int(typ)] for typ in shares["typ"].tolist()],
            "weight": shares[share_column].astype(np.float64),
        }
    )
    if not np.isfinite(frame["weight"]).all() or (frame["weight"] < 0).any():
        raise SpatialErrorError(f"{share_column} must be finite and non-negative")
    frame["cell"] = [f"{row}_{col}" for row, col in zip(frame["row"], frame["col"], strict=True)]
    weights = _aggregate(frame, ("cell", "group"), total=("weight",), first=("row", "col"))
    per_cell = _aggregate(weights, ("cell",), total=("weight",))
    if (_values(per_cell, "weight") > 1.0 + 1e-6).any():
        raise SpatialErrorError(f"{share_column} sums to more than 1 in a cell")
    weight = _values(weights, "weight")
    weights = _where(weights, (weight > 0) & (weight >= min_share))
    return pd.DataFrame({name: weights[name] for name in ("row", "col", "group", "weight")})


def group_scene_metrics(
    cell_errors: CellErrors,
    weights: pd.DataFrame,
    *,
    min_effective_cells: float = DEFAULT_MIN_EFFECTIVE_CELLS,
) -> pd.DataFrame:
    """Weighted bias, MAE and RMSE per (run, scene, group).

    ``effective_cells`` is the summed weight (area in units of one 100 m cell).
    ``supported`` marks scene-group pairs with at least ``min_effective_cells``;
    the support does not depend on the run. ``centred_bias_k`` is the group bias
    after the scene bias of the same run is removed.
    """
    if min_effective_cells <= 0:
        raise SpatialErrorError("min_effective_cells must be positive")
    cells = cell_errors.cells[["scene_id", "row", "col"]].copy()
    cells["cell_index"] = np.arange(len(cells))
    cells["row"] = cells["row"].astype(np.int64)
    cells["col"] = cells["col"].astype(np.int64)
    pairs = cells.merge(weights, on=["row", "col"], how="inner")
    if pairs.empty:
        raise SpatialErrorError("no evaluated cell carries a Stadtstruktur share")
    scene_code, scene_names = pd.factorize(pairs["scene_id"], sort=True)
    group_code, group_names = pd.factorize(pairs["group"], sort=True)
    n_groups = len(group_names)
    key = np.asarray(scene_code) * n_groups + np.asarray(group_code)
    n_keys = len(scene_names) * n_groups
    weight = pairs["weight"].to_numpy(dtype=np.float64)
    index = pairs["cell_index"].to_numpy()
    total = np.bincount(key, weights=weight, minlength=n_keys)
    touched = np.bincount(key, minlength=n_keys)
    present = np.nonzero(total > 0)[0]
    centred = centred_errors(cell_errors)
    frames = []
    for r, run in enumerate(cell_errors.runs):
        error = cell_errors.errors[r][index]

        def mean(values: np.ndarray) -> np.ndarray:
            return (
                np.bincount(key, weights=weight * values, minlength=n_keys)[present]
                / total[present]
            )

        mse = mean(error**2)
        frames.append(
            pd.DataFrame(
                {
                    "arm": run.arm,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "scene_id": np.asarray(scene_names)[present // n_groups],
                    "group": np.asarray(group_names)[present % n_groups],
                    "effective_cells": total[present],
                    "cells_touched": touched[present],
                    "bias_k": mean(error),
                    "centred_bias_k": mean(centred[r][index]),
                    "mae_k": mean(np.abs(error)),
                    "mse_k2": mse,
                    "rmse_k": np.sqrt(mse),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["supported"] = out["effective_cells"] >= min_effective_cells
    return out


_GROUP_METRICS = ("bias_k", "centred_bias_k", "mae_k", "mse_k2", "rmse_k")


def _arm_group_scenes(per_run: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean metrics per (arm, group, scene) on supported pairs only."""
    supported = _where(per_run, per_run["supported"].to_numpy(dtype=bool))
    return _aggregate(
        supported, ("arm", "group", "scene_id"), mean=_GROUP_METRICS, first=("effective_cells",)
    )


def group_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, group): scene-equal and area-pooled MAE, bias and RMSE."""
    rows = []
    for arm, arm_part in _arm_group_scenes(per_run).groupby("arm", sort=True):
        for group, part in arm_part.groupby("group", sort=True):
            weight = _values(part, "effective_cells")
            pooled = {
                column: float((weight * _values(part, column)).sum() / weight.sum())
                for column in ("mae_k", "bias_k", "mse_k2")
            }
            rows.append(
                {
                    "arm": str(arm),
                    "group": str(group),
                    "n_scenes": int(len(part)),
                    "effective_cells": float(weight.sum()),
                    "scene_equal_mae_k": float(_values(part, "mae_k").mean()),
                    "scene_equal_bias_k": float(_values(part, "bias_k").mean()),
                    "scene_equal_centred_bias_k": float(_values(part, "centred_bias_k").mean()),
                    "scene_equal_rmse_k": float(_values(part, "rmse_k").mean()),
                    "pooled_mae_k": pooled["mae_k"],
                    "pooled_bias_k": pooled["bias_k"],
                    "pooled_rmse_k": math.sqrt(pooled["mse_k2"]),
                }
            )
    return pd.DataFrame(rows)


def group_paired_differences(
    per_run: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    *,
    min_scenes: int = DEFAULT_MIN_SCENES,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    level: float = DEFAULT_LEVEL,
) -> list[dict[str, Any]]:
    """Paired MAE difference A - B per group over the scenes that support the group.

    A group with fewer than ``min_scenes`` supported scenes is listed without an
    interval. The family is exploratory and the intervals are unadjusted.
    """
    if min_scenes < 2:
        raise SpatialErrorError("min_scenes must be at least 2")
    scenes = _arm_group_scenes(per_run)
    arms = np.asarray(_labels(scenes, "arm"), dtype=object)
    if arm_a == arm_b or not {arm_a, arm_b} <= set(arms.tolist()):
        raise SpatialErrorError(f"arms {arm_a!r} and {arm_b!r} must be two loaded arms")
    part_a, part_b = _where(scenes, arms == arm_a), _where(scenes, arms == arm_b)
    pairs = list(zip(_labels(part_a, "group"), _labels(part_a, "scene_id"), strict=True))
    if pairs != list(zip(_labels(part_b, "group"), _labels(part_b, "scene_id"), strict=True)):
        raise SpatialErrorError("the two arms differ in supported scene-group pairs")
    groups = np.asarray([group for group, _scene in pairs], dtype=object)
    out = []
    for group in sorted(set(groups.tolist())):
        select = groups == group
        mae_a = _values(part_a, "mae_k")[select]
        mae_b = _values(part_b, "mae_k")[select]
        diff = mae_a - mae_b
        entry: dict[str, Any] = {
            "group": str(group),
            "family": "exploratory",
            "n_scenes": int(diff.size),
            "mae_a": float(mae_a.mean()),
            "mae_b": float(mae_b.mean()),
            "diff": float(diff.mean()),
            "relative_to_b": float(diff.mean() / mae_b.mean()),
            "fraction_scenes_a_better": float((diff < 0).mean()),
            "ci": None,
            "interval_note": None,
        }
        if diff.size >= min_scenes:
            ci = scene_bootstrap_ci(diff, n_bootstrap=n_bootstrap, rng_seed=rng_seed, level=level)
            entry["ci"] = [ci.ci_low, ci.ci_high]
        else:
            entry["interval_note"] = f"fewer than {min_scenes} supported scenes"
        out.append(entry)
    return out


def mapped_coverage(cell_errors: CellErrors, weights: pd.DataFrame) -> dict[str, Any]:
    """How much of the evaluated cell area carries a group weight."""
    per_cell: dict[tuple[int, int], float] = {}
    for row, col, weight in zip(
        weights["row"].tolist(), weights["col"].tolist(), weights["weight"].tolist(), strict=True
    ):
        per_cell[(int(row), int(col))] = per_cell.get((int(row), int(col)), 0.0) + float(weight)
    assigned = np.asarray(
        [
            per_cell.get((int(row), int(col)), 0.0)
            for row, col in zip(
                cell_errors.cells["row"].tolist(), cell_errors.cells["col"].tolist(), strict=True
            )
        ],
        dtype=np.float64,
    )
    return {
        "evaluated_cells": int(assigned.size),
        "cells_with_a_group": int((assigned > 0).sum()),
        "assigned_area_share": float(assigned.mean()),
    }


# ----------------------------------------------------------------- step 3: Moran's I


@dataclass(frozen=True)
class MoranResult:
    i: float | None
    expected: float | None
    n_cells: int
    n_islands: int


def morans_i(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    neighbourhood: Neighbourhood = "queen",
    weights: MoranWeights = "row",
) -> MoranResult:
    """Global Moran's I of ``values`` on grid cells ``(rows, cols)``.

    Neighbours are the 4 (rook) or 8 (queen) adjacent cells that are present.
    ``weights="row"`` row-standardises (each cell's neighbours sum to 1),
    ``"binary"`` gives every adjacency weight 1. Cells without a present
    neighbour (islands) stay in the mean and the variance with zero weight.
    ``I = (N / S0) * sum_ij w_ij z_i z_j / sum_i z_i**2`` with ``z`` centred on
    the mean of ``values``; ``i`` is ``None`` when it is undefined.
    """
    if neighbourhood not in NEIGHBOURHOODS:
        raise SpatialErrorError(f"neighbourhood must be one of {NEIGHBOURHOODS}")
    if weights not in MORAN_WEIGHTS:
        raise SpatialErrorError(f"weights must be one of {MORAN_WEIGHTS}")
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if not (rows.shape == cols.shape == values.shape) or rows.ndim != 1:
        raise SpatialErrorError("rows, cols and values must be 1-D arrays of equal length")
    n = int(values.size)
    if n == 0 or not np.isfinite(values).all():
        raise SpatialErrorError("Moran's I needs finite values on at least one cell")
    r = rows - rows.min() + 1
    c = cols - cols.min() + 1
    shape = (int(r.max()) + 2, int(c.max()) + 2)
    present = np.zeros(shape, dtype=bool)
    present[r, c] = True
    if int(present.sum()) != n:
        raise SpatialErrorError("a grid cell occurs more than once")
    z = np.zeros(shape, dtype=np.float64)
    z[r, c] = values - values.mean()
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbourhood == "queen":
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    lag = np.zeros(shape, dtype=np.float64)
    degree = np.zeros(shape, dtype=np.int64)
    for dr, dc in offsets:
        lag += np.roll(z, (dr, dc), axis=(0, 1))
        degree += np.roll(present, (dr, dc), axis=(0, 1))
    lag_cells = lag[r, c]
    degree_cells = degree[r, c]
    z_cells = z[r, c]
    islands = int((degree_cells == 0).sum())
    denominator = float((z_cells**2).sum())
    if weights == "row":
        connected = degree_cells > 0
        s0 = float(connected.sum())
        numerator = float(
            (z_cells[connected] * lag_cells[connected] / degree_cells[connected]).sum()
        )
    else:
        s0 = float(degree_cells.sum())
        numerator = float((z_cells * lag_cells).sum())
    if n < 2 or s0 == 0 or denominator == 0:
        return MoranResult(None, None, n, islands)
    return MoranResult(n / s0 * numerator / denominator, -1.0 / (n - 1), n, islands)


def scene_morans(
    cell_errors: CellErrors,
    *,
    neighbourhood: Neighbourhood = "queen",
    weights: MoranWeights = "row",
) -> pd.DataFrame:
    """Moran's I of the centred residuals per (run, scene)."""
    centred = centred_errors(cell_errors)
    codes, names = _scene_codes(cell_errors)
    grid_rows = cell_errors.cells["row"].to_numpy(dtype=np.int64)
    grid_cols = cell_errors.cells["col"].to_numpy(dtype=np.int64)
    out = []
    for s, scene in enumerate(names):
        select = codes == s
        for r, run in enumerate(cell_errors.runs):
            result = morans_i(
                grid_rows[select],
                grid_cols[select],
                centred[r][select],
                neighbourhood=neighbourhood,
                weights=weights,
            )
            out.append(
                {
                    "arm": run.arm,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "scene_id": scene,
                    "n_cells": result.n_cells,
                    "n_islands": result.n_islands,
                    "moran_i": result.i,
                    "expected_i": result.expected,
                }
            )
    return pd.DataFrame(out)


def moran_summary(per_run: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Seed-mean Moran's I per (arm, scene) and its spread over scenes per arm.

    A scene whose value is undefined for one seed is undefined for the arm.
    """
    scenes = _aggregate(
        per_run, ("arm", "scene_id"), mean=("moran_i",), first=("n_cells", "n_islands")
    ).rename(columns={"n": "n_seeds"})
    summary = []
    for arm, part in scenes.groupby("arm", sort=True):
        values = _values(part, "moran_i")
        valid = values[np.isfinite(values)]
        summary.append(
            {
                "arm": str(arm),
                "n_scenes": int(values.size),
                "n_scenes_defined": int(valid.size),
                "mean": float(valid.mean()) if valid.size else None,
                "median": float(np.median(valid)) if valid.size else None,
                "min": float(valid.min()) if valid.size else None,
                "max": float(valid.max()) if valid.size else None,
            }
        )
    return scenes, summary


# -------------------------------------------------------------------------- report


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs and reported choices of one analysis run (all paths are local files)."""

    run_labels: tuple[str, ...]
    arm_a: str
    arm_b: str
    patch_export_root: Path | None = None
    cell_export_root: Path | None = None
    stadtstruktur_shares: Path | None = None
    scene_properties: Path | None = None
    reference_a: Path | None = None
    reference_b: Path | None = None
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    rng_seed: int = DEFAULT_RNG_SEED
    min_effective_cells: float = DEFAULT_MIN_EFFECTIVE_CELLS
    min_scenes: int = DEFAULT_MIN_SCENES
    share_column: str = DEFAULT_SHARE_COLUMN
    min_share: float = DEFAULT_MIN_SHARE
    neighbourhood: Neighbourhood = "queen"
    moran_weights: MoranWeights = "row"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Frame rows as JSON-ready dicts (NaN becomes null, numpy scalars become Python)."""
    out = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            clean[str(key)] = value
        out.append(clean)
    return out


def _export_paths(root: Path, labels: Sequence[str], filename: str) -> list[Path]:
    return [Path(root) / label / filename for label in labels]


def _typ_to_group() -> tuple[dict[int, str], dict[str, str]]:
    # Imported here: the class review pulls in geopandas, which steps 4 and 5 do not need.
    from berlin_lst_downscaling.data.stadtstruktur.class_review import (
        PROPOSAL_B,
        PROPOSAL_B_GROUPS,
    )

    return dict(PROPOSAL_B), {spec.group_id: spec.role for spec in PROPOSAL_B_GROUPS}


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run every step whose inputs are present and report the skipped ones."""
    if not config.run_labels or len(set(config.run_labels)) != len(config.run_labels):
        raise SpatialErrorError("run labels must be a non-empty list without duplicates")
    report: dict[str, Any] = {
        "kind": "spatial_error_analysis",
        "split": SPLIT,
        "exploratory": True,
        "note": EXPLORATORY_NOTE,
        "run_labels": list(config.run_labels),
        "pair": {"arm_a": config.arm_a, "arm_b": config.arm_b, "difference": "A - B"},
        "parameters": {
            "n_bootstrap": config.n_bootstrap,
            "rng_seed": config.rng_seed,
            "level": DEFAULT_LEVEL,
            "min_effective_cells": config.min_effective_cells,
            "min_scenes": config.min_scenes,
            "share_column": config.share_column,
            "min_share": config.min_share,
            "typ_grouping": "proposal B (data/stadtstruktur/class_review.py PROPOSAL_B)",
            "moran_neighbourhood": config.neighbourhood,
            "moran_weights": config.moran_weights,
            "sign": "error = prediction - target (positive: too warm)",
        },
        "skipped": {},
    }
    skipped: dict[str, str] = report["skipped"]

    # Step 4: per-patch exports.
    patch_tables: dict[str, pa.Table] = {}
    if config.patch_export_root is None:
        skipped["scene_table"] = "no per-patch export root given"
    else:
        for label, path in zip(
            config.run_labels,
            _export_paths(config.patch_export_root, config.run_labels, PATCH_EXPORT_FILENAME),
            strict=True,
        ):
            patch_tables[label] = sb.load_patch_exports([path], split=SPLIT)
        by_arm: dict[str, list[pa.Table]] = {}
        for table in patch_tables.values():
            by_arm.setdefault(str(table.column("arm")[0].as_py()), []).append(table)
        if not {config.arm_a, config.arm_b} <= set(by_arm):
            raise SpatialErrorError(
                f"pair {config.arm_a!r}, {config.arm_b!r} is not among the loaded arms "
                f"{sorted(by_arm)}"
            )
        comparison = sb.compare_arms(
            pa.concat_tables(by_arm[config.arm_a]),
            pa.concat_tables(by_arm[config.arm_b]),
            metric="mae_100m",
            n_bootstrap=config.n_bootstrap,
            rng_seed=config.rng_seed,
        )
        properties = (
            load_scene_properties(config.scene_properties)
            if config.scene_properties is not None
            else None
        )
        table = scene_difference_table(comparison, properties)
        seeds = comparison.seed_diagnostics
        report["scene_table"] = {
            "source": "per-patch exports",
            "arm_a": sb.result_to_dict(comparison)["arm_a"],
            "arm_b": sb.result_to_dict(comparison)["arm_b"],
            "scene_properties": (
                "EDA lst_distribution (value lst_k, weighting observation, grain scene): mean and "
                "SD over all eligible cells of the scene"
                if properties is not None
                else None
            ),
            "compare_arms": {
                "point_estimate": comparison.point_estimate,
                "ci_95": [comparison.ci_low, comparison.ci_high],
            },
            "seed_pooled_mae": {
                "a": list(seeds.metric_a),
                "b": list(seeds.metric_b),
                "diffs_a_minus_b": list(seeds.diffs),
                "sd_a_ddof1": seeds.sd_a,
                "sd_b_ddof1": seeds.sd_b,
            },
            "estimators": pooled_vs_scene_equal(
                table, n_bootstrap=config.n_bootstrap, rng_seed=config.rng_seed
            ),
            "scenes": _records(table),
        }

    # Step 5: single-run reference pair.
    if config.reference_a is None or config.reference_b is None:
        skipped["reference_pair"] = "no pair of single-run reference exports given"
    else:
        report["reference_pair"] = compare_single_runs(
            sb.load_patch_exports([config.reference_a], split=SPLIT),
            sb.load_patch_exports([config.reference_b], split=SPLIT),
            n_bootstrap=config.n_bootstrap,
            rng_seed=config.rng_seed,
        )

    # Steps 1-3: per-cell exports.
    if config.cell_export_root is None:
        reason = "no per-cell export root given (the per-patch exports carry no signed cell error)"
        for step in ("scene_decomposition", "stadtstruktur_groups", "morans_i"):
            skipped[step] = reason
        return report
    cell_errors = load_cell_exports(
        _export_paths(config.cell_export_root, config.run_labels, CELL_EXPORT_FILENAME)
    )
    if not {config.arm_a, config.arm_b} <= set(cell_errors.arms()):
        raise SpatialErrorError("the pair is not among the arms of the loaded cell exports")
    for label, patch_table in patch_tables.items():
        check_cells_against_patch_export(cell_errors, label, patch_table)
    report["cells"] = {
        "runs": [vars(run) for run in cell_errors.runs],
        "n_cells": int(len(cell_errors.cells)),
        "n_scenes": len(set(_labels(cell_errors.cells, "scene_id"))),
        "checked_against_patch_exports": sorted(patch_tables),
    }

    decomposition = scene_decomposition(cell_errors)
    report["scene_decomposition"] = {
        "summary": decomposition_summary(decomposition),
        "scenes": _records(arm_scene_decomposition(decomposition)),
    }

    morans = scene_morans(
        cell_errors, neighbourhood=config.neighbourhood, weights=config.moran_weights
    )
    moran_scenes, moran_arms = moran_summary(morans)
    report["morans_i"] = {
        "values": "scene-centred residuals",
        "summary": moran_arms,
        "scenes": _records(moran_scenes),
    }

    if config.stadtstruktur_shares is None:
        skipped["stadtstruktur_groups"] = "no Stadtstruktur share table given"
        return report
    typ_to_group, roles = _typ_to_group()
    shares = pq.read_table(config.stadtstruktur_shares).to_pandas()
    weights = group_weights(
        shares, typ_to_group, share_column=config.share_column, min_share=config.min_share
    )
    per_group = group_scene_metrics(
        cell_errors, weights, min_effective_cells=config.min_effective_cells
    )
    summary = group_summary(per_group)
    summary["role"] = [roles.get(group) for group in _labels(summary, "group")]
    differences = group_paired_differences(
        per_group,
        config.arm_a,
        config.arm_b,
        min_scenes=config.min_scenes,
        n_bootstrap=config.n_bootstrap,
        rng_seed=config.rng_seed,
    )
    for entry in differences:
        entry["role"] = roles.get(entry["group"])
    report["stadtstruktur_groups"] = {
        "family": "exploratory",
        "vintage": int(shares["vintage"].iloc[0]),
        "coverage": mapped_coverage(cell_errors, weights),
        "summary": _records(summary),
        "paired_differences": differences,
    }
    return report


# ------------------------------------------------------------------------ markdown


def _f(value: float | int | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _signed(value: float | int | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def _interval(ci: Sequence[float] | None, digits: int = 3) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _estimator_lines(block: Mapping[str, Any], a: str, b: str) -> list[str]:
    lines = [
        "| estimator | MAE A | MAE B | A - B | 95 % | % of B |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, name in (("scene_equal", "scene-equal"), ("cell_pooled", "cell-pooled")):
        e = block[key]
        lines.append(
            f"| {name} | {_f(e['mae_a'])} | {_f(e['mae_b'])} | {_signed(e['point_estimate'])} | "
            f"{_interval(e['ci'])} | {_signed(100 * e['relative_to_b'], 1)} |"
        )
    lines.append("")
    lines.append(f"A = {a}, B = {b}. Both intervals resample scenes (percentile).")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report dict of ``run_analysis`` as Markdown."""
    a, b = report["pair"]["arm_a"], report["pair"]["arm_b"]
    lines = [
        f"# Spatial error analysis, validation: {a} minus {b}",
        "",
        str(report["note"]),
        "",
        "Runs: " + ", ".join(f"`{label}`" for label in report["run_labels"]) + ".",
        "",
    ]
    if "scene_table" in report:
        block = report["scene_table"]
        est = block["estimators"]
        lines += ["## Pooled against scene-equal", ""]
        lines += _estimator_lines(est, a, b)
        lines += [
            "",
            f"Gap (pooled minus scene-equal): {_signed(est['gap_pooled_minus_scene_equal'], 4)} K. "
            f"{a} is better in {est['fraction_scenes_a_better']:.0%} of {est['n_scenes']} scenes.",
            "",
            OVERPASS_NOTE,
            "",
            "| estimator | units | A - B | 95 % |",
            "| --- | --- | --- | --- |",
            *(
                f"| {name} | {est['sensitivity_overpass_units'][key]['n_units']} | "
                f"{_signed(est['sensitivity_overpass_units'][key]['point_estimate'])} | "
                f"{_interval(est['sensitivity_overpass_units'][key]['ci'])} |"
                for key, name in (("scene_equal", "scene-equal"), ("cell_pooled", "cell-pooled"))
            ),
            "",
            "| season | scenes | cell share | scene-equal A - B | pooled A - B | gap part |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in est["by_season"]:
            lines.append(
                f"| {s['season']} | {s['n_scenes']} | {s['cell_share']:.2f} | "
                f"{_signed(s['scene_equal_diff'])} | {_signed(s['pooled_diff'])} | "
                f"{_signed(s['gap_contribution'], 4)} |"
            )
        lines += ["", "Correlation of the scene difference with scene properties:", ""]
        lines += ["| property | n | Pearson | Spearman |", "| --- | --- | --- | --- |"]
        for name, c in est["correlation_of_diff_with"].items():
            lines.append(
                f"| {name} | {c['n']} | {_signed(c['pearson'], 2)} | {_signed(c['spearman'], 2)} |"
            )
        has_lst = any("scene_mean_lst_k" in row for row in block["scenes"])
        header = (
            "| scene | date | season | patches | cells | cell share | MAE A | MAE B | A - B "
            "| gap part |"
        )
        rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        if has_lst:
            header += " scene mean LST (K) |"
            rule += " --- |"
        lines += ["", header, rule]
        for row in block["scenes"]:
            line = (
                f"| {row['scene_id']} | {row['date']} | {row['season']} | {row['n_patches']} | "
                f"{row['n_cells']} | {row['cell_share']:.3f} | {_f(row['mae_a'])} | "
                f"{_f(row['mae_b'])} | {_signed(row['diff'])} | "
                f"{_signed(row['gap_contribution'], 4)} |"
            )
            if has_lst:
                line += f" {_f(row.get('scene_mean_lst_k'), 2)} |"
            lines.append(line)
        lines.append("")
    if "reference_pair" in report:
        block = report["reference_pair"]
        ra, rb = block["arm_a"]["arm"], block["arm_b"]["arm"]
        lines += [f"## Single-run reference pair: {ra} minus {rb}", ""]
        lines += _estimator_lines(block, ra, rb)
        lines += [
            "",
            f"{block['note']}. {ra} is better in {block['fractions']['a_better']:.0%} of "
            f"{block['scene_equal']['n_units']} scenes.",
            "",
        ]
    if "scene_decomposition" in report:
        lines += [
            "## Scene bias and centred residuals",
            "",
            "| arm | bias | abs. bias | SD of bias | MAE | centred MAE | RMSE | centred RMSE | "
            "bias share of pooled MSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in report["scene_decomposition"]["summary"]:
            e = s["scene_equal"]
            lines.append(
                f"| {s['arm']} | {_signed(e['bias_k'])} | {_f(e['abs_bias_k'])} | "
                f"{_f(e['bias_sd_over_scenes_k'])} | {_f(e['mae_k'])} | {_f(e['centred_mae_k'])} | "
                f"{_f(e['rmse_k'])} | {_f(e['centred_rmse_k'])} | "
                f"{s['cell_pooled_mse']['scene_bias_share']:.2f} |"
            )
        lines += ["", "Scene-equal means in K over the scenes; seeds are averaged first.", ""]
    if "stadtstruktur_groups" in report:
        block = report["stadtstruktur_groups"]
        cov = block["coverage"]
        lines += [
            "## Stadtstruktur groups (proposal B), exploratory",
            "",
            f"Vintage {block['vintage']}. {cov['cells_with_a_group']} of {cov['evaluated_cells']} "
            f"evaluated cells carry a group; assigned area share {cov['assigned_area_share']:.2f}.",
            "",
            "| arm | group | role | scenes | eff. cells | MAE | bias | centred bias | RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in block["summary"]:
            lines.append(
                f"| {row['arm']} | {row['group']} | {row['role']} | {row['n_scenes']} | "
                f"{row['effective_cells']:.0f} | {_f(row['scene_equal_mae_k'])} | "
                f"{_signed(row['scene_equal_bias_k'])} | "
                f"{_signed(row['scene_equal_centred_bias_k'])} | {_f(row['scene_equal_rmse_k'])} |"
            )
        lines += [
            "",
            f"Paired MAE difference {a} minus {b} per group (exploratory, unadjusted 95 %):",
            "",
            "| group | scenes | A - B | 95 % | % of B | A better |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in block["paired_differences"]:
            lines.append(
                f"| {row['group']} | {row['n_scenes']} | {_signed(row['diff'])} | "
                f"{_interval(row['ci'])} | {_signed(100 * row['relative_to_b'], 1)} | "
                f"{row['fraction_scenes_a_better']:.2f} |"
            )
        lines.append("")
    if "morans_i" in report:
        lines += [
            "## Moran's I of the centred residuals",
            "",
            f"Neighbourhood {report['parameters']['moran_neighbourhood']}, weights "
            f"{report['parameters']['moran_weights']}.",
            "",
            "| arm | scenes | mean | median | min | max |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in report["morans_i"]["summary"]:
            lines.append(
                f"| {s['arm']} | {s['n_scenes_defined']} | {_f(s['mean'])} | {_f(s['median'])} | "
                f"{_f(s['min'])} | {_f(s['max'])} |"
            )
        lines.append("")
    if report["skipped"]:
        lines += ["## Skipped steps", ""]
        lines += [f"- {step}: {reason}" for step, reason in report["skipped"].items()]
        lines.append("")
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write ``spatial_error_analysis.json`` and ``.md`` into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spatial_error_analysis.json"
    md_path = output_dir / "spatial_error_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
