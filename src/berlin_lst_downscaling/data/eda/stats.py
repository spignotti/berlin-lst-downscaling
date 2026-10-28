"""Pure reductions for the training-data EDA.

Empty inputs return a zero count and NaN statistics. They never invent a
value. Population standard deviation (``ddof=0``) matches the training scaler.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from berlin_lst_downscaling.data.eda.rules import (
    COARSE_WITHIN_TO_BETWEEN,
    COLD_QUANTILE,
    FLAT_STD,
    HOT_QUANTILE,
    MIN_WITHIN_SCENE_N,
    NEIGHBOR_OFFSETS,
    STAT_COLUMNS,
    scene_vintage,
    season_of_month,
)

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _empty_stats() -> dict[str, float | int]:
    stats: dict[str, float | int] = {name: math.nan for name in STAT_COLUMNS}
    stats["n"] = 0
    return stats


def finite_values(values: np.ndarray) -> np.ndarray:
    """Return a 1-d float64 array with non-finite values removed."""
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def distribution(values: np.ndarray) -> dict[str, float | int]:
    """Return n, mean, population std, and the fixed quantile set."""
    array = finite_values(values)
    if array.size == 0:
        return _empty_stats()
    quantiles = np.quantile(array, _QUANTILES)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(quantiles[2]),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def quantiles_monotone(stats: dict[str, float | int]) -> bool:
    """Return whether min ≤ q05 ≤ q25 ≤ median ≤ q75 ≤ q95 ≤ max."""
    if int(stats["n"]) == 0:
        return True
    order = ("min", "q05", "q25", "median", "q75", "q95", "max")
    values = [float(stats[name]) for name in order]
    if any(not math.isfinite(value) for value in values):
        return False
    return all(values[index] <= values[index + 1] + 1e-9 for index in range(len(values) - 1))


def weighted_sum(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    """Return ``(sum(value * weight), sum(weight), n)`` over finite positive weights."""
    value = np.asarray(values, dtype=np.float64).ravel()
    weight = np.asarray(weights, dtype=np.float64).ravel()
    if value.shape != weight.shape:
        raise ValueError("values and weights have different shapes")
    ok = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if not np.any(ok):
        return 0.0, 0.0, 0
    return float(np.sum(value[ok] * weight[ok])), float(np.sum(weight[ok])), int(np.sum(ok))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean, or NaN when the weight sum is zero."""
    total, weight, _count = weighted_sum(values, weights)
    if weight == 0.0:
        return math.nan
    return total / weight


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a weight-cumulative quantile. ``quantile`` is in ``[0, 1]``."""
    value = np.asarray(values, dtype=np.float64).ravel()
    weight = np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if not np.any(ok):
        return math.nan
    order = np.argsort(value[ok], kind="mergesort")
    ordered = value[ok][order]
    cumulative = np.cumsum(weight[ok][order])
    cutoff = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(ordered[min(index, ordered.size - 1)])


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN when either side has no spread."""
    pair_left = np.asarray(left, dtype=np.float64).ravel()
    pair_right = np.asarray(right, dtype=np.float64).ravel()
    ok = np.isfinite(pair_left) & np.isfinite(pair_right)
    if int(np.sum(ok)) < 3:
        return math.nan
    if float(pair_left[ok].std(ddof=0)) == 0.0 or float(pair_right[ok].std(ddof=0)) == 0.0:
        return math.nan
    return float(np.corrcoef(pair_left[ok], pair_right[ok])[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Spearman correlation via average ranks."""
    pair_left = np.asarray(left, dtype=np.float64).ravel()
    pair_right = np.asarray(right, dtype=np.float64).ravel()
    ok = np.isfinite(pair_left) & np.isfinite(pair_right)
    if int(np.sum(ok)) < 3:
        return math.nan
    left_rank = rankdata(pair_left[ok], method="average")
    right_rank = rankdata(pair_right[ok], method="average")
    return pearson(left_rank, right_rank)


def hot_cold_counts(values: np.ndarray) -> dict[str, float | int]:
    """Count cells at or beyond the scene's own q10 and q90."""
    array = finite_values(values)
    if array.size == 0:
        return {"n": 0, "n_hot": 0, "n_cold": 0, "q10": math.nan, "q90": math.nan}
    q10 = float(np.quantile(array, COLD_QUANTILE))
    q90 = float(np.quantile(array, HOT_QUANTILE))
    return {
        "n": int(array.size),
        "n_hot": int(np.sum(array >= q90)),
        "n_cold": int(np.sum(array <= q10)),
        "q10": q10,
        "q90": q90,
    }


def neighbor_abs_diffs(row: np.ndarray, col: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Absolute differences of finite 4-neighbors. Each unordered pair is once."""
    rows = np.asarray(row).ravel()
    cols = np.asarray(col).ravel()
    vals = np.asarray(values, dtype=np.float64).ravel()
    lookup: dict[tuple[int, int], float] = {}
    for cell_row, cell_col, value in zip(rows, cols, vals, strict=True):
        if math.isfinite(float(value)):
            lookup[(int(cell_row), int(cell_col))] = float(value)
    diffs: list[float] = []
    for (cell_row, cell_col), value in lookup.items():
        for d_row, d_col in NEIGHBOR_OFFSETS:
            other = lookup.get((cell_row + d_row, cell_col + d_col))
            if other is not None:
                diffs.append(abs(value - other))
    return np.asarray(diffs, dtype=np.float64)


def within_between(scene_ids: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    """Equal-scene within and between spread.

    Between is the population std of scene means. Within is the mean and the
    median of the within-scene population std. Scenes with no finite value
    are omitted. A scene is not weighted by its cell count.
    """
    ids = np.asarray(scene_ids).ravel()
    vals = np.asarray(values, dtype=np.float64).ravel()
    if ids.shape != vals.shape:
        raise ValueError("scene ids and values have different shapes")
    means: list[float] = []
    withins: list[float] = []
    kept: list[np.ndarray] = []
    for scene_id in dict.fromkeys(ids.tolist()):
        part = vals[ids == scene_id]
        finite = part[np.isfinite(part)]
        if finite.size == 0:
            continue
        means.append(float(finite.mean()))
        withins.append(float(finite.std(ddof=0)))
        kept.append(finite)
    if not means:
        return {
            "n_scenes": 0,
            "n_observations": 0,
            "between_std": math.nan,
            "mean_within_std": math.nan,
            "median_within_std": math.nan,
            "pooled_std": math.nan,
            "mean_of_scene_means": math.nan,
            "median_of_scene_means": math.nan,
            "coarse_scale": False,
            "globally_flat": False,
        }
    mean_array = np.asarray(means, dtype=np.float64)
    within_array = np.asarray(withins, dtype=np.float64)
    pooled = np.concatenate(kept)
    between = float(mean_array.std(ddof=0))
    median_within = float(np.median(within_array))
    pooled_std = float(pooled.std(ddof=0))
    coarse = between > 0.0 and median_within < COARSE_WITHIN_TO_BETWEEN * between
    return {
        "n_scenes": int(mean_array.size),
        "n_observations": int(pooled.size),
        "between_std": between,
        "mean_within_std": float(within_array.mean()),
        "median_within_std": median_within,
        "pooled_std": pooled_std,
        "mean_of_scene_means": float(mean_array.mean()),
        "median_of_scene_means": float(np.median(mean_array)),
        "coarse_scale": bool(coarse),
        "globally_flat": bool(pooled_std < FLAT_STD),
    }


def _grouped_pearson(
    codes: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    n_groups: int,
    constant: np.ndarray,
) -> np.ndarray:
    """Per-group Pearson from centered two-pass sums; NaN for constant groups."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)

    def centered(values: np.ndarray) -> np.ndarray:
        # Second pass removes the rounding error of the first group mean.
        dev = values - (np.bincount(codes, weights=values, minlength=n_groups) / counts)[codes]
        return dev - (np.bincount(codes, weights=dev, minlength=n_groups) / counts)[codes]

    with np.errstate(invalid="ignore", divide="ignore"):
        left_dev = centered(left)
        right_dev = centered(right)
        sxx = np.bincount(codes, weights=left_dev * left_dev, minlength=n_groups)
        syy = np.bincount(codes, weights=right_dev * right_dev, minlength=n_groups)
        sxy = np.bincount(codes, weights=left_dev * right_dev, minlength=n_groups)
        corr = np.clip(sxy / (np.sqrt(sxx) * np.sqrt(syy)), -1.0, 1.0)
    corr[constant | (sxx <= 0.0) | (syy <= 0.0)] = math.nan
    return corr


def within_scene_correlations(
    scene_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | int]:
    """Median of per-scene Pearson and Spearman correlations.

    Scenes with fewer than ``MIN_WITHIN_SCENE_N`` finite pairs are omitted
    and counted. The median does not weight a scene by its cell count.

    Raises:
        ValueError: If any scene ID is missing (NaN or None). Rows without a
            scene cannot be assigned to a within-scene correlation.
    """
    ids = np.asarray(scene_ids).ravel()
    n_missing = int(np.sum(pd.isna(ids)))
    if n_missing:
        raise ValueError(
            f"within_scene_correlations: {n_missing} row(s) have a missing scene ID "
            "(NaN/None); every row must belong to a scene"
        )
    pair_left = np.asarray(left, dtype=np.float64).ravel()
    pair_right = np.asarray(right, dtype=np.float64).ravel()
    all_codes, uniques = pd.factorize(ids, use_na_sentinel=False)
    n_groups = len(uniques)
    ok = np.isfinite(pair_left) & np.isfinite(pair_right)
    n_finite = np.bincount(all_codes[ok], minlength=n_groups)
    eligible = n_finite >= MIN_WITHIN_SCENE_N
    used = int(np.sum(eligible))
    skipped = n_groups - used
    keep = ok & eligible[all_codes]
    codes = all_codes[keep]
    frame = pd.DataFrame({"g": codes, "x": pair_left[keep], "y": pair_right[keep]})
    grouped = frame.groupby("g", sort=True)
    spread = grouped[["x", "y"]].agg(["min", "max"]).reindex(range(n_groups))
    constant = (
        (spread[("x", "min")] == spread[("x", "max")])
        | (spread[("y", "min")] == spread[("y", "max")])
    ).to_numpy()
    rank_left = np.asarray(grouped["x"].rank(method="average"), dtype=np.float64)
    rank_right = np.asarray(grouped["y"].rank(method="average"), dtype=np.float64)
    pearsons = _grouped_pearson(
        codes, frame["x"].to_numpy(), frame["y"].to_numpy(), n_groups, constant
    )
    spearmans = _grouped_pearson(codes, rank_left, rank_right, n_groups, constant)
    pearsons = pearsons[eligible & np.isfinite(pearsons)]
    spearmans = spearmans[eligible & np.isfinite(spearmans)]
    return {
        "n_scenes_used": used,
        "n_scenes_skipped": skipped,
        "median_pearson": float(np.median(pearsons)) if pearsons.size else math.nan,
        "median_spearman": float(np.median(spearmans)) if spearmans.size else math.nan,
    }


def self_check() -> list[str]:
    """Return failures of the fixed numeric rules. An empty list is a pass."""
    failures: list[str] = []
    if scene_vintage(2017) != 2020 or scene_vintage(2019) != 2020:
        failures.append("carry-back vintage is not 2020")
    if scene_vintage(2021) != 2021 or scene_vintage(2024) != 2024:
        failures.append("same-year vintage mapping drifted")
    if scene_vintage(2025) != 2024:
        failures.append("2025 carry-forward is not 2024")
    if scene_vintage(2026) is not None:
        failures.append("2026 received a Stadtstruktur vintage")
    if season_of_month(1) != "DJF" or season_of_month(7) != "JJA":
        failures.append("season mapping drifted")
    if season_of_month(None) != "unknown":
        failures.append("missing month did not stay unknown")

    stats = distribution(np.arange(100, dtype=np.float64))
    if stats["n"] != 100 or abs(float(stats["median"]) - 49.5) > 1e-9:
        failures.append(f"distribution median drifted: {stats['median']}")
    if not quantiles_monotone(stats):
        failures.append("distribution quantiles are not monotone")
    if distribution(np.array([math.nan]))["n"] != 0:
        failures.append("non-finite input was counted")

    if abs(weighted_mean(np.array([1.0, 3.0]), np.array([1.0, 1.0])) - 2.0) > 1e-12:
        failures.append("equal-weight mean drifted")
    if abs(weighted_mean(np.array([1.0, 3.0]), np.array([3.0, 1.0])) - 1.5) > 1e-12:
        failures.append("area weight drifted")
    if math.isfinite(weighted_mean(np.array([1.0]), np.array([0.0]))):
        failures.append("zero weight produced a mean")
    median = weighted_quantile(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), 0.5)
    if abs(median - 2.0) > 1e-12:
        failures.append(f"weighted median drifted: {median}")
    skewed = weighted_quantile(np.array([1.0, 3.0]), np.array([3.0, 1.0]), 0.5)
    if abs(skewed - 1.0) > 1e-12:
        failures.append(f"skewed weighted median drifted: {skewed}")

    counts = hot_cold_counts(np.arange(10, dtype=np.float64))
    if counts["n_hot"] != 1 or counts["n_cold"] != 1:
        failures.append(f"hot/cold counts drifted: {counts}")

    diffs = neighbor_abs_diffs(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([1.0, 3.0, 1.0, 5.0]),
    )
    if diffs.size != 4 or abs(float(diffs.mean()) - 2.0) > 1e-12:
        failures.append(f"neighbor pairs drifted: {diffs.tolist()}")

    diagonal = neighbor_abs_diffs(np.array([0, 1]), np.array([0, 1]), np.array([0.0, 4.0]))
    if diagonal.size != 0:
        failures.append("diagonal cells were treated as neighbors")

    spread = within_between(
        np.array(["a", "a", "b", "b"]),
        np.array([0.0, 0.0, 10.0, 10.0]),
    )
    if spread["n_scenes"] != 2 or abs(float(spread["between_std"]) - 5.0) > 1e-12:
        failures.append(f"between-scene std drifted: {spread}")
    if not spread["coarse_scale"]:
        failures.append("coarse-scale rule did not flag a flat scene pair")

    constant = pearson(np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0]))
    if math.isfinite(constant):
        failures.append("constant input produced a correlation")
    perfect = spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 6.0]))
    if abs(perfect - 1.0) > 1e-12:
        failures.append(f"spearman drifted: {perfect}")
    return failures


__all__ = [
    "distribution",
    "finite_values",
    "hot_cold_counts",
    "neighbor_abs_diffs",
    "pearson",
    "quantiles_monotone",
    "self_check",
    "spearman",
    "weighted_mean",
    "weighted_quantile",
    "weighted_sum",
    "within_between",
    "within_scene_correlations",
]
