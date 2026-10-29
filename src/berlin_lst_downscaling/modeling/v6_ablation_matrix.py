"""v6 ablation run matrix: which arms run with which seeds and epochs.

The matrix file only plans runs; nothing here submits a job. Estimates are
planning figures, not measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from berlin_lst_downscaling.modeling.channels import V6_ABLATION_ARMS


@dataclass(frozen=True)
class V6ArmPlan:
    seeds: tuple[int, ...]
    max_epochs: int


@dataclass(frozen=True)
class V6Matrix:
    arms: dict[str, V6ArmPlan]
    hours_per_epoch_estimate: float
    fixed_overhead_hours_estimate: float
    usd_per_hour: float


@dataclass(frozen=True)
class PlannedRun:
    arm_id: str
    config_name: str
    seed: int
    max_epochs: int


def _plain_int(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value!r}")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")
    return number


def _non_negative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative finite number, got {value!r}")
    return number


def parse_v6_matrix(raw: object) -> V6Matrix:
    """Validate a matrix mapping; raise on any deviation."""
    if not isinstance(raw, Mapping):
        raise ValueError("v6 matrix must be a mapping")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, Mapping):
        raise ValueError("v6 matrix needs an 'arms' mapping")
    given, expected = set(arms_raw), set(V6_ABLATION_ARMS)
    if given != expected:
        raise ValueError(
            f"v6 matrix arms differ from the registry: missing={sorted(expected - given)} "
            f"extra={sorted(given - expected)}"
        )
    arms: dict[str, V6ArmPlan] = {}
    for arm_id in V6_ABLATION_ARMS:
        entry = arms_raw[arm_id]
        if not isinstance(entry, Mapping):
            raise ValueError(f"arm {arm_id}: entry must be a mapping")
        seeds_raw = entry.get("seeds")
        if not isinstance(seeds_raw, (list, tuple)) or not seeds_raw:
            raise ValueError(f"arm {arm_id}: seeds must be a non-empty list")
        seeds = tuple(_plain_int(s, f"arm {arm_id}: seed", minimum=0) for s in seeds_raw)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"arm {arm_id}: seeds must be unique, got {list(seeds)}")
        epochs = _plain_int(entry.get("max_epochs"), f"arm {arm_id}: max_epochs", minimum=1)
        arms[arm_id] = V6ArmPlan(seeds=seeds, max_epochs=epochs)
    return V6Matrix(
        arms=arms,
        hours_per_epoch_estimate=_positive_float(
            raw.get("hours_per_epoch_estimate"), "hours_per_epoch_estimate"
        ),
        fixed_overhead_hours_estimate=_non_negative_float(
            raw.get("fixed_overhead_hours_estimate"), "fixed_overhead_hours_estimate"
        ),
        usd_per_hour=_positive_float(raw.get("usd_per_hour"), "usd_per_hour"),
    )


def load_v6_matrix(path: str | Path) -> V6Matrix:
    """Load and validate the matrix YAML at ``path``."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"v6 matrix file not found: {file}")
    return parse_v6_matrix(OmegaConf.to_container(OmegaConf.load(file), resolve=True))


def plan_runs(matrix: V6Matrix) -> list[PlannedRun]:
    """Expand the matrix into one planned run per (arm, seed)."""
    return [
        PlannedRun(arm_id, V6_ABLATION_ARMS[arm_id].config_name, seed, plan.max_epochs)
        for arm_id, plan in matrix.arms.items()
        for seed in plan.seeds
    ]


def run_hours_estimate(matrix: V6Matrix, run: PlannedRun) -> float:
    """Estimated GPU hours for one run: fixed overhead plus epochs times the per-epoch rate."""
    return matrix.fixed_overhead_hours_estimate + run.max_epochs * matrix.hours_per_epoch_estimate


def plan_totals(matrix: V6Matrix) -> tuple[int, float, float]:
    """Return (runs, estimated GPU hours, estimated USD)."""
    planned = plan_runs(matrix)
    hours = sum(run_hours_estimate(matrix, run) for run in planned)
    return len(planned), hours, hours * matrix.usd_per_hour


__all__ = [
    "PlannedRun",
    "V6ArmPlan",
    "V6Matrix",
    "load_v6_matrix",
    "parse_v6_matrix",
    "plan_runs",
    "plan_totals",
    "run_hours_estimate",
]
