"""Spatial holdout evaluation: scene-equal MAE by anchor stratum, G and K (pm #553).

Inputs are per-patch validation exports (schema_version 2, the files
``analysis.scene_bootstrap`` reads) and a committed holdout assignment
(``modeling.spatial_holdout``). Nothing here evaluates a model or reads a data
product, and only the validation split is accepted.

Strata (by the anchor of a patch): ``seen`` (not held out), ``unseen`` (held
out), and the two parts of ``unseen``: ``unseen_interior`` and ``unseen_edge``.

Estimand, as in the matrix: patch sums are pooled within a scene, the scene MAE
is averaged over the seeds of a model group, every scene counts once, and the
scene is the only resampling unit. A scene without a patch in a stratum has no
value there and is not part of that stratum's estimate; the output names the
number of scenes behind every figure.

- MAE per model group and stratum.
- ``G`` (transfer gap per arm) = holdout model minus matrix model of the same
  arm and seeds, on the unseen anchors.
- ``K`` (context under transfer) = L4 minus L1, both holdout models, on the
  unseen anchors.
- Paired differences of the matrix families (``notes/statistikplan-matrix.md``):
  A (L2-L1, L3-L2, L4-L3, L5-L4) and B (LOO-M-L4, LOO-S-L4, L3-L4,
  NO-NDVI-NDWI-L4, NO-RED-NIR-IDX-L4) for the matrix and the holdout models,
  and K for the holdout models, on every stratum. With seeds 0, 1 and 2 on
  both sides each carries the three criteria of the decision rule, read from
  ``scene_bootstrap.compare_arms`` run on the rows of the stratum.

The TESSERA arm (``ARM_TESSERA``, a sibling of L4 that replaces morphology and shadow channels,
docs/tessera-arm.md) drops patches with invalid embedding pixels, so its patch set is smaller than
L4's. Every comparison that involves it (``tessera_section``) is computed on the intersection of
the patch sets of all compared runs; the output names how many patches and scenes each side lost.
The quantities are ``G_TESSERA`` (holdout minus temporal TESSERA, unseen anchors) and TESSERA
minus L4 (holdout models on the unseen anchors, temporal models on all validation patches), each
with the decision-rule criteria when seeds 0, 1 and 2 exist on both sides. All other
comparisons keep the strict identical-patch-set check.

Intervals are 95% percentile intervals of the paired scene bootstrap with the
resampling of ``scene_bootstrap.compare_arms`` (same generator, batch layout and
quantile method; a test pins the two to each other). Arms without exports are
skipped with a note.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from berlin_lst_downscaling.analysis import scene_bootstrap as sb
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
)
from berlin_lst_downscaling.modeling.spatial_holdout import (
    CV_FOLDS,
    CV_PARTITION,
    POSITION_INTERIOR,
    Anchor,
    SpatialHoldoutAssignment,
    assignment_bytes,
)

N_BOOTSTRAP = 10_000
RNG_SEED = 0
SEEN = "seen"
UNSEEN = "unseen"
UNSEEN_INTERIOR = "unseen_interior"
UNSEEN_EDGE = "unseen_edge"
STRATA: tuple[str, ...] = (SEEN, UNSEEN, UNSEEN_INTERIOR, UNSEEN_EDGE)
ARM_SPECTRAL = "L1"
ARM_FULL = "L4"
ARM_TESSERA = "TESSERA"
# Labels a TESSERA export can carry: the arm name, and the Hydra config names the runs use
# (the arm has no ``ablation_arm``, so the export takes the config name).
TESSERA_ARM_LABELS = frozenset(
    {ARM_TESSERA, "tessera", "tessera_arm", "tessera_arm_spatial_holdout"}
)
ALL = "all"  # every validation patch; only the TESSERA comparisons use it
# Matrix families of notes/statistikplan-matrix.md as (A, B) of "A minus B": the
# ladder (A), the comparisons against L4 (B; LOO-ERA5 is L3) and K (L4 - L1).
FAMILY_A = (("L2", "L1"), ("L3", "L2"), ("L4", "L3"), ("L5", "L4"))
FAMILY_B = (
    ("LOO-M", "L4"),
    ("LOO-S", "L4"),
    ("L3", "L4"),
    ("NO-NDVI-NDWI", "L4"),
    ("NO-RED-NIR-IDX", "L4"),
)
FAMILY_K = ((ARM_FULL, ARM_SPECTRAL),)
PAIRING_FAMILIES = (("A", FAMILY_A), ("B", FAMILY_B), ("K", FAMILY_K))
# Stage-2 trigger of the pre-registered plan: an interval of G or K excludes 0,
# or the absolute point estimate reaches this many kelvin.
STAGE2_ABS_THRESHOLD_K = 0.03

_CELL_ID = re.compile(r":E(-?\d+)N(-?\d+)$")


class SpatialHoldoutEvalError(ValueError):
    """Raised for exports that cannot be compared under the holdout assignment."""


def anchor_from_patch_id(patch_id: str) -> Anchor:
    """Return the global ``(row, col)`` anchor encoded in ``<scene_id>:E<x>N<y>``."""
    match = _CELL_ID.search(patch_id)
    if match is None:
        raise SpatialHoldoutEvalError(f"patch_id {patch_id!r} carries no E<x>N<y> cell id")
    col, col_rest = divmod(int(match.group(1)) - int(CANON_GRID_ORIGIN_X), CELL_SIZE_M)
    row, row_rest = divmod(int(CANON_GRID_ORIGIN_Y) - int(match.group(2)), CELL_SIZE_M)
    if col_rest or row_rest:
        raise SpatialHoldoutEvalError(f"patch_id {patch_id!r} is off the canonical 100 m grid")
    return row, col


@dataclass(frozen=True)
class PatchScore:
    scene_id: str
    anchor: Anchor
    abs_error_sum: float
    valid_cells: int


@dataclass(frozen=True)
class Run:
    """One export: one model run scored on the validation patches."""

    arm: str
    seed: int | None
    run_id: str
    patches: dict[str, PatchScore]
    # The validated export rows; the decision-rule criteria hand them to ``scene_bootstrap``.
    table: pa.Table | None = field(default=None, repr=False, compare=False)
    # Where the export was read from, and the assignment the run itself recorded
    # (``load_fold_runs``); both stay ``None`` for a plain ``load_runs``.
    source: str | None = field(default=None, compare=False)
    assignment: str | None = field(default=None, compare=False)
    assignment_sha256: str | None = field(default=None, compare=False)

    @property
    def label(self) -> str:
        return f"{self.arm}/seed{self.seed}/{self.run_id}"


def load_runs(paths: Sequence[Path]) -> list[Run]:
    """Load validation exports (validated by ``scene_bootstrap``) as one ``Run`` per file."""
    runs: list[Run] = []
    for path in paths:
        table = sb.load_patch_exports([Path(path)], split="validation")
        cols = {
            name: table.column(name).to_pylist()
            for name in ("arm", "seed", "run_id", "patch_id", "scene_id")
        }
        abs_sum = table.column("abs_error_sum").to_pylist()
        cells = table.column("valid_cells").to_pylist()
        patches = {
            pid: PatchScore(
                scene_id=cols["scene_id"][i],
                anchor=anchor_from_patch_id(pid),
                abs_error_sum=float(abs_sum[i]),
                valid_cells=int(cells[i]),
            )
            for i, pid in enumerate(cols["patch_id"])
        }
        arm = cols["arm"][0]
        runs.append(
            Run(
                arm=ARM_TESSERA if arm in TESSERA_ARM_LABELS else arm,
                seed=cols["seed"][0],
                run_id=cols["run_id"][0],
                patches=patches,
                table=table,
            )
        )
    return runs


def read_tessera_invalid(path: Path) -> frozenset[str]:
    """Validation patch ids a TESSERA run recorded as skipped for invalid embedding pixels.

    ``path`` is the run's ``data_scope.json`` (``skipped_refs`` per split with a reason).
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    skipped = record.get("skipped_refs") if isinstance(record, dict) else None
    rows = skipped.get("validation") if isinstance(skipped, dict) else None
    if not isinstance(rows, list):
        raise SpatialHoldoutEvalError(
            f"{path}: no skipped_refs.validation list (a data_scope.json)"
        )
    return frozenset(str(row["patch_id"]) for row in rows if row.get("reason") == "tessera_invalid")


def read_recorded_assignment(path: Path) -> tuple[str, str]:
    """Return ``(assignment, sha256)`` from a run record: the evidence or the raw summary.

    The evidence record keeps the summary under ``full.summary``; ``full_summary.json``
    holds it at the top level. Both carry ``spatial_holdout.assignment`` and
    ``spatial_holdout.assignment_sha256``, written by the run itself.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = record.get("full", {}).get("summary", record) if isinstance(record, dict) else None
    block = summary.get("spatial_holdout") if isinstance(summary, dict) else None
    name = block.get("assignment") if isinstance(block, dict) else None
    sha = block.get("assignment_sha256") if isinstance(block, dict) else None
    if not isinstance(name, str) or not isinstance(sha, str):
        raise SpatialHoldoutEvalError(
            f"{path}: the run record holds no spatial_holdout.assignment and assignment_sha256"
        )
    return name, sha


def load_fold_runs(paths: Sequence[Path]) -> list[Run]:
    """Load cross-validation exports together with the record of the run that wrote them.

    The record sits next to the export as ``<stem>.evidence.json`` (the run's evidence record
    or its ``full_summary.json``) and is required: ``evaluate_cv`` checks it against the fold.
    """
    runs: list[Run] = []
    for path in map(Path, paths):
        record = path.with_suffix(".evidence.json")
        if not record.is_file():
            raise SpatialHoldoutEvalError(
                f"{path}: the run record {record.name} is missing next to the export; the "
                "fold the run trained without is read from it, never from a label"
            )
        try:
            name, sha = read_recorded_assignment(record)
        except (SpatialHoldoutEvalError, ValueError) as exc:
            raise SpatialHoldoutEvalError(f"{path}: {exc}") from exc
        runs += [
            dataclasses.replace(run, source=str(path), assignment=name, assignment_sha256=sha)
            for run in load_runs([path])
        ]
    return runs


def strata_of(anchor: Anchor, assignment: SpatialHoldoutAssignment) -> tuple[str, ...]:
    """Return the strata an anchor belongs to; an anchor outside the assignment raises."""
    return _strata(anchor, assignment, assignment.positions())


@dataclass(frozen=True)
class _Group:
    """Seed-averaged scene MAE of one model group on one stratum."""

    label: str
    seeds: tuple[int | None, ...]
    values: dict[str, float]
    patch_sets: dict[str, frozenset[str]]


def _group(
    runs: Sequence[Run], assignment: SpatialHoldoutAssignment, stratum: str, label: str
) -> _Group:
    if stratum not in STRATA:
        raise SpatialHoldoutEvalError(f"unknown stratum {stratum!r}; expected one of {STRATA}")
    positions = assignment.positions()
    return _group_where(
        runs,
        lambda score: stratum in _strata(score.anchor, assignment, positions),
        stratum,
        label,
    )


def _group_where(
    runs: Sequence[Run], keep: Callable[[PatchScore], bool], stratum: str, label: str
) -> _Group:
    """Seed-averaged scene MAE of the patches ``keep`` selects (``stratum`` only names them)."""
    if not runs:
        raise SpatialHoldoutEvalError(f"{label}: no runs")
    arms = {run.arm for run in runs}
    if len(arms) != 1:
        raise SpatialHoldoutEvalError(f"{label}: runs mix arms {sorted(arms)}")
    seeds = tuple(run.seed for run in runs)
    if len(set(seeds)) != len(seeds):
        raise SpatialHoldoutEvalError(f"{label}: duplicate seeds {list(seeds)}")
    per_run: list[dict[str, tuple[float, int]]] = []
    patch_sets: dict[str, frozenset[str]] | None = None
    cells_ref: dict[str, int] = {}
    for run in runs:
        sums: dict[str, list[float]] = {}
        members: dict[str, set[str]] = {}
        for pid in sorted(run.patches):
            score = run.patches[pid]
            if not keep(score):
                continue
            if cells_ref.setdefault(pid, score.valid_cells) != score.valid_cells:
                raise SpatialHoldoutEvalError(
                    f"{label}: patch {pid!r} has differing valid_cells across runs"
                )
            entry = sums.setdefault(score.scene_id, [0.0, 0])
            entry[0] += score.abs_error_sum
            entry[1] += score.valid_cells
            members.setdefault(score.scene_id, set()).add(pid)
        frozen = {scene: frozenset(ids) for scene, ids in members.items()}
        if patch_sets is None:
            patch_sets = frozen
        elif frozen != patch_sets:
            raise SpatialHoldoutEvalError(
                f"{label}: run {run.label} scored a different patch set on stratum {stratum!r}"
            )
        per_run.append({scene: (v[0], int(v[1])) for scene, v in sums.items()})
    patch_sets = patch_sets or {}
    values = {
        scene: float(np.mean([run[scene][0] / run[scene][1] for run in per_run]))
        for scene in sorted(patch_sets)
    }
    return _Group(label=label, seeds=seeds, values=values, patch_sets=patch_sets)


def _strata(
    anchor: Anchor, assignment: SpatialHoldoutAssignment, positions: Mapping[Anchor, str]
) -> tuple[str, ...]:
    if anchor not in assignment.anchors:
        raise SpatialHoldoutEvalError(
            f"anchor {anchor} is not in the assignment's patch index; the export and the "
            "assignment come from different indexes"
        )
    position = positions.get(anchor)
    if position is None:
        return (SEEN,)
    return (UNSEEN, UNSEEN_INTERIOR if position == POSITION_INTERIOR else UNSEEN_EDGE)


def scene_bootstrap_mean(
    values: Sequence[float], *, n_bootstrap: int = N_BOOTSTRAP, rng_seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` of one value per scene.

    The resampling of ``scene_bootstrap.compare_arms``: scenes are drawn with
    replacement from ``numpy.random.default_rng(rng_seed)`` in the same batch
    layout, and the interval is the 2.5% / 97.5% percentile (linear).
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise SpatialHoldoutEvalError("n_bootstrap must be a positive int")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or rng_seed < 0:
        raise SpatialHoldoutEvalError("rng_seed must be a non-negative int")
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    if array.ndim != 1 or n == 0:
        raise SpatialHoldoutEvalError("the scene bootstrap needs at least one scene value")
    if not np.isfinite(array).all():
        raise SpatialHoldoutEvalError("scene values must be finite")
    rng = np.random.default_rng(rng_seed)
    batch = max(1, sb._BATCH_INDICES // n)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        b = min(batch, n_bootstrap - done)
        idx = rng.integers(0, n, size=(b, n))
        reps[done : done + b] = array[idx].mean(axis=1)
        done += b
    low, high = np.quantile(reps, [0.025, 0.975], method="linear")
    return float(array.mean()), float(low), float(high)


@dataclass(frozen=True)
class Estimate:
    """A scene-equal mean with its scene-bootstrap interval."""

    quantity: str
    label: str
    stratum: str
    seeds: tuple[int | None, ...]
    n_scenes: int
    n_patches: int
    point: float
    ci_low: float
    ci_high: float
    scenes: tuple[tuple[str, float], ...]

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "label": self.label,
            "stratum": self.stratum,
            "seeds": list(self.seeds),
            "n_scenes": self.n_scenes,
            "n_patches": self.n_patches,
            "point": self.point,
            "ci_95": [self.ci_low, self.ci_high],
            "scenes": [{"scene_id": scene, "value": value} for scene, value in self.scenes],
        }


def _estimate(
    quantity: str,
    label: str,
    stratum: str,
    seeds: tuple[int | None, ...],
    values: Mapping[str, float],
    n_patches: int,
    *,
    n_bootstrap: int,
    rng_seed: int,
) -> Estimate:
    if not values:
        raise SpatialHoldoutEvalError(f"{label}: no scene has a patch on stratum {stratum!r}")
    scenes = sorted(values)
    point, low, high = scene_bootstrap_mean(
        [values[s] for s in scenes], n_bootstrap=n_bootstrap, rng_seed=rng_seed
    )
    return Estimate(
        quantity=quantity,
        label=label,
        stratum=stratum,
        seeds=seeds,
        n_scenes=len(scenes),
        n_patches=n_patches,
        point=point,
        ci_low=low,
        ci_high=high,
        scenes=tuple((s, values[s]) for s in scenes),
    )


def stratum_mae(
    runs: Sequence[Run],
    assignment: SpatialHoldoutAssignment,
    stratum: str,
    *,
    label: str,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> Estimate:
    """Scene-equal MAE of one model group on one stratum."""
    group = _group(runs, assignment, stratum, label)
    return _estimate(
        "mae_100m",
        label,
        stratum,
        group.seeds,
        group.values,
        sum(len(ids) for ids in group.patch_sets.values()),
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )


def paired_difference(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    assignment: SpatialHoldoutAssignment,
    stratum: str,
    *,
    label: str,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> Estimate:
    """Paired scene difference A minus B on one stratum.

    Both groups must hold the same seeds and must have scored the same patches
    in every scene of the stratum; anything else raises instead of comparing
    different universes.
    """
    return _paired_groups(
        _group(runs_a, assignment, stratum, f"{label} (A)"),
        _group(runs_b, assignment, stratum, f"{label} (B)"),
        stratum,
        label=label,
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )


def _paired_groups(
    a: _Group,
    b: _Group,
    stratum: str,
    *,
    label: str,
    n_bootstrap: int,
    rng_seed: int,
) -> Estimate:
    if sorted(a.seeds, key=str) != sorted(b.seeds, key=str):
        raise SpatialHoldoutEvalError(
            f"{label}: seeds differ between A {list(a.seeds)} and B {list(b.seeds)}"
        )
    if a.patch_sets != b.patch_sets:
        differing = sorted(
            scene
            for scene in set(a.patch_sets) | set(b.patch_sets)
            if a.patch_sets.get(scene) != b.patch_sets.get(scene)
        )
        raise SpatialHoldoutEvalError(
            f"{label}: A and B scored different patches on stratum {stratum!r} in "
            f"{len(differing)} scene(s), e.g. {differing[:5]}"
        )
    values = {scene: a.values[scene] - b.values[scene] for scene in a.values}
    return _estimate(
        "mae_100m_difference",
        label,
        stratum,
        a.seeds,
        values,
        sum(len(ids) for ids in a.patch_sets.values()),
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )


def stratum_gap(
    runs: Sequence[Run],
    assignment: SpatialHoldoutAssignment,
    *,
    label: str,
    stratum_a: str = UNSEEN,
    stratum_b: str = SEEN,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> Estimate:
    """Within-model scene difference ``stratum_a`` minus ``stratum_b``.

    Only scenes with a patch in both strata have a difference. For a matrix
    model this shows whether the held-out anchors are easier or harder than the
    rest, before any holdout run.
    """
    a = _group(runs, assignment, stratum_a, label)
    b = _group(runs, assignment, stratum_b, label)
    common = sorted(set(a.values) & set(b.values))
    values = {scene: a.values[scene] - b.values[scene] for scene in common}
    return _estimate(
        "mae_100m_difference",
        label,
        f"{stratum_a}-{stratum_b}",
        a.seeds,
        values,
        sum(len(a.patch_sets[s]) + len(b.patch_sets[s]) for s in common),
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )


def _stratum_table(
    runs: Sequence[Run], assignment: SpatialHoldoutAssignment, stratum: str
) -> pa.Table:
    """Concatenate the export rows of ``runs`` that lie on ``stratum``."""
    positions = assignment.positions()
    parts: list[pa.Table] = []
    for run in runs:
        if run.table is None:
            raise SpatialHoldoutEvalError(f"{run.label}: the run holds no export rows")
        keep = [
            stratum in _strata(run.patches[pid].anchor, assignment, positions)
            for pid in run.table.column("patch_id").to_pylist()
        ]
        parts.append(run.table.filter(pa.array(keep, type=pa.bool_())))
    return pa.concat_tables(parts)


def seed_rule_criteria(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    assignment: SpatialHoldoutAssignment,
    stratum: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> dict[str, Any]:
    """The three criteria of the matrix decision rule for A minus B on one stratum.

    ``notes/statistikplan-matrix.md``: (a) the interval excludes 0, (b) the sign
    is the same in all seeds, (c) ``|difference| >= 2 x`` the larger seed SD of
    the two arms. The numbers come from ``scene_bootstrap.compare_arms`` run on
    the rows of the stratum, so the interval, the seed SDs (ddof 1, pooled seed
    metric) and the sign count are the matrix's own. ``compare_arms`` needs
    seeds 0, 1 and 2 on both sides; with fewer the criteria are not available.
    """
    return _rule_criteria(
        runs_a,
        runs_b,
        lambda: (
            _stratum_table(runs_a, assignment, stratum),
            _stratum_table(runs_b, assignment, stratum),
        ),
        stratum,
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )


def _rule_criteria(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    tables: Callable[[], tuple[pa.Table, pa.Table]],
    stratum: str,
    *,
    n_bootstrap: int,
    rng_seed: int,
) -> dict[str, Any]:
    """``seed_rule_criteria`` on the export rows ``tables`` selects."""
    seeds = {
        side: sorted((run.seed for run in runs), key=str)
        for side, runs in (("A", runs_a), ("B", runs_b))
    }
    wanted = list(sb.SEEDS)
    if seeds["A"] != wanted or seeds["B"] != wanted:
        return {
            "available": False,
            "reason": (
                f"needs seeds {wanted} on both sides (A has {seeds['A']}, B has {seeds['B']})"
            ),
        }
    try:
        table_a, table_b = tables()
        result = sb.compare_arms(
            table_a,
            table_b,
            metric="mae_100m",
            n_bootstrap=n_bootstrap,
            rng_seed=rng_seed,
        )
    except sb.SceneBootstrapError as exc:
        raise SpatialHoldoutEvalError(f"decision-rule criteria on {stratum!r}: {exc}") from exc
    diag = result.seed_diagnostics
    excludes = result.ci_low > 0.0 or result.ci_high < 0.0
    same_sign = max(diag.n_positive, diag.n_negative) == len(diag.diffs)
    seed_sd = max(diag.sd_a, diag.sd_b)
    large = abs(result.point_estimate) >= 2.0 * seed_sd
    return {
        "available": True,
        "a_interval_excludes_zero": excludes,
        "b_same_sign_all_seeds": same_sign,
        "c_abs_difference_at_least_2_seed_sd": large,
        "all_met": excludes and same_sign and large,
        "sign_consistency": diag.sign_consistency,
        "seed_differences": list(diag.diffs),
        "larger_seed_sd": seed_sd,
    }


def _runs_for_seeds(runs: Sequence[Run], seeds: Sequence[int | None], label: str) -> list[Run]:
    by_seed = {run.seed: run for run in runs}
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise SpatialHoldoutEvalError(f"{label}: no matrix run for seed(s) {missing}")
    return [by_seed[seed] for seed in seeds]


def _stage2_check(estimates: Mapping[str, Estimate]) -> dict[str, Any]:
    checks = {
        name: {
            "point": est.point,
            "ci_95": [est.ci_low, est.ci_high],
            "interval_excludes_zero": est.excludes_zero,
            "abs_point_reaches_threshold": abs(est.point) >= STAGE2_ABS_THRESHOLD_K,
        }
        for name, est in estimates.items()
    }
    return {
        "rule": (
            "stage 2 (seeds 1 and 2) only if an interval of G or K excludes 0 or "
            f"|point| >= {STAGE2_ABS_THRESHOLD_K} K"
        ),
        "checks": checks,
        "triggered": any(
            c["interval_excludes_zero"] or c["abs_point_reaches_threshold"] for c in checks.values()
        ),
    }


def _pairings(
    assignment: SpatialHoldoutAssignment,
    arms: Mapping[str, Sequence[Run]],
    model: str,
    notes: list[str],
    *,
    n_bootstrap: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    """Paired differences of the matrix families on every stratum, one model family.

    A pairing whose arms lack an export, or whose seeds differ, is skipped with a note.
    """
    kwargs = {"n_bootstrap": n_bootstrap, "rng_seed": rng_seed}
    out: list[dict[str, Any]] = []
    for family, pairs in PAIRING_FAMILIES:
        if family == "K" and model != "holdout":
            continue  # K is defined on the holdout models; the matrix L4 - L1 is in L4_minus_L1
        for arm_a, arm_b in pairs:
            name = f"{arm_a}-{arm_b}"
            absent = [arm for arm in (arm_a, arm_b) if arm not in arms]
            if absent:
                notes.append(
                    f"{model} {name} ({family}): skipped, no export of {', '.join(absent)}"
                )
                continue
            seeds = {
                arm: sorted((run.seed for run in arms[arm]), key=str) for arm in (arm_a, arm_b)
            }
            if seeds[arm_a] != seeds[arm_b]:
                notes.append(
                    f"{model} {name} ({family}): skipped, seeds differ "
                    f"({arm_a} {seeds[arm_a]}, {arm_b} {seeds[arm_b]})"
                )
                continue
            strata: dict[str, Any] = {}
            for stratum in STRATA:
                label = f"{model}: {name}"
                estimate = paired_difference(
                    arms[arm_a], arms[arm_b], assignment, stratum, label=label, **kwargs
                ).to_dict()
                estimate["criteria"] = seed_rule_criteria(
                    arms[arm_a], arms[arm_b], assignment, stratum, **kwargs
                )
                strata[stratum] = estimate
            out.append({"name": name, "family": family, "a": arm_a, "b": arm_b, "strata": strata})
    return out


def _key(score: PatchScore) -> tuple[str, Anchor]:
    return score.scene_id, score.anchor


def _side_report(
    runs: Sequence[Run], keep: Callable[[PatchScore], bool], common: frozenset[tuple[str, Anchor]]
) -> dict[str, int]:
    """Patches and scenes of one side on the stratum, and how many the intersection lost."""
    patches = {_key(sc) for run in runs for sc in run.patches.values() if keep(sc)}
    scenes = {scene for scene, _anchor in patches}
    kept_scenes = {scene for scene, _anchor in patches & common}
    return {
        "patches": len(patches),
        "patches_lost": len(patches - common),
        "scenes": len(scenes),
        "scenes_lost": len(scenes - kept_scenes),
    }


def _intersection_table(
    runs: Sequence[Run], common: frozenset[tuple[str, Anchor]], arm: str
) -> pa.Table:
    """Export rows of ``runs`` on the common patches, under one arm label."""
    parts: list[pa.Table] = []
    for run in runs:
        if run.table is None:
            raise SpatialHoldoutEvalError(f"{run.label}: the run holds no export rows")
        keep = [
            _key(run.patches[pid]) in common for pid in run.table.column("patch_id").to_pylist()
        ]
        part = run.table.filter(pa.array(keep, type=pa.bool_()))
        index = part.schema.get_field_index("arm")
        parts.append(part.set_column(index, "arm", pa.array([arm] * part.num_rows, pa.string())))
    return pa.concat_tables(parts)


def _lost_ids(
    runs: Sequence[Run], keep: Callable[[PatchScore], bool], common: frozenset[tuple[str, Anchor]]
) -> set[str]:
    return {
        pid
        for run in runs
        for pid, sc in run.patches.items()
        if keep(sc) and _key(sc) not in common
    }


def _examples(ids: Iterable[str]) -> str:
    ordered = sorted(ids)
    return ", ".join(ordered[:5]) + (
        f", ... (+{len(ordered) - 5} more)" if len(ordered) > 5 else ""
    )


def _check_losses(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    keep: Callable[[PatchScore], bool],
    common: frozenset[tuple[str, Anchor]],
    *,
    same_arm: bool,
    label: str,
    tessera_invalid: frozenset[str] | None,
) -> bool:
    """Fail closed on a loss the TESSERA arm cannot explain; return whether the scope was checked.

    ``same_arm`` (G_TESSERA): neither side may lose a patch. Otherwise A is TESSERA and B is
    L4: TESSERA loses none, and L4 only patches absent from the TESSERA export of the whole
    split (dropped at admission). With ``tessera_invalid`` (the validation patches a TESSERA
    run recorded as skipped for invalid embedding pixels) L4's loss must equal that set on the
    stratum.
    """
    lost_a, lost_b = _lost_ids(runs_a, keep, common), _lost_ids(runs_b, keep, common)
    if same_arm:
        if lost_a or lost_b:
            raise SpatialHoldoutEvalError(
                f"{label}: the holdout and the temporal TESSERA runs must score the same patches, "
                f"but side A lost {len(lost_a)} patch(es) and side B lost {len(lost_b)} on the "
                f"intersection, e.g. {_examples(lost_a | lost_b)}"
            )
        return False
    if lost_a:
        raise SpatialHoldoutEvalError(
            f"{label}: the TESSERA side lost {len(lost_a)} patch(es) on the intersection with L4 "
            f"(L4 lacks them or a TESSERA seed differs), e.g. {_examples(lost_a)}"
        )
    in_tessera = {_key(sc) for run in runs_a for sc in run.patches.values()}
    present = {
        pid
        for run in runs_b
        for pid, sc in run.patches.items()
        if keep(sc) and pid in lost_b and _key(sc) in in_tessera
    }
    if present:
        raise SpatialHoldoutEvalError(
            f"{label}: L4 lost {len(present)} patch(es) that the TESSERA export holds, "
            f"e.g. {_examples(present)}"
        )
    if tessera_invalid is None:
        return False
    expected = {
        pid
        for run in runs_b
        for pid, sc in run.patches.items()
        if keep(sc) and pid in tessera_invalid
    }
    if lost_b != expected:
        raise SpatialHoldoutEvalError(
            f"{label}: L4 lost {len(lost_b)} patch(es) but the TESSERA run recorded "
            f"{len(expected)} tessera_invalid validation patch(es) on this stratum, "
            f"e.g. {_examples(lost_b ^ expected)}"
        )
    return True


def _intersection_difference(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    keep: Callable[[PatchScore], bool],
    stratum: str,
    *,
    label: str,
    same_arm: bool,
    tessera_invalid: frozenset[str] | None,
    n_bootstrap: int,
    rng_seed: int,
) -> dict[str, Any]:
    """A minus B on the patches every run of both sides scored on the stratum.

    The scene-equal paired difference of ``paired_difference`` and the decision-rule
    criteria, both computed on the intersection; the entry names what each side lost.
    A loss ``_check_losses`` cannot explain raises.
    """
    per_run = [{_key(sc) for sc in run.patches.values() if keep(sc)} for run in (*runs_a, *runs_b)]
    common = frozenset(set.intersection(*per_run))
    verified = _check_losses(
        runs_a,
        runs_b,
        keep,
        common,
        same_arm=same_arm,
        label=label,
        tessera_invalid=tessera_invalid,
    )
    group_a = _group_where(runs_a, lambda sc: _key(sc) in common, stratum, f"{label} (A)")
    group_b = _group_where(runs_b, lambda sc: _key(sc) in common, stratum, f"{label} (B)")
    estimate = _paired_groups(
        group_a, group_b, stratum, label=label, n_bootstrap=n_bootstrap, rng_seed=rng_seed
    ).to_dict()
    estimate["intersection"] = {
        "patches": len(common),
        "a": _side_report(runs_a, keep, common),
        "b": _side_report(runs_b, keep, common),
        "scope_verified": verified,
    }
    estimate["criteria"] = _rule_criteria(
        runs_a,
        runs_b,
        lambda: (
            _intersection_table(runs_a, common, f"{label} (A)"),
            _intersection_table(runs_b, common, f"{label} (B)"),
        ),
        stratum,
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )
    return estimate


def tessera_section(
    assignment: SpatialHoldoutAssignment,
    matrix: Mapping[str, Sequence[Run]],
    holdout: Mapping[str, Sequence[Run]],
    notes: list[str],
    *,
    tessera_invalid: frozenset[str] | None = None,
    n_bootstrap: int,
    rng_seed: int,
) -> dict[str, Any]:
    """The TESSERA quantities on patch-set intersections; empty without a TESSERA export.

    Fails closed: G_TESSERA loses no patch on either side; in TESSERA minus L4 the TESSERA side
    loses none and L4 only the patches the TESSERA arm dropped at admission
    (``tessera_invalid`` checks that against the run's own record, else it is unverified).

    ``G_TESSERA``: holdout TESSERA minus temporal TESSERA on the unseen anchors.
    ``TESSERA_minus_L4_holdout`` (unseen anchors) and ``TESSERA_minus_L4_matrix`` (all
    validation patches). A quantity without both sides is skipped with a note.
    """
    if ARM_TESSERA not in matrix and ARM_TESSERA not in holdout:
        return {}
    positions = assignment.positions()

    def on(stratum: str) -> Callable[[PatchScore], bool]:
        if stratum == ALL:
            return lambda score: bool(_strata(score.anchor, assignment, positions))
        return lambda score: stratum in _strata(score.anchor, assignment, positions)

    out: dict[str, Any] = {
        "rule": (
            "every comparison is computed on the intersection of the patch sets of all compared "
            "runs; 'intersection' names the patches and scenes each side lost"
        ),
    }
    # as for G of the nine arms, each holdout seed is paired with the matrix run of that seed
    paired = (
        _runs_for_seeds(
            matrix[ARM_TESSERA], [run.seed for run in holdout[ARM_TESSERA]], "G_TESSERA"
        )
        if ARM_TESSERA in matrix and ARM_TESSERA in holdout
        else None
    )
    plan = (
        ("G_TESSERA", holdout.get(ARM_TESSERA), paired, UNSEEN, "G TESSERA", True),
        (
            "TESSERA_minus_L4_holdout",
            holdout.get(ARM_TESSERA),
            holdout.get(ARM_FULL),
            UNSEEN,
            "holdout: TESSERA - L4",
            False,
        ),
        (
            "TESSERA_minus_L4_matrix",
            matrix.get(ARM_TESSERA),
            matrix.get(ARM_FULL),
            ALL,
            "matrix: TESSERA - L4",
            False,
        ),
    )
    for name, runs_a, runs_b, stratum, label, same_arm in plan:
        if not runs_a or not runs_b:
            notes.append(f"{name}: skipped, the runs of one side are missing")
            continue
        seeds = [sorted((run.seed for run in runs), key=str) for runs in (runs_a, runs_b)]
        if seeds[0] != seeds[1]:
            notes.append(f"{name}: skipped, seeds differ ({seeds[0]} against {seeds[1]})")
            continue
        out[name] = {
            stratum: _intersection_difference(
                runs_a,
                runs_b,
                on(stratum),
                stratum,
                label=label,
                same_arm=same_arm,
                tessera_invalid=tessera_invalid,
                n_bootstrap=n_bootstrap,
                rng_seed=rng_seed,
            )
        }
    if tessera_invalid is None and any(not q.endswith("G_TESSERA") for q in out if q != "rule"):
        notes.append(
            "TESSERA minus L4: L4's patch loss is unverified against the TESSERA run's own "
            "tessera_invalid record (pass its data_scope.json: --tessera-scope)"
        )
    return out


def evaluate(
    assignment: SpatialHoldoutAssignment,
    *,
    matrix: Mapping[str, Sequence[Run]],
    holdout: Mapping[str, Sequence[Run]] | None = None,
    tessera_invalid: frozenset[str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Evaluate matrix runs, and holdout runs when given, under one assignment.

    ``matrix`` and ``holdout`` map an arm (any of the nine matrix arms) to its
    runs. Without holdout runs the result holds the matrix models' MAE per
    stratum and their unseen-minus-seen gap. With them it adds the holdout
    models, ``G`` per arm and ``K``, and applies the stage-2 trigger of the plan
    to ``G`` of L1 and L4 and to ``K``. ``pairings`` holds the matrix families
    for each model family; ``notes`` names what was skipped for lack of exports.
    """
    kwargs = {"n_bootstrap": n_bootstrap, "rng_seed": rng_seed}
    groups: dict[str, Sequence[Run]] = {f"matrix/{arm}": runs for arm, runs in matrix.items()}
    for arm, runs in (holdout or {}).items():
        groups[f"holdout/{arm}"] = runs
    result: dict[str, Any] = {
        "split": "validation",
        "metric": "mae_100m",
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "rng_seed": rng_seed,
            "method": "percentile",
            "level": 0.95,
            "unit": "scene",
        },
        "assignment": {
            "seed": assignment.seed,
            "neighbourhood": assignment.neighbourhood,
            "anchors_total": len(assignment.anchors),
            "holdout_anchors": len(assignment.holdout),
            "blocks": [list(block) for block in assignment.blocks],
            "patch_index": dict(assignment.patch_index),
        },
        "mae": {
            name: {
                stratum: stratum_mae(runs, assignment, stratum, label=name, **kwargs).to_dict()
                for stratum in STRATA
            }
            for name, runs in groups.items()
        },
        "unseen_minus_seen": {
            name: stratum_gap(
                runs, assignment, label=name, n_bootstrap=n_bootstrap, rng_seed=rng_seed
            ).to_dict()
            for name, runs in groups.items()
        },
    }
    for family, arms in (("matrix", matrix), ("holdout", holdout or {})):
        if ARM_SPECTRAL in arms and ARM_FULL in arms:
            if {run.seed for run in arms[ARM_FULL]} != {run.seed for run in arms[ARM_SPECTRAL]}:
                continue
            result.setdefault("L4_minus_L1", {})[family] = {
                stratum: paired_difference(
                    arms[ARM_FULL],
                    arms[ARM_SPECTRAL],
                    assignment,
                    stratum,
                    label=f"{family}: {ARM_FULL} - {ARM_SPECTRAL}",
                    **kwargs,
                ).to_dict()
                for stratum in STRATA
            }
    notes: list[str] = []
    result["pairings"] = {}
    for family, arms in (("matrix", matrix), ("holdout", holdout or {})):
        if arms:
            result["pairings"][family] = _pairings(assignment, arms, family, notes, **kwargs)
    result["notes"] = notes
    section = tessera_section(
        assignment,
        matrix,
        holdout or {},
        notes,
        tessera_invalid=tessera_invalid,
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
    )
    if section:
        result["tessera"] = section
    if not holdout:
        return result

    unseen_strata = (UNSEEN, UNSEEN_INTERIOR, UNSEEN_EDGE)
    decisive: dict[str, Estimate] = {}
    result["G"] = {}
    for arm in sorted(set(matrix) - set(holdout) - {ARM_TESSERA}):
        notes.append(f"G {arm}: skipped, no holdout export")
    for arm, runs in holdout.items():
        if arm == ARM_TESSERA:
            continue  # G_TESSERA is computed on patch-set intersections (tessera_section)
        if arm not in matrix:
            notes.append(f"G {arm}: skipped, no matrix export")
            continue
        paired = _runs_for_seeds(matrix[arm], [run.seed for run in runs], f"G {arm}")
        per_stratum = {
            stratum: paired_difference(
                runs, paired, assignment, stratum, label=f"G {arm}: holdout - matrix", **kwargs
            )
            for stratum in unseen_strata
        }
        if arm in (ARM_SPECTRAL, ARM_FULL):
            decisive[f"G_{arm}"] = per_stratum[UNSEEN]
        result["G"][arm] = {stratum: est.to_dict() for stratum, est in per_stratum.items()}
    if ARM_SPECTRAL in holdout and ARM_FULL in holdout:
        per_stratum = {
            stratum: paired_difference(
                holdout[ARM_FULL],
                holdout[ARM_SPECTRAL],
                assignment,
                stratum,
                label=f"K: holdout {ARM_FULL} - holdout {ARM_SPECTRAL}",
                **kwargs,
            )
            for stratum in unseen_strata
        }
        decisive["K"] = per_stratum[UNSEEN]
        result["K"] = {stratum: est.to_dict() for stratum, est in per_stratum.items()}
    if decisive:
        result["stage2"] = _stage2_check(decisive)
    return result


CV_KIND = "spatial_holdout_cv"
CV_ARMS = (ARM_SPECTRAL, ARM_FULL)
CV_SEED = 0
OUT_OF_FOLD = "out_of_fold"


def _cv_folds(folds: Mapping[str, SpatialHoldoutAssignment]) -> dict[str, SpatialHoldoutAssignment]:
    """Check the five fold assignments and return them in fold order."""
    expected = {f"{CV_PARTITION}-fold{k}" for k in range(CV_FOLDS)}
    if set(folds) != expected:
        raise SpatialHoldoutEvalError(
            f"the cross-validation needs the assignments {sorted(expected)}, got {sorted(folds)}"
        )
    ordered = {name: folds[name] for name in sorted(expected)}
    first = ordered[f"{CV_PARTITION}-fold0"]
    for name, fold in ordered.items():
        if fold.fold != int(name.removeprefix(f"{CV_PARTITION}-fold")):
            raise SpatialHoldoutEvalError(f"{name} is not the fold file of that name")
        if fold.anchors != first.anchors or fold.patch_index != first.patch_index:
            raise SpatialHoldoutEvalError(
                f"{name} was drawn from another anchor table or patch index than the other folds"
            )
    unseen_in: dict[Anchor, list[str]] = {a: [] for a in first.anchors}
    for name, fold in ordered.items():
        for anchor in fold.holdout_anchors:
            unseen_in[anchor].append(name)
    bad = {
        a: names
        for a, names in unseen_in.items()
        if first.anchors[a].validation > 0 and len(names) != 1
    }
    if bad:
        raise SpatialHoldoutEvalError(
            f"{len(bad)} anchor(s) with validation patches are not unseen in exactly one fold, "
            f"e.g. {dict(sorted(bad.items())[:3])}"
        )
    return ordered


def _spread(points: Mapping[str, float]) -> dict[str, Any]:
    """Mean and range of one point estimate over the folds that have it."""
    values = list(points.values())
    return {
        "folds": list(points),
        "n_folds": len(values),
        "points": dict(points),
        "mean": float(np.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _out_of_fold_runs(
    folds: Mapping[str, SpatialHoldoutAssignment],
    holdout: Mapping[str, Mapping[str, Run]],
    arm: str,
) -> Run:
    """One run of ``arm`` that scores every patch with the fold model that did not train on it."""
    patches: dict[str, PatchScore] = {}
    for name, fold in folds.items():
        for pid, score in holdout[name][arm].patches.items():
            if score.anchor not in fold.holdout_anchors:
                continue
            if pid in patches:
                raise SpatialHoldoutEvalError(f"patch {pid!r} is out of fold in two folds")
            patches[pid] = score
    return Run(arm=arm, seed=CV_SEED, run_id=f"{OUT_OF_FOLD}/{arm}", patches=patches)


def _out_of_fold(
    folds: Mapping[str, SpatialHoldoutAssignment],
    matrix: Mapping[str, Run],
    holdout: Mapping[str, Mapping[str, Run]],
    *,
    n_bootstrap: int,
    rng_seed: int,
) -> dict[str, Any]:
    kwargs = {"n_bootstrap": n_bootstrap, "rng_seed": rng_seed}
    oof = {arm: _out_of_fold_runs(folds, holdout, arm) for arm in CV_ARMS}

    def keep_all(_score: PatchScore) -> bool:
        return True

    groups = {
        arm: _group_where([oof[arm]], keep_all, OUT_OF_FOLD, f"{OUT_OF_FOLD} {arm}")
        for arm in CV_ARMS
    }
    base = {
        arm: _group_where([matrix[arm]], keep_all, OUT_OF_FOLD, f"matrix {arm}") for arm in CV_ARMS
    }
    mae = {
        arm: _estimate(
            "mae_100m",
            f"{OUT_OF_FOLD} {arm}",
            OUT_OF_FOLD,
            group.seeds,
            group.values,
            sum(len(ids) for ids in group.patch_sets.values()),
            **kwargs,
        ).to_dict()
        for arm, group in groups.items()
    }
    k_cv = _paired_groups(
        groups[ARM_FULL],
        groups[ARM_SPECTRAL],
        OUT_OF_FOLD,
        label=f"K_cv: {OUT_OF_FOLD} {ARM_FULL} - {OUT_OF_FOLD} {ARM_SPECTRAL}",
        **kwargs,
    )
    g_cv = {
        arm: _paired_groups(
            groups[arm],
            base[arm],
            OUT_OF_FOLD,
            label=f"G_cv {arm}: {OUT_OF_FOLD} - matrix",
            **kwargs,
        )
        for arm in CV_ARMS
    }
    return {
        "complete": True,
        "patches": len(oof[ARM_SPECTRAL].patches),
        "mae": mae,
        "K_cv": k_cv.to_dict(),
        "G_cv": {arm: est.to_dict() for arm, est in g_cv.items()},
    }


def evaluate_cv(
    folds: Mapping[str, SpatialHoldoutAssignment],
    *,
    matrix: Mapping[str, Sequence[Run]],
    holdout: Mapping[str, Mapping[str, Sequence[Run]]],
    fold_sha256: Mapping[str, str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Evaluate the five-fold spatial cross-validation of L1 and L4 (seed 0).

    ``folds`` maps ``cv5-fold0`` ... ``cv5-fold4`` to the fold assignments;
    ``holdout[fold][arm]`` holds the run of that arm trained without the fold's
    anchors, read with ``load_fold_runs``; ``matrix[arm]`` holds the matrix runs, of
    which seed 0 is used. Every holdout run must carry the assignment and the SHA-256
    its own record names, and both must be the fold's it is filed under:
    ``fold_sha256`` (fold -> SHA-256 of the committed file) defaults to the hash of the
    canonical file bytes. A run whose record names another fold, another file or none
    (an image that predates the folds trains ``v1``) raises.

    - ``folds``: per fold the evaluation of ``evaluate`` on that fold's unseen
      anchors (MAE by stratum, G per arm against the matrix run of the same
      arm and seed, K = L4 - L1).
    - ``fold_summary``: mean and range of K and of G per arm over the folds
      that have them (the unseen stratum).
    - ``out_of_fold``: every validation patch scored by the fold model that did
      not train on its anchor: scene-equal MAE of L1 and L4, ``K_cv`` and
      ``G_cv`` per arm against the matrix seed-0 models on the same patches.
      Only computed when all five folds have both arms; otherwise it is marked
      incomplete and names what is missing.
    """
    kwargs = {"n_bootstrap": n_bootstrap, "rng_seed": rng_seed}
    ordered = _cv_folds(folds)
    unknown = sorted(set(holdout) - set(ordered))
    if unknown:
        raise SpatialHoldoutEvalError(f"holdout runs for unknown fold(s) {unknown}")
    base: dict[str, Run] = {}
    for arm in CV_ARMS:
        runs = [run for run in matrix.get(arm, ()) if run.seed == CV_SEED]
        if len(runs) != 1:
            raise SpatialHoldoutEvalError(
                f"the cross-validation needs exactly one matrix seed-{CV_SEED} run of {arm}, "
                f"got {len(runs)}"
            )
        base[arm] = runs[0]
    fold_runs: dict[str, dict[str, Run]] = {}
    for name, by_arm in holdout.items():
        fold_runs[name] = {}
        for arm, runs in by_arm.items():
            if arm not in CV_ARMS or len(runs) != 1 or runs[0].seed != CV_SEED:
                raise SpatialHoldoutEvalError(
                    f"{name} {arm}: the cross-validation holds one seed-{CV_SEED} run of each of "
                    f"{list(CV_ARMS)}, got {len(runs)} run(s) with seeds {[r.seed for r in runs]}"
                )
            fold_runs[name][arm] = runs[0]

    expected = {
        name: (fold_sha256 or {}).get(name) or hashlib.sha256(assignment_bytes(fold)).hexdigest()
        for name, fold in ordered.items()
    }
    for name, by_arm in fold_runs.items():
        for arm, run in by_arm.items():
            where = f"{run.source or run.label} ({name} {arm})"
            if run.assignment is None or run.assignment_sha256 is None:
                raise SpatialHoldoutEvalError(
                    f"{where}: no recorded assignment; read the export with load_fold_runs"
                )
            if run.assignment != name:
                raise SpatialHoldoutEvalError(
                    f"{where}: filed under {name}, but its run record names "
                    f"{run.assignment!r}; that run did not train without {name}"
                )
            if run.assignment_sha256 != expected[name]:
                raise SpatialHoldoutEvalError(
                    f"{where}: its run record has sha256 {run.assignment_sha256[:16]}, the "
                    f"committed {name} file has {expected[name][:16]}"
                )

    result: dict[str, Any] = {
        "kind": CV_KIND,
        "partition": CV_PARTITION,
        "split": "validation",
        "metric": "mae_100m",
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "rng_seed": rng_seed,
            "method": "percentile",
            "level": 0.95,
            "unit": "scene",
        },
        "folds": {},
    }
    notes: list[str] = []
    missing: dict[str, list[str]] = {}
    for name, fold in ordered.items():
        held = fold.holdout_anchors
        present = fold_runs.get(name, {})
        absent = [arm for arm in CV_ARMS if arm not in present]
        if absent:
            missing[name] = absent
            notes.append(f"{name}: no holdout export of {', '.join(absent)}")
        result["folds"][name] = {
            "blocks": len(fold.blocks),
            "anchors": len(held),
            "train_patches": sum(fold.anchors[a].train for a in held),
            "validation_patches": sum(fold.anchors[a].validation for a in held),
            "evaluation": (
                evaluate(
                    fold,
                    matrix={arm: [base[arm]] for arm in CV_ARMS},
                    holdout={arm: [run] for arm, run in present.items()},
                    n_bootstrap=n_bootstrap,
                    rng_seed=rng_seed,
                )
                if present
                else None
            ),
        }
    result["missing"] = missing

    summary: dict[str, Any] = {}
    k_points = {
        name: ev["K"][UNSEEN]["point"]
        for name, entry in result["folds"].items()
        if (ev := entry["evaluation"]) is not None and "K" in ev
    }
    if k_points:
        summary["K"] = _spread(k_points)
    for arm in CV_ARMS:
        g_points = {
            name: ev["G"][arm][UNSEEN]["point"]
            for name, entry in result["folds"].items()
            if (ev := entry["evaluation"]) is not None and arm in ev.get("G", {})
        }
        if g_points:
            summary.setdefault("G", {})[arm] = _spread(g_points)
    result["fold_summary"] = summary

    if missing:
        result[OUT_OF_FOLD] = {
            "complete": False,
            "reason": "folds without both arms: "
            + "; ".join(f"{name} lacks {', '.join(arms)}" for name, arms in missing.items()),
        }
    else:
        result[OUT_OF_FOLD] = _out_of_fold(ordered, base, fold_runs, **kwargs)
    result["notes"] = notes
    return result


def _fmt(est: Mapping[str, Any]) -> str:
    low, high = est["ci_95"]
    return f"{est['point']:.4f} [{low:.4f}, {high:.4f}]"


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the evaluation as Markdown tables (kelvin; 95% scene-bootstrap intervals)."""
    boot = result["bootstrap"]
    lines = [
        "# Spatial holdout evaluation",
        "",
        f"Split: {result['split']}; metric: {result['metric']} in K, scene-equal. Intervals: 95% "
        f"percentile, paired scene bootstrap (B={boot['n_bootstrap']}, "
        f"rng_seed={boot['rng_seed']}). n = scenes with a patch on the stratum / patches.",
        "",
        "## MAE by stratum",
        "",
        "| model | seeds | " + " | ".join(STRATA) + " |",
        "| --- | --- | " + " | ".join("---" for _ in STRATA) + " |",
    ]
    for name, strata in result["mae"].items():
        seeds = ",".join(str(s) for s in strata[SEEN]["seeds"])
        cells = [
            f"{_fmt(strata[s])} (n={strata[s]['n_scenes']}/{strata[s]['n_patches']})"
            for s in STRATA
        ]
        lines.append(f"| {name} | {seeds} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Unseen minus seen, same model (scenes with both)",
        "",
        "| model | difference | scenes |",
        "| --- | --- | --- |",
    ]
    for name, est in result["unseen_minus_seen"].items():
        lines.append(f"| {name} | {_fmt(est)} | {est['n_scenes']} |")
    for family, strata in result.get("L4_minus_L1", {}).items():
        lines += ["", f"## L4 minus L1, {family} models", ""]
        lines += ["| stratum | difference | scenes |", "| --- | --- | --- |"]
        for stratum in STRATA:
            est = strata[stratum]
            lines.append(f"| {stratum} | {_fmt(est)} | {est['n_scenes']} |")
    if "G" in result:
        lines += ["", "## G: holdout model minus matrix model, unseen anchors", ""]
        lines += ["| arm | stratum | G | scenes |", "| --- | --- | --- | --- |"]
        for arm, strata in result["G"].items():
            for stratum, est in strata.items():
                lines.append(f"| {arm} | {stratum} | {_fmt(est)} | {est['n_scenes']} |")
    if "tessera" in result:
        lines += ["", "## TESSERA, on patch-set intersections", "", result["tessera"]["rule"], ""]
        lines += [
            "| quantity | stratum | difference | scenes | patches | A lost (patches, scenes) "
            "| B lost (patches, scenes) | a | b | c |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, strata in result["tessera"].items():
            if name == "rule":
                continue
            for stratum, est in strata.items():
                side = est["intersection"]
                lost_a = f"{side['a']['patches_lost']}, {side['a']['scenes_lost']}"
                lost_b = f"{side['b']['patches_lost']}, {side['b']['scenes_lost']}"
                crit = est["criteria"]
                flags = (
                    [
                        "yes" if crit[key] else "no"
                        for key in (
                            "a_interval_excludes_zero",
                            "b_same_sign_all_seeds",
                            "c_abs_difference_at_least_2_seed_sd",
                        )
                    ]
                    if crit["available"]
                    else ["n/a"] * 3
                )
                lines.append(
                    f"| {name} | {stratum} | {_fmt(est)} | {est['n_scenes']} | "
                    f"{side['patches']} | {lost_a} | {lost_b} | " + " | ".join(flags) + " |"
                )
    if "K" in result:
        lines += ["", "## K: holdout L4 minus holdout L1, unseen anchors", ""]
        lines += ["| stratum | K | scenes |", "| --- | --- | --- |"]
        for stratum, est in result["K"].items():
            lines.append(f"| {stratum} | {_fmt(est)} | {est['n_scenes']} |")
    for model, pairings in result.get("pairings", {}).items():
        if not pairings:
            continue
        lines += ["", f"## Paired differences A - B, {model} models", ""]
        lines += [
            "Criteria (decision rule of the matrix): a = interval excludes 0, b = same sign in "
            "all seeds, c = |difference| >= 2 x the larger seed SD; n/a without seeds 0, 1, 2.",
            "",
            "| pairing | family | stratum | difference | scenes | a | b | c |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for pairing in pairings:
            for stratum in STRATA:
                est = pairing["strata"][stratum]
                crit = est["criteria"]
                flags = (
                    [
                        "yes" if crit[key] else "no"
                        for key in (
                            "a_interval_excludes_zero",
                            "b_same_sign_all_seeds",
                            "c_abs_difference_at_least_2_seed_sd",
                        )
                    ]
                    if crit["available"]
                    else ["n/a"] * 3
                )
                lines.append(
                    f"| {pairing['name']} | {pairing['family']} | {stratum} | {_fmt(est)} | "
                    f"{est['n_scenes']} | " + " | ".join(flags) + " |"
                )
    if result.get("notes"):
        lines += ["", "## Skipped", ""]
        lines += [f"- {note}" for note in result["notes"]]
    if "stage2" in result:
        stage2 = result["stage2"]
        lines += ["", "## Stage-2 trigger", "", stage2["rule"] + ".", ""]
        for name, check in stage2["checks"].items():
            lines.append(
                f"- {name}: interval excludes 0: {check['interval_excludes_zero']}; "
                f"|point| >= threshold: {check['abs_point_reaches_threshold']}"
            )
        lines += ["", f"Triggered: {stage2['triggered']}."]
    return "\n".join(lines) + "\n"


def render_cv_markdown(result: Mapping[str, Any]) -> str:
    """Render the cross-validation: fold table, fold summary, out-of-fold, then each fold."""
    boot = result["bootstrap"]
    lines = [
        "# Spatial holdout cross-validation (cv5, L1 and L4, seed 0)",
        "",
        f"Split: {result['split']}; metric: {result['metric']} in K, scene-equal. Intervals: 95% "
        f"percentile, paired scene bootstrap (B={boot['n_bootstrap']}, "
        f"rng_seed={boot['rng_seed']}).",
        "",
        "## Folds",
        "",
        "| fold | blocks | anchors | train patches | validation patches | evaluated |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, fold in result["folds"].items():
        done = "no" if fold["evaluation"] is None else "yes"
        lines.append(
            f"| {name} | {fold['blocks']} | {fold['anchors']} | {fold['train_patches']} | "
            f"{fold['validation_patches']} | {done} |"
        )
    if result["missing"]:
        lines += [
            "",
            "Missing: "
            + "; ".join(f"{n} lacks {', '.join(a)}" for n, a in result["missing"].items())
            + ".",
        ]
    summary = result["fold_summary"]
    if summary:
        lines += [
            "",
            "## K and G over the folds (unseen anchors of each fold)",
            "",
            "| quantity | folds | mean | min | max |",
            "| --- | --- | --- | --- | --- |",
        ]
        rows = ([("K = L4 - L1", summary["K"])] if "K" in summary else []) + [
            (f"G {arm}", spread) for arm, spread in summary.get("G", {}).items()
        ]
        for label, spread in rows:
            lines.append(
                f"| {label} | {spread['n_folds']} | {spread['mean']:.4f} | "
                f"{spread['min']:.4f} | {spread['max']:.4f} |"
            )
    oof = result[OUT_OF_FOLD]
    lines += ["", "## Out of fold", ""]
    if not oof["complete"]:
        lines.append(f"Incomplete, not computed: {oof['reason']}.")
    else:
        lines += [
            f"Every validation patch ({oof['patches']}) scored by the fold model that did not "
            "train on its anchor.",
            "",
            "| quantity | value | scenes |",
            "| --- | --- | --- |",
        ]
        for arm, est in oof["mae"].items():
            lines.append(f"| MAE {arm} | {_fmt(est)} | {est['n_scenes']} |")
        lines.append(f"| K_cv = L4 - L1 | {_fmt(oof['K_cv'])} | {oof['K_cv']['n_scenes']} |")
        for arm, est in oof["G_cv"].items():
            lines.append(f"| G_cv {arm} (out of fold - matrix) | {_fmt(est)} | {est['n_scenes']} |")
    for name, fold in result["folds"].items():
        if fold["evaluation"] is None:
            continue
        text = render_markdown(fold["evaluation"]).splitlines()
        lines += ["", f"# Fold {name}", ""]
        lines += [f"#{line}" if line.startswith("#") else line for line in text[2:]]
    if result["notes"]:
        lines += ["", "## Skipped", ""] + [f"- {note}" for note in result["notes"]]
    return "\n".join(lines) + "\n"


def write_outputs(result: Mapping[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    """Write ``<prefix>.json`` and ``<prefix>.md``."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.parent / (prefix.name + ".json")
    md_path = prefix.parent / (prefix.name + ".md")
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render_cv_markdown(result) if result.get("kind") == CV_KIND else render_markdown(result)
    md_path.write_text(text, encoding="utf-8")
    return json_path, md_path


__all__ = [
    "ALL",
    "ARM_TESSERA",
    "CV_ARMS",
    "CV_KIND",
    "FAMILY_A",
    "FAMILY_B",
    "FAMILY_K",
    "N_BOOTSTRAP",
    "RNG_SEED",
    "STAGE2_ABS_THRESHOLD_K",
    "STRATA",
    "Estimate",
    "PatchScore",
    "Run",
    "SpatialHoldoutEvalError",
    "anchor_from_patch_id",
    "evaluate",
    "evaluate_cv",
    "load_fold_runs",
    "load_runs",
    "paired_difference",
    "read_recorded_assignment",
    "read_tessera_invalid",
    "render_cv_markdown",
    "render_markdown",
    "scene_bootstrap_mean",
    "seed_rule_criteria",
    "strata_of",
    "tessera_section",
    "stratum_gap",
    "stratum_mae",
    "write_outputs",
]
