"""CLI: spatial holdout evaluation from existing per-patch validation exports.

Reads explicit local export files and a committed holdout assignment; it never
launches an evaluation and accepts the validation split only.

Usage
-----
    # matrix models only: are the held-out anchors easier or harder than the rest?
    uv run python scripts/analyze_spatial_holdout.py \\
        --matrix-export <L1 seed0> <L1 seed1> <L1 seed2> <L4 seed0> <L4 seed1> <L4 seed2> \\
        --output-prefix data/analysis/spatial-holdout/matrix

    # after the holdout runs: adds the holdout models, G per arm and K, and the
    # paired differences of the matrix families A and B (any of the nine arms)
    uv run python scripts/analyze_spatial_holdout.py \\
        --matrix-export <...> --holdout-export <holdout seed0 of each arm> <...> \\
        --output-prefix data/analysis/spatial-holdout/stage1

Spatial cross-validation of L1 and L4 (seed 0, assignments ``cv5-fold0`` ...
``cv5-fold4``): name the fold of each holdout export explicitly, once per fold,
and pass the matrix exports of L1 and L4 (seed 0 is used):

    uv run python scripts/analyze_spatial_holdout.py \\
        --matrix-export <L1 seed0> <L4 seed0> \\
        --fold-export cv5-fold0 <fold0 L1> <fold0 L4> \\
        --fold-export cv5-fold1 <fold1 L1> <fold1 L4> ... \\
        --output-prefix data/analysis/spatial-holdout/cv5

The output holds each fold's evaluation, the mean and range of K and G over the
folds, and the out-of-fold summary (incomplete when a fold or an arm is missing).

The arm and seed of an export are read from the file. G pairs each holdout run
with the matrix run of the same arm and seed. An arm without exports is skipped
with a note under "Skipped" in the output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from berlin_lst_downscaling.analysis.scene_bootstrap import SceneBootstrapError
from berlin_lst_downscaling.analysis.spatial_holdout_eval import (
    N_BOOTSTRAP,
    RNG_SEED,
    Run,
    SpatialHoldoutEvalError,
    evaluate,
    evaluate_cv,
    load_runs,
    render_cv_markdown,
    render_markdown,
    write_outputs,
)
from berlin_lst_downscaling.modeling.spatial_holdout import (
    SPATIAL_HOLDOUT_CV_ASSIGNMENTS,
    SpatialHoldoutError,
    load_assignment,
    resolve_assignment_path,
)


def _by_arm(runs: Sequence[Run]) -> dict[str, list[Run]]:
    grouped: dict[str, list[Run]] = {}
    for run in sorted(runs, key=lambda r: (r.arm, str(r.seed))):
        grouped.setdefault(run.arm, []).append(run)
    return grouped


def _main_cv(args: argparse.Namespace) -> int:
    by_fold: dict[str, dict[str, list[Run]]] = {}
    for fold, *files in args.fold_export:
        if fold not in SPATIAL_HOLDOUT_CV_ASSIGNMENTS:
            raise SpatialHoldoutEvalError(
                f"unknown fold {fold!r}; expected one of {list(SPATIAL_HOLDOUT_CV_ASSIGNMENTS)}"
            )
        if fold in by_fold or not files:
            raise SpatialHoldoutEvalError(f"fold {fold!r} is given twice or without exports")
        by_fold[fold] = _by_arm(load_runs([Path(f) for f in files]))
    result = evaluate_cv(
        {
            name: load_assignment(resolve_assignment_path(name))
            for name in SPATIAL_HOLDOUT_CV_ASSIGNMENTS
        },
        matrix=_by_arm(load_runs(args.matrix_export)),
        holdout=by_fold,
        n_bootstrap=args.n_bootstrap,
        rng_seed=args.rng_seed,
    )
    json_path, md_path = write_outputs(result, args.output_prefix)
    print(render_cv_markdown(result))
    print(f"wrote {json_path} and {md_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", default="v1", help="assignment version or .json path")
    parser.add_argument("--matrix-export", nargs="+", type=Path, required=True, metavar="FILE")
    parser.add_argument("--holdout-export", nargs="+", type=Path, default=[], metavar="FILE")
    parser.add_argument(
        "--fold-export",
        nargs="+",
        action="append",
        default=[],
        metavar=("FOLD", "FILE"),
        help="cross-validation: a fold (cv5-fold0 ... cv5-fold4) followed by its holdout exports",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fold_export and args.holdout_export:
        parser.error("--fold-export replaces --holdout-export; pass one of them")
    try:
        if args.fold_export:
            return _main_cv(args)
        assignment = load_assignment(resolve_assignment_path(args.assignment))
        holdout = _by_arm(load_runs(args.holdout_export)) if args.holdout_export else None
        result = evaluate(
            assignment,
            matrix=_by_arm(load_runs(args.matrix_export)),
            holdout=holdout,
            n_bootstrap=args.n_bootstrap,
            rng_seed=args.rng_seed,
        )
    except (
        SceneBootstrapError,
        SpatialHoldoutEvalError,
        SpatialHoldoutError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_outputs(result, args.output_prefix)
    print(render_markdown(result))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
