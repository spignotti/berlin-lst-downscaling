"""CLI: spatial holdout evaluation from existing per-patch validation exports.

Reads explicit local export files and a committed holdout assignment; it never
launches an evaluation and accepts the validation split only.

Usage
-----
    # matrix models only: are the held-out anchors easier or harder than the rest?
    uv run python scripts/analyze_spatial_holdout.py \\
        --matrix-export <L1 seed0> <L1 seed1> <L1 seed2> <L4 seed0> <L4 seed1> <L4 seed2> \\
        --output-prefix data/analysis/spatial-holdout/matrix

    # after the holdout runs: adds the holdout models, G per arm and K
    uv run python scripts/analyze_spatial_holdout.py \\
        --matrix-export <...> --holdout-export <L1 holdout seed0> <L4 holdout seed0> \\
        --output-prefix data/analysis/spatial-holdout/stage1

The arm and seed of an export are read from the file. G pairs each holdout run
with the matrix run of the same arm and seed.
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
    load_runs,
    render_markdown,
    write_outputs,
)
from berlin_lst_downscaling.modeling.spatial_holdout import (
    SpatialHoldoutError,
    load_assignment,
    resolve_assignment_path,
)


def _by_arm(runs: Sequence[Run]) -> dict[str, list[Run]]:
    grouped: dict[str, list[Run]] = {}
    for run in sorted(runs, key=lambda r: (r.arm, str(r.seed))):
        grouped.setdefault(run.arm, []).append(run)
    return grouped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", default="v1", help="assignment version or .json path")
    parser.add_argument("--matrix-export", nargs="+", type=Path, required=True, metavar="FILE")
    parser.add_argument("--holdout-export", nargs="+", type=Path, default=[], metavar="FILE")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
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
