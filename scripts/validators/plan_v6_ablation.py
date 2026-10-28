"""List the planned v6 ablation runs from the matrix file; submits nothing."""

from __future__ import annotations

import argparse
from pathlib import Path

from berlin_lst_downscaling.modeling.v6_ablation_matrix import (
    load_v6_matrix,
    plan_runs,
    plan_totals,
)

_DEFAULT_MATRIX = Path(__file__).resolve().parents[2] / "configs" / "ablation" / "v6_matrix.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=_DEFAULT_MATRIX)
    args = parser.parse_args(argv)
    matrix = load_v6_matrix(args.matrix)
    for run in plan_runs(matrix):
        print(f"arm={run.arm_id} config={run.config_name} seed={run.seed} epochs={run.max_epochs}")
    runs, hours, usd = plan_totals(matrix)
    print(
        f"TOTAL runs={runs} est_gpu_hours={hours:.1f} est_usd={usd:.1f} "
        "(planning estimates, nothing submitted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
