"""CLI: paired scene bootstrap between two ablation arms from existing patch exports.

This tool only reads explicit local export files; it never launches evaluation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from berlin_lst_downscaling.analysis.scene_bootstrap import (
    METRICS,
    SceneBootstrapError,
    compare_arms,
    load_patch_exports,
    render_markdown,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    arm_help = (
        "three seed exports, or one export of a single-run reference arm (baseline or "
        "random forest), which is replicated against each seed of the other arm"
    )
    p.add_argument("--arm-a", nargs="+", type=Path, required=True, metavar="FILE", help=arm_help)
    p.add_argument("--arm-b", nargs="+", type=Path, required=True, metavar="FILE", help=arm_help)
    p.add_argument("--split", choices=("validation",), default="validation")
    p.add_argument(
        "--metric",
        choices=METRICS,
        default="mae_100m",
        help=(
            "mae_100m, ssim_100m (main SSIM, fixed artifact data range; also reports the "
            "ssim_100m_L250 appendix comparison), or ssim_100m_L250 alone"
        ),
    )
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--allow-scene-mismatch", action="store_true")
    p.add_argument("--output-prefix", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for flag, files in (("--arm-a", args.arm_a), ("--arm-b", args.arm_b)):
        if len(files) not in (1, 3):
            parser.error(f"{flag} takes 3 seed exports or 1 reference export, got {len(files)}")
    inputs = [*args.arm_a, *args.arm_b]
    prefix: Path = args.output_prefix
    outs = {
        (prefix.parent / (prefix.name + ".json")).resolve(),
        (prefix.parent / (prefix.name + ".md")).resolve(),
    }
    if outs & {Path(p).resolve() for p in inputs}:
        print("error: output path collides with an input file", file=sys.stderr)
        return 2
    try:
        table_a = load_patch_exports(args.arm_a, split=args.split)
        table_b = load_patch_exports(args.arm_b, split=args.split)
        result = compare_arms(
            table_a,
            table_b,
            metric=args.metric,
            n_bootstrap=args.n_bootstrap,
            rng_seed=args.rng_seed,
            allow_scene_mismatch=args.allow_scene_mismatch,
        )
        write_outputs(result, prefix)
    except (SceneBootstrapError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
