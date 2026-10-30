"""CLI: paired scene bootstrap between two ablation arms from existing patch exports.

This tool only reads explicit local export files; it never launches evaluation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from berlin_lst_downscaling.analysis.scene_bootstrap import (
    SceneBootstrapError,
    compare_arms,
    load_patch_exports,
    render_markdown,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-a", nargs=3, type=Path, required=True, metavar="FILE")
    p.add_argument("--arm-b", nargs=3, type=Path, required=True, metavar="FILE")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--i-know-test-is-frozen", action="store_true")
    p.add_argument("--metric", choices=("mae_100m", "ssim_100m"), default="mae_100m")
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--allow-scene-mismatch", action="store_true")
    p.add_argument("--output-prefix", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.split == "test" and not args.i_know_test_is_frozen:
        print(
            "error: --split test reads frozen test exports; pass --i-know-test-is-frozen",
            file=sys.stderr,
        )
        return 2
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
