# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare arithmetic vs T4 prior aggregation (issue #80, manual operator script)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from berlin_lst_downscaling.analysis.training_aggregation import (
    estimate_cost,
    run_synthetic_analysis,
    write_outputs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="fresh local output directory",
    )
    parser.add_argument(
        "--train-scope",
        choices=("full", "sample30"),
        default="full",
        help="train scene selection (validation/test always complete)",
    )
    parser.add_argument("--runtime-seconds", type=float, default=None, help="bound processing time")
    parser.add_argument("--training-root", default=None, help="published training release root")
    parser.add_argument("--patch-index-root", default=None, help="strict 244/256 patch index root")
    parser.add_argument("--ard-root", default=None, help="Landsat ARD ledger root")
    parser.add_argument(
        "--prediction-raster",
        default=None,
        help="optional versioned 10 m prediction raster URI (comparison uncomputed when absent)",
    )
    parser.add_argument(
        "--prediction-source",
        default=None,
        help="label for the optional prediction raster provenance",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run bounded synthetic analysis (CPU tests / dry run without GCS)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or new: {args.output_dir}")

    if args.synthetic or not args.training_root:
        print(
            json.dumps(
                {
                    "cost_estimate": estimate_cost(
                        train_scene_count=3 if args.train_scope == "full" else 2,
                        val_scene_count=1,
                        test_scene_count=1,
                    ),
                    "note": "Pre-analysis cost estimate (synthetic path).",
                },
                indent=2,
            )
        )
        result = run_synthetic_analysis(
            train_scope=args.train_scope,
            runtime_seconds=args.runtime_seconds,
            strict_patch_blocks={"S2017": {"S2017:B1"}},
        )
        if args.prediction_raster:
            result.prediction_10m_comparison = {
                "status": "uncomputed",
                "reason": "synthetic run does not read rasters",
                "source": args.prediction_source,
                "uri": args.prediction_raster,
            }
        paths = write_outputs(result, args.output_dir)
        print(json.dumps({"written": paths, "incomplete": result.incomplete}, indent=2))
        return 0

    missing = [
        name
        for name, value in (
            ("training-root", args.training_root),
            ("patch-index-root", args.patch_index_root),
            ("ard-root", args.ard_root),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"real-data analysis requires: {', '.join(missing)} (or pass --synthetic)")

    raise SystemExit(
        "Real-data execution is manual and not run from CI. "
        "Use --synthetic for offline validation or run on the VM with the required roots."
    )


if __name__ == "__main__":
    raise SystemExit(main())
