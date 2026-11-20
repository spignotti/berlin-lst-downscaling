"""Write per-patch test exports for selected frozen v6 matrix checkpoints.

Inference only, one opening of the 2025 test split for the frozen matrix: no
optimizer step, no new checkpoint, and no upload. Reads the checkpoints and the
published test split (plus the validation split for the best-score identity
recheck) and writes local files only. Not run in CI.

Usage
-----
    uv run python scripts/operators/export_v6_test_patches.py --list
    uv run python scripts/operators/export_v6_test_patches.py \\
        --all --output-root data/runs/v6-test-patches --device cpu
    uv run python scripts/operators/export_v6_test_patches.py \\
        --run-labels v6m-070b33c-l1-seed0-ep40-cos v6m-070b33c-l4-seed0-ep40-cos \\
        --output-root data/runs/v6-test-patches --device cpu

``--list`` prints the 27 matrix run labels and exits without touching GCS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from berlin_lst_downscaling.modeling.v6_test_score import run_test_export
from berlin_lst_downscaling.modeling.v6_val_rescore import rescore_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-labels", nargs="+", metavar="LABEL")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_runs",
        help="export every frozen matrix run label",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--list", action="store_true", help="print the matrix run labels")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        for run in rescore_inventory():
            print(f"{run.arm_id} seed{run.seed} {run.run_label}")
        return 0
    if args.all_runs and args.run_labels:
        parser.error("--all and --run-labels are mutually exclusive")
    if args.output_root is None or not (args.all_runs or args.run_labels):
        parser.error("--run-labels (or --all) and --output-root are required unless --list")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but CUDA is not available")
    labels = (
        [run.run_label for run in rescore_inventory()] if args.all_runs else list(args.run_labels)
    )
    manifest = run_test_export(run_labels=labels, output_root=args.output_root, device=args.device)
    runs = manifest["runs"]
    if not isinstance(runs, list):
        raise SystemExit("ERROR: manifest carries no run list")
    print(f"runs exported: {len(runs)}; split: test; uploaded: false")
    for entry in runs:
        print(
            json.dumps(
                {
                    key: entry[key]
                    for key in ("run_label", "patches", "scenes", "recomputed_test_mae_100m")
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
