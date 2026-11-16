"""Write per-cell validation exports for selected frozen v6 matrix checkpoints.

Inference only on validation 2024: no optimizer step, no new checkpoint, and no
test (2025) ref, metric, or export. Reads the checkpoints and the published
validation sources from GCS; writes local files only (nothing is uploaded).
Not run in CI.

Usage
-----
    uv run python scripts/operators/export_v6_validation_cells.py --list
    uv run python scripts/operators/export_v6_validation_cells.py \\
        --run-labels v6m-070b33c-l1-seed0-ep40-cos v6m-070b33c-l4-seed0-ep40-cos \\
        --output-root data/runs/v6-val-cells/<label> --device cpu

``--list`` prints the 27 matrix run labels and exits without touching GCS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from berlin_lst_downscaling.modeling.v6_val_cell_export import run_cell_export
from berlin_lst_downscaling.modeling.v6_val_rescore import rescore_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-labels", nargs="+", metavar="LABEL")
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
    if not args.run_labels or args.output_root is None:
        parser.error("--run-labels and --output-root are required unless --list")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but CUDA is not available")
    manifest = run_cell_export(
        run_labels=args.run_labels, output_root=args.output_root, device=args.device
    )
    runs = manifest["runs"]
    if not isinstance(runs, list):
        raise SystemExit("ERROR: manifest carries no run list")
    print(f"runs exported: {len(runs)}; split: validation; evaluate_test: false")
    for entry in runs:
        print(json.dumps({k: entry[k] for k in ("run_label", "cells", "patches", "scenes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
