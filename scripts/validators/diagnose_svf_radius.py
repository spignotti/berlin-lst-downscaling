#!/usr/bin/env python3
"""Read-only SVF radius diagnostic.

Computes sky-view factor at several search radii on a small sample and
prints summary statistics, pairwise correlations, and a runtime estimate.
It writes no raster and does not read GCS.

Examples
--------
    uv run python scripts/validators/diagnose_svf_radius.py --synthetic
"""

from __future__ import annotations

import argparse

import numpy as np

from berlin_lst_downscaling.data.secondary.svf_radius_diagnostic import (
    SVF_DIAGNOSTIC_RADII_M,
    diagnose_svf_radii,
    synthetic_dsm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SVF radius diagnostic")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a built-in 64x64 DSM. No file is read or written.",
    )
    parser.add_argument(
        "--npy",
        default=None,
        help="Path to a local 2-D float .npy DSM sample. Not a full-AOI raster.",
    )
    parser.add_argument(
        "--radii-m",
        type=float,
        nargs="+",
        default=list(SVF_DIAGNOSTIC_RADII_M),
        help="Search radii in metres (default: 30 50 100 200).",
    )
    args = parser.parse_args()
    if args.synthetic == bool(args.npy):
        parser.error("pass exactly one of --synthetic or --npy")
    if args.synthetic:
        dsm = synthetic_dsm()
    else:
        dsm = np.load(args.npy)
    report = diagnose_svf_radii(dsm, radii_m=tuple(args.radii_m))
    print(report.format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
