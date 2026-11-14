"""CLI: exploratory spatial error analysis of validation runs from existing exports.

Reads explicit local files only; it never launches evaluation and never reads a
test export. Runs are named by their run label; every step whose inputs are
present runs, the others are listed as skipped:

- per-patch exports ``<patch-export-root>/<label>/eval_validation_patches.parquet``:
  the per-scene table of A - B and the pooled against scene-equal comparison;
- per-cell exports ``<cell-export-root>/<label>/eval_validation_cells.parquet``:
  scene bias and centred residuals, Moran's I, and, with ``--stadtstruktur-shares``,
  the metrics per Stadtstruktur group;
- ``--reference-a``/``--reference-b``: paired scene bootstrap between two
  single-run reference exports (random forest stages), A - B.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from berlin_lst_downscaling.analysis import spatial_error as se
from berlin_lst_downscaling.analysis.scene_bootstrap import SceneBootstrapError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-labels", nargs="+", required=True, metavar="LABEL")
    p.add_argument("--arm-a", required=True, help="arm A of the paired difference A - B")
    p.add_argument("--arm-b", required=True, help="arm B of the paired difference A - B")
    p.add_argument("--patch-export-root", type=Path)
    p.add_argument("--cell-export-root", type=Path)
    p.add_argument(
        "--stadtstruktur-shares",
        type=Path,
        help="local copy of evaluation/stadtstruktur/v1/grid/100m/<vintage>/shares.parquet",
    )
    p.add_argument(
        "--scene-properties",
        type=Path,
        help="local copy of evaluation/eda/v3/tables/lst_distribution.parquet",
    )
    p.add_argument("--reference-a", type=Path, help="single-run per-patch export, arm A")
    p.add_argument("--reference-b", type=Path, help="single-run per-patch export, arm B")
    p.add_argument("--n-bootstrap", type=int, default=se.DEFAULT_N_BOOTSTRAP)
    p.add_argument("--rng-seed", type=int, default=se.DEFAULT_RNG_SEED)
    p.add_argument(
        "--min-effective-cells",
        type=float,
        default=se.DEFAULT_MIN_EFFECTIVE_CELLS,
        help="minimum summed group weight for a scene to count for a group",
    )
    p.add_argument(
        "--min-scenes",
        type=int,
        default=se.DEFAULT_MIN_SCENES,
        help="minimum supported scenes for a group interval",
    )
    p.add_argument("--share-column", choices=se.SHARE_COLUMNS, default=se.DEFAULT_SHARE_COLUMN)
    p.add_argument(
        "--min-share",
        type=float,
        default=se.DEFAULT_MIN_SHARE,
        help="drop a group in a cell below this share (0 keeps every mixed cell)",
    )
    p.add_argument(
        "--moran-neighbourhood", choices=se.NEIGHBOURHOODS, default=se.DEFAULT_NEIGHBOURHOOD
    )
    p.add_argument("--moran-weights", choices=se.MORAN_WEIGHTS, default=se.DEFAULT_MORAN_WEIGHTS)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.reference_a is None) != (args.reference_b is None):
        parser.error("--reference-a and --reference-b go together")
    config = se.AnalysisConfig(
        run_labels=tuple(args.run_labels),
        arm_a=args.arm_a,
        arm_b=args.arm_b,
        patch_export_root=args.patch_export_root,
        cell_export_root=args.cell_export_root,
        stadtstruktur_shares=args.stadtstruktur_shares,
        scene_properties=args.scene_properties,
        reference_a=args.reference_a,
        reference_b=args.reference_b,
        n_bootstrap=args.n_bootstrap,
        rng_seed=args.rng_seed,
        min_effective_cells=args.min_effective_cells,
        min_scenes=args.min_scenes,
        share_column=args.share_column,
        min_share=args.min_share,
        neighbourhood=args.moran_neighbourhood,
        moran_weights=args.moran_weights,
    )
    try:
        report = se.run_analysis(config)
        se.write_outputs(report, args.output_dir)
    except (se.SpatialErrorError, SceneBootstrapError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(se.render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
