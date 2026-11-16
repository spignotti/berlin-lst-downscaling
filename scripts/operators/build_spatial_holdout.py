"""Draw the spatial holdout assignment from a published patch index (pm #553).

Reads only the patch index (``row``, ``col``, ``split``, ``year``) and its
completion marker; no raster, no metric. The rule lives in
``berlin_lst_downscaling.modeling.spatial_holdout``. The written file is the
input of the holdout training option and of the holdout evaluation.

Usage
-----
    # print the assignment and its statistics, write nothing
    uv run python scripts/operators/build_spatial_holdout.py

    # write the committed file (refuses to overwrite a different one)
    uv run python scripts/operators/build_spatial_holdout.py \\
        --output configs/modeling/spatial_holdout/v1.json

    # verify the committed file against the published index
    uv run python scripts/operators/build_spatial_holdout.py \\
        --check configs/modeling/spatial_holdout/v1.json

``--patch-index-dir`` reads a local copy of ``patch_index.parquet`` and
``complete.json`` instead of the ``gs://`` root; the recorded root stays the
published one.

Spatial cross-validation (``--partition cv5``): the five fold files
``cv5-fold0.json`` ... ``cv5-fold4.json`` partition ALL anchors, no random draw.
``--anchors-from`` takes the anchor table and the index fingerprints from a
committed assignment (``v1.json``, verified against the published index with
``--check``) instead of reading the index; nothing is read from the cloud then.

    uv run python scripts/operators/build_spatial_holdout.py --partition cv5 \
        --anchors-from configs/modeling/spatial_holdout/v1.json \
        --output-dir configs/modeling/spatial_holdout
    uv run python scripts/operators/build_spatial_holdout.py --partition cv5 \
        --anchors-from configs/modeling/spatial_holdout/v1.json \
        --check-dir configs/modeling/spatial_holdout
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from berlin_lst_downscaling.modeling.spatial_holdout import (
    CV_PARTITION,
    MAX_FRACTION,
    MIN_FRACTION,
    POSITION_EDGE,
    POSITION_INTERIOR,
    AnchorCounts,
    SpatialHoldoutAssignment,
    anchor_counts,
    assignment_bytes,
    build_assignment,
    build_fold_assignments,
    load_assignment,
    render_fold_map,
    render_map,
)

DEFAULT_PATCH_INDEX_ROOT = "gs://berlin-lst-training-data/training/patch-index/v3"
_COLUMNS = ["row", "col", "split", "year"]
_MARKER_KEYS = ("index_policy_hash", "source_policy_hash", "patches_total", "published_at")


def _read_index(root: str, local_dir: Path | None) -> tuple[list[dict], dict[str, str]]:
    """Return the index rows (four columns) and the completion-marker fingerprints."""
    if local_dir is not None:
        table = pq.read_table(local_dir / "patch_index.parquet", columns=_COLUMNS)
        marker = json.loads((local_dir / "complete.json").read_text(encoding="utf-8"))
    else:
        from berlin_lst_downscaling.data.io import read_bytes
        from berlin_lst_downscaling.data.training.paths import (
            patch_index_completion,
            patch_index_parquet,
        )

        table = pq.read_table(io.BytesIO(read_bytes(patch_index_parquet(root))), columns=_COLUMNS)
        marker = json.loads(read_bytes(patch_index_completion(root)))
    if int(marker.get("patches_total", -1)) != table.num_rows:
        raise SystemExit(
            f"ERROR: completion marker reports {marker.get('patches_total')} patches, "
            f"the table holds {table.num_rows}"
        )
    fingerprints = {"root": root, **{key: str(marker.get(key, "")) for key in _MARKER_KEYS}}
    return table.to_pylist(), fingerprints


def assignment_statistics(assignment: SpatialHoldoutAssignment, rows: list[dict]) -> dict:
    """Held-out and total index rows per split and year, and per position (train/validation)."""
    positions = assignment.positions()
    per_year: dict[str, dict[str, int]] = {}
    per_position: Counter[str] = Counter()
    for record in rows:
        split = str(record["split"])
        if split not in ("train", "validation"):
            continue
        key = f"{split}/{int(record['year'])}"
        entry = per_year.setdefault(key, {"held_out": 0, "total": 0})
        entry["total"] += 1
        position = positions.get((int(record["row"]), int(record["col"])))
        if position is not None:
            entry["held_out"] += 1
            per_position[f"{split}/{position}"] += 1
    anchor_positions = Counter(a.position for a in assignment.holdout)
    return {
        "anchors_total": len(assignment.anchors),
        "anchors_held_out": len(assignment.holdout),
        "fraction": assignment.fraction,
        "anchors_interior": anchor_positions[POSITION_INTERIOR],
        "anchors_edge": anchor_positions[POSITION_EDGE],
        "held_out_without_train": sorted(
            a for a in assignment.holdout_anchors if assignment.anchors[a].train == 0
        ),
        "held_out_without_validation": sorted(
            a for a in assignment.holdout_anchors if assignment.anchors[a].validation == 0
        ),
        "patches_by_split_year": dict(sorted(per_year.items())),
        "patches_by_split_position": dict(sorted(per_position.items())),
    }


def _print_report(assignment: SpatialHoldoutAssignment, stats: dict) -> None:
    print(
        f"anchors: {stats['anchors_held_out']} of {stats['anchors_total']} held out "
        f"({stats['fraction']:.1%}); interior {stats['anchors_interior']}, "
        f"edge {stats['anchors_edge']} (neighbourhood {assignment.neighbourhood})"
    )
    print(f"eligible complete blocks: {len(assignment.eligible_blocks)}; seed {assignment.seed}")
    print("blocks (block_row, block_col) -> anchor rows x cols:")
    for block in assignment.blocks:
        print(
            f"  {block}: rows {block[0] * 48}-{block[0] * 48 + 32}, "
            f"cols {block[1] * 48}-{block[1] * 48 + 32}"
        )
    print("patches (held out / total):")
    for key, entry in stats["patches_by_split_year"].items():
        share = entry["held_out"] / entry["total"] if entry["total"] else 0.0
        print(f"  {key}: {entry['held_out']} / {entry['total']} ({share:.1%})")
    for key, count in stats["patches_by_split_position"].items():
        print(f"  {key}: {count}")
    print(f"held-out anchors without train patches: {stats['held_out_without_train']}")
    print(f"held-out anchors without validation patches: {stats['held_out_without_validation']}")
    print("map (I interior, E edge, o kept, . no anchor; north up, one cell = 1.6 km):")
    print(render_map(assignment))


def _print_fold_report(folds: Sequence[SpatialHoldoutAssignment]) -> None:
    print(f"partition {CV_PARTITION}: {len(folds)} folds over {len(folds[0].anchors)} anchors")
    print("fold | blocks | anchors | train | validation | interior | edge")
    for k, fold in enumerate(folds):
        held = fold.holdout_anchors
        positions = Counter(a.position for a in fold.holdout)
        print(
            f"{k} | {len(fold.blocks)} | {len(held)} | "
            f"{sum(fold.anchors[a].train for a in held)} | "
            f"{sum(fold.anchors[a].validation for a in held)} | "
            f"{positions[POSITION_INTERIOR]} | {positions[POSITION_EDGE]}"
        )
    print("map (fold of each anchor; . no anchor; north up, one cell = 1.6 km):")
    print(render_fold_map(folds))


def _main_cv(args: argparse.Namespace) -> int:
    if args.output is not None or args.check is not None:
        raise SystemExit("ERROR: --partition cv5 writes or checks a directory: use --output-dir")
    if args.output_dir is not None and args.check_dir is not None:
        raise SystemExit("ERROR: pass --output-dir or --check-dir, not both")
    if args.anchors_from is not None:
        source = load_assignment(args.anchors_from)
        anchors: dict[tuple[int, int], AnchorCounts] = dict(source.anchors)
        fingerprints = dict(source.patch_index)
    else:
        rows, fingerprints = _read_index(args.patch_index_root, args.patch_index_dir)
        anchors = anchor_counts(rows)
    folds = build_fold_assignments(
        anchors, neighbourhood=args.neighbourhood, patch_index=fingerprints
    )
    _print_fold_report(folds)
    files = {f"{CV_PARTITION}-fold{k}.json": assignment_bytes(fold) for k, fold in enumerate(folds)}
    if args.check_dir is not None:
        bad = [
            name
            for name, payload in files.items()
            if not (args.check_dir / name).is_file()
            or (args.check_dir / name).read_bytes() != payload
        ]
        if bad:
            print(f"FAIL: {bad} in {args.check_dir} differ from the partition drawn from the table")
            return 1
        print(f"OK: the {len(files)} fold files in {args.check_dir} match the rule")
        return 0
    if args.output_dir is not None:
        for name, payload in files.items():
            target = args.output_dir / name
            if target.exists() and target.read_bytes() != payload:
                raise SystemExit(
                    f"ERROR: {target} exists with different content; an assignment that a "
                    "run may have used is never overwritten. Write a new version file."
                )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (args.output_dir / name).write_bytes(payload)
            print(f"wrote {args.output_dir / name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-index-root", default=DEFAULT_PATCH_INDEX_ROOT)
    parser.add_argument("--patch-index-dir", type=Path, help="local copy of the index files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbourhood", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-fraction", type=float, default=MIN_FRACTION)
    parser.add_argument("--max-fraction", type=float, default=MAX_FRACTION)
    parser.add_argument("--output", type=Path, help="write the assignment file here")
    parser.add_argument("--check", type=Path, help="compare with an existing assignment file")
    parser.add_argument(
        "--partition",
        choices=("draw", CV_PARTITION),
        default="draw",
        help="draw: the seeded block draw (default); cv5: the five-fold partition of all anchors",
    )
    parser.add_argument("--anchors-from", type=Path, help="cv5: read the anchor table from here")
    parser.add_argument("--output-dir", type=Path, help="cv5: write the five fold files here")
    parser.add_argument("--check-dir", type=Path, help="cv5: compare the five fold files here")
    args = parser.parse_args(argv)
    if args.partition == CV_PARTITION:
        return _main_cv(args)
    if args.output_dir is not None or args.check_dir is not None or args.anchors_from is not None:
        raise SystemExit("ERROR: --output-dir, --check-dir and --anchors-from need --partition cv5")
    if args.output is not None and args.check is not None:
        raise SystemExit("ERROR: pass --output or --check, not both")

    rows, fingerprints = _read_index(args.patch_index_root, args.patch_index_dir)
    assignment = build_assignment(
        anchor_counts(rows),
        seed=args.seed,
        neighbourhood=args.neighbourhood,
        min_fraction=args.min_fraction,
        max_fraction=args.max_fraction,
        patch_index=fingerprints,
    )
    _print_report(assignment, assignment_statistics(assignment, rows))
    payload = assignment_bytes(assignment)

    if args.check is not None:
        if args.check.read_bytes() != payload:
            print(f"FAIL: {args.check} differs from the assignment drawn from the index")
            return 1
        print(f"OK: {args.check} matches the published index and the rule")
        return 0
    if args.output is not None:
        if args.output.exists() and args.output.read_bytes() != payload:
            raise SystemExit(
                f"ERROR: {args.output} exists with different content; an assignment that a "
                "run may have used is never overwritten. Write a new version file."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
