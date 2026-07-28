# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""LoD historical-vintage runner — 2017 / 2021 / 2022 morphology.

Streams the locally-supplied Berlin LoD1 (2017) and LoD2 (2021/2022)
CityGML tiles into GCS, publishes the canonical-grid
``lod2_morphology`` products for each historical vintage, derives the
morphology-dependent products (building DSM, combined DSM, building
horizon, SVF) per vintage, and writes a year → vintage carry-forward
mapping.

Usage
-----
    uv run python scripts/run_lod_vintages.py \\
        --source-root gs://berlin-lst-data/static/sources/full \\
        --derived-root gs://berlin-lst-data/static/derived/full \\
        --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \\
        --vintages 2017,2021,2022

    # Smoke run on a tiny bbox (200×200 m), no GCS writes:
    uv run python scripts/run_lod_vintages.py --dry-run --smoke-tile-count 4

    # Delete local raw inputs after a successful run:
    uv run python scripts/run_lod_vintages.py ... --cleanup
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

from berlin_lst_downscaling.common.grid import smoke_grid
from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.secondary.lod_vintages import (
    _VINTAGE_SOURCES,
    derive_vintage_products,
    load_lod1_footprints,
    publish_geometry_mapping,
    publish_vintage_morphology,
    stream_vintage_to_gcs,
    vintage_geometry_id,
)

_logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        default="gs://berlin-lst-data/static/sources/full",
        help="Pipeline A source product root (local path or gs:// URI).",
    )
    parser.add_argument(
        "--derived-root",
        default="gs://berlin-lst-data/static/derived/full",
        help="Pipeline B derived product root (local path or gs:// URI).",
    )
    parser.add_argument(
        "--metadata-root",
        default="gs://berlin-lst-data/static/geometry_vintages/v1",
        help="Bucket-level root for the geometry_mapping.json artifact.",
    )
    parser.add_argument(
        "--vintages",
        default="2017,2021,2022",
        help="Comma-separated list of vintages to process.",
    )
    parser.add_argument(
        "--smoke-tile-count",
        type=int,
        default=None,
        help="Restrict each vintage to this many leading tiles.",
    )
    parser.add_argument(
        "--smoke-bbox",
        default=None,
        help="WGS84 bbox (west,south,east,north) for local smoke runs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (default: uuid4 hex).",
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip the morphology-dependent derived products.",
    )
    parser.add_argument(
        "--skip-raw-upload",
        action="store_true",
        help="Skip the raw archive upload (assume already uploaded).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing to GCS; useful for local smoke testing.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete local raw inputs after a successful run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _resolve_vintages(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vintage = int(chunk)
        if vintage not in _VINTAGE_SOURCES:
            raise ValueError(
                f"Unsupported vintage {vintage}; supported: {sorted(_VINTAGE_SOURCES)}"
            )
        out.append(vintage)
    return out


def _delete_local_inputs() -> int:
    """Delete every local raw input; return the number of files removed."""
    removed = 0
    targets: list[Path] = [
        Path("data/LoD2/2017"),
        Path("data/LoD2/LoD2_BE_1_33_2021"),
        Path("data/LoD2/LoD2_2022.zip"),
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed += 1
            log_event(_logger, logging.INFO, "local_input_deleted", path=str(target), kind="dir")
        elif target.is_file():
            target.unlink()
            removed += 1
            log_event(_logger, logging.INFO, "local_input_deleted", path=str(target), kind="file")
    return removed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vintages = _resolve_vintages(args.vintages)
    run_id = args.run_id or uuid4().hex[:8]
    level = getattr(logging, args.log_level)

    if args.dry_run:
        log_root = args.source_root
    else:
        log_root = args.source_root

    grid = None
    if args.smoke_bbox:
        west, south, east, north = (float(p) for p in args.smoke_bbox.split(","))
        grid = smoke_grid((west, south, east, north))

    with RunLogSession(log_root, pipeline="lod-vintages", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "run_start",
            run_id=run_id,
            vintages=vintages,
            source_root=args.source_root,
            derived_root=args.derived_root,
            metadata_root=args.metadata_root,
            dry_run=args.dry_run,
        )

        t0 = time.perf_counter()
        failures: list[str] = []

        # Pre-load LoD1 2017 once for the 2017 stock filter.
        lod1_index: dict | None = None
        if 2017 in vintages:
            log_event(_logger, logging.INFO, "lod1_index_load_start")
            lod1_index = load_lod1_footprints(2017)
            log_event(
                _logger,
                logging.INFO,
                "lod1_index_load_done",
                n_tiles=len(lod1_index),
                n_footprints=sum(len(v) for v in lod1_index.values()),
            )

        vintage_artifacts: dict[int, dict] = {}

        for vintage in vintages:
            try:
                if not args.skip_raw_upload and not args.dry_run:
                    stream_vintage_to_gcs(vintage, args.source_root)

                artifacts, _stats = publish_vintage_morphology(
                    vintage,
                    args.source_root,
                    run_id,
                    lod1_index=lod1_index if vintage == 2017 else None,
                    grid=grid,
                    smoke_tile_count=args.smoke_tile_count,
                )

                vintage_artifacts[vintage] = {
                    "geometry_id": vintage_geometry_id(vintage),
                    "lod2_morphology": artifacts.cog_uri,
                    "stac": artifacts.stac_uri,
                    "provenance": artifacts.provenance_uri,
                    "completion": artifacts.completion_uri,
                }
            except Exception as exc:
                log_event(
                    _logger,
                    logging.ERROR,
                    "vintage_failed",
                    vintage=vintage,
                    error=str(exc),
                )
                failures.append(f"{vintage}: {exc}")

        # Derived products (only after every vintage succeeded)
        if not args.skip_derived and not failures and not args.dry_run:
            for vintage in vintages:
                try:
                    derive_vintage_products(
                        vintage,
                        args.source_root,
                        args.derived_root,
                        run_id,
                        grid=grid,
                    )
                except Exception as exc:
                    log_event(
                        _logger,
                        logging.ERROR,
                        "derive_vintage_failed",
                        vintage=vintage,
                        error=str(exc),
                    )
                    failures.append(f"{vintage}-derived: {exc}")

        if not args.dry_run and vintage_artifacts:
            try:
                publish_geometry_mapping(args.metadata_root, vintage_artifacts=vintage_artifacts)
            except Exception as exc:
                log_event(
                    _logger,
                    logging.ERROR,
                    "geometry_mapping_failed",
                    error=str(exc),
                )
                failures.append(f"mapping: {exc}")

        # Cleanup only on full success and only on explicit request.
        if args.cleanup and not failures and not args.dry_run:
            removed = _delete_local_inputs()
            log_event(_logger, logging.INFO, "cleanup_done", removed_paths=removed)
        elif args.cleanup and failures:
            log_event(
                _logger,
                logging.WARNING,
                "cleanup_skipped_due_to_failures",
                n_failures=len(failures),
            )

        elapsed = time.perf_counter() - t0
        log_event(
            _logger,
            logging.INFO,
            "duration",
            elapsed_s=round(elapsed, 1),
            vintages=vintages,
            n_failures=len(failures),
        )

        if failures:
            for line in failures:
                print(line, file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())