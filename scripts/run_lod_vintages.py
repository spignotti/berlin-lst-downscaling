# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""LoD historical-vintage runner — archive-first GCS workflow.

Streams one ZIP per vintage from GCS into a per-vintage processing
session, publishes the canonical-grid ``lod2_morphology`` products,
derives the morphology-dependent bundle (building DSM, combined DSM,
building horizon, SVF) per vintage, and writes the year → vintage
carry-forward mapping.

Usage
-----
    # Full archive-backed production run on the VM
    uv run python scripts/run_lod_vintages.py \\
        --source-root gs://berlin-lst-data/static/sources/full \\
        --derived-root gs://berlin-lst-data/static/derived/full \\
        --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \\
        --raw-root gs://berlin-lst-data \\
        --vintages 2017,2021,2022

    # Local smoke against an arbitrary local archive
    uv run python scripts/run_lod_vintages.py --smoke-archive \\
        data/LoD2/LoD2_2022.zip --vintages 2022 \\
        --smoke-tile-count 8 --smoke-bbox 13.586225,52.467717,13.616234,52.486040 \\
        --skip-derived

    # Delete the source-dir copy after a successful build of one archive
    uv run python scripts/run_lod_vintages.py --stage-local 2017 \\
        --local-source-dir data/LoD2/2017 --raw-root gs://berlin-lst-data
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
    RawManifest,
    derive_vintage_products,
    materialize_vintage_archive,
    publish_geometry_mapping,
    publish_raw_archive_manifest,
    publish_vintage_morphology,
    stage_local_archive,
    stream_archive_to_gcs,
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
        "--raw-root",
        default="gs://berlin-lst-data",
        help="Bucket-level root for raw vintage archives.",
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
        "--smoke-archive",
        default=None,
        help="Local path to a single archive for smoke runs (default: download from GCS).",
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
        "--stage-local",
        choices=("2017", "2021"),
        default=None,
        help="Build the canonical archive ZIP from a local source directory, "
        "upload to GCS, then delete the source dir on success.",
    )
    parser.add_argument(
        "--local-source-dir",
        default=None,
        help="Local directory containing the .xml files to archive (used with --stage-local).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip raw upload + GCS writes; useful for local smoke testing.",
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


def _stage_and_upload(
    vintage: int,
    local_source_dir: Path,
    raw_root: str,
    source_root: str,
) -> RawManifest:
    """Build the archive ZIP from *local_source_dir*, upload to GCS, return manifest."""
    spec = _VINTAGE_SOURCES[vintage]
    archive_path = stage_local_archive(spec, local_source_dir)
    try:
        manifest = stream_archive_to_gcs(spec, archive_path, raw_root)
        # Mirror the architecture of the published product layout:
        # the manifest lives next to other source artefacts under
        # ``ard/static/sources/lod_vintages/`` of the configured
        # source root.
        manifest_uri_base = source_root.rstrip("/")
        publish_raw_archive_manifest(manifest, manifest_uri_base)
    finally:
        # Always remove the local archive copy; the bucket is the
        # immutable source of truth.
        try:
            archive_path.unlink()
        except FileNotFoundError:
            pass
    return manifest


def _delete_local_dir(path: Path) -> bool:
    """Delete *path* if it exists. Return True if anything was deleted."""
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log_event(_logger, logging.INFO, "local_input_deleted", path=str(path))
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vintages = _resolve_vintages(args.vintages)
    run_id = args.run_id or uuid4().hex[:8]
    level = getattr(logging, args.log_level)

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
            raw_root=args.raw_root,
            dry_run=args.dry_run,
        )

        t0 = time.perf_counter()
        failures: list[str] = []

        # Optional: stage one local directory into an archive, upload, delete local copy.
        if args.stage_local and args.local_source_dir:
            try:
                if not args.dry_run:
                    _stage_and_upload(
                        int(args.stage_local),
                        Path(args.local_source_dir),
                        args.raw_root,
                        args.source_root,
                    )
                    # After successful upload, free the local directory.
                    if _delete_local_dir(Path(args.local_source_dir)):
                        log_event(
                            _logger,
                            logging.INFO,
                            "local_source_released",
                            vintage=int(args.stage_local),
                            path=args.local_source_dir,
                        )
            except Exception as exc:
                log_event(
                    _logger,
                    logging.ERROR,
                    "stage_local_failed",
                    error=str(exc),
                )
                failures.append(f"stage-{args.stage_local}: {exc}")

        vintage_artifacts: dict[int, dict] = {}

        for vintage in vintages:
            try:
                # 1. Materialise the LoD2 archive for this vintage
                #    (always the 2021 archive for vintage=2017).
                source_vintage = 2021 if vintage == 2017 else vintage

                if args.smoke_archive:
                    lod2_archive_path = Path(args.smoke_archive)
                elif args.dry_run:
                    # Dry-run without smoke-archive: nothing to read.
                    log_event(
                        _logger,
                        logging.WARNING,
                        "dry_run_no_archive",
                        vintage=vintage,
                    )
                    continue
                else:
                    mat, mat_tmp = materialize_vintage_archive(
                        _VINTAGE_SOURCES[source_vintage], args.raw_root
                    )
                    lod2_archive_path = mat.local_path

                # 2. Materialise the LoD1 archive when filtering 2017
                lod1_archive_path = None
                if vintage == 2017:
                    if args.smoke_archive:
                        lod1_archive_path = Path(args.smoke_archive)
                    else:
                        lod1_mat, lod1_tmp = materialize_vintage_archive(
                            _VINTAGE_SOURCES[2017], args.raw_root
                        )
                        lod1_archive_path = lod1_mat.local_path

                # 3. Publish morphology product
                if args.dry_run:
                    log_event(
                        _logger,
                        logging.INFO,
                        "vintage_skipped_dry_run",
                        vintage=vintage,
                    )
                    continue

                artifacts, _stats, _archive = publish_vintage_morphology(
                    vintage,
                    args.source_root,
                    run_id,
                    lod1_archive=lod1_archive_path,
                    lod2_archive=lod2_archive_path,
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