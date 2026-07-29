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
import json
import logging
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from berlin_lst_downscaling.common.grid import smoke_grid
from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.secondary.citygml import iter_xml_members
from berlin_lst_downscaling.data.secondary.lod_vintages import (
    _VINTAGE_SOURCES,
    ArchiveMaterialization,
    RawManifest,
    VintageSpec,
    _stream_sha256_file,
    archive_uri_for,
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
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
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
        "--upstream-source-root",
        default=None,
        help="Override root used to resolve terrain/vh/2024 lod2 source "
        "products when deriving historical vintage products. Default: "
        "--source-root.",
    )
    parser.add_argument(
        "--upstream-derived-root",
        default=None,
        help="Override root used to resolve the vintage-agnostic "
        "vegetation_dsm/2024 product. Default: --derived-root.",
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
        "--reconcile-only",
        action="store_true",
        help=(
            "Validate finalized artifacts and reconcile them into both "
            "Static ledgers.  Downloads no archives, computes no raster, "
            "derives no products."
        ),
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


def _smoke_archive_ctx(spec: VintageSpec, archive_path: Path):
    """Context manager: yield an ArchiveMaterialization built from a local file.

    Used for local smoke runs where the caller already has the archive on
    disk.  No GCS download, no cleanup — the caller owns the file.
    """

    @contextmanager
    def _ctx():
        byte_count = archive_path.stat().st_size
        sha = _stream_sha256_file(archive_path)
        member_names = sorted(iter_xml_members(archive_path))
        materialization = ArchiveMaterialization(
            spec=spec,
            local_path=archive_path,
            byte_count=byte_count,
            sha256=sha,
            member_count=len(member_names),
            member_names=member_names,
        )
        yield materialization

    return _ctx()


def _publish_archive_manifest(
    raw_root: str,
    source_root: str,
    mat: ArchiveMaterialization,
    raw: RawManifest | None = None,
) -> str:
    """Build a :class:`RawManifest` from *mat* and persist it under *source_root*."""
    if raw is None:
        raw = RawManifest(
            vintage=mat.spec.vintage,
            source_kind=mat.spec.level,
            feed_label=mat.spec.feed_label,
            archive_uri=archive_uri_for(raw_root, mat.spec),
            archive_sha256=mat.sha256,
            archive_byte_count=mat.byte_count,
            member_count=mat.member_count,
            member_names=list(mat.member_names),
        )
    return publish_raw_archive_manifest(raw, source_root)


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
        publish_raw_archive_manifest(manifest, source_root)
    finally:
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


def _publish_one_vintage(
    vintage: int,
    args: argparse.Namespace,
    run_id: str,
    grid,
    lod1_mat: ArchiveMaterialization | None,
    lod2_mat: ArchiveMaterialization,
):
    """Call publish_vintage_morphology once and return its artifacts tuple."""
    return publish_vintage_morphology(
        vintage,
        args.source_root,
        run_id,
        lod1_mat=lod1_mat,
        lod2_mat=lod2_mat,
        raw_root=args.raw_root,
        grid=grid,
        smoke_tile_count=args.smoke_tile_count,
    )


def _reconcile_only(
    args: argparse.Namespace,
    run_id: str,
    vintages: list[int],
) -> int:
    """Validate finalized artifacts and reconcile them into both ledgers.

    Downloads no archives, computes no raster, derives no products.
    Fails if any required artifact is incomplete or missing.
    """
    from berlin_lst_downscaling.data.io.storage import exists, read_bytes
    from berlin_lst_downscaling.data.secondary.idempotency import reconcile
    from berlin_lst_downscaling.data.secondary.ledger import (
        SecondaryLedger,
        SecondaryLedgerRow,
    )
    from berlin_lst_downscaling.data.secondary.paths import (
        derived_ledger_path,
        derived_product_cog,
        source_product_cog,
        source_product_provenance,
    )

    failures: list[str] = []
    source_ledger_path = (
        f"{args.source_root.rstrip('/')}/ledger.parquet"
    )

    for vintage in vintages:
        # ── source artifacts ────────────────────────────────────────
        cog_uri = source_product_cog(
            args.source_root, "lod2_morphology", str(vintage)
        )
        prov_uri = source_product_provenance(
            args.source_root, "lod2_morphology", str(vintage)
        )
        stac_uri = (
            f"{args.source_root.rstrip('/')}/ard/static/sources/"
            f"lod2_morphology/{vintage}/lod2_morphology_{vintage}.stac.json"
        )
        completion_uri = (
            f"{args.source_root.rstrip('/')}/ard/static/sources/"
            f"lod2_morphology/{vintage}/complete.json"
        )

        for name, uri in [
            ("COG", cog_uri),
            ("provenance", prov_uri),
            ("STAC", stac_uri),
            ("complete", completion_uri),
        ]:
            if not exists(uri):
                failures.append(f"{vintage}: source {name} missing: {uri}")

        if failures:
            continue

        # Read config_hash from provenance
        config_hash = ""
        try:
            prov = json.loads(read_bytes(prov_uri))
            config_hash = str(prov.get("config_hash", ""))
        except Exception:  # noqa: S110 — fall through, empty hash acceptable
            pass

        # ── derived artifacts ───────────────────────────────────────
        geometry_id = vintage_geometry_id(vintage)
        expected_products = [
            "building_dsm",
            "combined_dsm",
            "horizon_building",
            "svf",
        ]
        derived_ok = True
        for product in expected_products:
            prod_cog = derived_product_cog(
                args.derived_root, product, geometry_id
            )
            prod_prov = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/provenance.json"
            )
            prod_stac = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/"
                f"{product}_{geometry_id}.stac.json"
            )
            prod_complete = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/complete.json"
            )
            for name, uri in [
                ("COG", prod_cog),
                ("provenance", prod_prov),
                ("STAC", prod_stac),
                ("complete", prod_complete),
            ]:
                if not exists(uri):
                    failures.append(
                        f"{vintage}: derived {product} {name} missing: {uri}"
                    )
                    derived_ok = False

        if not derived_ok:
            continue

        # ── upsert source ledger ────────────────────────────────────
        led = SecondaryLedger.open(source_ledger_path)
        item_id = f"lod2_morphology_{vintage}"
        todo = reconcile(
            [(item_id, "lod2_morphology", str(vintage))],
            led,
            config_hash,
        )
        if todo:
            led.upsert(
                SecondaryLedgerRow(
                    item_id=item_id,
                    source="lod2_morphology",
                    period_or_vintage=str(vintage),
                    status="done",
                    run_id=run_id,
                    config_hash=config_hash,
                    output_uri=cog_uri,
                    stac_uri=stac_uri,
                    provenance_uri=prov_uri,
                    completion_uri=completion_uri,
                )
            )
            log_event(
                _logger,
                logging.INFO,
                "source_ledger_upserted",
                vintage=vintage,
                item_id=item_id,
            )

        # ── upsert derived ledger ───────────────────────────────────
        deriv_led = SecondaryLedger.open(
            derived_ledger_path(args.derived_root)
        )
        for product in expected_products:
            prod_prov = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/provenance.json"
            )
            prod_hash = ""
            try:
                pp = json.loads(read_bytes(prod_prov))
                prod_hash = str(pp.get("config_hash", ""))
            except Exception:  # noqa: S110 — fall through, empty hash acceptable
                pass

            prod_cog = derived_product_cog(
                args.derived_root, product, geometry_id
            )
            prod_stac = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/"
                f"{product}_{geometry_id}.stac.json"
            )
            prod_complete = (
                f"{args.derived_root.rstrip('/')}/ard/static/derived/"
                f"{product}/{geometry_id}/complete.json"
            )

            deriv_todo = reconcile(
                [(product, product, geometry_id)],
                deriv_led,
                prod_hash,
            )
            if deriv_todo:
                deriv_led.upsert(
                    SecondaryLedgerRow(
                        item_id=product,
                        source=product,
                        period_or_vintage=geometry_id,
                        status="done",
                        run_id=run_id,
                        config_hash=prod_hash,
                        output_uri=prod_cog,
                        stac_uri=prod_stac,
                        provenance_uri=prod_prov,
                        completion_uri=prod_complete,
                    )
                )
                log_event(
                    _logger,
                    logging.INFO,
                    "derived_ledger_upserted",
                    vintage=vintage,
                    product=product,
                    geometry_id=geometry_id,
                )

        log_event(
            _logger,
            logging.INFO,
            "reconcile_done",
            vintage=vintage,
            geometry_id=geometry_id,
        )

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    # ── publish geometry mapping after all ledgers are reconciled ────
    vintage_artifacts: dict[int, dict] = {}
    for vintage in vintages:
        cog_uri = source_product_cog(
            args.source_root, "lod2_morphology", str(vintage)
        )
        prov_uri = source_product_provenance(
            args.source_root, "lod2_morphology", str(vintage)
        )
        geometry_id = vintage_geometry_id(vintage)
        vintage_artifacts[vintage] = {
            "geometry_id": geometry_id,
            "lod2_morphology": cog_uri,
            "stac": (
                f"{args.source_root.rstrip('/')}/ard/static/sources/"
                f"lod2_morphology/{vintage}/"
                f"lod2_morphology_{vintage}.stac.json"
            ),
            "provenance": prov_uri,
            "completion": (
                f"{args.source_root.rstrip('/')}/ard/static/sources/"
                f"lod2_morphology/{vintage}/complete.json"
            ),
        }

    publish_geometry_mapping(
        args.metadata_root,
        vintage_artifacts=vintage_artifacts,
        legacy_source_root=args.source_root,
        legacy_derived_root=args.derived_root,
    )

    # ── ledger summary ──────────────────────────────────────────────
    led_src = SecondaryLedger.open(source_ledger_path)
    led_drv = SecondaryLedger.open(
        derived_ledger_path(args.derived_root)
    )
    log_event(
        _logger,
        logging.INFO,
        "reconcile_summary",
        source_counts=led_src.status_counts(),
        derived_counts=led_drv.status_counts(),
    )
    print(f"OK: reconciled {len(vintages)} vintage(s) into both ledgers")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vintages = _resolve_vintages(args.vintages)
    run_id = args.run_id or uuid4().hex[:8]
    level = getattr(logging, args.log_level)

    if args.reconcile_only:
        with RunLogSession(
            args.source_root,
            pipeline="lod-vintages",
            run_id=run_id,
            level=level,
        ):
            log_event(
                _logger,
                logging.INFO,
                "run_start",
                run_id=run_id,
                vintages=vintages,
                mode="reconcile_only",
                source_root=args.source_root,
                derived_root=args.derived_root,
                metadata_root=args.metadata_root,
            )
            t0 = time.perf_counter()
            rc = _reconcile_only(args, run_id, vintages)
            log_event(
                _logger,
                logging.INFO,
                "duration",
                elapsed_s=round(time.perf_counter() - t0, 1),
            )
            return rc

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
                    archive_ctx = _smoke_archive_ctx(
                        _VINTAGE_SOURCES[source_vintage],
                        Path(args.smoke_archive),
                    )
                elif args.dry_run:
                    log_event(
                        _logger,
                        logging.WARNING,
                        "dry_run_no_archive",
                        vintage=vintage,
                    )
                    continue
                else:
                    archive_ctx = materialize_vintage_archive(
                        _VINTAGE_SOURCES[source_vintage], args.raw_root
                    )

                with archive_ctx as lod2_mat:
                    # 2. Materialise the LoD1 archive when filtering 2017
                    lod1_mat = None
                    lod1_ctx = None
                    if vintage == 2017:
                        if args.smoke_archive:
                            lod1_ctx = _smoke_archive_ctx(
                                _VINTAGE_SOURCES[2017],
                                Path(args.smoke_archive),
                            )
                        else:
                            lod1_ctx = materialize_vintage_archive(
                                _VINTAGE_SOURCES[2017], args.raw_root
                            )

                    if lod1_ctx is not None:
                        with lod1_ctx as lod1_mat_in:
                            lod1_mat = lod1_mat_in
                            artifacts, _stats = _publish_one_vintage(
                                vintage,
                                args,
                                run_id,
                                grid,
                                lod1_mat,
                                lod2_mat,
                            )
                    else:
                        artifacts, _stats = _publish_one_vintage(
                            vintage,
                            args,
                            run_id,
                            grid,
                            None,
                            lod2_mat,
                        )

                    raw_manifest_uri = _publish_archive_manifest(
                        args.raw_root, args.source_root, lod2_mat
                    )
                    if lod1_mat is not None:
                        _publish_archive_manifest(
                            args.raw_root, args.source_root, lod1_mat
                        )

                    vintage_artifacts[vintage] = {
                        "geometry_id": vintage_geometry_id(vintage),
                        "lod2_morphology": artifacts.cog_uri,
                        "stac": artifacts.stac_uri,
                        "provenance": artifacts.provenance_uri,
                        "completion": artifacts.completion_uri,
                        "raw_manifest": raw_manifest_uri,
                        "archive_uri": artifacts.archive_uri,
                        "archive_sha256": artifacts.archive_sha256,
                        "archive_byte_count": artifacts.archive_byte_count,
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
        derived_success: dict[int, dict] = {}
        if not args.skip_derived and not failures and not args.dry_run:
            upstream_src = args.upstream_source_root or args.source_root
            upstream_drv = args.upstream_derived_root or args.derived_root
            for vintage in vintages:
                try:
                    artefact = derive_vintage_products(
                        vintage,
                        args.source_root,
                        args.derived_root,
                        run_id,
                        upstream_source_root=upstream_src,
                        upstream_derived_root=upstream_drv,
                        grid=grid,
                    )
                    derived_success[vintage] = artefact
                except Exception as exc:
                    log_event(
                        _logger,
                        logging.ERROR,
                        "derive_vintage_failed",
                        vintage=vintage,
                        error=str(exc),
                    )
                    failures.append(f"{vintage}-derived: {exc}")

        # Geometry mapping only published after every vintage + every
        # derived product has finalised successfully, so the artefact can
        # never point at half-published state.
        # Skip mapping during smoke runs (legacy 2024 roots may not
        # resolve from local/smoke roots) and --skip-derived.
        is_smoke = args.smoke_archive or args.smoke_tile_count is not None
        if (
            not args.dry_run
            and not is_smoke
            and not args.skip_derived
            and vintage_artifacts
            and not failures
            and len(derived_success) == len(vintages)
        ):
            try:
                publish_geometry_mapping(
                    args.metadata_root,
                    vintage_artifacts=vintage_artifacts,
                    legacy_source_root=args.source_root,
                    legacy_derived_root=args.derived_root,
                )
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