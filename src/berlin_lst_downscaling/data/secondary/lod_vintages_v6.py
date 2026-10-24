"""Historical LoD vintages into the v6 roots (republish step 4b).

Step 3 (``configs/static_sources/v6.yaml``) rebuilds only the 2024
ATOM-feed LoD2 product into
``static/sources/lod2-area-exact-v2`` and step 4 derives only the
``dgm1-2021__lod2-2024__vh-2020`` geometry bundle. Steps 5–7 resolve every
scene year (2017–2025) through ``geometry_mapping.json`` and therefore
need the 2017/2021/2022 area-exact morphology, raw manifests, and
per-vintage derived bundles in the same v6 roots.

This module is the runnable path for that backfill. It reuses
:mod:`berlin_lst_downscaling.data.secondary.lod_vintages` for the
archive-first morphology and the per-vintage derived products (no
duplicated rasterisation logic) and records both runs in the v6 source
and derived ledgers, so downstream steps resolve them exactly like the
2024 products. Contracts are identical to the 2024 v6 product
(``contract_for_lod2_morphology``: mean/max ``[0, 400]`` m, std
``[0, 200]`` m).

Run order: after step 4 (needs ``terrain_height/2021``,
``vegetation_height/2020`` in the v6 source root and the vintage-agnostic
``vegetation_dsm/2024`` in the v6 derived root), before step 5.
"""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.lod2 import config_hash_for_vintage
from berlin_lst_downscaling.data.secondary.lod_vintages import (
    _DERIVED_PRODUCTS,
    _VINTAGE_SOURCES,
    RawManifest,
    archive_uri_for,
    derive_vintage_products,
    materialize_vintage_archive,
    publish_raw_archive_manifest,
    publish_vintage_morphology,
    vintage_geometry_id,
)
from berlin_lst_downscaling.data.secondary.paths import derived_ledger_path
from berlin_lst_downscaling.data.secondary.source_pipeline import find_not_done_items

_logger = logging.getLogger(__name__)

# Historical vintages backfilled into the v6 roots. 2024 is produced by
# steps 3–4; the carry-forward mapping serves exactly these four.
HISTORICAL_VINTAGES: tuple[int, ...] = (2017, 2021, 2022)


def expected_source_items(cfg) -> list[tuple[str, str, str]]:
    """Return the expected ``(item_id, source, period)`` source rows."""
    vintages: list[int] = [int(v) for v in list(cfg.get("vintages", list(HISTORICAL_VINTAGES)))]
    return [(f"lod2_morphology_{v}", "lod2_morphology", str(v)) for v in vintages]


def expected_derived_items(cfg) -> list[tuple[str, str, str]]:
    """Return the expected ``(item_id, source, period)`` derived rows."""
    vintages: list[int] = [int(v) for v in list(cfg.get("vintages", list(HISTORICAL_VINTAGES)))]
    out: list[tuple[str, str, str]] = []
    for vintage in vintages:
        geometry_id = vintage_geometry_id(vintage)
        for product in _DERIVED_PRODUCTS:
            out.append((product, product, geometry_id))
    return out


def run_lod_vintages_v6(cfg: DictConfig, run_id: str | None = None) -> int:
    """Build the historical morphology + derived bundles into the v6 roots.

    Returns 0 only when every expected source and derived row is done
    after the run, 1 otherwise. The closing ledger audit is fail-closed.
    """
    if run_id is None:
        run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    derived_root = str(cfg.derived_root)
    raw_root = str(cfg.get("raw_root", "gs://berlin-lst-training-data"))
    vintages: list[int] = [int(v) for v in list(cfg.get("vintages", list(HISTORICAL_VINTAGES)))]
    smoke_tile_count = cfg.get("smoke_tile_count")
    t0 = time.perf_counter()

    _banner(cfg, run_id, source_root, derived_root)

    src_led = SecondaryLedger.open(f"{source_root.rstrip('/')}/ledger.parquet")

    failed = 0
    for vintage in vintages:
        failed += _run_vintage_morphology(
            src_led, cfg, run_id, raw_root, source_root, vintage, smoke_tile_count
        )
        failed += _run_vintage_derived(cfg, run_id, source_root, derived_root, vintage)

    report_errors = find_not_done_items(src_led, expected_source_items(cfg))
    derived_led = SecondaryLedger.open(derived_ledger_path(derived_root))
    report_errors += find_not_done_items(derived_led, expected_derived_items(cfg))
    if report_errors:
        log_event(
            _logger,
            logging.ERROR,
            "ledger_not_all_done",
            errors=report_errors,
        )
        print(f"FAIL: historical LoD ledger not all done ({len(report_errors)} items).")
        for err in report_errors:
            print(f"  - {err}")

    elapsed = time.perf_counter() - t0
    log_event(_logger, logging.INFO, "duration", elapsed_s=round(elapsed, 1))
    if report_errors:
        return 1
    return 0 if failed == 0 else 1


def _run_vintage_morphology(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    raw_root: str,
    source_root: str,
    vintage: int,
    smoke_tile_count: int | None,
) -> int:
    """Materialise the archives and publish one vintage morphology product."""
    if vintage not in _VINTAGE_SOURCES:
        log_event(_logger, logging.ERROR, "unknown_vintage", vintage=vintage)
        return 1
    item_id = f"lod2_morphology_{vintage}"
    c_hash = config_hash_for_vintage(vintage)

    todo = reconcile([(item_id, "lod2_morphology", str(vintage))], led, c_hash)
    if not todo:
        log_event(_logger, logging.INFO, "skipped", source="lod2_morphology", vintage=vintage)
        return 0

    reason = todo[0][3]
    led.upsert(
        SecondaryLedgerRow(
            item_id=item_id,
            source="lod2_morphology",
            period_or_vintage=str(vintage),
            status="exporting",
            run_id=run_id,
        )
    )
    log_event(
        _logger,
        logging.INFO,
        "processing",
        source="lod2_morphology",
        vintage=vintage,
        reason=reason,
    )

    spec = _VINTAGE_SOURCES[vintage]
    try:
        # Vintage 2017 filters the LoD2-2021 stock against the LoD1-2017
        # footprints, so both archives are materialised for that run.
        source_vintage = 2021 if vintage == 2017 else vintage
        lod2_spec = _VINTAGE_SOURCES[source_vintage]
        with ExitStack() as stack:
            lod2_mat = stack.enter_context(materialize_vintage_archive(lod2_spec, raw_root))
            lod1_mat = None
            if vintage == 2017:
                lod1_mat = stack.enter_context(materialize_vintage_archive(spec, raw_root))
            _publish_manifests(source_root, raw_root, vintage, lod1_mat, lod2_mat)
            artifacts, _stats = publish_vintage_morphology(
                vintage,
                source_root,
                run_id,
                lod1_mat=lod1_mat,
                lod2_mat=lod2_mat,
                raw_root=raw_root,
                smoke_tile_count=smoke_tile_count,
            )
    except Exception as exc:
        log_event(
            _logger,
            logging.ERROR,
            "failed",
            source="lod2_morphology",
            vintage=vintage,
            error=str(exc),
        )
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source="lod2_morphology",
                period_or_vintage=str(vintage),
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            )
        )
        return 1

    led.upsert(
        SecondaryLedgerRow(
            item_id=item_id,
            source="lod2_morphology",
            period_or_vintage=str(vintage),
            status="done",
            run_id=run_id,
            config_hash=c_hash,
            output_uri=artifacts.cog_uri,
            stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        )
    )
    log_event(
        _logger,
        logging.INFO,
        "done",
        source="lod2_morphology",
        vintage=vintage,
        output_uri=artifacts.cog_uri,
    )
    return 0


def _publish_manifests(source_root: str, raw_root: str, vintage: int, lod1_mat, lod2_mat) -> None:
    """Publish the raw archive manifest(s) the coverage resolver expects."""
    mats = [(vintage, lod1_mat), (lod2_mat.spec.vintage, lod2_mat)]
    seen: set[int] = set()
    for manifest_vintage, mat in mats:
        if mat is None or manifest_vintage in seen:
            continue
        seen.add(manifest_vintage)
        manifest = RawManifest(
            vintage=manifest_vintage,
            source_kind=mat.spec.level,
            feed_label=mat.spec.feed_label,
            archive_uri=archive_uri_for(raw_root, mat.spec),
            archive_sha256=mat.sha256,
            archive_byte_count=mat.byte_count,
            member_count=mat.member_count,
            member_names=list(mat.member_names),
        )
        uri = publish_raw_archive_manifest(manifest, source_root)
        log_event(
            _logger,
            logging.INFO,
            "raw_manifest_published",
            vintage=manifest_vintage,
            uri=uri,
        )


def _run_vintage_derived(
    cfg: DictConfig,
    run_id: str,
    source_root: str,
    derived_root: str,
    vintage: int,
) -> int:
    """Build the per-vintage derived bundle (idempotent on the ledger)."""
    geometry_id = vintage_geometry_id(vintage)
    led = SecondaryLedger.open(derived_ledger_path(derived_root))

    missing = [
        product for product in _DERIVED_PRODUCTS if _row_needs_work(led, product, geometry_id)
    ]
    if not missing:
        log_event(
            _logger,
            logging.INFO,
            "skipped",
            source="vintage_derived",
            vintage=vintage,
            geometry_id=geometry_id,
        )
        return 0

    # The morphology row must be done (with resolvable artifacts) before
    # its derived bundle can be built; otherwise the final audit reports
    # it fail-closed.
    src_led = SecondaryLedger.open(f"{source_root.rstrip('/')}/ledger.parquet")
    if find_not_done_items(
        src_led, [(f"lod2_morphology_{vintage}", "lod2_morphology", str(vintage))]
    ):
        log_event(
            _logger,
            logging.ERROR,
            "morphology_not_done",
            vintage=vintage,
        )
        return 1

    log_event(
        _logger,
        logging.INFO,
        "processing",
        source="vintage_derived",
        vintage=vintage,
        geometry_id=geometry_id,
        products=sorted(missing),
    )
    try:
        derive_vintage_products(
            vintage,
            source_root,
            derived_root,
            run_id,
            upstream_source_root=source_root,
            upstream_derived_root=derived_root,
            grid=canon_grid_10m(),
            max_radius_m=float(cfg.get("horizon_max_radius_m", 200.0)),
            svf_max_radius=int(cfg.get("svf_max_radius", 10)),
            svf_n_directions=int(cfg.get("svf_n_directions", 16)),
            products=sorted(missing),
        )
    except Exception as exc:
        log_event(
            _logger,
            logging.ERROR,
            "failed",
            source="vintage_derived",
            vintage=vintage,
            error=str(exc),
        )
        for product in missing:
            if _row_needs_work(led, product, geometry_id):
                led.upsert(
                    SecondaryLedgerRow(
                        item_id=product,
                        source=product,
                        period_or_vintage=geometry_id,
                        status="failed",
                        run_id=run_id,
                        last_error=str(exc),
                    )
                )
        return 1
    return 0


def _row_needs_work(led: SecondaryLedger, source: str, period: str) -> bool:
    """Return True when the derived row is missing or not done."""
    row = led.get(source, source, period)
    return row is None or row.status != "done"


def _banner(cfg: DictConfig, run_id: str, source_root: str, derived_root: str) -> None:
    log_event(
        _logger,
        logging.INFO,
        "run_start",
        pipeline="lod-vintages-v6",
        run_id=run_id,
        source_root=source_root,
        derived_root=derived_root,
        vintages=[int(v) for v in list(cfg.get("vintages", list(HISTORICAL_VINTAGES)))],
    )


__all__ = [
    "HISTORICAL_VINTAGES",
    "expected_derived_items",
    "expected_source_items",
    "run_lod_vintages_v6",
]
