"""Secondary-data pipeline — execution orchestration.

Supports two modes:

* ``fixture`` — small synthetic products per registered source,
  validates the full lifecycle without downloading real datasets.
* ``full`` — real source processing; produces the four final
  artifacts (COG, STAC, provenance, completion marker) for each
  configured source/vintage.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import ledger_path
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)

_logger = logging.getLogger(__name__)

# ── public entry ──────────────────────────────────────────────────────


def run(cfg: DictConfig) -> int:
    """Execute the secondary-data pipeline.

    Returns 0 on success, 1 if any items failed.

    Parameters
    ----------
    cfg :
        Hydra config.  Must contain at least ``mode`` and ``output_root``.
    """
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    t0 = time.perf_counter()

    _banner(cfg, run_id, output_root)

    led = SecondaryLedger.open(ledger_path(output_root))

    if cfg.mode == "fixture":
        rc = _run_fixture(led, cfg, run_id, output_root)
        elapsed = time.perf_counter() - t0
        log_event(_logger, logging.INFO, "duration", elapsed_s=round(elapsed, 1))
        return rc

    if cfg.mode == "full":
        rc = _run_full(led, cfg, run_id, output_root)
        elapsed = time.perf_counter() - t0
        log_event(_logger, logging.INFO, "duration", elapsed_s=round(elapsed, 1))
        return rc

    log_event(_logger, logging.ERROR, "unknown_mode", mode=cfg.mode)
    return 1


# ── fixture ───────────────────────────────────────────────────────────


def _run_fixture(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Run source-registered synthetic fixtures.

    Iterates :func:`fixtures.registry` and finalises each product via the
    same path as real sources.  No upstream downloads.  The smoke run
    validates the full contract (COG + STAC + provenance + completion
    marker + ledger + QA report) for every registered source.
    """
    from berlin_lst_downscaling.data.secondary.fixtures import registry

    grid = canon_grid_10m()
    failed = 0

    for source, factory in registry().items():
        item_id = f"fixture_{source}"
        period = "fixture"
        c_hash = "fixture"

        items = [(item_id, source, period)]
        todo = reconcile(items, led, c_hash)
        if not todo:
            log_event(_logger, logging.INFO, "skipped", source=source)
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source=source,
            period_or_vintage=period,
            status="exporting",
            run_id=run_id,
        ))

        try:
            log_event(_logger, logging.INFO, "finalising_fixture", source=source)
            prepared = factory(output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            log_event(_logger, logging.ERROR, "fixture_failed", source=source, error=str(exc))
            led.upsert(SecondaryLedgerRow(
                item_id=item_id,
                source=source,
                period_or_vintage=period,
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source=source,
            period_or_vintage=period,
            status="done",
            run_id=run_id,
            config_hash=c_hash,
            output_uri=artifacts.cog_uri,
            stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))
        log_event(
            _logger, logging.INFO, "fixture_done",
            source=source, output_uri=artifacts.cog_uri,
        )

    report = secondary_qa_report(led, run_id, sources=list(registry().keys()))
    log_event(_logger, logging.INFO, "qa_report", report=format_secondary_report(report))
    report_uri = persist_secondary_report(report, output_root)
    log_event(_logger, logging.INFO, "qa_report_path", path=report_uri)
    return 0 if failed == 0 else 1


# ── full mode ─────────────────────────────────────────────────────────


def _run_full(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Run the secondary pipeline with configured sources."""
    sources: list[str] = list(cfg.get("sources", []))
    if not sources:
        log_event(_logger, logging.INFO, "no_sources")
        return 0

    peak_scratch_gb = cfg.get("peak_scratch_gb", 999)
    disk_budget_gb = cfg.get("disk_budget_gb", 20)
    if peak_scratch_gb > disk_budget_gb:
        log_event(_logger, logging.ERROR, "disk_budget_exceeded",
            peak_scratch_gb=peak_scratch_gb, disk_budget_gb=disk_budget_gb)
        return 1

    failed = 0

    for source in sources:
        if source == "imperviousness":
            failed += _run_imperviousness(led, cfg, run_id, output_root)
        elif source == "vegetation_height":
            failed += _run_vegetation_height(led, cfg, run_id, output_root)
        elif source == "terrain_height":
            failed += _run_terrain_height(led, cfg, run_id, output_root)
        elif source == "lod2_morphology":
            failed += _run_lod2_morphology(led, cfg, run_id, output_root)
        else:
            log_event(_logger, logging.WARNING, "unknown_source", source=source)
            failed += 1

    # Final QA report — printed and persisted under qa/secondary/{run_id}/
    report = secondary_qa_report(led, run_id)
    log_event(_logger, logging.INFO, "qa_report", report=format_secondary_report(report))
    report_uri = persist_secondary_report(report, output_root)
    log_event(_logger, logging.INFO, "qa_report_path", path=report_uri)

    return 0 if failed == 0 else 1


def _run_imperviousness(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Process both imperviousness vintages. Returns the failed-count."""
    from berlin_lst_downscaling.data.secondary.imperviousness import (
        config_hash_for_vintage,
        prepare_imperviousness,
    )

    vintages: list[int] = list(cfg.get("vintages", [2016, 2021]))
    grid = canon_grid_10m()
    failed = 0

    for vintage in vintages:
        item_id = f"imperviousness_{vintage}"
        c_hash = config_hash_for_vintage(vintage)

        items = [(item_id, "imperviousness", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="imperviousness", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="imperviousness",
            period_or_vintage=str(vintage),
            status="exporting",
            run_id=run_id,
        ))

        try:
            log_event(
                _logger, logging.INFO, "processing",
                source="imperviousness",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_imperviousness(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            log_event(
                _logger, logging.ERROR, "failed",
                source="imperviousness",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(SecondaryLedgerRow(
                item_id=item_id,
                source="imperviousness",
                period_or_vintage=str(vintage),
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="imperviousness",
            period_or_vintage=str(vintage),
            status="done",
            run_id=run_id,
            config_hash=c_hash,
            output_uri=artifacts.cog_uri,
            stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))

        log_event(
            _logger, logging.INFO, "done",
            source="imperviousness",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


# ── helpers ───────────────────────────────────────────────────────────


def _run_vegetation_height(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Process the 2020 vegetation-height vintage. Returns the failed-count."""
    from berlin_lst_downscaling.data.secondary.vegetation_height import (
        config_hash_for_vintage,
        prepare_vegetation_height,
    )

    vintages: list[int] = list(cfg.get("vintages", [2020]))
    grid = canon_grid_10m()
    failed = 0

    for vintage in vintages:
        item_id = f"vegetation_height_{vintage}"
        c_hash = config_hash_for_vintage(vintage)

        items = [(item_id, "vegetation_height", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="vegetation_height", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="vegetation_height",
            period_or_vintage=str(vintage),
            status="exporting",
            run_id=run_id,
        ))

        try:
            log_event(
                _logger, logging.INFO, "processing",
                source="vegetation_height",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_vegetation_height(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            log_event(
                _logger, logging.ERROR, "failed",
                source="vegetation_height",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(SecondaryLedgerRow(
                item_id=item_id,
                source="vegetation_height",
                period_or_vintage=str(vintage),
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="vegetation_height",
            period_or_vintage=str(vintage),
            status="done",
            run_id=run_id,
            config_hash=c_hash,
            output_uri=artifacts.cog_uri,
            stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))

        log_event(
            _logger, logging.INFO, "done",
            source="vegetation_height",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _run_terrain_height(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Process DGM terrain height vintage 2021. Returns the failed-count."""
    from berlin_lst_downscaling.data.secondary.dgm import (
        config_hash_for_vintage,
        prepare_terrain_height,
    )

    vintages: list[int] = list(cfg.get("vintages", [2021]))
    grid = canon_grid_10m()
    failed = 0

    for vintage in vintages:
        item_id = f"terrain_height_{vintage}"
        c_hash = config_hash_for_vintage(vintage)

        items = [(item_id, "terrain_height", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="terrain_height", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="terrain_height",
            period_or_vintage=str(vintage),
            status="exporting",
            run_id=run_id,
        ))

        try:
            log_event(
                _logger, logging.INFO, "processing",
                source="terrain_height",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_terrain_height(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            log_event(
                _logger, logging.ERROR, "failed",
                source="terrain_height",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(SecondaryLedgerRow(
                item_id=item_id,
                source="terrain_height",
                period_or_vintage=str(vintage),
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="terrain_height",
            period_or_vintage=str(vintage),
            status="done",
            run_id=run_id,
            config_hash=c_hash,
            output_uri=artifacts.cog_uri,
            stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))

        log_event(
            _logger, logging.INFO, "done",
            source="terrain_height",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _run_lod2_morphology(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> int:
    """Process LoD2 morphology vintage. Returns the failed-count."""
    from berlin_lst_downscaling.data.secondary.lod2 import (
        config_hash_for_vintage,
        prepare_lod2_morphology,
    )

    vintages: list[int] = list(cfg.get("vintages", [2026]))
    grid = canon_grid_10m()
    failed = 0

    for vintage in vintages:
        item_id = f"lod2_morphology_{vintage}"
        c_hash = config_hash_for_vintage(vintage)

        items = [(item_id, "lod2_morphology", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="lod2_morphology", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source="lod2_morphology",
            period_or_vintage=str(vintage),
            status="exporting",
            run_id=run_id,
        ))

        try:
            log_event(
                _logger, logging.INFO, "processing",
                source="lod2_morphology",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_lod2_morphology(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            log_event(
                _logger, logging.ERROR, "failed",
                source="lod2_morphology",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(SecondaryLedgerRow(
                item_id=item_id,
                source="lod2_morphology",
                period_or_vintage=str(vintage),
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
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
        ))

        log_event(
            _logger, logging.INFO, "done",
            source="lod2_morphology",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _banner(cfg: DictConfig, run_id: str, output_root: str) -> None:
    """Log a pipeline header."""
    log_event(_logger, logging.INFO, "run_start",
        pipeline="secondary", mode=cfg.mode, run_id=run_id, output_root=output_root)
