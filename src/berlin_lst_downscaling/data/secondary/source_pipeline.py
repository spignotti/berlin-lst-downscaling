"""Pipeline A — static source product acquisition and preparation.

Downloads official source archives, processes them to the canonical
10 m EPSG:25833 grid, and publishes the four final artifacts (COG,
STAC, provenance, completion marker) for each source.

Pipeline A is strictly independent of Pipeline B; it does not
depend on any derived geometry products.

Sources handled: imperviousness, vegetation_height, terrain_height,
lod2_morphology.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m, smoke_grid
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import source_product_dir
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)


def run_sources(cfg: DictConfig, run_id: str | None = None) -> int:
    """Execute the static source pipeline (Pipeline A).

    Returns 0 only when every expected ledger row is done after the run,
    1 otherwise. The closing ledger audit is fail-closed: a run that
    processed zero items but left a failed row behind still exits non-zero.
    """
    if run_id is None:
        run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    t0 = time.perf_counter()

    _banner(cfg, run_id, source_root)

    led = SecondaryLedger.open(f"{source_root.rstrip('/')}/ledger.parquet")

    sources: list[str] = list(cfg.get("sources", []))
    if not sources:
        log_event(_logger, logging.INFO, "no_sources")
        return 0

    failed = 0
    smoke_count = cfg.get("smoke_tile_count")
    grid = _resolve_grid(cfg)

    for source in sources:
        if source == "imperviousness":
            failed += _run_imperviousness(led, cfg, run_id, source_root, grid)
        elif source == "vegetation_height":
            failed += _run_vegetation_height(led, cfg, run_id, source_root, grid)
        elif source == "terrain_height":
            failed += _run_terrain_height(
                led,
                cfg,
                run_id,
                source_root,
                smoke_count,
                grid,
            )
        elif source == "lod2_morphology":
            failed += _run_lod2_morphology(
                led,
                cfg,
                run_id,
                source_root,
                smoke_count,
                grid,
            )
        else:
            log_event(_logger, logging.WARNING, "unknown_source", source=source)
            failed += 1

    report = secondary_qa_report(led, run_id, sources=sources)
    log_event(_logger, logging.INFO, "qa_report", report=format_secondary_report(report))
    persist_secondary_report(report, source_root)

    # ── fail-closed ledger audit ─────────────────────────────────────
    # ``failed`` counts items that raised during this run, but a row can
    # also be non-done without raising here (interrupted previous run,
    # partial publication, or a runner bug that skips without reconcile).
    # Re-read the ledger and refuse success unless every expected row is
    # done with resolvable artifacts.
    audit_errors = find_not_done_items(led, expected_source_items(cfg))
    if audit_errors:
        log_event(
            _logger,
            logging.ERROR,
            "ledger_not_all_done",
            errors=audit_errors,
        )
        print(f"FAIL: static sources ledger not all done ({len(audit_errors)} items).")
        for err in audit_errors:
            print(f"  - {err}")

    elapsed = time.perf_counter() - t0
    log_event(_logger, logging.INFO, "duration", elapsed_s=round(elapsed, 1))
    if audit_errors:
        return 1
    return 0 if failed == 0 else 1


# ── fail-closed ledger audit ─────────────────────────────────────────

# Default vintages mirror the per-source ``cfg.get("vintages", [...])``
# fallbacks in the runners above. A shared ``vintages`` key overrides all.
_DEFAULT_VINTAGES: dict[str, list[int]] = {
    "imperviousness": [2016, 2021],
    "vegetation_height": [2020],
    "terrain_height": [2021],
    "lod2_morphology": [2024],
}


def expected_source_items(cfg) -> list[tuple[str, str, str]]:
    """Return the expected ``(item_id, source, period)`` rows for this config."""
    sources: list[str] = list(cfg.get("sources", []))
    out: list[tuple[str, str, str]] = []
    for source in sources:
        defaults = _DEFAULT_VINTAGES.get(source, [])
        vintages: list[int] = list(cfg.get("vintages", defaults))
        for vintage in vintages:
            out.append((f"{source}_{vintage}", source, str(vintage)))
    return out


def find_not_done_items(led: SecondaryLedger, expected: list[tuple[str, str, str]]) -> list[str]:
    """Return human-readable errors for every expected row that is not done.

    A ``done`` row must still resolve to a real COG and completion marker;
    otherwise it is reported (partial publication is never success).
    """
    from berlin_lst_downscaling.data.io import exists, resolve_canonical_uri

    errors: list[str] = []
    for item_id, source, period in expected:
        row = led.get(item_id, source, period)
        if row is None:
            errors.append(f"{item_id}: missing ledger row")
            continue
        if row.status != "done":
            errors.append(
                f"{item_id}: status={row.status} run_id={row.run_id} last_error={row.last_error}"
            )
            continue
        try:
            output_ok = bool(row.output_uri) and exists(resolve_canonical_uri(row.output_uri))
            completion_ok = bool(row.completion_uri) and exists(
                resolve_canonical_uri(row.completion_uri)
            )
        except Exception as exc:  # fail closed on unreadable artifacts
            errors.append(f"{item_id}: artifact check failed: {exc}")
            continue
        if not output_ok or not completion_ok:
            errors.append(
                f"{item_id}: done but artifacts missing "
                f"(output_ok={output_ok} completion_ok={completion_ok})"
            )
    return errors


_logger = logging.getLogger(__name__)

# ── runners ──────────────────────────────────────────────────────────


def _run_imperviousness(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    source_root: str,
    grid=None,
) -> int:
    """Process both imperviousness vintages."""
    from berlin_lst_downscaling.data.secondary.imperviousness import (
        config_hash_for_vintage,
        load_imperviousness_auxiliaries,
        prepare_imperviousness,
    )

    vintages: list[int] = list(cfg.get("vintages", [2016, 2021]))
    grid = grid or canon_grid_10m()
    failed = 0
    aux = load_imperviousness_auxiliaries(source_root)

    for vintage in vintages:
        item_id = f"imperviousness_{vintage}"
        c_hash = config_hash_for_vintage(
            vintage,
            street_content_sha256=aux.street_content_sha256,
            blocks_content_sha256=aux.blocks_content_sha256,
        )

        items = [(item_id, "imperviousness", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="imperviousness", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source="imperviousness",
                period_or_vintage=str(vintage),
                status="exporting",
                run_id=run_id,
            )
        )

        try:
            log_event(
                _logger,
                logging.INFO,
                "processing",
                source="imperviousness",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_imperviousness(vintage, source_root, run_id, grid=grid, aux=aux)
            prod_dir = source_product_dir(source_root, "imperviousness", str(vintage))
            artifacts = finalize_secondary_product(
                prepared,
                grid,
                prod_dir,
                run_id,
            )
        except Exception as exc:
            log_event(
                _logger,
                logging.ERROR,
                "failed",
                source="imperviousness",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(
                SecondaryLedgerRow(
                    item_id=item_id,
                    source="imperviousness",
                    period_or_vintage=str(vintage),
                    status="failed",
                    run_id=run_id,
                    last_error=str(exc),
                )
            )
            failed += 1
            continue

        led.upsert(
            SecondaryLedgerRow(
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
            )
        )
        log_event(
            _logger,
            logging.INFO,
            "done",
            source="imperviousness",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _run_vegetation_height(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    source_root: str,
    grid=None,
) -> int:
    """Process vegetation-height vintage 2020."""
    from berlin_lst_downscaling.data.secondary.vegetation_height import (
        config_hash_for_vintage,
        prepare_vegetation_height,
    )

    vintages: list[int] = list(cfg.get("vintages", [2020]))
    grid = grid or canon_grid_10m()
    failed = 0
    raw_threshold = cfg.get("canopy_cover_threshold_m", None)
    canopy_threshold = None if raw_threshold is None else float(raw_threshold)

    for vintage in vintages:
        item_id = f"vegetation_height_{vintage}"
        c_hash = config_hash_for_vintage(vintage, canopy_cover_threshold_m=canopy_threshold)

        items = [(item_id, "vegetation_height", str(vintage))]
        todo = reconcile(items, led, c_hash)

        if not todo:
            log_event(_logger, logging.INFO, "skipped", source="vegetation_height", vintage=vintage)
            continue

        reason = todo[0][3]
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source="vegetation_height",
                period_or_vintage=str(vintage),
                status="exporting",
                run_id=run_id,
            )
        )

        try:
            log_event(
                _logger,
                logging.INFO,
                "processing",
                source="vegetation_height",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_vegetation_height(
                vintage,
                source_root,
                run_id,
                grid=grid,
                canopy_cover_threshold_m=canopy_threshold,
            )
            prod_dir = source_product_dir(source_root, "vegetation_height", str(vintage))
            artifacts = finalize_secondary_product(
                prepared,
                grid,
                prod_dir,
                run_id,
            )
        except Exception as exc:
            log_event(
                _logger,
                logging.ERROR,
                "failed",
                source="vegetation_height",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(
                SecondaryLedgerRow(
                    item_id=item_id,
                    source="vegetation_height",
                    period_or_vintage=str(vintage),
                    status="failed",
                    run_id=run_id,
                    last_error=str(exc),
                )
            )
            failed += 1
            continue

        led.upsert(
            SecondaryLedgerRow(
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
            )
        )
        log_event(
            _logger,
            logging.INFO,
            "done",
            source="vegetation_height",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _run_terrain_height(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    source_root: str,
    smoke_tile_count: int | None,
    grid=None,
) -> int:
    """Process DGM terrain-height vintage 2021."""
    from berlin_lst_downscaling.data.secondary.dgm import (
        config_hash_for_vintage,
        prepare_terrain_height,
    )

    vintages: list[int] = list(cfg.get("vintages", [2021]))
    grid = grid or canon_grid_10m()
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
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source="terrain_height",
                period_or_vintage=str(vintage),
                status="exporting",
                run_id=run_id,
            )
        )

        try:
            log_event(
                _logger,
                logging.INFO,
                "processing",
                source="terrain_height",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_terrain_height(
                vintage,
                source_root,
                run_id,
                smoke_tile_count=smoke_tile_count,
                grid=grid,
            )
            prod_dir = source_product_dir(source_root, "terrain_height", str(vintage))
            artifacts = finalize_secondary_product(
                prepared,
                grid,
                prod_dir,
                run_id,
            )
        except Exception as exc:
            log_event(
                _logger,
                logging.ERROR,
                "failed",
                source="terrain_height",
                vintage=vintage,
                error=str(exc),
            )
            led.upsert(
                SecondaryLedgerRow(
                    item_id=item_id,
                    source="terrain_height",
                    period_or_vintage=str(vintage),
                    status="failed",
                    run_id=run_id,
                    last_error=str(exc),
                )
            )
            failed += 1
            continue

        led.upsert(
            SecondaryLedgerRow(
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
            )
        )
        log_event(
            _logger,
            logging.INFO,
            "done",
            source="terrain_height",
            vintage=vintage,
            output_uri=artifacts.cog_uri,
        )

    return failed


def _run_lod2_morphology(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    source_root: str,
    smoke_tile_count: int | None,
    grid=None,
) -> int:
    """Process LoD2 morphology vintage 2024."""
    from berlin_lst_downscaling.data.secondary.lod2 import (
        config_hash_for_vintage,
        prepare_lod2_morphology,
    )

    vintages: list[int] = list(cfg.get("vintages", [2024]))
    grid = grid or canon_grid_10m()
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
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source="lod2_morphology",
                period_or_vintage=str(vintage),
                status="exporting",
                run_id=run_id,
            )
        )

        try:
            log_event(
                _logger,
                logging.INFO,
                "processing",
                source="lod2_morphology",
                vintage=vintage,
                reason=reason,
            )
            prepared = prepare_lod2_morphology(
                vintage,
                source_root,
                run_id,
                smoke_tile_count=smoke_tile_count,
                grid=grid,
            )
            prod_dir = source_product_dir(source_root, "lod2_morphology", str(vintage))
            artifacts = finalize_secondary_product(
                prepared,
                grid,
                prod_dir,
                run_id,
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
            failed += 1
            continue

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

    return failed


# ── grid resolution ──────────────────────────────────────────────────


def _resolve_grid(cfg: DictConfig):
    """Return the output grid for this run.

    Uses ``smoke_grid_bbox`` if configured (for local smoke tests with a
    cropped subset), otherwise the full canonical 10 m Berlin grid.
    """
    bbox = cfg.get("smoke_grid_bbox")
    if bbox is not None:
        return smoke_grid(tuple(bbox))
    return canon_grid_10m()


# ── banner ───────────────────────────────────────────────────────────


def _banner(cfg: DictConfig, run_id: str, source_root: str) -> None:
    log_event(
        _logger,
        logging.INFO,
        "run_start",
        pipeline="static-sources",
        mode=cfg.mode,
        run_id=run_id,
        source_root=source_root,
    )
