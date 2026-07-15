"""Secondary-data pipeline — execution orchestration.

Supports two modes:

* ``fixture`` — small synthetic products per registered source,
  validates the full lifecycle without downloading real datasets.
* ``full`` — real source processing; produces the four final
  artifacts (COG, STAC, provenance, completion marker) for each
  configured source/vintage.
"""

from __future__ import annotations

import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import ledger_path
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)

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
        print(f"  Duration: {elapsed:.1f}s")
        return rc

    if cfg.mode == "full":
        rc = _run_full(led, cfg, run_id, output_root)
        elapsed = time.perf_counter() - t0
        print(f"  Duration: {elapsed:.1f}s")
        return rc

    print(f"Unknown mode '{cfg.mode}'.")
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
            print(f"  fixture[{source}] already done — skipping.")
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source=source,
            period_or_vintage=period,
            status="exporting",
            run_id=run_id,
        ))

        try:
            print(f"  Finalising fixture[{source}]...")
            prepared = factory(output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            print(f"  fixture[{source}] FAILED: {exc}")
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
        print(f"  fixture[{source}] OK — {artifacts.cog_uri}")

    report = secondary_qa_report(led, run_id, sources=list(registry().keys()))
    print(format_secondary_report(report))
    report_uri = persist_secondary_report(report, output_root)
    print(f"  QA report  : {report_uri}")
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
        print("  No sources configured — nothing to do.")
        return 0

    peak_scratch_gb = cfg.get("peak_scratch_gb", 999)
    disk_budget_gb = cfg.get("disk_budget_gb", 20)
    if peak_scratch_gb > disk_budget_gb:
        print(
            f"  DISK BUDGET EXCEEDED: peak_scratch_gb={peak_scratch_gb} > "
            f"disk_budget_gb={disk_budget_gb}"
        )
        return 1

    failed = 0

    for source in sources:
        if source == "imperviousness":
            failed += _run_imperviousness(led, cfg, run_id, output_root)
        elif source == "vegetation_height":
            failed += _run_vegetation_height(led, cfg, run_id, output_root)
        else:
            print(f"  Unknown source '{source}' — skipping.")
            failed += 1

    # Final QA report — printed and persisted under qa/secondary/{run_id}/
    report = secondary_qa_report(led, run_id)
    print(format_secondary_report(report))
    report_uri = persist_secondary_report(report, output_root)
    print(f"  QA report  : {report_uri}")

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
            print(f"  imperviousness {vintage} already done — skipping.")
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
            print(f"  Processing imperviousness {vintage} (reason={reason})...")
            prepared = prepare_imperviousness(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            print(f"  imperviousness {vintage} FAILED: {exc}")
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

        print(f"  imperviousness {vintage} OK — {artifacts.cog_uri}")

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
            print(f"  vegetation_height {vintage} already done — skipping.")
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
            print(f"  Processing vegetation_height {vintage} (reason={reason})...")
            prepared = prepare_vegetation_height(vintage, output_root, run_id)
            artifacts = finalize_secondary_product(
                prepared, grid, output_root, run_id,
            )
        except Exception as exc:
            print(f"  vegetation_height {vintage} FAILED: {exc}")
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

        print(f"  vegetation_height {vintage} OK — {artifacts.cog_uri}")

    return failed


def _banner(cfg: DictConfig, run_id: str, output_root: str) -> None:
    """Print a pipeline header."""
    width = 60
    print("=" * width, flush=True)
    print(f"Secondary Pipeline — mode={cfg.mode}", flush=True)
    print(f"  run_id      : {run_id}", flush=True)
    print(f"  output_root : {output_root}", flush=True)
    print("=" * width, flush=True)
