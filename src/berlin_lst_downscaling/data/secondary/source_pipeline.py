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

import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m, smoke_grid
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import (
    ledger_path,
    source_product_dir,
)
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)


def run_sources(cfg: DictConfig) -> int:
    """Execute the static source pipeline (Pipeline A).

    Returns 0 on success, 1 if any items failed.
    """
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    t0 = time.perf_counter()

    _banner(cfg, run_id, source_root)

    led = SecondaryLedger.open(ledger_path(source_root))

    sources: list[str] = list(cfg.get("sources", []))
    if not sources:
        print("  No sources configured — nothing to do.")
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
                led, cfg, run_id, source_root, smoke_count, grid,
            )
        elif source == "lod2_morphology":
            failed += _run_lod2_morphology(
                led, cfg, run_id, source_root, smoke_count, grid,
            )
        else:
            print(f"  Unknown source '{source}' — skipping.")
            failed += 1

    report = secondary_qa_report(led, run_id, sources=sources)
    print(format_secondary_report(report))
    persist_secondary_report(report, source_root)

    elapsed = time.perf_counter() - t0
    print(f"  Duration: {elapsed:.1f}s")
    return 0 if failed == 0 else 1


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
        prepare_imperviousness,
    )

    vintages: list[int] = list(cfg.get("vintages", [2016, 2021]))
    grid = grid or canon_grid_10m()
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
            item_id=item_id, source="imperviousness",
            period_or_vintage=str(vintage), status="exporting",
            run_id=run_id,
        ))

        try:
            print(f"  Processing imperviousness {vintage} (reason={reason})...")
            prepared = prepare_imperviousness(vintage, source_root, run_id, grid=grid)
            prod_dir = source_product_dir(source_root, "imperviousness", str(vintage))
            artifacts = finalize_secondary_product(
                prepared, grid, source_root, run_id,
                product_dir_override=prod_dir,
            )
        except Exception as exc:
            print(f"  imperviousness {vintage} FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=item_id, source="imperviousness",
                period_or_vintage=str(vintage), status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="imperviousness",
            period_or_vintage=str(vintage), status="done",
            run_id=run_id, config_hash=c_hash,
            output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))
        print(f"  imperviousness {vintage} OK — {artifacts.cog_uri}")

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
            item_id=item_id, source="vegetation_height",
            period_or_vintage=str(vintage), status="exporting",
            run_id=run_id,
        ))

        try:
            print(f"  Processing vegetation_height {vintage} (reason={reason})...")
            prepared = prepare_vegetation_height(vintage, source_root, run_id, grid=grid)
            prod_dir = source_product_dir(source_root, "vegetation_height", str(vintage))
            artifacts = finalize_secondary_product(
                prepared, grid, source_root, run_id,
                product_dir_override=prod_dir,
            )
        except Exception as exc:
            print(f"  vegetation_height {vintage} FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=item_id, source="vegetation_height",
                period_or_vintage=str(vintage), status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="vegetation_height",
            period_or_vintage=str(vintage), status="done",
            run_id=run_id, config_hash=c_hash,
            output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))
        print(f"  vegetation_height {vintage} OK — {artifacts.cog_uri}")

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
            print(f"  terrain_height {vintage} already done — skipping.")
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="terrain_height",
            period_or_vintage=str(vintage), status="exporting",
            run_id=run_id,
        ))

        try:
            print(f"  Processing terrain_height {vintage} (reason={reason})...")
            prepared = prepare_terrain_height(
                vintage, source_root, run_id, smoke_tile_count=smoke_tile_count, grid=grid,
            )
            prod_dir = source_product_dir(source_root, "terrain_height", str(vintage))
            artifacts = finalize_secondary_product(
                prepared, grid, source_root, run_id,
                product_dir_override=prod_dir,
            )
        except Exception as exc:
            print(f"  terrain_height {vintage} FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=item_id, source="terrain_height",
                period_or_vintage=str(vintage), status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="terrain_height",
            period_or_vintage=str(vintage), status="done",
            run_id=run_id, config_hash=c_hash,
            output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))
        print(f"  terrain_height {vintage} OK — {artifacts.cog_uri}")

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
            print(f"  lod2_morphology {vintage} already done — skipping.")
            continue

        reason = todo[0][3]
        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="lod2_morphology",
            period_or_vintage=str(vintage), status="exporting",
            run_id=run_id,
        ))

        try:
            print(f"  Processing lod2_morphology {vintage} (reason={reason})...")
            prepared = prepare_lod2_morphology(
                vintage, source_root, run_id, smoke_tile_count=smoke_tile_count, grid=grid,
            )
            prod_dir = source_product_dir(source_root, "lod2_morphology", str(vintage))
            artifacts = finalize_secondary_product(
                prepared, grid, source_root, run_id,
                product_dir_override=prod_dir,
            )
        except Exception as exc:
            print(f"  lod2_morphology {vintage} FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=item_id, source="lod2_morphology",
                period_or_vintage=str(vintage), status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1
            continue

        led.upsert(SecondaryLedgerRow(
            item_id=item_id, source="lod2_morphology",
            period_or_vintage=str(vintage), status="done",
            run_id=run_id, config_hash=c_hash,
            output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
            provenance_uri=artifacts.provenance_uri,
            completion_uri=artifacts.completion_uri,
        ))
        print(f"  lod2_morphology {vintage} OK — {artifacts.cog_uri}")

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
    width = 60
    print("=" * width, flush=True)
    print(f"Static Sources Pipeline (A) — mode={cfg.mode}", flush=True)
    print(f"  run_id      : {run_id}", flush=True)
    print(f"  source_root : {source_root}", flush=True)
    print("=" * width, flush=True)
