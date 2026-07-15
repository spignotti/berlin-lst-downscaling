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

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.validate import validate_cog
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import ledger_path
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    finalize_secondary_product,
)
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
    """Run a source-neutral fixture: write a dummy COG and validate.

    Exercises the full pipeline lifecycle without downloading any real
    dataset.  The fixture is idempotent — a second identical run skips
    all work.
    """
    item_id = "fixture_001"
    source = "fixture"
    period = "2024"

    contract = Contract(
        source=source,
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="feature",
                dtype="float32",
                nodata=float("nan"),
                description="Fixture output band (random uniform)",
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )
    config_hash = "v1"

    # ── reconcile ──────────────────────────────────────────────────
    items = [(item_id, source, period)]
    todo = reconcile(items, led, config_hash)

    if not todo:
        print("  Fixture already done — nothing to process.")
        report = secondary_qa_report(led, run_id, sources=[source])
        print(format_secondary_report(report))
        return 0

    # ── generate fixture data on canonical grid ────────────────────
    grid = canon_grid_10m()
    rng = np.random.default_rng(42)
    data = rng.random((grid.shape.y, grid.shape.x)).astype(np.float32)

    # Create xr.Dataset with spatial references matching the canonical grid
    # pixel-center coordinates derived from the GeoBox affine transform
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"feature": (("y", "x"), data)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    # ── mark exporting ──────────────────────────────────────────────
    led.upsert(SecondaryLedgerRow(
        item_id=item_id,
        source=source,
        period_or_vintage=period,
        status="exporting",
        run_id=run_id,
    ))

    # ── write COG ──────────────────────────────────────────────────
    cog_dst = f"{output_root}/fixture/{item_id}.tif"
    print(f"  Writing {cog_dst}")
    write_cog_atomic(ds, cog_dst, contract, overwrite=True)

    # ── validate ────────────────────────────────────────────────────
    vig = validate_cog(cog_dst, contract, grid)
    if not vig.ok:
        led.upsert(SecondaryLedgerRow(
            item_id=item_id,
            source=source,
            period_or_vintage=period,
            status="failed",
            run_id=run_id,
            last_error="; ".join(vig.errors),
        ))
        print(f"  Fixture FAILED: {'; '.join(vig.errors)}")
        return 1

    print(f"  Validation OK — {cog_dst}")

    # ── finalise ledger ─────────────────────────────────────────────
    led.upsert(SecondaryLedgerRow(
        item_id=item_id,
        source=source,
        period_or_vintage=period,
        status="done",
        run_id=run_id,
        config_hash=config_hash,
        output_uri=cog_dst,
    ))

    # ── QA report ──────────────────────────────────────────────────
    report = secondary_qa_report(led, run_id, sources=[source])
    print(format_secondary_report(report))

    return 0


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

    all_qa: list[dict] = []
    failed = 0

    for source in sources:
        if source == "imperviousness":
            rc, qa = _run_imperviousness(led, cfg, run_id, output_root)
            if qa:
                all_qa.extend(qa)
            failed += rc
        elif source == "vegetation_height":
            rc, qa = _run_vegetation_height(led, cfg, run_id, output_root)
            if qa:
                all_qa.extend(qa)
            failed += rc
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
) -> tuple[int, list[dict]]:
    """Process both imperviousness vintages.

    Returns ``(failed_count, qa_payloads)``.  The caller is responsible
    for recording artifact URIs in the ledger and emitting the QA report.
    """
    from berlin_lst_downscaling.data.secondary.imperviousness import (
        config_hash_for_vintage,
        prepare_imperviousness,
    )

    vintages: list[int] = list(cfg.get("vintages", [2016, 2021]))
    grid = canon_grid_10m()

    qa_payloads: list[dict] = []
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
        qa_payloads.append(_qa_payload(prepared))

    return failed, qa_payloads


# ── helpers ───────────────────────────────────────────────────────────


def _run_vegetation_height(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    output_root: str,
) -> tuple[int, list[dict]]:
    """Process the 2020 vegetation-height vintage."""
    from berlin_lst_downscaling.data.secondary.vegetation_height import (
        config_hash_for_vintage,
        prepare_vegetation_height,
    )

    vintages: list[int] = list(cfg.get("vintages", [2020]))
    grid = canon_grid_10m()

    qa_payloads: list[dict] = []
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
        qa_payloads.append(_qa_payload(prepared))

    return failed, qa_payloads


def _banner(cfg: DictConfig, run_id: str, output_root: str) -> None:
    """Print a pipeline header."""
    width = 60
    print("=" * width, flush=True)
    print(f"Secondary Pipeline — mode={cfg.mode}", flush=True)
    print(f"  run_id      : {run_id}", flush=True)
    print(f"  output_root : {output_root}", flush=True)
    print("=" * width, flush=True)


def _qa_payload(prepared: PreparedSecondaryProduct) -> dict:
    """Extract the flat QA payload from a prepared product."""
    return {
        "source": prepared.source,
        "item_key": prepared.item_key,
        "config_hash": prepared.config_hash,
        **prepared.qa_stats,
    }
