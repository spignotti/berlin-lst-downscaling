"""Secondary-data pipeline — execution orchestration.

Supports three modes:

* ``fixture`` — source-neutral dummy COG, validates the full lifecycle
  (contract → COG write → validation → ledger → QA report) without
  downloading any real dataset.
* ``cloud_smoke`` — same fixture targeting GCS.
* ``full`` — real source processing (future tasks).
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
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
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

    print(f"Mode '{cfg.mode}' not yet implemented — no data sources configured.")
    return 0


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


# ── helpers ───────────────────────────────────────────────────────────


def _banner(cfg: DictConfig, run_id: str, output_root: str) -> None:
    """Print a pipeline header."""
    width = 60
    print("=" * width, flush=True)
    print(f"Secondary Pipeline — mode={cfg.mode}", flush=True)
    print(f"  run_id      : {run_id}", flush=True)
    print(f"  output_root : {output_root}", flush=True)
    print("=" * width, flush=True)
