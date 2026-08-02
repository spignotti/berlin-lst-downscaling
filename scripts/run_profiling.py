#!/usr/bin/env python3
"""WB2c-1 Data Profiling Runner.

Read-only, repeatable profiling of all published COGs.
Validates structural alignment, computes descriptive statistics,
and emits a fixed artifact bundle.

Usage::

    # Full profiling on VM
    uv run python scripts/run_profiling.py --config-name full

    # Smoke profiling locally
    uv run python scripts/run_profiling.py --config-name smoke

    # Override specific values
    uv run python scripts/run_profiling.py --config-name full \\
        output_root=gs://berlin-lst-data/profiling/wb2c-1
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io.run_logging import RunLogSession, log_event
from berlin_lst_downscaling.data.profiling.inspection import inspect_asset
from berlin_lst_downscaling.data.profiling.inventory import build_all_assets
from berlin_lst_downscaling.data.profiling.models import ProfileRow
from berlin_lst_downscaling.data.profiling.report import emit_artifacts
from berlin_lst_downscaling.data.profiling.statistics import profile_row_statistics

_logger = logging.getLogger(__name__)


def run_profiling(cfg: DictConfig, run_id: str) -> int:
    """Execute the profiling pipeline."""
    output_root = str(cfg.output_root)

    # Build asset inventory
    log_event(_logger, logging.INFO, "building_inventory")
    assets = build_all_assets(
        manifest_uri=cfg.manifest_uri,
        ard_ledger_uri=cfg.ard_ledger_uri,
        dynamic_root=cfg.dynamic_root,
        inference_root=cfg.inference_root,
    )
    log_event(_logger, logging.INFO, "inventory_built", total_assets=len(assets))

    # Profile each asset
    rows: list[ProfileRow] = []
    for i, asset in enumerate(assets, 1):
        log_event(
            _logger,
            logging.INFO,
            "profiling_asset",
            index=i,
            total=len(assets),
            item_id=asset.item_id,
            source=asset.source,
        )

        row = inspect_asset(asset)
        row = profile_row_statistics(row, asset)
        rows.append(row)

    # Emit artifacts
    log_event(_logger, logging.INFO, "emitting_artifacts")
    emit_artifacts(rows, output_root)

    # Summary
    hard_failures = sum(1 for r in rows if r.has_hard_failure)
    log_event(
        _logger,
        logging.INFO,
        "profiling_completed",
        total_assets=len(rows),
        hard_failures=hard_failures,
    )

    if hard_failures > 0:
        log_event(_logger, logging.WARNING, "hard_failures_detected", count=hard_failures)
        return 1

    return 0


@hydra.main(config_path="../configs/profiling", config_name="full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to WB2c-1 profiling."""
    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri:
        raise SystemExit(
            "manifest_uri is required — provide the published bundle, e.g.\n"
            "  manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet"
        )
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="profiling", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "profiling_started",
            manifest_uri=manifest_uri,
            output_root=output_root,
        )
        return run_profiling(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
