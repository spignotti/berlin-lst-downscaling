"""Pipeline B — derived geometry product computation.

Consumes finalized Pipeline A source products (local or GCS) and produces:
- building_dsm, vegetation_dsm, combined_dsm
- horizon_building, horizon_vegetation (36-band cubes)
- svf

Pipeline B refuses any input that is not a finalized product with
valid COG, STAC, provenance, and completion marker.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import (
    derived_ledger_path,
    derived_product_dir,
)
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)
from berlin_lst_downscaling.data.secondary.source_products import (
    resolve_source_products,
)

_logger = logging.getLogger(__name__)


def run_derived(cfg: DictConfig) -> int:
    """Execute the derived geometry pipeline (Pipeline B).

    Returns 0 on success, 1 if any items failed.
    """
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    derived_root = str(cfg.derived_root)
    geometry_id = str(cfg.geometry_id)
    t0 = time.perf_counter()

    _banner(cfg, run_id, source_root, derived_root)

    # ── 0. preflight: resolve source products ────────────────────────
    upstream_cfg = dict(cfg.get("upstream", {}))
    report = resolve_source_products(source_root, upstream_cfg or None)
    if not report.ok:
        log_event(_logger, logging.ERROR, "source_resolution_failed")
        for err in report.errors:
            log_event(_logger, logging.ERROR, "source_resolution_error", error=str(err))
        return 1

    log_event(_logger, logging.INFO, "sources_resolved", n=len(report.resolved))
    for r in report.resolved:
        log_event(
            _logger, logging.INFO, "source_resolved",
            source=r.source,
            revision=r.revision,
            cog_uri=r.cog_uri,
        )

    # ── 1. infer grid from terrain COG ───────────────────────────────
    terrain_resolved = next(
        (r for r in report.resolved if r.source == "terrain_height"), None,
    )
    if terrain_resolved is None:
        log_event(_logger, logging.ERROR, "terrain_height_not_resolved")
        return 1

    from berlin_lst_downscaling.common.grid import grid_from_cog
    grid = grid_from_cog(terrain_resolved.cog_uri)
    log_event(
        _logger, logging.INFO, "grid",
        shape=grid.shape,
        xoff=grid.transform.xoff,
        yoff=grid.transform.yoff,
    )

    # Build upstream map: "source/revision" → ResolvedSource
    src_map = {f"{r.source}/{r.revision}": r for r in report.resolved}

    led = SecondaryLedger.open(derived_ledger_path(derived_root))
    failed = 0

    # ── 2. derive DSMs ───────────────────────────────────────────────
    failed += _run_dsm_products(
        led, cfg, run_id, derived_root, geometry_id, src_map, grid,
    )

    # ── 3. derive horizons + SVF ─────────────────────────────────────
    failed += _run_horizon_svf(
        led, cfg, run_id, derived_root, geometry_id, src_map, grid,
    )

    # ── 4. final report ──────────────────────────────────────────────
    report = secondary_qa_report(led, run_id)
    log_event(_logger, logging.INFO, "qa_report", report=format_secondary_report(report))
    persist_secondary_report(report, derived_root)

    elapsed = time.perf_counter() - t0
    log_event(_logger, logging.INFO, "duration", elapsed_s=round(elapsed, 1))
    return 0 if failed == 0 else 1


# ── DSM stage ────────────────────────────────────────────────────────


def _run_dsm_products(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    derived_root: str,
    geometry_id: str,
    src_map: dict,
    grid,
) -> int:
    """Compute and publish building/vegetation/combined DSMs."""
    from berlin_lst_downscaling.data.secondary.dsm import (
        config_hash_for_dsm,
        prepare_building_dsm,
        prepare_combined_dsm,
        prepare_vegetation_dsm,
    )

    failed = 0

    terrain = src_map.get("terrain_height/2021")
    vh = src_map.get("vegetation_height/2020")
    lod2 = src_map.get("lod2_morphology/2024")

    if terrain is None or vh is None or lod2 is None:
        missing = [s for s, v in [
            ("terrain_height/2021", terrain),
            ("vegetation_height/2020", vh),
            ("lod2_morphology/2024", lod2),
        ] if v is None]
        log_event(_logger, logging.WARNING, "dsm_skipped_missing_upstream", missing=missing)
        return 1

    upstream_hashes = {
        "terrain_hash": terrain.config_hash,
        "lod2_hash": lod2.config_hash,
        "vh_hash": vh.config_hash,
    }

    # Building DSM
    bldg_item = "building_dsm"
    bldg_hash = config_hash_for_dsm("building", **upstream_hashes)
    todo = reconcile([(bldg_item, bldg_item, geometry_id)], led, bldg_hash)

    if todo:
        log_event(_logger, logging.INFO, "processing", product="building_dsm")
        led.upsert(SecondaryLedgerRow(
            item_id=bldg_item, source="building_dsm",
            period_or_vintage=geometry_id, status="exporting",
            run_id=run_id,
        ))
        try:
            prod_dir = derived_product_dir(derived_root, bldg_item, geometry_id)
            prepared = prepare_building_dsm(
                terrain.cog_uri, lod2.cog_uri, derived_root, run_id,
                item_key=geometry_id, upstream_hashes=upstream_hashes, grid=grid,
            )
            artifacts = finalize_secondary_product(
                prepared, grid, derived_root, run_id,
                product_dir_override=prod_dir,
            )
            led.upsert(SecondaryLedgerRow(
                item_id=bldg_item, source="building_dsm",
                period_or_vintage=geometry_id, status="done",
                run_id=run_id, config_hash=bldg_hash,
                output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            ))
            log_event(
                _logger, logging.INFO, "done",
                product="building_dsm", output_uri=artifacts.cog_uri,
            )
        except Exception as exc:
            log_event(_logger, logging.ERROR, "failed", product="building_dsm", error=str(exc))
            led.upsert(SecondaryLedgerRow(
                item_id=bldg_item, source="building_dsm",
                period_or_vintage=geometry_id, status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1

    # Vegetation DSM
    veg_item = "vegetation_dsm"
    veg_hash = config_hash_for_dsm("vegetation", **upstream_hashes)
    todo = reconcile([(veg_item, veg_item, geometry_id)], led, veg_hash)

    if todo:
        log_event(_logger, logging.INFO, "processing", product="vegetation_dsm")
        led.upsert(SecondaryLedgerRow(
            item_id=veg_item, source="vegetation_dsm",
            period_or_vintage=geometry_id, status="exporting",
            run_id=run_id,
        ))
        try:
            prod_dir = derived_product_dir(derived_root, veg_item, geometry_id)
            prepared = prepare_vegetation_dsm(
                terrain.cog_uri, vh.cog_uri, derived_root, run_id,
                item_key=geometry_id, upstream_hashes=upstream_hashes, grid=grid,
            )
            artifacts = finalize_secondary_product(
                prepared, grid, derived_root, run_id,
                product_dir_override=prod_dir,
            )
            led.upsert(SecondaryLedgerRow(
                item_id=veg_item, source="vegetation_dsm",
                period_or_vintage=geometry_id, status="done",
                run_id=run_id, config_hash=veg_hash,
                output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            ))
            log_event(
                _logger, logging.INFO, "done",
                product="vegetation_dsm", output_uri=artifacts.cog_uri,
            )
        except Exception as exc:
            log_event(_logger, logging.ERROR, "failed", product="vegetation_dsm", error=str(exc))
            led.upsert(SecondaryLedgerRow(
                item_id=veg_item, source="vegetation_dsm",
                period_or_vintage=geometry_id, status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1

    # Combined DSM — depends on building and vegetation DSMs
    from berlin_lst_downscaling.data.io.storage import exists
    bldg_cog = _product_cog(derived_root, bldg_item, geometry_id)
    veg_cog = _product_cog(derived_root, veg_item, geometry_id)

    if exists(bldg_cog) and exists(veg_cog):
        combined_item = "combined_dsm"
        combined_hash = config_hash_for_dsm("combined", **upstream_hashes)
        todo = reconcile([(combined_item, combined_item, geometry_id)], led, combined_hash)

        if todo:
            log_event(_logger, logging.INFO, "processing", product="combined_dsm")
            led.upsert(SecondaryLedgerRow(
                item_id=combined_item, source="combined_dsm",
                period_or_vintage=geometry_id, status="exporting",
                run_id=run_id,
            ))
            try:
                prod_dir = derived_product_dir(derived_root, combined_item, geometry_id)
                prepared = prepare_combined_dsm(
                    bldg_cog, veg_cog, derived_root, run_id,
                    item_key=geometry_id, upstream_hashes=upstream_hashes, grid=grid,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                    product_dir_override=prod_dir,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id=combined_item, source="combined_dsm",
                    period_or_vintage=geometry_id, status="done",
                    run_id=run_id, config_hash=combined_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                log_event(
                    _logger, logging.INFO, "done",
                    product="combined_dsm", output_uri=artifacts.cog_uri,
                )
            except Exception as exc:
                log_event(_logger, logging.ERROR, "failed", product="combined_dsm", error=str(exc))
                led.upsert(SecondaryLedgerRow(
                    item_id=combined_item, source="combined_dsm",
                    period_or_vintage=geometry_id, status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1

    return failed


# ── Horizons + SVF stage ────────────────────────────────────────────


def _run_horizon_svf(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    derived_root: str,
    geometry_id: str,
    src_map: dict,
    grid,
) -> int:
    """Compute horizons from component DSMs and SVF from combined DSM."""
    from berlin_lst_downscaling.data.io.storage import exists
    from berlin_lst_downscaling.data.secondary.horizon import (
        config_hash_for_horizon,
        prepare_horizon,
    )
    from berlin_lst_downscaling.data.secondary.svf import (
        config_hash_for_svf,
        prepare_svf,
    )

    failed = 0
    max_radius_m = cfg.get("horizon_max_radius_m", 200)
    svf_max_radius = cfg.get("svf_max_radius", 3)
    svf_n_dir = cfg.get("svf_n_directions", 16)

    # Component DSM COGs
    bldg_dsm_cog = _product_cog(derived_root, "building_dsm", geometry_id)
    veg_dsm_cog = _product_cog(derived_root, "vegetation_dsm", geometry_id)
    combined_cog = _product_cog(derived_root, "combined_dsm", geometry_id)

    # Building horizon — from building_dsm
    if exists(bldg_dsm_cog):
        horizon_bldg_hash = config_hash_for_horizon("building", max_radius_m, geometry_id)
        todo = reconcile(
            [("horizon_building", "horizon_building", geometry_id)],
            led, horizon_bldg_hash,
        )
        if todo:
            log_event(_logger, logging.INFO, "processing", product="horizon_building")
            led.upsert(SecondaryLedgerRow(
                item_id="horizon_building", source="horizon_building",
                period_or_vintage=geometry_id, status="exporting",
                run_id=run_id,
            ))
            try:
                prod_dir = derived_product_dir(derived_root, "horizon_building", geometry_id)
                prepared = prepare_horizon(
                    bldg_dsm_cog, derived_root, run_id,
                    item_key=geometry_id, component="building",
                    upstream_hash=geometry_id, max_radius_m=max_radius_m, grid=grid,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                    product_dir_override=prod_dir,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id="horizon_building", source="horizon_building",
                    period_or_vintage=geometry_id, status="done",
                    run_id=run_id, config_hash=horizon_bldg_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                log_event(
                    _logger, logging.INFO, "done",
                    product="horizon_building", output_uri=artifacts.cog_uri,
                )
            except Exception as exc:
                log_event(
                    _logger, logging.ERROR, "failed",
                    product="horizon_building", error=str(exc),
                )
                led.upsert(SecondaryLedgerRow(
                    item_id="horizon_building", source="horizon_building",
                    period_or_vintage=geometry_id, status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1
    else:
        log_event(
            _logger, logging.INFO, "skipped",
            product="horizon_building", reason="building_dsm not available",
        )

    # Vegetation horizon — from vegetation_dsm
    if exists(veg_dsm_cog):
        horizon_veg_hash = config_hash_for_horizon("vegetation", max_radius_m, geometry_id)
        todo = reconcile(
            [("horizon_vegetation", "horizon_vegetation", geometry_id)],
            led, horizon_veg_hash,
        )
        if todo:
            log_event(_logger, logging.INFO, "processing", product="horizon_vegetation")
            led.upsert(SecondaryLedgerRow(
                item_id="horizon_vegetation", source="horizon_vegetation",
                period_or_vintage=geometry_id, status="exporting",
                run_id=run_id,
            ))
            try:
                prod_dir = derived_product_dir(derived_root, "horizon_vegetation", geometry_id)
                prepared = prepare_horizon(
                    veg_dsm_cog, derived_root, run_id,
                    item_key=geometry_id, component="vegetation",
                    upstream_hash=geometry_id, max_radius_m=max_radius_m, grid=grid,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                    product_dir_override=prod_dir,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id="horizon_vegetation", source="horizon_vegetation",
                    period_or_vintage=geometry_id, status="done",
                    run_id=run_id, config_hash=horizon_veg_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                log_event(
                    _logger, logging.INFO, "done",
                    product="horizon_vegetation", output_uri=artifacts.cog_uri,
                )
            except Exception as exc:
                log_event(
                    _logger, logging.ERROR, "failed",
                    product="horizon_vegetation", error=str(exc),
                )
                led.upsert(SecondaryLedgerRow(
                    item_id="horizon_vegetation", source="horizon_vegetation",
                    period_or_vintage=geometry_id, status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1
    else:
        log_event(
            _logger, logging.INFO, "skipped",
            product="horizon_vegetation", reason="vegetation_dsm not available",
        )

    # SVF — from combined DSM
    if exists(combined_cog):
        svf_hash = config_hash_for_svf(svf_max_radius, svf_n_dir, geometry_id)
        todo = reconcile(
            [("svf", "svf", geometry_id)],
            led, svf_hash,
        )
        if todo:
            log_event(_logger, logging.INFO, "processing", product="svf")
            led.upsert(SecondaryLedgerRow(
                item_id="svf", source="svf",
                period_or_vintage=geometry_id, status="exporting",
                run_id=run_id,
            ))
            try:
                prod_dir = derived_product_dir(derived_root, "svf", geometry_id)
                prepared = prepare_svf(
                    combined_cog, derived_root, run_id,
                    item_key=geometry_id, upstream_hash=geometry_id,
                    max_radius=svf_max_radius, n_directions=svf_n_dir, grid=grid,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                    product_dir_override=prod_dir,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id="svf", source="svf",
                    period_or_vintage=geometry_id, status="done",
                    run_id=run_id, config_hash=svf_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                log_event(
                    _logger, logging.INFO, "done",
                    product="svf", output_uri=artifacts.cog_uri,
                )
            except Exception as exc:
                log_event(_logger, logging.ERROR, "failed", product="svf", error=str(exc))
                led.upsert(SecondaryLedgerRow(
                    item_id="svf", source="svf",
                    period_or_vintage=geometry_id, status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1
    else:
        log_event(
            _logger, logging.INFO, "skipped",
            product="svf", reason="combined_dsm not available",
        )

    return failed


# ── helpers ──────────────────────────────────────────────────────────


def _product_cog(root: str, product: str, geometry_id: str) -> str:
    """Build the COG URI for a derived product."""
    from berlin_lst_downscaling.data.secondary.paths import derived_product_cog
    return derived_product_cog(root, product, geometry_id)


def _banner(
    cfg: DictConfig, run_id: str, source_root: str, derived_root: str,
) -> None:
    log_event(_logger, logging.INFO, "run_start",
        pipeline="static-derived", mode=cfg.mode, run_id=run_id,
        source_root=source_root, derived_root=derived_root)
