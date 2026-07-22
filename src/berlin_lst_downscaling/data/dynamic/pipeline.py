"""Dynamic pipeline — per-scene ERA5-Land and shadow product generation.

Orchestrates the full lifecycle for each Landsat anchor scene:
1. Validate manifest and static geometry products
2. For each scene: prepare ERA5 meteorology COG
3. For each scene: prepare building + vegetation shadow COGs
4. Publish through shared finalizer (COG + STAC + provenance + complete)
5. Produce dynamic QA report with coverage, vintage distribution, DWD validation
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.dynamic.geometry import resolve_geometry
from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors
from berlin_lst_downscaling.data.dynamic.paths import ledger_path, scene_product_dir
from berlin_lst_downscaling.data.dynamic.schema import (
    GEOMETRY_TEMPORAL_MODE,
    GEOMETRY_VINTAGES,
    config_hash_for_dynamic,
    config_hash_for_era5,
)
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product

_logger = logging.getLogger(__name__)


def run_dynamic(cfg: DictConfig, run_id: str | None = None) -> int:
    """Execute the dynamic scene pipeline.

    Returns 0 on success, 1 if any items failed.
    """
    if run_id is None:
        run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    manifest_uri = str(cfg.manifest_uri)
    source_root = str(cfg.source_root)
    derived_root = str(cfg.derived_root)
    geometry_id = str(cfg.geometry_id)
    t0 = time.perf_counter()

    _banner(cfg, run_id, output_root, manifest_uri)

    # ── 0. preflight ─────────────────────────────────────────────────
    scene_ids = list(cfg.scene_ids) if cfg.scene_ids else None
    years = list(cfg.years) if cfg.get("years") else None
    dataset_role = cfg.get("dataset_role")
    expected_count = cfg.get("expected_scene_count")

    # When scene_ids are explicitly provided, skip year filter to allow
    # the child to find scenes across the full manifest range.
    filter_years = None if scene_ids else years

    manifest_report = load_landsat_anchors(
        manifest_uri,
        years=filter_years,
        scene_ids=scene_ids,
        dataset_role=dataset_role,
    )
    if not manifest_report.ok:
        log_event(_logger, logging.ERROR, "manifest_load_failed",
                  errors=manifest_report.errors)
        return 1

    # Validate expected scene count (skip when scene_ids override)
    if (expected_count is not None and not scene_ids
            and len(manifest_report.scenes) != expected_count):
        log_event(_logger, logging.ERROR, "scene_count_mismatch",
                  expected=expected_count, actual=len(manifest_report.scenes))
        return 1

    log_event(_logger, logging.INFO, "manifest_loaded",
              n_scenes=len(manifest_report.scenes),
              total_rows=manifest_report.total_rows,
              manifest_hash=manifest_report.manifest_hash,
              dataset_role=dataset_role)

    geom_report = resolve_geometry(source_root, derived_root, geometry_id)
    if not geom_report.ok or geom_report.resolved is None:
        log_event(_logger, logging.ERROR, "geometry_resolution_failed",
                  errors=geom_report.errors)
        return 1

    geo = geom_report.resolved
    log_event(_logger, logging.INFO, "geometry_resolved",
              geometry_id=geometry_id)

    # Separate config hashes: ERA5 uses v2 hash (triggers reprocessing),
    # shadows use the unchanged dynamic hash (existing products preserved).
    era5_hash = config_hash_for_era5(
        manifest_report.manifest_hash, geometry_id, output_root,
    )
    shadow_hash = config_hash_for_dynamic(
        manifest_report.manifest_hash, geometry_id, output_root,
    )

    grid = canon_grid_10m()
    led = SecondaryLedger.open(ledger_path(output_root))
    failed = 0
    processed = 0

    # ── 1. process scenes ────────────────────────────────────────────
    # Group scenes by acquisition month for bounded ERA5 local cache
    scenes_by_month: dict[tuple[int, int], list] = defaultdict(list)
    for scene in manifest_report.scenes:
        key = (scene.acquisition_datetime.year, scene.acquisition_datetime.month)
        scenes_by_month[key].append(scene)

    for (ym_year, ym_month), month_scenes in sorted(scenes_by_month.items()):
        log_event(_logger, logging.INFO, "month_group_start",
                  year=ym_year, month=ym_month, n_scenes=len(month_scenes))

        with tempfile.TemporaryDirectory(prefix=f"era5_{ym_year}{ym_month:02d}_") as tmp_dir:
            local_dir = Path(tmp_dir)

            for scene in month_scenes:
                log_event(_logger, logging.INFO, "scene_start",
                          scene_id=scene.scene_id,
                          year=scene.year,
                          doy=scene.day_of_year,
                          dt=scene.acquisition_datetime.isoformat())

                # ── 1a. ERA5 meteorology ─────────────────────────────
                era5_source = "era5_land"
                era5_item_id = f"era5_land_{scene.scene_id}"
                era5_todo = reconcile([(era5_item_id, era5_source, scene.scene_id)], led, era5_hash)

                if era5_todo:
                    led.upsert(SecondaryLedgerRow(
                        item_id=era5_item_id, source=era5_source,
                        period_or_vintage=scene.scene_id,
                        status="exporting", run_id=run_id,
                        role=dataset_role))
                    try:
                        from berlin_lst_downscaling.data.dynamic.era5 import prepare_era5_scene

                        prepared = prepare_era5_scene(
                            scene.scene_id, scene.acquisition_datetime,
                            output_root, run_id, grid=grid, local_dir=local_dir)
                        prod_dir = scene_product_dir(output_root, era5_source, scene.scene_id)
                        artifacts = finalize_secondary_product(
                            prepared, grid, output_root, run_id,
                            product_dir_override=prod_dir)
                        led.upsert(SecondaryLedgerRow(
                            item_id=era5_item_id, source=era5_source,
                            period_or_vintage=scene.scene_id,
                            status="done", run_id=run_id, config_hash=era5_hash,
                            output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                            provenance_uri=artifacts.provenance_uri,
                            completion_uri=artifacts.completion_uri,
                            role=dataset_role))
                        processed += 1
                        log_event(_logger, logging.INFO, "era5_done",
                                  scene_id=scene.scene_id, output_uri=artifacts.cog_uri)
                    except Exception as exc:
                        log_event(_logger, logging.ERROR, "era5_failed",
                                  scene_id=scene.scene_id, error=str(exc))
                        led.upsert(SecondaryLedgerRow(
                            item_id=era5_item_id, source=era5_source,
                            period_or_vintage=scene.scene_id,
                            status="failed", run_id=run_id, last_error=str(exc),
                            role=dataset_role))
                        failed += 1
                else:
                    log_event(_logger, logging.INFO, "era5_skipped",
                              scene_id=scene.scene_id)

                # ── 1b. Shadow masks (building + vegetation) ─────────
                azimuth = scene.solar_azimuth
                elevation = scene.solar_elevation

                if azimuth is None or elevation is None:
                    log_event(_logger, logging.WARNING, "shadow_skipped_no_solar",
                              scene_id=scene.scene_id)
                else:
                    for component in ("building", "vegetation"):
                        shadow_source = f"shadow_{component}"
                        shadow_item_id = f"shadow_{component}_{scene.scene_id}"
                        shadow_todo = reconcile(
                            [(shadow_item_id, shadow_source, scene.scene_id)], led, shadow_hash)

                        if shadow_todo:
                            led.upsert(SecondaryLedgerRow(
                                item_id=shadow_item_id, source=shadow_source,
                                period_or_vintage=scene.scene_id,
                                status="exporting", run_id=run_id,
                                role=dataset_role))
                            try:
                                from berlin_lst_downscaling.data.dynamic.shadows import (
                                    prepare_shadow,
                                )

                                horizon_uri = (
                                    geo.horizon_building_cog if component == "building"
                                    else geo.horizon_vegetation_cog
                                )
                                prepared = prepare_shadow(
                                    component=component,
                                    horizon_uri=horizon_uri,
                                    azimuth_deg=azimuth,
                                    elevation_deg=elevation,
                                    scene_id=scene.scene_id,
                                    output_root=output_root,
                                    run_id=run_id,
                                    grid=grid,
                                    geometry_id=geometry_id,
                                    geometry_hash=shadow_hash,
                                )
                                # Pass acquisition datetime into the product
                                prepared.acquisition_datetime = scene.acquisition_datetime
                                prepared.stac_properties = {
                                    **(prepared.stac_properties or {}),
                                    "acquisition:datetime": scene.acquisition_datetime.isoformat(),
                                    "acquisition:doy": scene.day_of_year,
                                    "acquisition:year": scene.year,
                                }

                                prod_dir = scene_product_dir(
                                    output_root, shadow_source, scene.scene_id)
                                artifacts = finalize_secondary_product(
                                    prepared, grid, output_root, run_id,
                                    product_dir_override=prod_dir)
                                led.upsert(SecondaryLedgerRow(
                                    item_id=shadow_item_id, source=shadow_source,
                                    period_or_vintage=scene.scene_id,
                                    status="done", run_id=run_id, config_hash=shadow_hash,
                                    output_uri=artifacts.cog_uri,
                                    stac_uri=artifacts.stac_uri,
                                    provenance_uri=artifacts.provenance_uri,
                                    completion_uri=artifacts.completion_uri,
                                    role=dataset_role))
                                processed += 1
                                log_event(_logger, logging.INFO, "shadow_done",
                                          scene_id=scene.scene_id,
                                          component=component,
                                          output_uri=artifacts.cog_uri)
                            except Exception as exc:
                                log_event(_logger, logging.ERROR, "shadow_failed",
                                          scene_id=scene.scene_id,
                                          component=component, error=str(exc))
                                led.upsert(SecondaryLedgerRow(
                                    item_id=shadow_item_id, source=shadow_source,
                                    period_or_vintage=scene.scene_id,
                                    status="failed", run_id=run_id,
                                    last_error=str(exc),
                                    role=dataset_role))
                                failed += 1

        log_event(_logger, logging.INFO, "month_group_done",
                  year=ym_year, month=ym_month)

    # ── 2. final report ──────────────────────────────────────────────
    from berlin_lst_downscaling.data.dynamic.reports import (
        dynamic_qa_report,
        format_dynamic_report,
        persist_dynamic_report,
    )

    report = dynamic_qa_report(
        led, run_id,
        manifest_hash=manifest_report.manifest_hash,
        geometry_id=geometry_id,
    )
    log_event(_logger, logging.INFO, "qa_report",
              report=format_dynamic_report(report))
    report_uri = persist_dynamic_report(report, output_root)
    log_event(_logger, logging.INFO, "qa_report_path", path=report_uri)

    elapsed = time.perf_counter() - t0
    log_event(_logger, logging.INFO, "duration",
              elapsed_s=round(elapsed, 1),
              scenes_processed=processed, scenes_failed=failed)

    return 0 if failed == 0 else 1


def _banner(
    cfg: DictConfig, run_id: str, output_root: str, manifest_uri: str,
) -> None:
    log_event(_logger, logging.INFO, "run_start",
              pipeline="dynamic", run_id=run_id,
              output_root=output_root, manifest_uri=manifest_uri,
              geometry_temporal_mode=GEOMETRY_TEMPORAL_MODE,
              geometry_vintages=GEOMETRY_VINTAGES)
