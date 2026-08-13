"""Dynamic pipeline — per-scene ERA5-Land and shadow product generation.

Orchestrates the full lifecycle for each Landsat anchor scene:
1. Load and validate geometry_mapping.json
2. Validate manifest and resolve per-scene geometry
3. For each scene: prepare ERA5 meteorology COG (8 bands)
4. For each scene: prepare building + vegetation shadow COGs
5. Publish through shared finalizer (COG + STAC + provenance + complete)
6. Produce dynamic QA report with coverage and vintage distribution
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
from berlin_lst_downscaling.data.dynamic.geometry import (
    load_geometry_mapping,
    resolve_scene_geometry,
)
from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors
from berlin_lst_downscaling.data.dynamic.paths import ledger_path, scene_product_dir
from berlin_lst_downscaling.data.dynamic.schema import (
    config_hash_for_dynamic,
    config_hash_for_era5,
    config_hash_for_shadow_vegetation,
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
    geometry_mapping_uri = str(cfg.geometry_mapping_uri)
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
        log_event(_logger, logging.ERROR, "manifest_load_failed", errors=manifest_report.errors)
        return 1

    # Validate expected scene count (skip when scene_ids override)
    if (
        expected_count is not None
        and not scene_ids
        and len(manifest_report.scenes) != expected_count
    ):
        log_event(
            _logger,
            logging.ERROR,
            "scene_count_mismatch",
            expected=expected_count,
            actual=len(manifest_report.scenes),
        )
        return 1

    log_event(
        _logger,
        logging.INFO,
        "manifest_loaded",
        n_scenes=len(manifest_report.scenes),
        total_rows=manifest_report.total_rows,
        manifest_hash=manifest_report.manifest_hash,
        dataset_role=dataset_role,
    )

    # ── 1. load geometry mapping ─────────────────────────────────────
    mapping_report = load_geometry_mapping(geometry_mapping_uri)
    if not mapping_report.ok or mapping_report.mapping is None:
        log_event(
            _logger,
            logging.ERROR,
            "geometry_mapping_failed",
            errors=mapping_report.errors,
        )
        return 1
    mapping = mapping_report.mapping

    # Preflight: validate all scene years are covered
    scene_years = {s.year for s in manifest_report.scenes}
    for sy in scene_years:
        try:
            resolve_scene_geometry(sy, mapping)
        except ValueError as exc:
            log_event(_logger, logging.ERROR, "geometry_resolution_failed", error=str(exc))
            return 1

    log_event(
        _logger,
        logging.INFO,
        "geometry_mapping_validated",
        uri=geometry_mapping_uri,
        hash=mapping.content_hash,
        version=mapping.version,
        n_years=len(mapping.year_to_vintage),
    )

    # ── 2. config hashes ─────────────────────────────────────────────
    era5_hash = config_hash_for_era5(
        manifest_report.manifest_hash,
        mapping.content_hash,
        output_root,
    )
    # Building shadows keep the legacy dynamic fingerprint; vegetation
    # shadows additionally track the finalised vegetation horizon's hash
    # so a corrected horizon invalidates only the vegetation shadows.
    shadow_hash_building = config_hash_for_dynamic(
        manifest_report.manifest_hash,
        mapping.content_hash,
        output_root,
    )
    shadow_hash_vegetation = config_hash_for_shadow_vegetation(
        manifest_report.manifest_hash,
        mapping.content_hash,
        output_root,
        mapping.vegetation_horizon_hash,
    )

    grid = canon_grid_10m()
    led = SecondaryLedger.open(ledger_path(output_root))
    failed = 0
    processed = 0

    # ── 3. process scenes ────────────────────────────────────────────
    # Group scenes by acquisition month for bounded ERA5 local cache
    scenes_by_month: dict[tuple[int, int], list] = defaultdict(list)
    for scene in manifest_report.scenes:
        key = (scene.acquisition_datetime.year, scene.acquisition_datetime.month)
        scenes_by_month[key].append(scene)

    for (ym_year, ym_month), month_scenes in sorted(scenes_by_month.items()):
        log_event(
            _logger,
            logging.INFO,
            "month_group_start",
            year=ym_year,
            month=ym_month,
            n_scenes=len(month_scenes),
        )

        with tempfile.TemporaryDirectory(prefix=f"era5_{ym_year}{ym_month:02d}_") as tmp_dir:
            local_dir = Path(tmp_dir)

            for scene in month_scenes:
                log_event(
                    _logger,
                    logging.INFO,
                    "scene_start",
                    scene_id=scene.scene_id,
                    year=scene.year,
                    doy=scene.day_of_year,
                    dt=scene.acquisition_datetime.isoformat(),
                )

                # Resolve per-scene geometry
                scene_geom = resolve_scene_geometry(scene.year, mapping)

                # ── 3a. ERA5 meteorology ─────────────────────────────
                era5_source = "era5_land"
                era5_item_id = f"era5_land_{scene.scene_id}"
                era5_todo = reconcile([(era5_item_id, era5_source, scene.scene_id)], led, era5_hash)

                if era5_todo:
                    led.upsert(
                        SecondaryLedgerRow(
                            item_id=era5_item_id,
                            source=era5_source,
                            period_or_vintage=scene.scene_id,
                            status="exporting",
                            run_id=run_id,
                            role=dataset_role,
                        )
                    )
                    try:
                        from berlin_lst_downscaling.data.dynamic.era5 import prepare_era5_scene

                        prepared = prepare_era5_scene(
                            scene.scene_id,
                            scene.acquisition_datetime,
                            output_root,
                            run_id,
                            grid=grid,
                            local_dir=local_dir,
                        )
                        prod_dir = scene_product_dir(output_root, era5_source, scene.scene_id)
                        artifacts = finalize_secondary_product(
                            prepared, grid, prod_dir, run_id
                        )
                        led.upsert(
                            SecondaryLedgerRow(
                                item_id=era5_item_id,
                                source=era5_source,
                                period_or_vintage=scene.scene_id,
                                status="done",
                                run_id=run_id,
                                config_hash=era5_hash,
                                output_uri=artifacts.cog_uri,
                                stac_uri=artifacts.stac_uri,
                                provenance_uri=artifacts.provenance_uri,
                                completion_uri=artifacts.completion_uri,
                                role=dataset_role,
                            )
                        )
                        processed += 1
                        log_event(
                            _logger,
                            logging.INFO,
                            "era5_done",
                            scene_id=scene.scene_id,
                            output_uri=artifacts.cog_uri,
                        )
                    except Exception as exc:
                        log_event(
                            _logger,
                            logging.ERROR,
                            "era5_failed",
                            scene_id=scene.scene_id,
                            error=str(exc),
                        )
                        led.upsert(
                            SecondaryLedgerRow(
                                item_id=era5_item_id,
                                source=era5_source,
                                period_or_vintage=scene.scene_id,
                                status="failed",
                                run_id=run_id,
                                last_error=str(exc),
                                role=dataset_role,
                            )
                        )
                        failed += 1
                else:
                    log_event(_logger, logging.INFO, "era5_skipped", scene_id=scene.scene_id)

                # ── 3b. Shadow masks (building + vegetation) ─────────
                # Circuit breaker: skip shadows if ERA5 failed for this scene
                era5_failed = led.get(
                    era5_item_id, era5_source, scene.scene_id
                )
                era5_is_failed = era5_failed is not None and era5_failed.status == "failed"

                if era5_todo and era5_is_failed:
                    log_event(
                        _logger,
                        logging.WARNING,
                        "shadow_skipped_era5_failed",
                        scene_id=scene.scene_id,
                    )
                elif scene.solar_azimuth is None or scene.solar_elevation is None:
                    log_event(
                        _logger, logging.WARNING, "shadow_skipped_no_solar", scene_id=scene.scene_id
                    )
                else:
                    azimuth = scene.solar_azimuth
                    elevation = scene.solar_elevation

                    for component in ("building", "vegetation"):
                        shadow_source = f"shadow_{component}"
                        shadow_item_id = f"shadow_{component}_{scene.scene_id}"

                        # Per-component config hash: depends on which horizon was used
                        if component == "building":
                            horizon_uri = scene_geom.building_horizon_uri
                            horizon_geom_id = scene_geom.building_geometry_id
                            comp_shadow_hash = shadow_hash_building
                            comp_horizon_hash = ""
                        else:
                            horizon_uri = scene_geom.vegetation_horizon_uri
                            horizon_geom_id = "dgm1-2021__lod2-2024__vh-2020"
                            comp_shadow_hash = shadow_hash_vegetation
                            comp_horizon_hash = scene_geom.vegetation_horizon_hash

                        shadow_todo = reconcile(
                            [(shadow_item_id, shadow_source, scene.scene_id)],
                            led,
                            comp_shadow_hash,
                        )

                        if shadow_todo:
                            led.upsert(
                                SecondaryLedgerRow(
                                    item_id=shadow_item_id,
                                    source=shadow_source,
                                    period_or_vintage=scene.scene_id,
                                    status="exporting",
                                    run_id=run_id,
                                    role=dataset_role,
                                )
                            )
                            try:
                                from berlin_lst_downscaling.data.dynamic.shadows import (
                                    prepare_shadow,
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
                                    geometry_id=horizon_geom_id,
                                    config_hash=comp_shadow_hash,
                                    horizon_config_hash=comp_horizon_hash,
                                    acquisition_datetime=scene.acquisition_datetime,
                                )

                                prod_dir = scene_product_dir(
                                    output_root, shadow_source, scene.scene_id
                                )
                                artifacts = finalize_secondary_product(
                                    prepared,
                                    grid,
                                    prod_dir,
                                    run_id,
                                )
                                led.upsert(
                                    SecondaryLedgerRow(
                                        item_id=shadow_item_id,
                                        source=shadow_source,
                                        period_or_vintage=scene.scene_id,
                                        status="done",
                                        run_id=run_id,
                                        config_hash=comp_shadow_hash,
                                        output_uri=artifacts.cog_uri,
                                        stac_uri=artifacts.stac_uri,
                                        provenance_uri=artifacts.provenance_uri,
                                        completion_uri=artifacts.completion_uri,
                                        role=dataset_role,
                                    )
                                )
                                processed += 1
                                log_event(
                                    _logger,
                                    logging.INFO,
                                    "shadow_done",
                                    scene_id=scene.scene_id,
                                    component=component,
                                    output_uri=artifacts.cog_uri,
                                )
                            except Exception as exc:
                                log_event(
                                    _logger,
                                    logging.ERROR,
                                    "shadow_failed",
                                    scene_id=scene.scene_id,
                                    component=component,
                                    error=str(exc),
                                )
                                led.upsert(
                                    SecondaryLedgerRow(
                                        item_id=shadow_item_id,
                                        source=shadow_source,
                                        period_or_vintage=scene.scene_id,
                                        status="failed",
                                        run_id=run_id,
                                        last_error=str(exc),
                                        role=dataset_role,
                                    )
                                )
                                failed += 1

        log_event(_logger, logging.INFO, "month_group_done", year=ym_year, month=ym_month)

    import gc
    gc.collect()

    # ── 4. final report ──────────────────────────────────────────────
    from berlin_lst_downscaling.data.dynamic.reports import (
        dynamic_qa_report,
        format_dynamic_report,
        persist_dynamic_report,
    )

    report = dynamic_qa_report(
        led,
        run_id,
        manifest_hash=manifest_report.manifest_hash,
        geometry_mapping=mapping,
    )
    log_event(_logger, logging.INFO, "qa_report", report=format_dynamic_report(report))
    report_uri = persist_dynamic_report(report, output_root)
    log_event(_logger, logging.INFO, "qa_report_path", path=report_uri)

    elapsed = time.perf_counter() - t0
    log_event(
        _logger,
        logging.INFO,
        "duration",
        elapsed_s=round(elapsed, 1),
        scenes_processed=processed,
        scenes_failed=failed,
    )

    return 0 if failed == 0 else 1

def _banner(
    cfg: DictConfig,
    run_id: str,
    output_root: str,
    manifest_uri: str,
) -> None:
    log_event(
        _logger,
        logging.INFO,
        "run_start",
        pipeline="dynamic",
        run_id=run_id,
        output_root=output_root,
        manifest_uri=manifest_uri,
        geometry_mapping_uri=str(cfg.get("geometry_mapping_uri", "")),
    )
