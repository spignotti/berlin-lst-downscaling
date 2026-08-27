"""WB2c-4 training-data pipeline — per-scene eligibility publication.

Resolves the published training universe via the Stage-1 inventory,
hard-rejects any input that is not Feature Release V3 (root, ledger rows,
and per-scene provenance config hash), computes the strict
``training_eligible@100m`` mask per assessable scene, and publishes the
mask + provenance + completion marker under ``<output_root>/<scene_id>/``
with a create-only visibility gate and per-scene publish lock.

2026 inference scenes are never processed: they are recorded as metadata
rows with split ``inference`` and reason ``inference_deferred``.

Top-level release artifacts (manifest, cells, scaler, release marker) are
assembled by ``training.release``; this module owns the per-scene
contract and the run report.
"""

from __future__ import annotations

import json
import logging

from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.features.paths import (
    feature_mask_cog,
    feature_provenance,
)
from berlin_lst_downscaling.data.io import exists, log_event, read_bytes
from berlin_lst_downscaling.data.qa.inventory import (
    INFERENCE_EXCLUSION_REASON,
    ResolvedScene,
    build_inventory,
)
from berlin_lst_downscaling.data.qa.stage1_raw import analysis_grid_10m
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.training.contracts import (
    EXPECTED_V3_CONFIG_HASH,
    INFERENCE_DEFERRED_REASON,
    NO_ELIGIBLE_CELLS_REASON,
    split_for_year,
    training_policy_hash,
)
from berlin_lst_downscaling.data.training.eligibility import (
    compute_eligibility,
)
from berlin_lst_downscaling.data.training.paths import ledger_path
from berlin_lst_downscaling.data.training.product import (
    publish_eligibility,
)
from berlin_lst_downscaling.data.training.release import (
    V3_FEATURES_ROOT,
    _verify_v3_release,
    publish_release,
)
from berlin_lst_downscaling.data.training.report import (
    SceneTrainingResult,
    TrainingRunReport,
    new_run_id,
    now_iso,
)

_logger = logging.getLogger(__name__)


# ── orchestration ────────────────────────────────────────────────────


def run_training_data(cfg, *, run_id: str | None = None) -> TrainingRunReport:
    """Run the training-data eligibility pipeline for the configured universe."""
    if run_id is None:
        run_id = new_run_id()
    timestamp = now_iso()

    features_root = str(cfg.features_root)
    output_root = str(cfg.output_root)
    scene_ids = [str(s) for s in cfg.get("scene_ids", []) or []]
    bbox = tuple(cfg.get("bbox", None)) if cfg.get("bbox") else None
    expected_count = cfg.get("expected_scene_count")

    log_event(
        _logger,
        logging.INFO,
        "training_start",
        run_id=run_id,
        features_root=features_root,
        output_root=output_root,
    )

    # ── 0. V3 release gate ───────────────────────────────────────────
    policy_hash = training_policy_hash(v3_config_hash=EXPECTED_V3_CONFIG_HASH)
    _verify_v3_release(
        features_root=features_root,
        expected_scene_count=int(expected_count) if expected_count is not None else None,
        scene_ids=scene_ids,
    )

    # ── 1. inventory (same resolver as the QA gates) ─────────────────
    inventory = build_inventory(
        manifest_uri=str(cfg.manifest_uri),
        ard_root=str(cfg.ard_root),
        static_sources_root=str(cfg.static_sources_root),
        static_derived_root=str(cfg.static_derived_root),
        dynamic_root=str(cfg.dynamic_root),
        geometry_mapping_uri=str(cfg.geometry_mapping_uri),
        scene_ids=scene_ids,
    )
    if not inventory.ok:
        raise RuntimeError(f"Training inventory failed: {inventory.errors}")
    if expected_count is not None and not scene_ids and inventory.assessed != int(expected_count):
        raise RuntimeError(
            f"expected {expected_count} assessable scenes, inventory found {inventory.assessed}"
        )

    # ── 2. grids ─────────────────────────────────────────────────────
    grid_10m: GeoBox = analysis_grid_10m(bbox)
    grid_100m: GeoBox = grid_10m.zoom_out(10)
    log_event(
        _logger,
        logging.INFO,
        "grid",
        shape_10m=[grid_10m.shape.x, grid_10m.shape.y],
        shape_100m=[grid_100m.shape.x, grid_100m.shape.y],
        origin_100m=[grid_100m.transform.xoff, grid_100m.transform.yoff],
    )

    # ── 3. per-scene eligibility + publication ───────────────────────
    led = SecondaryLedger.open(ledger_path(output_root))
    results: list[SceneTrainingResult] = []
    processed = failed = 0
    agg = {"eligible_cells": 0, "target_valid_cells": 0}
    exclusion_reasons: dict[str, int] = {}

    for scene in inventory.scenes:
        if not scene.assessable:
            split = split_for_year(scene.year)
            reason = (
                INFERENCE_DEFERRED_REASON
                if scene.year >= 2026
                else (scene.exclusion_reason or INFERENCE_EXCLUSION_REASON)
            )
            results.append(
                SceneTrainingResult(
                    scene_id=scene.scene_id,
                    year=scene.year,
                    s2_scene_id=scene.s2_scene_id,
                    split=split,
                    status="excluded",
                    sensor=_sensor(scene.scene_id),
                    exclusion_reason=reason,
                )
            )
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
            continue

        result = _process_scene(
            scene=scene,
            grid_10m=grid_10m,
            grid_100m=grid_100m,
            features_root=features_root,
            output_root=output_root,
            policy_hash=policy_hash,
            run_id=run_id,
            led=led,
        )
        results.append(result)
        if result.status == "done":
            processed += 1
            agg["eligible_cells"] += result.eligible_cells or 0
            agg["target_valid_cells"] += result.target_valid_cells or 0
            # A scene with zero eligible cells is excluded from training
            # with the single documented reason (no sparse category).
            if (result.eligible_cells or 0) == 0:
                reason = NO_ELIGIBLE_CELLS_REASON
                result.exclusion_reason = reason
                result.status = "excluded"
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        elif result.status == "failed":
            failed += 1

    report = TrainingRunReport(
        run_id=run_id,
        timestamp=timestamp,
        inputs={
            "manifest_uri": str(cfg.manifest_uri),
            "features_root": features_root,
            "output_root": output_root,
        },
        fingerprints={
            "manifest": inventory.fingerprints["manifest"],
            "v3_config_hash": EXPECTED_V3_CONFIG_HASH,
            "policy_hash": policy_hash,
        },
        grid={
            "crs": str(grid_100m.crs),
            "shape": [grid_100m.shape.y, grid_100m.shape.x],
            "origin": [grid_100m.transform.xoff, grid_100m.transform.yoff],
            "bbox_subset": bbox is not None,
        },
        policy_hash=policy_hash,
        v3_config_hash=EXPECTED_V3_CONFIG_HASH,
        total_pairings=inventory.total_pairings,
        assessed=inventory.assessed,
        processed=processed,
        failed=failed,
        excluded=inventory.excluded,
        exclusion_reasons=exclusion_reasons,
        aggregate=agg,
        scenes=results,
    )

    log_event(
        _logger,
        logging.INFO,
        "training_done",
        run_id=run_id,
        processed=processed,
        failed=failed,
        eligible_cells=agg["eligible_cells"],
    )

    # ── 4. release assembly (manifest, cells, scaler, marker) ─────────
    # Idempotent for the same policy hash; refuses a policy change under
    # an existing release marker (immutable release). Fail closed: an
    # incomplete release (any failed assessable scene) must never publish
    # the release artifacts or its completion marker.
    if report.ok:
        report.release_uris = publish_release(
            report,
            features_root=features_root,
            output_root=output_root,
            grid_10m=grid_10m,
            run_id=run_id,
            policy_hash=policy_hash,
        )
    else:
        log_event(
            _logger,
            logging.ERROR,
            "release_skipped",
            run_id=run_id,
            failed=failed,
            reason="report not ok — release not assembled",
        )
    return report


def _process_scene(
    *,
    scene: ResolvedScene,
    grid_10m: GeoBox,
    grid_100m: GeoBox,
    features_root: str,
    output_root: str,
    policy_hash: str,
    run_id: str,
    led: SecondaryLedger,
) -> SceneTrainingResult:
    """Compute + publish one assessable scene; returns the report row."""
    item_id = f"training_{scene.scene_id}"
    source = "training_eligibility"
    todo = reconcile([(item_id, source, scene.scene_id)], led, policy_hash)

    base_result = SceneTrainingResult(
        scene_id=scene.scene_id,
        year=scene.year,
        s2_scene_id=scene.s2_scene_id,
        split=split_for_year(scene.year),
        status="failed",
        sensor=_sensor(scene.scene_id),
    )

    if not todo:
        # Idempotent skip: reuse the published counts so the run report
        # stays deterministic across re-runs. The published artifact must
        # be genuinely present and readable — a ledger row whose
        # provenance is missing/corrupt must not silently degrade to zero
        # counts (which would wrongly mark the scene no_eligible_cells).
        row = led.get(item_id, source, scene.scene_id)
        if row is None or not row.provenance_uri or not row.completion_uri:
            raise RuntimeError(
                f"scene {scene.scene_id}: ledger done row lacks provenance/completion URI"
            )
        if not exists(row.completion_uri):
            raise RuntimeError(
                f"scene {scene.scene_id}: ledger done row without completion marker "
                f"({row.completion_uri}) — re-finalise"
            )
        if not row.output_uri or not exists(row.output_uri):
            raise RuntimeError(
                f"scene {scene.scene_id}: ledger done row without readable eligibility "
                f"COG ({row.output_uri}) — re-finalise"
            )
        try:
            prov = json.loads(read_bytes(row.provenance_uri))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"scene {scene.scene_id}: published provenance unreadable: {exc}"
            ) from exc
        if prov.get("policy_hash") != policy_hash:
            raise RuntimeError(
                f"scene {scene.scene_id}: published provenance policy_hash "
                f"{prov.get('policy_hash')!r} != {policy_hash!r}"
            )
        if prov.get("scene_id") != scene.scene_id:
            raise RuntimeError(
                f"scene {scene.scene_id}: published provenance scene_id "
                f"{prov.get('scene_id')!r} != {scene.scene_id}"
            )
        counts = prov.get("counts", {})
        if "eligible_cells" not in counts or "target_valid_cells" not in counts:
            raise RuntimeError(f"scene {scene.scene_id}: published provenance lacks counts")
        base_result.status = "done"
        base_result.eligible_cells = int(counts.get("eligible_cells", 0))
        base_result.target_valid_cells = int(counts.get("target_valid_cells", 0))
        return base_result

    led.upsert(
        SecondaryLedgerRow(
            item_id=item_id,
            source=source,
            period_or_vintage=scene.scene_id,
            status="exporting",
            run_id=run_id,
        )
    )

    try:
        mask_uri = feature_mask_cog(features_root, scene.scene_id)
        computed = compute_eligibility(
            scene=scene,
            analysis_10=grid_10m,
            mask_uri=mask_uri,
        )
        if computed.errors:
            raise RuntimeError("; ".join(computed.errors))

        artifacts = publish_eligibility(
            result=computed,
            grid_100m=grid_100m,
            output_root=output_root,
            run_id=run_id,
            policy_hash=policy_hash,
            v3_config_hash=EXPECTED_V3_CONFIG_HASH,
            feature_valid_uri=mask_uri,
            feature_provenance_uri=feature_provenance(features_root, scene.scene_id),
            landsat_cog_uri=scene.landsat_cog,
            landsat_flag_uri=scene.landsat_flag,
            geometry_id=scene.geometry_id,
        )
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source=source,
                period_or_vintage=scene.scene_id,
                status="done",
                run_id=run_id,
                config_hash=policy_hash,
                output_uri=artifacts.cog_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            )
        )
        log_event(
            _logger,
            logging.INFO,
            "eligibility_done",
            scene_id=scene.scene_id,
            eligible_cells=computed.eligible_cells,
            output_uri=artifacts.cog_uri,
        )
        base_result.status = "done"
        base_result.eligible_cells = computed.eligible_cells
        base_result.target_valid_cells = computed.target_valid_cells
        return base_result
    except Exception as exc:  # per-scene failure, never crash the run
        log_event(
            _logger,
            logging.ERROR,
            "eligibility_failed",
            scene_id=scene.scene_id,
            error=str(exc),
        )
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source=source,
                period_or_vintage=scene.scene_id,
                status="failed",
                run_id=run_id,
                last_error=str(exc),
            )
        )
        base_result.error = str(exc)
        return base_result


def _sensor(scene_id: str) -> str:
    """Return the Landsat sensor family from a scene ID (``LC08_...`` -> ``LC08``)."""
    return scene_id.split("_", 1)[0]


__all__ = [
    "SceneTrainingResult",
    "TrainingRunReport",
    "V3_FEATURES_ROOT",
    "publish_release",
    "run_training_data",
]
