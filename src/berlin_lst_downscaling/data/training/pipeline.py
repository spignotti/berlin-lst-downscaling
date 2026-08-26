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
assembled by the P2 stage; this module owns the per-scene contract and the
run report.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pyarrow.parquet as pq
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.features.paths import (
    feature_mask_cog,
    feature_provenance,
)
from berlin_lst_downscaling.data.features.paths import (
    ledger_path as features_ledger_path,
)
from berlin_lst_downscaling.data.io import atomic_write, log_event, read_bytes
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
from berlin_lst_downscaling.data.training.paths import (
    ledger_path,
    qa_report_path,
)
from berlin_lst_downscaling.data.training.product import (
    publish_eligibility,
)

_logger = logging.getLogger(__name__)

# Only Feature Release V3 may be consumed (user-mandated). The canonical
# root is ``gs://berlin-lst-data/features/v3``; smoke configs use the same
# published root (they read real GCS stacks).
V3_FEATURES_ROOT = "gs://berlin-lst-data/features/v3"


# ── report types ─────────────────────────────────────────────────────


@dataclass
class SceneTrainingResult:
    """One per-scene row of the training run report."""

    scene_id: str
    year: int
    s2_scene_id: str
    split: str  # train | validation | test | inference
    status: str  # done | excluded | failed
    sensor: str
    eligible_cells: int | None = None
    target_valid_cells: int | None = None
    exclusion_reason: str | None = None
    error: str | None = None


@dataclass
class TrainingRunReport:
    """Complete training-data run report (serialised to report.json)."""

    run_id: str
    timestamp: str
    inputs: dict
    fingerprints: dict
    grid: dict
    policy_hash: str
    v3_config_hash: str
    total_pairings: int
    assessed: int
    processed: int
    failed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    aggregate: dict
    scenes: list[SceneTrainingResult]

    @property
    def ok(self) -> bool:
        return self.failed == 0


# ── V3 release gate ───────────────────────────────────────────────────


def _verify_v3_release(
    *,
    features_root: str,
    expected_scene_count: int | None,
    scene_ids: list[str],
) -> dict:
    """Hard-reject anything that is not Feature Release V3.

    Checks, in order:

    - ``features_root`` must be the canonical V3 root;
    - the features ledger must hold exactly ``expected_scene_count`` done
      rows, all with the pinned V3 config hash ``d9eb25995b2f4911``;
    - every in-scope scene's published ``provenance.json`` must carry the
      pinned config hash and the canonical 28-channel order.

    Returns the ledger table columns for reuse (``{scene_id: row}``).
    """
    if features_root.rstrip("/") != V3_FEATURES_ROOT:
        raise RuntimeError(
            f"features_root {features_root!r} is not Feature Release V3 "
            f"({V3_FEATURES_ROOT!r}) — V1/V2 must never be consumed"
        )

    led_uri = features_ledger_path(features_root)
    table = pq.read_table(io.BytesIO(read_bytes(led_uri)))
    cols = table.to_pydict()
    statuses = {str(s) for s in cols["status"]}
    if statuses != {"done"}:
        raise RuntimeError(f"features ledger {led_uri}: unexpected statuses {sorted(statuses)}")
    # Ledger row-count invariant applies to full runs only — a scene_ids-
    # restricted (smoke) run reads the same published ledger with 324 rows.
    full_run = not scene_ids
    if expected_scene_count is not None and full_run and table.num_rows != expected_scene_count:
        raise RuntimeError(
            f"features ledger {led_uri}: {table.num_rows} rows, expected {expected_scene_count}"
        )
    hashes = {str(h) for h in cols.get("config_hash", []) if h is not None}
    if hashes != {EXPECTED_V3_CONFIG_HASH}:
        raise RuntimeError(
            f"features ledger {led_uri}: config hashes {sorted(hashes)}, "
            f"expected {{ {EXPECTED_V3_CONFIG_HASH} }}"
        )

    rows = {}
    for i in range(table.num_rows):
        rows[str(cols["period_or_vintage"][i])] = {name: cols[name][i] for name in cols}

    # Per-scene provenance: pinned config hash + canonical channel order.
    for scene_id in scene_ids:
        prov_uri = feature_provenance(features_root, scene_id)
        prov = json.loads(read_bytes(prov_uri))
        if prov.get("config_hash") != EXPECTED_V3_CONFIG_HASH:
            raise RuntimeError(
                f"{scene_id}: provenance config_hash {prov.get('config_hash')!r}, "
                f"expected {EXPECTED_V3_CONFIG_HASH}"
            )
        if list(prov.get("channel_order", [])) != list(FEATURE_CHANNEL_NAMES):
            raise RuntimeError(
                f"{scene_id}: provenance channel_order does not match the canonical "
                f"28-channel order"
            )

    return rows


# ── orchestration ────────────────────────────────────────────────────


def run_training_data(cfg, *, run_id: str | None = None) -> TrainingRunReport:
    """Run the training-data eligibility pipeline for the configured universe."""
    if run_id is None:
        run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()

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
        # stays deterministic across re-runs.
        row = led.get(item_id, source, scene.scene_id)
        prov = _read_provenance(row.provenance_uri) if row and row.provenance_uri else {}
        counts = prov.get("counts", {})
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


def _read_provenance(uri: str) -> dict:
    try:
        return json.loads(read_bytes(uri))
    except Exception:  # best-effort skip metadata
        return {}


# ── report serialisation ─────────────────────────────────────────────


def write_report(report: TrainingRunReport, output_root: str) -> str:
    """Persist the run report JSON under ``qa/training/<run_id>/report.json``."""
    uri = qa_report_path(output_root, report.run_id)
    payload = {
        "pipeline": "training-data",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "ok": report.ok,
        "inputs": report.inputs,
        "fingerprints": report.fingerprints,
        "grid": report.grid,
        "policy_hash": report.policy_hash,
        "v3_config_hash": report.v3_config_hash,
        "scenes": {
            "total_pairings": report.total_pairings,
            "assessed": report.assessed,
            "processed": report.processed,
            "failed": report.failed,
            "excluded": report.excluded,
            "exclusion_reasons": report.exclusion_reasons,
        },
        "aggregate": report.aggregate,
        "scene_results": [
            {
                "scene_id": s.scene_id,
                "year": s.year,
                "s2_scene_id": s.s2_scene_id,
                "split": s.split,
                "status": s.status,
                "sensor": s.sensor,
                "eligible_cells": s.eligible_cells,
                "target_valid_cells": s.target_valid_cells,
                "exclusion_reason": s.exclusion_reason,
                "error": s.error,
            }
            for s in report.scenes
        ],
    }
    atomic_write(uri, json.dumps(payload, indent=2, default=str), overwrite=True)
    return uri


__all__ = [
    "SceneTrainingResult",
    "TrainingRunReport",
    "V3_FEATURES_ROOT",
    "run_training_data",
    "write_report",
]
