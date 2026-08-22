"""Feature-stack pipeline — manifest-driven per-anchor orchestration.

For every assessable paired Landsat anchor (2017-2025, role=anchor):

1. Resolve inputs via the Stage-1 inventory (S2 ARD + flag, per-vintage
   morphology, ERA5-Land, shadows) plus the fixed vegetation_dsm
   carry-forward COG from the static derived ledger and the exact Berlin
   AOI mask.
2. Reconcile against the features ledger (config-hash + completion-marker
   gated idempotency).
3. Compose the 28-band stack + feature_valid mask, finalise the five
   artifacts (data COG, mask COG, provenance, STAC, complete), and record
   the ledger row.

Excluded scenes (2026 role=inference) are reported, never processed.
A run report is persisted under ``<root>/qa/features/<run_id>/report.json``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.dynamic.geometry import load_geometry_mapping
from berlin_lst_downscaling.data.features.lod_coverage import (
    rasterize_lod_coverage,
    resolve_lod_coverage_artifacts,
)
from berlin_lst_downscaling.data.features.paths import (
    ledger_path,
    qa_report_path,
    scene_product_dir,
)
from berlin_lst_downscaling.data.features.product import (
    PreparedFeatureProduct,
    finalize_feature_product,
)
from berlin_lst_downscaling.data.features.schema import config_hash_for_features
from berlin_lst_downscaling.data.features.stack import (
    FeatureInputs,
    compose_feature_stack,
    load_aoi_mask_on_grid,
)
from berlin_lst_downscaling.data.io import atomic_write, log_event, read_bytes
from berlin_lst_downscaling.data.qa.inventory import ResolvedScene, build_inventory
from berlin_lst_downscaling.data.qa.stage1_raw import analysis_grid_10m
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow

_logger = logging.getLogger(__name__)

# Static derived ledger path (same layout the Stage-1 inventory reads).
_DERIVED_LEDGER_REL = "_state/static/derived/ledger.parquet"


# ── report types ─────────────────────────────────────────────────────


@dataclass
class SceneFeatureResult:
    """One per-scene row of the run report."""

    scene_id: str
    year: int
    s2_scene_id: str
    geometry_id: str
    status: str  # done | failed | excluded
    config_hash: str | None = None
    exclusion_reason: str | None = None
    coverage: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class FeatureRunReport:
    """Complete features run report (serialised to report.json)."""

    run_id: str
    timestamp: str
    fingerprints: dict
    grid: dict
    inputs: dict
    vegetation_carry_forward: dict
    total_pairings: int
    assessed: int
    processed: int
    failed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    aggregate_coverage: dict
    scenes: list[SceneFeatureResult]

    @property
    def ok(self) -> bool:
        return self.failed == 0


# ── ledger helpers ───────────────────────────────────────────────────


def _read_table(uri: str):
    """Read a Parquet table from a local path or GCS URI."""
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _resolve_vegetation_carry_forward(
    static_derived_root: str,
    geometry_id: str,
) -> str:
    """Return the vegetation_dsm COG URI for *geometry_id* (done ledger row)."""
    ledger_uri = f"{static_derived_root.rstrip('/')}/{_DERIVED_LEDGER_REL}"
    table = _read_table(ledger_uri)
    cols = table.to_pydict()
    for i in range(table.num_rows):
        if (
            str(cols["source"][i]) == "vegetation_dsm"
            and str(cols["period_or_vintage"][i]) == geometry_id
            and str(cols["status"][i]) == "done"
        ):
            uri = cols["output_uri"][i]
            if uri:
                return str(uri)
    raise RuntimeError(
        f"vegetation_dsm not done for carry-forward geometry {geometry_id!r} "
        f"(ledger {ledger_uri})"
    )


# ── orchestration ────────────────────────────────────────────────────


def run_features(cfg, *, run_id: str | None = None) -> FeatureRunReport:
    """Run the feature-stack pipeline for the configured training universe."""
    if run_id is None:
        run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()

    manifest_uri = str(cfg.manifest_uri)
    ard_root = str(cfg.ard_root)
    static_sources_root = str(cfg.static_sources_root)
    static_derived_root = str(cfg.static_derived_root)
    dynamic_root = str(cfg.dynamic_root)
    geometry_mapping_uri = str(cfg.geometry_mapping_uri)
    aoi_uri = str(cfg.aoi_mask_uri)
    output_root = str(cfg.output_root)
    scene_ids = [str(s) for s in cfg.get("scene_ids", []) or []]
    bbox = tuple(cfg.get("bbox", None)) if cfg.get("bbox") else None
    carry_forward_vintage = int(cfg.get("vegetation_carry_forward_vintage", 2024))
    expected_count = cfg.get("expected_scene_count")

    log_event(
        _logger,
        logging.INFO,
        "features_start",
        run_id=run_id,
        manifest=manifest_uri,
        output_root=output_root,
    )

    # ── 0. inventory + geometry mapping ──────────────────────────────
    inventory = build_inventory(
        manifest_uri=manifest_uri,
        ard_root=ard_root,
        static_sources_root=static_sources_root,
        static_derived_root=static_derived_root,
        dynamic_root=dynamic_root,
        geometry_mapping_uri=geometry_mapping_uri,
        scene_ids=scene_ids,
    )
    if not inventory.ok:
        raise RuntimeError(f"Feature inventory failed: {inventory.errors}")

    mapping_report = load_geometry_mapping(geometry_mapping_uri)
    if not mapping_report.ok or mapping_report.mapping is None:
        raise RuntimeError(f"Geometry mapping failed: {mapping_report.errors}")
    mapping = mapping_report.mapping

    if carry_forward_vintage not in mapping.vintages:
        raise RuntimeError(
            f"carry-forward vintage {carry_forward_vintage} not in geometry mapping"
        )
    veg_geometry_id = str(mapping.vintages[carry_forward_vintage].get("geometry_id", ""))
    if not veg_geometry_id:
        raise RuntimeError(f"vintage {carry_forward_vintage} has no geometry_id")

    if expected_count is not None and not scene_ids and inventory.assessed != expected_count:
        raise RuntimeError(
            f"expected {expected_count} assessable scenes, inventory found {inventory.assessed}"
        )

    # ── 1. vegetation carry-forward + AOI ────────────────────────────
    veg_dsm_uri = _resolve_vegetation_carry_forward(static_derived_root, veg_geometry_id)
    aoi_fingerprint = sha256_bytes(read_bytes(aoi_uri))[:16]
    vegetation_policy = (
        f"carry_forward_vintage_{carry_forward_vintage}_geometry_{veg_geometry_id}"
    )
    log_event(
        _logger,
        logging.INFO,
        "inputs_resolved",
        vegetation_dsm_uri=veg_dsm_uri,
        vegetation_carry_forward_geometry=veg_geometry_id,
        aoi_fingerprint=aoi_fingerprint,
    )

    # ── 2. grid + AOI mask on grid ───────────────────────────────────
    grid: GeoBox = analysis_grid_10m(bbox)
    aoi = load_aoi_mask_on_grid(aoi_uri, grid)
    log_event(
        _logger,
        logging.INFO,
        "grid",
        shape=[grid.shape.x, grid.shape.y],
        origin=[grid.transform.xoff, grid.transform.yoff],
        aoi_inside_px=int(aoi.sum()),
    )

    # ── 3. LoD source-coverage resolution ─────────────────────────────
    # Immutable evidence (raw archive manifests / LoD provenance) decides
    # which cells are "covered, no building" (zero) vs "true source gap"
    # (NaN) at composition time. Rasterized once per vintage on the
    # analysis grid, then selected per scene by its LoD vintage.
    lod_artifacts = resolve_lod_coverage_artifacts(static_sources_root)
    lod_coverage_fingerprints = {str(v): a.fingerprint for v, a in lod_artifacts.items()}
    lod_cog_fingerprints = {str(v): a.cog_fingerprint for v, a in lod_artifacts.items()}
    lod_coverage_evidence = {
        v: {
            "uris": list(a.uris),
            "fingerprint": a.fingerprint,
            "cog_uri": a.cog_uri,
            "cog_fingerprint": a.cog_fingerprint,
        }
        for v, a in lod_artifacts.items()
    }
    coverage_masks = {
        v: rasterize_lod_coverage(a, grid) for v, a in lod_artifacts.items()
    }

    # ── 4. config hash ───────────────────────────────────────────────
    config_hash = config_hash_for_features(
        manifest_hash=inventory.fingerprints["manifest"],
        geometry_mapping_hash=mapping.content_hash,
        ard_ledger_hash=inventory.fingerprints["ard_ledger"],
        static_sources_ledger_hash=inventory.fingerprints.get("static_sources_ledger", ""),
        static_derived_ledger_hash=inventory.fingerprints["static_derived_ledger"],
        dynamic_ledger_hash=inventory.fingerprints["dynamic_ledger"],
        aoi_fingerprint=aoi_fingerprint,
        vegetation_carry_forward_geometry_id=veg_geometry_id,
        lod_coverage_fingerprints=lod_coverage_fingerprints,
        lod_cog_fingerprints=lod_cog_fingerprints,
    )

    # S2 acquisition datetimes from the manifest (STAC datetime per stack).
    s2_datetimes = _manifest_datetimes(manifest_uri)

    # ── 4. process scenes ────────────────────────────────────────────
    led = SecondaryLedger.open(ledger_path(output_root))
    results: list[SceneFeatureResult] = []
    processed = failed = 0
    agg = {"feature_valid_px": 0, "inside_aoi_px": 0, "outside_aoi_px": 0}

    for scene in inventory.scenes:
        if not scene.assessable:
            reason = scene.exclusion_reason or "excluded"
            results.append(
                SceneFeatureResult(
                    scene_id=scene.scene_id,
                    year=scene.year,
                    s2_scene_id=scene.s2_scene_id,
                    geometry_id=scene.geometry_id,
                    status="excluded",
                    exclusion_reason=reason,
                )
            )
            continue

        result = _process_scene(
            scene=scene,
            grid=grid,
            aoi=aoi,
            config_hash=config_hash,
            veg_dsm_uri=veg_dsm_uri,
            vegetation_policy=vegetation_policy,
            aoi_uri=aoi_uri,
            aoi_fingerprint=aoi_fingerprint,
            acquisition_datetime=s2_datetimes.get(scene.s2_scene_id, ""),
            output_root=output_root,
            led=led,
            run_id=run_id,
            coverage_masks=coverage_masks,
            lod_coverage_evidence=lod_coverage_evidence,
        )
        results.append(result)
        if result.status == "done":
            processed += 1
            agg["feature_valid_px"] += result.coverage.get("feature_valid_px", 0)
            agg["inside_aoi_px"] += result.coverage.get("inside_aoi_px", 0)
            agg["outside_aoi_px"] += result.coverage.get("outside_aoi_px", 0)
        elif result.status == "failed":
            failed += 1

    exclusion_reasons: dict[str, int] = {}
    for r in results:
        if r.status == "excluded" and r.exclusion_reason:
            exclusion_reasons[r.exclusion_reason] = exclusion_reasons.get(
                r.exclusion_reason, 0
            ) + 1

    report = FeatureRunReport(
        run_id=run_id,
        timestamp=timestamp,
        fingerprints={
            **inventory.fingerprints,
            "lod_coverage": lod_coverage_fingerprints,
            "lod_cog": lod_cog_fingerprints,
        },
        grid={
            "crs": str(grid.crs),
            "shape": [grid.shape.x, grid.shape.y],
            "origin": [grid.transform.xoff, grid.transform.yoff],
            "bbox_subset": bbox is not None,
        },
        inputs={
            "manifest_uri": manifest_uri,
            "ard_root": ard_root,
            "static_sources_root": static_sources_root,
            "static_derived_root": static_derived_root,
            "dynamic_root": dynamic_root,
            "geometry_mapping_uri": geometry_mapping_uri,
            "aoi_uri": aoi_uri,
        },
        vegetation_carry_forward={
            "vintage": carry_forward_vintage,
            "geometry_id": veg_geometry_id,
            "vegetation_dsm_uri": veg_dsm_uri,
        },
        total_pairings=inventory.total_pairings,
        assessed=inventory.assessed,
        processed=processed,
        failed=failed,
        excluded=inventory.excluded,
        exclusion_reasons=exclusion_reasons,
        aggregate_coverage=agg,
        scenes=results,
    )

    log_event(
        _logger,
        logging.INFO,
        "features_done",
        run_id=run_id,
        processed=processed,
        failed=failed,
        excluded=inventory.excluded,
        ok=report.ok,
    )
    return report


def _process_scene(
    *,
    scene: ResolvedScene,
    grid: GeoBox,
    aoi,
    config_hash: str,
    veg_dsm_uri: str,
    vegetation_policy: str,
    aoi_uri: str,
    aoi_fingerprint: str,
    acquisition_datetime: str,
    output_root: str,
    led: SecondaryLedger,
    run_id: str,
    coverage_masks: dict[int, np.ndarray],
    lod_coverage_evidence: dict[int, dict],
) -> SceneFeatureResult:
    """Compose + finalise one assessable scene; returns the report row."""
    item_id = f"feature_{scene.scene_id}"
    source = "feature_stack"
    todo = reconcile([(item_id, source, scene.scene_id)], led, config_hash)

    if scene.lod_vintage not in coverage_masks:
        raise RuntimeError(
            f"scene {scene.scene_id}: no LoD coverage for vintage {scene.lod_vintage!r}"
        )
    lod_coverage = coverage_masks[scene.lod_vintage]

    base_result = SceneFeatureResult(
        scene_id=scene.scene_id,
        year=scene.year,
        s2_scene_id=scene.s2_scene_id,
        geometry_id=scene.geometry_id,
        status="failed",
    )

    if not todo:
        # Idempotent skip: reuse the published product's coverage so the
        # run report stays deterministic across re-runs.
        coverage = _existing_coverage(led, item_id, source, scene.scene_id)
        return SceneFeatureResult(
            scene_id=scene.scene_id,
            year=scene.year,
            s2_scene_id=scene.s2_scene_id,
            geometry_id=scene.geometry_id,
            status="done",
            config_hash=config_hash,
            coverage=coverage,
        )

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
        # Each morphology channel maps to (COG URI, band number).
        # Multi-band source COGs: lod2_morphology (bands 1-4), vegetation_height (bands 1-2).
        # Single-band: imperviousness (band 1), svf (band 1).
        lod_uri = scene.static_sources["lod2_morphology"]
        vh_uri = scene.static_sources["vegetation_height"]
        morphology = {
            "building_height_mean": (lod_uri, 1),
            "building_height_std": (lod_uri, 2),
            "building_coverage_ratio": (lod_uri, 3),
            "building_height_max": (lod_uri, 4),
            "vegetation_height_mean": (vh_uri, 1),
            "vegetation_height_max": (vh_uri, 2),
            "imperviousness": (scene.static_sources["imperviousness"], 1),
            "svf": (scene.static_derived["svf"], 1),
        }
        inputs = FeatureInputs(
            s2_cog=scene.s2_cog,
            s2_flag=scene.s2_flag,
            morphology=morphology,
            era5_cog=scene.dynamic["era5_land"],
            shadows={
                "shadow_building": scene.dynamic["shadow_building"],
                "shadow_vegetation": scene.dynamic["shadow_vegetation"],
            },
            lod_coverage=lod_coverage,
        )

        composed = compose_feature_stack(inputs, aoi, grid)
        prepared = PreparedFeatureProduct(
            scene_id=scene.scene_id,
            dataset=composed.dataset,
            mask=composed.mask,
            config_hash=config_hash,
            acquisition_datetime=acquisition_datetime,
            source_metadata={
                "aoi_uri": aoi_uri,
                "aoi_fingerprint": aoi_fingerprint,
                "vegetation_height_policy": vegetation_policy,
                "lod_vintage": scene.lod_vintage,
                "lod_coverage": lod_coverage_evidence,
                "inputs": {
                    "s2_cog": scene.s2_cog,
                    "s2_flag": scene.s2_flag,
                    "era5_cog": scene.dynamic["era5_land"],
                    "morphology": morphology,
                    "shadows": inputs.shadows,
                },
            },
            coverage=composed.coverage,
        )
        product_dir = scene_product_dir(output_root, scene.scene_id)
        artifacts = finalize_feature_product(prepared, grid, product_dir, run_id)
        led.upsert(
            SecondaryLedgerRow(
                item_id=item_id,
                source=source,
                period_or_vintage=scene.scene_id,
                status="done",
                run_id=run_id,
                config_hash=config_hash,
                output_uri=artifacts.cog_uri,
                stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            )
        )
        log_event(
            _logger,
            logging.INFO,
            "feature_done",
            scene_id=scene.scene_id,
            output_uri=artifacts.cog_uri,
            feature_valid_px=composed.coverage["feature_valid_px"],
        )
        return SceneFeatureResult(
            scene_id=scene.scene_id,
            year=scene.year,
            s2_scene_id=scene.s2_scene_id,
            geometry_id=scene.geometry_id,
            status="done",
            config_hash=config_hash,
            coverage=composed.coverage,
        )
    except Exception as exc:  # per-scene failure, never crash the run
        log_event(
            _logger,
            logging.ERROR,
            "feature_failed",
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


def _existing_coverage(
    led: SecondaryLedger,
    item_id: str,
    source: str,
    scene_id: str,
) -> dict:
    """Return the coverage dict of an already-published stack (or empty)."""
    row = led.get(item_id, source, scene_id)
    if row is None or not row.provenance_uri:
        return {}
    try:
        import json

        prov = json.loads(read_bytes(row.provenance_uri))
        return dict(prov.get("coverage", {}))
    except Exception:  # best-effort skip metadata
        return {}


def _manifest_datetimes(manifest_uri: str) -> dict[str, str]:
    """Return ``{scene_id: acquisition_datetime RFC 3339}`` from the manifest."""
    table = _read_table(manifest_uri)
    cols = table.to_pydict()
    out: dict[str, str] = {}
    dt_col = cols.get("acquisition_datetime")
    if dt_col is None:
        return out
    ids = cols.get("scene_id", [])
    for i in range(table.num_rows):
        value = dt_col[i]
        if value is not None:
            # pa.timestamp values carry a datetime; isoformat() yields the
            # RFC 3339 'T' separator STAC 1.0.0 requires (str() uses a space).
            out[str(ids[i])] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return out


def write_report(report: FeatureRunReport, output_root: str) -> str:
    """Persist the run report JSON under ``qa/features/<run_id>/report.json``."""
    uri = qa_report_path(output_root, report.run_id)
    payload = {
        "pipeline": "features",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "ok": report.ok,
        "grid": report.grid,
        "inputs": report.inputs,
        "fingerprints": report.fingerprints,
        "vegetation_carry_forward": report.vegetation_carry_forward,
        "scenes": {
            "total_pairings": report.total_pairings,
            "assessed": report.assessed,
            "processed": report.processed,
            "failed": report.failed,
            "excluded": report.excluded,
            "exclusion_reasons": report.exclusion_reasons,
        },
        "aggregate_coverage": report.aggregate_coverage,
        "scene_results": [
            {
                "scene_id": s.scene_id,
                "year": s.year,
                "s2_scene_id": s.s2_scene_id,
                "geometry_id": s.geometry_id,
                "status": s.status,
                "config_hash": s.config_hash,
                "exclusion_reason": s.exclusion_reason,
                "coverage": s.coverage,
                "error": s.error,
            }
            for s in report.scenes
        ],
    }
    atomic_write(uri, _json_dumps(payload), overwrite=True, if_generation_match=0)
    return uri


def _json_dumps(data) -> str:
    import json

    return json.dumps(data, indent=2, default=str)


__all__ = [
    "FeatureRunReport",
    "SceneFeatureResult",
    "run_features",
    "write_report",
]
