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

import pyarrow.parquet as pq
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.features.paths import (
    feature_mask_cog,
    feature_provenance,
)
from berlin_lst_downscaling.data.features.paths import (
    ledger_path as features_ledger_path,
)
from berlin_lst_downscaling.data.io import atomic_write, exists, log_event, read_bytes
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
from berlin_lst_downscaling.data.training.index import (
    CELLS_FIELDNAMES,
    MANIFEST_FIELDNAMES,
    build_cells_rows,
    build_manifest_rows,
)
from berlin_lst_downscaling.data.training.paths import (
    cells_csv,
    cells_parquet,
    eligibility_cog,
    eligibility_completion,
    ledger_path,
    manifest_csv,
    manifest_parquet,
    release_completion,
    scaler_json,
)
from berlin_lst_downscaling.data.training.product import (
    publish_eligibility,
)
from berlin_lst_downscaling.data.training.report import (
    SceneTrainingResult,
    TrainingRunReport,
    new_run_id,
    now_iso,
)
from berlin_lst_downscaling.data.training.scaler import fit_scaler

_logger = logging.getLogger(__name__)

# Only Feature Release V3 may be consumed (user-mandated). The canonical
# root is ``gs://berlin-lst-data/features/v3``; smoke configs use the same
# published root (they read real GCS stacks).
V3_FEATURES_ROOT = "gs://berlin-lst-data/features/v3"


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
    - every features-ledger row must be ``done`` and carry the pinned V3
      config hash ``d9eb25995b2f4911`` (individual row check — a missing
      hash on any row is a hard failure, never filtered out);
    - on full runs the ledger must hold exactly ``expected_scene_count``
      rows;
    - the per-scene ``provenance.json`` of every assessable scene (all
      324 on full runs, the configured subset on smoke) must carry the
      pinned config hash and the canonical 28-channel order.

    Returns the ledger rows keyed by scene ID for reuse.
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

    rows = {}
    for i in range(table.num_rows):
        period = str(cols["period_or_vintage"][i])
        config_hash = cols["config_hash"][i]
        if config_hash is None or str(config_hash) != EXPECTED_V3_CONFIG_HASH:
            raise RuntimeError(
                f"features ledger {led_uri}: row {period} config_hash "
                f"{config_hash!r}, expected {EXPECTED_V3_CONFIG_HASH}"
            )
        rows[period] = {name: cols[name][i] for name in cols}

    # Per-scene provenance: pinned config hash + canonical channel order —
    # full runs verify every ledger row; smoke runs verify the configured
    # subset (the only scenes it processes).
    targets = scene_ids if scene_ids else sorted(rows)
    for scene_id in targets:
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


# ── release assembly ──────────────────────────────────────────────────


def publish_release(
    report: TrainingRunReport,
    *,
    features_root: str,
    output_root: str,
    grid_10m: GeoBox,
    run_id: str,
    policy_hash: str,
) -> dict[str, str]:
    """Publish manifest, cells, scaler, and the release completion marker.

    Refuses to overwrite once the top-level completion marker exists
    (the release is immutable). If the marker exists **and** carries the
    same policy hash, the release is already complete and this is an
    idempotent no-op (smoke re-runs). A different policy hash is a hard
    error — a policy change requires a new release root.
    """
    completion_uri = release_completion(output_root)
    if exists(completion_uri):
        marker = json.loads(read_bytes(completion_uri))
        if marker.get("policy_hash") == policy_hash:
            # Idempotent re-run of a complete release: still verify the
            # published artifacts so the run report carries readback evidence.
            report.readback = _readback_release(
                report=report,
                output_root=output_root,
                policy_hash=policy_hash,
                marker_expected=True,
            )
            return {"complete": completion_uri}
        raise RuntimeError(
            f"training release already published under a different policy hash "
            f"({marker.get('policy_hash')!r} != {policy_hash!r}) — immutable"
        )

    # Top-level publish lock (atomic create-only): exactly one publisher
    # may assemble the release. Held through the completion-marker write,
    # released in ``finally``. A stale lock after a hard kill is an
    # explicit operator state (inspect, then delete).
    lock_uri = f"{output_root.rstrip('/')}/.release.lock"
    if lock_uri.startswith("gs://"):
        try:
            atomic_write(
                lock_uri,
                json.dumps({"run_id": run_id, "policy_hash": policy_hash}, indent=2),
                overwrite=False,
                if_generation_match=0,
            )
        except FileExistsError:
            raise RuntimeError(
                f"training release is being published by another run (lock {lock_uri})"
            ) from None
    else:
        import os

        lock_path = os.path.expanduser(lock_uri)
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"run_id": run_id, "policy_hash": policy_hash}, indent=2))
        except FileExistsError:
            raise RuntimeError(
                f"training release is being published by another run (lock {lock_uri})"
            ) from None

    try:
        # Write the release artifacts (manifest, cells, scaler) while
        # holding the top-level publish lock — the completion marker is
        # NOT written here.
        uris = _publish_release_locked(
            report=report,
            features_root=features_root,
            output_root=output_root,
            grid_10m=grid_10m,
            run_id=run_id,
            policy_hash=policy_hash,
        )
    finally:
        _delete_uri(lock_uri)

    # Publisher-side readback: verify every per-scene eligibility COG and
    # every top-level artifact is actually published and contract-conform
    # BEFORE the release completion marker is written. A release whose
    # readback fails is never marked complete (it stays invisible).
    report.readback = _readback_release(
        report=report,
        output_root=output_root,
        policy_hash=policy_hash,
        marker_expected=False,
    )
    if not report.readback.get("ok"):
        raise RuntimeError(
            f"release readback failed — completion marker NOT written: "
            f"{report.readback.get('errors')}"
        )

    # Release completion marker written last (create-only): the release
    # becomes visible only when every artifact is in place AND readback
    # passed.
    marker = {"published_at": now_iso(), "run_id": run_id, "policy_hash": policy_hash}
    atomic_write(
        completion_uri,
        json.dumps(marker, indent=2),
        overwrite=False,
        if_generation_match=0,
    )

    # Post-write marker verification: the marker must read back with the
    # policy hash before the release is reported as complete.
    report.readback = _readback_release(
        report=report,
        output_root=output_root,
        policy_hash=policy_hash,
        marker_expected=True,
    )
    if not report.readback.get("ok"):
        raise RuntimeError(f"release marker verification failed: {report.readback.get('errors')}")
    uris["complete"] = completion_uri
    return uris


def _publish_release_locked(
    *,
    report: TrainingRunReport,
    features_root: str,
    output_root: str,
    grid_10m: GeoBox,
    run_id: str,
    policy_hash: str,
) -> dict[str, str]:
    """Write the release artifacts while holding the top-level publish lock.

    The completion marker is deliberately NOT written here — the caller
    runs the publisher readback first and only then commits the marker.
    """
    uris: dict[str, str] = {}

    _validate_split_leakage(report.scenes)

    manifest_rows = build_manifest_rows(
        report.scenes, features_root=features_root, output_root=output_root
    )
    cells_rows = build_cells_rows(report.scenes, output_root=output_root)
    scaler = fit_scaler(
        report.scenes,
        features_root=features_root,
        output_root=output_root,
        grid_10m=grid_10m,
    )
    scaler["policy_hash"] = policy_hash
    scaler["v3_config_hash"] = EXPECTED_V3_CONFIG_HASH
    scaler["split_hash"] = _split_hash(report.scenes)
    scaler["training_years"] = sorted({s.year for s in report.scenes if s.split == "train"})
    scaler["transform_policy"] = (
        "zscore: continuous channels; log1p+zscore: tp_0_24h/tp_24_48h/tp_48_72h; "
        "identity: shadow_building/shadow_vegetation"
    )

    _write_table(
        manifest_rows,
        MANIFEST_FIELDNAMES,
        manifest_parquet(output_root),
        manifest_csv(output_root),
    )
    uris["manifest_parquet"] = manifest_parquet(output_root)
    uris["manifest_csv"] = manifest_csv(output_root)

    _write_table(
        cells_rows,
        CELLS_FIELDNAMES,
        cells_parquet(output_root),
        cells_csv(output_root),
    )
    uris["cells_parquet"] = cells_parquet(output_root)
    uris["cells_csv"] = cells_csv(output_root)

    scaler_uri = scaler_json(output_root)
    atomic_write(scaler_uri, json.dumps(scaler, indent=2), overwrite=True)
    uris["scaler"] = scaler_uri

    return uris


def _delete_uri(uri: str) -> None:
    """Delete one object best-effort (publish-lock cleanup)."""
    if uri.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

        bucket_name, key = _parse_gs_uri(uri)
        try:
            _gcs_client().bucket(bucket_name).blob(key).delete()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            _logger.warning("release-lock cleanup failed for %s: %s", uri, exc)
    else:
        import os

        try:
            os.remove(os.path.expanduser(uri))
        except OSError as exc:
            _logger.warning("release-lock cleanup failed for %s: %s", uri, exc)


def _readback_release(
    *,
    report: TrainingRunReport,
    output_root: str,
    policy_hash: str,
    marker_expected: bool,
) -> dict:
    """Verify the published release by re-reading every artifact.

    Publisher-side readback (independent re-derivation is the P3
    validator's job): every published scene's eligibility COG must exist
    with the canonical 100 m grid contract and 0/1 values; the per-scene
    completion markers must exist; the top-level manifest/cells/scaler
    must parse. When ``marker_expected`` (idempotent re-run or post-write
    verification) the release completion marker must carry the policy
    hash. Returns a ``{ok, artifacts, scenes, errors}`` evidence dict.
    """
    import rasterio

    errors: list[str] = []
    artifacts: dict[str, bool] = {}
    scene_checks: dict[str, bool] = {}

    # Per-scene COG + marker readback (published scenes only).
    for s in report.scenes:
        if s.status != "done":
            continue
        cog_uri = eligibility_cog(output_root, s.scene_id)
        try:
            with rasterio.open(cog_uri) as src:
                ok_grid = (
                    str(src.crs) == "EPSG:25833" and src.count == 1 and src.dtypes[0] == "uint8"
                )
                vals = set(int(v) for v in src.read(1).flatten().tolist())
                ok_vals = vals.issubset({0, 1})
            ok_marker = exists(eligibility_completion(output_root, s.scene_id))
            ok = ok_grid and ok_vals and ok_marker
            scene_checks[s.scene_id] = ok
            artifacts[cog_uri] = ok
            if not ok:
                errors.append(
                    f"{s.scene_id}: eligibility readback failed "
                    f"(grid {ok_grid}, values {ok_vals}, marker {ok_marker})"
                )
        except Exception as exc:  # noqa: BLE001 — readback probe
            scene_checks[s.scene_id] = False
            artifacts[cog_uri] = False
            errors.append(f"{s.scene_id}: eligibility readback error: {exc}")

    # Top-level artifacts.
    for label, uri in (
        ("manifest_parquet", manifest_parquet(output_root)),
        ("manifest_csv", manifest_csv(output_root)),
        ("cells_parquet", cells_parquet(output_root)),
        ("cells_csv", cells_csv(output_root)),
        ("scaler", scaler_json(output_root)),
    ):
        exists_ok = exists(uri)
        artifacts[uri] = exists_ok
        if not exists_ok:
            errors.append(f"{label}: missing {uri}")

    # Content parse checks.
    try:
        import io

        import pyarrow.parquet as pq

        pq.read_table(io.BytesIO(read_bytes(manifest_parquet(output_root))))
        pq.read_table(io.BytesIO(read_bytes(cells_parquet(output_root))))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manifest/cells parquet unreadable: {exc}")
    try:
        import json as _json

        scaler = _json.loads(read_bytes(scaler_json(output_root)))
        if scaler.get("policy_hash") != policy_hash:
            errors.append("scaler.json policy_hash mismatch")
        if len(scaler.get("channels", [])) != len(FEATURE_CHANNEL_NAMES):
            errors.append("scaler.json channel count mismatch")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"scaler.json unreadable: {exc}")

    # Release completion marker carries the policy hash — checked only when
    # the marker is expected to exist (idempotent re-run / post-write).
    if marker_expected:
        try:
            marker = json.loads(read_bytes(release_completion(output_root)))
            if marker.get("policy_hash") != policy_hash:
                errors.append("release complete.json policy_hash mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"release complete.json unreadable: {exc}")

    return {
        "ok": not errors,
        "artifacts": artifacts,
        "scenes": scene_checks,
        "errors": errors,
    }


def _validate_split_leakage(results: list[SceneTrainingResult]) -> None:
    """Fail if any s2_scene_id occurs in more than one non-inference split.

    The temporal contract requires every Landsat anchor sharing the same
    Sentinel-2 scene to stay in exactly one split (user-mandated). 2026
    inference scenes are excluded from this invariant (they are not
    assigned to a training split).
    """
    split_of: dict[str, str] = {}
    scene_of: dict[str, str] = {}
    for s in results:
        if s.split == "inference":
            continue
        prior = split_of.get(s.s2_scene_id)
        if prior is not None and prior != s.split:
            raise RuntimeError(
                f"s2_scene_id {s.s2_scene_id!r} spans splits {prior!r} and "
                f"{s.split!r} (scenes {scene_of[s.s2_scene_id]!r}, {s.scene_id!r}) "
                f"— temporal leakage"
            )
        split_of.setdefault(s.s2_scene_id, s.split)
        scene_of.setdefault(s.s2_scene_id, s.scene_id)


def _split_hash(results: list[SceneTrainingResult]) -> str:
    """Return a stable hash over the scene -> split assignment."""
    mapping = {s.scene_id: s.split for s in results}
    payload = json.dumps(mapping, sort_keys=True)
    return sha256_bytes(payload.encode())[:16]


def _write_table(
    rows: list[dict],
    fieldnames: list[str],
    parquet_uri: str,
    csv_uri: str,
) -> None:
    """Write a row list as Parquet + CSV via atomic writes."""
    import csv as _csv
    import io as _io

    import pyarrow as pa
    import pyarrow.parquet as _pq

    table = pa.Table.from_pylist(rows)
    buf = pa.BufferOutputStream()
    _pq.write_table(table, buf)
    atomic_write(parquet_uri, buf.getvalue().to_pybytes(), overwrite=True)

    csv_buf = _io.StringIO()
    writer = _csv.DictWriter(csv_buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(csv_uri, csv_buf.getvalue().encode("utf-8"), overwrite=True)


__all__ = [
    "SceneTrainingResult",
    "TrainingRunReport",
    "V3_FEATURES_ROOT",
    "publish_release",
    "run_training_data",
]
