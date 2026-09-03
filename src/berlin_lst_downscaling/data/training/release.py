"""WB2c-4 training release — input gate, assembly, and readback.

Owns the Feature-Release-V3 input gate, the top-level release artifacts
(manifest, cells, scaler, release completion marker), the split-leakage
check, and the publisher-side readback evidence. Per-scene eligibility
computation and publication live in ``training.pipeline``.
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
    feature_provenance,
)
from berlin_lst_downscaling.data.features.paths import (
    ledger_path as features_ledger_path,
)
from berlin_lst_downscaling.data.io import atomic_write, exists, publish_lock, read_bytes
from berlin_lst_downscaling.data.training.contracts import EXPECTED_V3_CONFIG_HASH
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
    manifest_csv,
    manifest_parquet,
    release_completion,
    scaler_json,
)
from berlin_lst_downscaling.data.training.report import (
    SceneTrainingResult,
    TrainingRunReport,
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
    lock_payload = {"run_id": run_id, "policy_hash": policy_hash}

    try:
        with publish_lock(lock_uri, lock_payload):
            # Recheck after acquiring lock: another publisher may have
            # completed the release while we waited.
            if exists(completion_uri):
                marker = json.loads(read_bytes(completion_uri))
                if marker.get("policy_hash") == policy_hash:
                    report.readback = _readback_release(
                        report=report,
                        output_root=output_root,
                        policy_hash=policy_hash,
                        marker_expected=True,
                    )
                    return {"complete": completion_uri}
                raise RuntimeError(
                    f"training release already published under a different "
                    f"policy hash ({marker.get('policy_hash')!r} != "
                    f"{policy_hash!r}) — immutable"
                )
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

            # Publisher-side readback: verify every per-scene eligibility COG
            # and every top-level artifact is actually published and
            # contract-conform BEFORE the release completion marker is
            # written. A release whose readback fails is never marked complete
            # (it stays invisible).
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

            # Release completion marker written last (create-only): the
            # release becomes visible only when every artifact is in place AND
            # readback passed.
            marker = {
                "published_at": now_iso(), "run_id": run_id,
                "policy_hash": policy_hash,
            }
            atomic_write(
                completion_uri,
                json.dumps(marker, indent=2),
                overwrite=False,
                if_generation_match=0,
            )

            # Post-write marker verification: the marker must read back with
            # the policy hash before the release is reported as complete.
            report.readback = _readback_release(
                report=report,
                output_root=output_root,
                policy_hash=policy_hash,
                marker_expected=True,
            )
            if not report.readback.get("ok"):
                raise RuntimeError(
                    f"release marker verification failed: {report.readback.get('errors')}"
                )
            uris["complete"] = completion_uri
            return uris
    except FileExistsError:
        raise RuntimeError(
            f"training release is being published by another run (lock {lock_uri})"
        ) from None


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
    "V3_FEATURES_ROOT",
    "publish_release",
]
