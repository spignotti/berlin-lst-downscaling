#!/usr/bin/env python3
"""Targeted S2 snow/ice flag repair — one-off production maintenance CLI.

Source: published S2 ARD flag COGs that predate ``FLAG_SNOW_ICE``.
Purpose: OR bit 5 (snow/ice) into the existing flag COG for exactly the
75 audited scenes with SCL=11, without touching reflectance COGs or the
main ARD pipeline. Grain: one report row per target S2 scene.

Mechanics live in ``data/qa/repair_s2_snow_ice.py``; this script is the
CLI + orchestration (dry-run / apply / restore).

Modes
-----
- Dry-run (default): full preflight, candidate staging, GCS snapshots,
  and a local receipt. Zero GCS writes.
- Apply: ``--apply`` — shared repair lock, immutable backup,
  generation-guarded publication (all flags/sidecars first, all
  completion markers last, then the ledger), after-state verification,
  and a post-audit that must report zero unflagged SCL11 pixels.
- Restore: ``--restore <repair-id> --apply`` — restore every backed-up
  object byte-identically (ledger first, then each scene's
  flag/provenance/STAC/completion as a set).

Usage
-----
    # Dry-run (default)
    uv run python scripts/repair_s2_snow_ice_flags.py \\
        --config configs/repair/s2_snow_ice.yaml

    # Apply (after explicit manual approval of the dry-run receipt)
    GIT_HEAD=$(git rev-parse HEAD) uv run python scripts/repair_s2_snow_ice_flags.py \\
        --config configs/repair/s2_snow_ice.yaml --apply

    # Restore a repair from its evidence prefix
    uv run python scripts/repair_s2_snow_ice_flags.py \\
        --config configs/repair/s2_snow_ice.yaml --restore <repair-id> --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from berlin_lst_downscaling.data.qa.repair_s2_snow_ice import (
    build_receipt,
    build_scene_candidate,
    collect_snapshots,
    copy_backup,
    delete_generation,
    gcs_client,
    git_head,
    load_audit,
    read_gcs_bytes,
    select_targets,
    sha256_bytes,
    snapshot_object,
    stage_ledger,
    upload_bytes_generation,
    upload_file_generation,
    verify_after_state,
    verify_completions,
)

_logger = logging.getLogger(__name__)

# Fixed shared lock name so concurrent repairs exclude one another.
_ACTIVE_LOCK = "s2-snow-ice-ACTIVE.lock"


# ── CLI / config ──────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted S2 snow/ice flag repair")
    parser.add_argument(
        "--config",
        default="configs/repair/s2_snow_ice.yaml",
        help="Repair configuration YAML",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write production GCS objects (default: read-only dry-run)",
    )
    parser.add_argument(
        "--restore",
        metavar="REPAIR_ID",
        default=None,
        help="Restore a previously applied repair from its evidence prefix",
    )
    parser.add_argument(
        "--finalize-failed",
        metavar="REPAIR_ID",
        default=None,
        help="Finalize a failed apply whose flags/sidecars/ledger were already "
        "published (read-only preflight by default; --apply writes only the "
        "missing completion markers + recovery evidence and releases the lock)",
    )
    parser.add_argument(
        "--dry-receipt",
        metavar="PATH",
        default=None,
        help="Local dry-run receipt path for --finalize-failed (required)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    required = [
        "manifest_uri",
        "ard_ledger_uri",
        "ard_root",
        "aoi_mask_uri",
        "audit_root",
        "audit_run_id",
        "evidence_prefix",
        "source",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Config {path!r} missing required keys: {missing}")
    return cfg


def evidence_root(cfg: dict[str, Any], repair_id: str) -> str:
    return f"{str(cfg['evidence_prefix']).rstrip('/')}/{repair_id}"


def active_lock_uri(cfg: dict[str, Any]) -> str:
    return f"{str(cfg['evidence_prefix']).rstrip('/')}/{_ACTIVE_LOCK}"


# ── dry-run ───────────────────────────────────────────────────────────


def run_dry_run(cfg: dict[str, Any]) -> int:
    """Full preflight + local staging. Never writes to GCS."""
    # git ancestor verification is a builder pre-apply step
    _, rows = load_audit(cfg)
    targets = select_targets(cfg, rows)

    from berlin_lst_downscaling.common.grid import canon_grid_10m
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.ledger import Ledger
    from berlin_lst_downscaling.data.qa.repair_s2_snow_ice import load_aoi_mask
    from berlin_lst_downscaling.data.selection.validate import load_bundle

    bundle, bundle_result = load_bundle(str(cfg["manifest_uri"]), require_item_href=True)
    if not bundle_result.ok:
        raise RuntimeError(f"manifest bundle invalid: {bundle_result.errors}")
    manifest_lookup = {
        row["scene_id"]: row
        for row in bundle.manifest_table.to_pylist()
        if row["source"] == str(cfg["source"])
    }
    ledger = Ledger.open(str(cfg["ard_ledger_uri"]))
    contract = contract_for_source(str(cfg["source"]))
    gbox = canon_grid_10m()
    aoi_mask = load_aoi_mask(str(cfg["aoi_mask_uri"]), gbox)
    head = git_head()
    run_id = uuid4().hex[:8]

    with tempfile.TemporaryDirectory(prefix="repair-s2-staging-") as tmp:
        tmpdir = Path(tmp)
        candidates = []
        for i, audit_row in enumerate(targets, 1):
            scene_id = str(audit_row["scene_id"])
            manifest_row = manifest_lookup.get(scene_id)
            if manifest_row is None:
                raise RuntimeError(f"{scene_id}: not in manifest {cfg['source']} rows")
            led = ledger.get(scene_id, str(cfg["source"]))
            if led is None or led.status != "done":
                raise RuntimeError(f"{scene_id}: ledger row not done")
            print(f"  [{i}/{len(targets)}] staging {scene_id}")
            candidates.append(
                build_scene_candidate(
                    cfg, audit_row, manifest_row, led.__dict__, gbox, aoi_mask,
                    contract, tmpdir, run_id, head,
                )
            )

        _, ledger_bytes = stage_ledger(cfg, candidates, tmpdir, run_id, contract)
        client = gcs_client()
        snapshots = collect_snapshots(cfg, candidates, client)
        receipt = build_receipt(cfg, candidates, snapshots, run_id, head, "dry-run")

        receipt_dir = Path(tempfile.gettempdir()) / "opencode"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"repair-s2-snow-ice-{run_id}.receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2))
        ledger_path = receipt_dir / f"repair-s2-snow-ice-{run_id}.ledger.parquet"
        ledger_path.write_bytes(ledger_bytes)

    total_unflagged = sum(c.unflagged_px for c in candidates)
    print(f"\nDry-run complete — repair {run_id}")
    print(f"  targets: {len(candidates)} (expected {cfg['expected_scenes_with_scl11']})")
    print(f"  unflagged SCL11 pixels to fix: {total_unflagged} "
          f"(expected {cfg['expected_scl11_unflagged_px']})")
    print(f"  reflectance COGs untouched: {len(candidates)}")
    print(f"  receipt: {receipt_path}")
    print(f"  staged ledger: {ledger_path}")
    print("  zero GCS writes performed")

    if total_unflagged != int(cfg["expected_scl11_unflagged_px"]):
        raise RuntimeError(
            f"total unflagged {total_unflagged} != expected "
            f"{cfg['expected_scl11_unflagged_px']}"
        )
    return 0


# ── apply ─────────────────────────────────────────────────────────────


def run_apply(cfg: dict[str, Any]) -> int:
    """Guarded production repair: lock → backup → scenes → completions → ledger."""
    from berlin_lst_downscaling.common.grid import canon_grid_10m
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.ledger import Ledger
    from berlin_lst_downscaling.data.io.run_logging import RunLogSession, log_event
    from berlin_lst_downscaling.data.io.storage import atomic_write
    from berlin_lst_downscaling.data.qa.repair_s2_snow_ice import load_aoi_mask
    from berlin_lst_downscaling.data.selection.validate import load_bundle

    head = git_head()
    if not head:
        raise RuntimeError(
            "GIT_HEAD env is required for --apply — the builder sets it after "
            "verifying the pinned ancestor commits (configs/repair/s2_snow_ice.yaml)"
        )
    _, rows = load_audit(cfg)
    targets = select_targets(cfg, rows)

    bundle, bundle_result = load_bundle(str(cfg["manifest_uri"]), require_item_href=True)
    if not bundle_result.ok:
        raise RuntimeError(f"manifest bundle invalid: {bundle_result.errors}")
    manifest_lookup = {
        row["scene_id"]: row
        for row in bundle.manifest_table.to_pylist()
        if row["source"] == str(cfg["source"])
    }
    ledger = Ledger.open(str(cfg["ard_ledger_uri"]))
    contract = contract_for_source(str(cfg["source"]))
    gbox = canon_grid_10m()
    aoi_mask = load_aoi_mask(str(cfg["aoi_mask_uri"]), gbox)
    client = gcs_client()
    repair_id = f"s2-snow-ice-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    run_id = uuid4().hex[:8]
    evidence = evidence_root(cfg, repair_id)
    backup_root = f"{evidence}/backup"
    lock_uri = active_lock_uri(cfg)

    with RunLogSession(evidence, pipeline="qa-repair-s2-snow-ice", run_id=run_id):
        log_event(_logger, logging.INFO, "repair_started", repair_id=repair_id, run_id=run_id)

        # ── shared lock ──────────────────────────────────────────────
        lock_payload = json.dumps(
            {
                "repair_id": repair_id,
                "run_id": run_id,
                "started": datetime.now(UTC).isoformat(),
                "git_head": head,
                "pid": os.getpid(),
            },
            indent=2,
        )
        try:
            atomic_write(lock_uri, lock_payload, overwrite=True, if_generation_match=0)
        except Exception as exc:
            raise RuntimeError(
                f"cannot acquire shared repair lock {lock_uri} — another repair "
                f"may be active: {exc}"
            ) from exc
        log_event(_logger, logging.INFO, "repair_lock_acquired", lock_uri=lock_uri)

        try:
            with tempfile.TemporaryDirectory(prefix="repair-s2-apply-") as tmp:
                tmpdir = Path(tmp)
                candidates = []
                for audit_row in targets:
                    scene_id = str(audit_row["scene_id"])
                    manifest_row = manifest_lookup.get(scene_id)
                    if manifest_row is None:
                        raise RuntimeError(f"{scene_id}: not in manifest")
                    led = ledger.get(scene_id, str(cfg["source"]))
                    if led is None or led.status != "done":
                        raise RuntimeError(f"{scene_id}: ledger row not done")
                    candidates.append(
                        build_scene_candidate(
                            cfg, audit_row, manifest_row, led.__dict__, gbox, aoi_mask,
                            contract, tmpdir, run_id, head,
                        )
                    )

                snapshots = collect_snapshots(cfg, candidates, client)

                # ── immutable backup ─────────────────────────────────
                backup_paths: dict[str, str] = {}
                for uri, snap in snapshots.items():
                    dst = f"{backup_root}/{uri.removeprefix('gs://berlin-lst-data/')}"
                    copy_backup(client, uri, snap, dst)
                    backup_paths[uri] = dst
                for uri, dst in backup_paths.items():
                    verify = snapshot_object(client, dst)
                    if (
                        verify is None
                        or verify["crc32c"] != snapshots[uri]["crc32c"]
                        or verify["size"] != snapshots[uri]["size"]
                    ):
                        raise RuntimeError(f"backup verification failed for {dst}")
                log_event(
                    _logger,
                    logging.INFO,
                    "backup_complete",
                    objects=len(backup_paths),
                    backup_root=backup_root,
                )

                # ── phase A0: remove ALL completion markers first ────
                # No scene advertises completion while the repair is
                # partial; they are recreated last (phase B).
                for cand in candidates:
                    delete_generation(
                        client,
                        cand.comp_uri,
                        if_generation_match=snapshots[cand.comp_uri]["generation"],
                    )
                log_event(
                    _logger,
                    logging.INFO,
                    "completions_removed",
                    scenes=len(candidates),
                )

                # ── phase A: flags + sidecars (no completions) ───────
                for cand in candidates:
                    _publish_scene(client, cand, snapshots)
                    log_event(
                        _logger,
                        logging.INFO,
                        "scene_repaired",
                        scene_id=cand.scene_id,
                        added_snow_px=cand.added_snow_px,
                    )

                # ── ledger (single CAS write) ────────────────────────
                _, ledger_bytes = stage_ledger(cfg, candidates, tmpdir, run_id, contract)
                upload_bytes_generation(
                    client,
                    ledger_bytes,
                    str(cfg["ard_ledger_uri"]),
                    if_generation_match=snapshots[str(cfg["ard_ledger_uri"])]["generation"],
                )
                reopened = Ledger.open(str(cfg["ard_ledger_uri"]))
                for cand in candidates:
                    if reopened.get(cand.scene_id, str(cfg["source"])) is None:
                        raise RuntimeError(f"ledger reload missing {cand.scene_id}")
                log_event(_logger, logging.INFO, "ledger_published", rows=len(candidates))

                # ── core after-state verification (no completions) ───
                after = verify_after_state(client, cfg, candidates, ledger_bytes, snapshots)
                log_event(_logger, logging.INFO, "after_state_verified")

                # ── post-audit must be zero (evidence prefix only) ───
                post_summary = _run_post_audit(cfg, evidence, run_id)
                if post_summary["scenes_failed"] != 0:
                    raise RuntimeError(
                        f"post-audit has failed scenes: {post_summary['scenes_failed']}"
                    )
                if post_summary["scenes_total"] != int(cfg["expected_scenes_total"]):
                    raise RuntimeError("post-audit scenes_total != expected")
                if post_summary["scl11_px"] != int(cfg["expected_scl11_px"]):
                    raise RuntimeError("post-audit SCL11 total drifted from baseline")
                if post_summary["scl11_unflagged_px"] != 0:
                    raise RuntimeError(
                        f"post-audit still has unflagged SCL11: "
                        f"{post_summary['scl11_unflagged_px']}"
                    )
                log_event(_logger, logging.INFO, "post_audit_zero")

                # ── phase B: all completion markers LAST ─────────────
                for cand in candidates:
                    atomic_write(
                        cand.comp_uri,
                        json.dumps(cand.comp_payload, indent=2),
                        overwrite=True,
                        if_generation_match=0,
                    )
                verify_completions(client, candidates)
                log_event(
                    _logger,
                    logging.INFO,
                    "completions_published",
                    scenes=len(candidates),
                )

                # ── receipts + summary ───────────────────────────────
                receipt = build_receipt(cfg, candidates, snapshots, run_id, head, "apply")
                receipt["repair_id"] = repair_id
                receipt["backup_root"] = backup_root
                receipt["lock_uri"] = lock_uri
                receipt["after_state"] = after
                receipt["post_audit"] = post_summary
                atomic_write(
                    f"{evidence}/receipt.json",
                    json.dumps(receipt, indent=2),
                    overwrite=True,
                    if_generation_match=0,
                )
                summary_payload = {
                    "repair_id": repair_id,
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "git_head": head,
                    "evidence_root": evidence,
                    "backup_root": backup_root,
                    "post_audit_root": f"{evidence}/post_audit",
                    "baseline": {
                        "audit_run_id": cfg.get("audit_run_id"),
                        "scenes": len(candidates),
                        "unflagged_px": sum(c.unflagged_px for c in candidates),
                    },
                    "post_audit": {
                        "scenes_total": post_summary["scenes_total"],
                        "scenes_failed": post_summary["scenes_failed"],
                        "scl11_px": post_summary["scl11_px"],
                        "scl11_unflagged_px": post_summary["scl11_unflagged_px"],
                    },
                    "per_scene": [
                        {
                            "scene_id": c.scene_id,
                            "candidate_flag_sha": c.candidate_flag_sha,
                            "added_snow_px": c.added_snow_px,
                            "flag_uri": c.flag_uri,
                        }
                        for c in candidates
                    ],
                }
                atomic_write(
                    f"{evidence}/summary.json",
                    json.dumps(summary_payload, indent=2),
                    overwrite=True,
                    if_generation_match=0,
                )

                # ── release lock (ownership-checked) ─────────────────
                _release_lock(client, lock_uri, repair_id)

        except BaseException:
            log_event(
                _logger,
                logging.ERROR,
                "repair_failed",
                repair_id=repair_id,
                lock_uri=lock_uri,
                note="lock and backup retained; restore via --restore",
            )
            raise

    print(f"\nRepair applied — {repair_id}")
    print(f"  scenes: {len(candidates)}")
    print(f"  evidence: {evidence}")
    return 0


def _run_post_audit(cfg: dict[str, Any], evidence: str, run_id: str) -> dict[str, Any]:
    """Run the S2 audit against the repaired state into the evidence prefix."""
    from berlin_lst_downscaling.data.qa.s2_snow_ice import run_s2_snow_ice_audit

    result = run_s2_snow_ice_audit(
        manifest_uri=str(cfg["manifest_uri"]),
        ard_ledger_uri=str(cfg["ard_ledger_uri"]),
        aoi_mask_uri=str(cfg["aoi_mask_uri"]),
        output_root=f"{evidence}/post_audit",
        run_id=run_id,
    )
    return result.summary


def _publish_scene(client, cand, snapshots: dict[str, dict[str, Any]]) -> None:
    """Publish one scene's flag + sidecars with generation guards.

    Completion markers are managed by the caller: all are deleted before
    any scene is published (phase A0) and all are recreated only after
    flags, sidecars, ledger, verification, and post-audit succeeded
    (phase B).
    """
    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    from berlin_lst_downscaling.data.io.storage import atomic_write
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    snap_flag = snapshots[cand.flag_uri]
    snap_prov = snapshots[cand.prov_uri]
    snap_stac = snapshots[cand.stac_uri]

    # 1. flag COG (exact generation precondition)
    upload_file_generation(
        client,
        cand.local_flag_path,
        cand.flag_uri,
        if_generation_match=snap_flag["generation"],
    )

    # 3. validate the published flag remotely. The staged candidate was
    #    already bit-asserted and strict-COG validated; byte-hash equality
    #    proves the remote flag is that candidate.
    remote_bytes = read_gcs_bytes(client, cand.flag_uri)
    if sha256_bytes(remote_bytes) != cand.candidate_flag_sha:
        raise RuntimeError(f"{cand.scene_id}: published flag hash mismatch")
    strict_ok = validate_strict_cog(gdal_uri(cand.flag_uri))
    if not strict_ok.valid:
        raise RuntimeError(
            f"{cand.scene_id}: remote strict COG failed: "
            f"{strict_ok.errors + strict_ok.warnings}"
        )

    # 4. provenance + STAC (exact generation preconditions)
    atomic_write(
        cand.prov_uri,
        json.dumps(cand.prov_payload, indent=2),
        overwrite=True,
        if_generation_match=snap_prov["generation"],
    )
    atomic_write(
        cand.stac_uri,
        json.dumps(cand.stac_payload, indent=2),
        overwrite=True,
        if_generation_match=snap_stac["generation"],
    )


def _release_lock(client, lock_uri: str, repair_id: str) -> None:
    """Delete the shared lock only if it belongs to *repair_id*."""
    payload = json.loads(read_gcs_bytes(client, lock_uri))
    if payload.get("repair_id") != repair_id:
        raise RuntimeError(
            f"lock {lock_uri} belongs to {payload.get('repair_id')!r}, not {repair_id!r}"
        )
    snap = snapshot_object(client, lock_uri)
    if snap is not None:
        delete_generation(client, lock_uri, if_generation_match=snap["generation"])


# ── restore ───────────────────────────────────────────────────────────


def run_restore(cfg: dict[str, Any], repair_id: str) -> int:
    """Restore every backed-up object byte-identically (explicit only)."""
    from berlin_lst_downscaling.data.io.storage import atomic_write

    evidence = evidence_root(cfg, repair_id)
    client = gcs_client()
    summary = json.loads(read_gcs_bytes(client, f"{evidence}/summary.json"))
    receipt = json.loads(read_gcs_bytes(client, f"{evidence}/receipt.json"))
    backup_root = str(summary["backup_root"])

    # Lock ownership: restoring a repair whose lock is active elsewhere
    # must fail; a lock belonging to this repair is fine (it is the
    # failed repair's own lock).
    lock_uri = active_lock_uri(cfg)
    lock_snap = snapshot_object(client, lock_uri)
    if lock_snap is not None:
        payload = json.loads(read_gcs_bytes(client, lock_uri))
        if payload.get("repair_id") != repair_id:
            raise RuntimeError(
                f"active lock belongs to {payload.get('repair_id')!r}; "
                f"refusing to restore {repair_id!r}"
            )

    # Build the ordered restore plan from the receipt: every object that
    # was backed up (flags, sidecars, completions, reflectance COGs,
    # ledger, audit artifacts, manifest bundle). Ledger first, then the
    # rest.
    restore_uris: set[str] = set()
    restore_uris.add(str(cfg["ard_ledger_uri"]))
    restore_uris.update(receipt["reflectance_snapshots"].keys())
    restore_uris.update(receipt["flag_snapshots"].keys())
    restore_uris.update(receipt["sidecar_snapshots"].keys())
    # Non-completion objects first (ledger first), completion markers
    # last — no scene advertises completeness mid-restore.
    non_completions = sorted(u for u in restore_uris if not u.endswith("complete.json"))
    completions = sorted(u for u in restore_uris if u.endswith("complete.json"))
    restore_plan = [str(cfg["ard_ledger_uri"])] + [
        u for u in non_completions if u != str(cfg["ard_ledger_uri"])
    ] + completions

    print(f"Restoring repair {repair_id} from {backup_root}")
    # Mirror publication: remove every existing completion marker up
    # front (generation-guarded) so no scene advertises completeness
    # while earlier objects are being restored. They are recreated from
    # their backups as the final restore step.
    for uri in completions:
        snap = snapshot_object(client, uri)
        if snap is not None:
            delete_generation(client, uri, if_generation_match=snap["generation"])
            print(f"  removed stale {uri}")
    for uri in restore_plan:
        backup_uri = f"{backup_root}/{uri.removeprefix('gs://berlin-lst-data/')}"
        data = read_gcs_bytes(client, backup_uri)
        snap = snapshot_object(client, uri)
        if snap is None or uri.endswith("complete.json"):
            if snap is not None:
                delete_generation(client, uri, if_generation_match=snap["generation"])
            atomic_write(uri, data, overwrite=True, if_generation_match=0)
        else:
            atomic_write(uri, data, overwrite=True, if_generation_match=snap["generation"])
        if read_gcs_bytes(client, uri) != data:
            raise RuntimeError(f"restore verification failed for {uri}")
        print(f"  restored {uri}")

    if lock_snap is not None:
        _release_lock(client, lock_uri, repair_id)
        print(f"  released {lock_uri}")

    print("Restore complete — verify with validate_ard and a fresh audit.")
    return 0


# ── finalize a failed apply ──────────────────────────────────────────


def _finalize_targets(cfg: dict[str, Any], receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive finalizer targets from the dry receipt + deterministic paths."""
    from berlin_lst_downscaling.data.ard.paths import (
        completion_path,
        provenance_path,
        stac_path,
    )

    root = str(cfg["ard_root"])
    source = str(cfg["source"])
    targets: list[dict[str, Any]] = []
    for t in receipt["target_scenes"]:
        scene_id = str(t["scene_id"])
        year = int(t["year"])
        targets.append(
            {
                "scene_id": scene_id,
                "year": year,
                "flag_uri": str(t["flag_uri"]),
                "cog_uri": str(t["cog_uri"]),
                "stac_uri": stac_path(root, source, year, scene_id),
                "prov_uri": provenance_path(root, source, year, scene_id),
                "comp_uri": completion_path(root, source, year, scene_id),
                "candidate_flag_sha": str(t["candidate_flag_sha"]),
                "stac_payload_sha": str(t["stac_payload_sha"]),
                "prov_payload_sha": str(t["prov_payload_sha"]),
            }
        )
    return targets


def _receipt_snapshot_map(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Unique pre-repair snapshot map from the receipt (382 objects)."""
    snaps: dict[str, dict[str, Any]] = {}
    snaps.update(receipt["reflectance_snapshots"])
    snaps.update(receipt["flag_snapshots"])
    snaps.update(receipt["sidecar_snapshots"])
    snaps[receipt["ledger_snapshot"]["uri"]] = receipt["ledger_snapshot"]
    return snaps


def _ledger_rows(uri: str) -> list[dict[str, Any]]:
    """Read a parquet ledger URI into pylist rows."""
    import io

    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.io.storage import read_bytes

    return pq.read_table(io.BytesIO(read_bytes(uri))).to_pylist()


def _finalize_preflight(
    cfg: dict[str, Any],
    repair_id: str,
    receipt_path: str,
    client,
    evidence: str,
    frozen: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify current state against the dry receipt; return the frozen plan.

    Read-only. Raises on any mismatch; no GCS write is performed here.
    """
    import hashlib
    import json as _json
    from pathlib import Path

    finalize = cfg["finalize"]

    # ── receipt binding ─────────────────────────────────────────────
    receipt_bytes = Path(receipt_path).read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != str(finalize["dry_receipt_sha256"]):
        raise RuntimeError(
            f"receipt sha256 {receipt_sha} != expected {finalize['dry_receipt_sha256']}"
        )
    receipt = _json.loads(receipt_bytes)
    if receipt.get("mode") != "dry-run":
        raise RuntimeError(f"receipt mode {receipt.get('mode')!r} != 'dry-run'")
    if len(receipt["target_scenes"]) != int(finalize["expected_scenes_with_scl11"]):
        raise RuntimeError("receipt target count != expected")
    targets = _finalize_targets(cfg, receipt)

    # ── lock ownership ──────────────────────────────────────────────
    lock_uri = active_lock_uri(cfg)
    lock_snap = snapshot_object(client, lock_uri)
    if lock_snap is None:
        raise RuntimeError(f"lock {lock_uri} is absent — nothing to finalize")
    lock_payload = _json.loads(read_gcs_bytes(client, lock_uri))
    if lock_payload.get("repair_id") != repair_id:
        raise RuntimeError(
            f"lock belongs to {lock_payload.get('repair_id')!r}, not {repair_id!r}"
        )
    if receipt.get("git_head") != lock_payload.get("git_head"):
        raise RuntimeError("receipt git_head != lock git_head")

    # ── backup inventory (382 unique objects, no extras) ────────────
    snap_map = _receipt_snapshot_map(receipt)
    unique_uris = set(snap_map)
    if len(receipt["reflectance_snapshots"]) != len(targets):
        raise RuntimeError("receipt reflectance snapshot count != target count")
    backup_root = f"{evidence}/backup"
    prefix = backup_root.removeprefix("gs://")
    bucket_name, backup_key_prefix = prefix.split("/", 1)
    listed = {
        b.name for b in client.bucket(bucket_name).list_blobs(prefix=backup_key_prefix)
    }
    if len(listed) != len(unique_uris):
        raise RuntimeError(
            f"backup inventory {len(listed)} != expected {len(unique_uris)}"
        )
    for uri in unique_uris:
        backup_uri = f"{backup_root}/{uri.removeprefix('gs://berlin-lst-data/')}"
        snap = snapshot_object(client, backup_uri)
        if snap is None:
            raise RuntimeError(f"backup object missing: {backup_uri}")
        want = snap_map[uri]
        if snap["crc32c"] != want["crc32c"] or snap["size"] != want["size"]:
            raise RuntimeError(f"backup mismatch vs receipt: {backup_uri}")

    # ── protected objects unchanged (81: 75 reflectance + 6 audit/manifest) ──
    protected_uris = [t["cog_uri"] for t in targets] + [
        str(cfg["manifest_uri"]),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "pairings.parquet"),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "manifest_report.json"),
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.parquet",
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.csv",
        f"{str(cfg['audit_root']).rstrip('/')}/summary.json",
    ]
    protected_snapshot: dict[str, dict[str, Any]] = {}
    for uri in protected_uris:
        now = snapshot_object(client, uri)
        want = snap_map[uri]
        if (
            now is None
            or now["generation"] != want["generation"]
            or now["crc32c"] != want["crc32c"]
            or now["size"] != want["size"]
        ):
            raise RuntimeError(f"protected object changed since receipt: {uri}")
        protected_snapshot[uri] = now

    # ── published flag/sidecar state matches the apply's candidates ──
    # flag + STAC are deterministic → byte-hash against the receipt.
    # provenance carries apply-time fields (run_id, completed_at) so it
    # is verified field-by-field instead.
    from berlin_lst_downscaling.data.ard.contract import contract_for_source

    contract = contract_for_source(str(cfg["source"]))
    apply_run_id = str(lock_payload["run_id"])
    apply_git_head = str(lock_payload["git_head"])
    for t in targets:
        flag = read_gcs_bytes(client, t["flag_uri"])
        if sha256_bytes(flag) != t["candidate_flag_sha"]:
            raise RuntimeError(f"{t['scene_id']}: flag hash != receipt candidate")
        stac = read_gcs_bytes(client, t["stac_uri"])
        if sha256_bytes(stac) != t["stac_payload_sha"]:
            raise RuntimeError(f"{t['scene_id']}: STAC hash != receipt candidate")
        prov = json.loads(read_gcs_bytes(client, t["prov_uri"]))
        if prov.get("scene_id") != t["scene_id"]:
            raise RuntimeError(f"{t['scene_id']}: provenance scene_id mismatch")
        if prov.get("source") != str(cfg["source"]):
            raise RuntimeError(f"{t['scene_id']}: provenance source mismatch")
        if int(prov.get("year", -1)) != t["year"]:
            raise RuntimeError(f"{t['scene_id']}: provenance year mismatch")
        if prov.get("schema_version") != contract.schema_version:
            raise RuntimeError(f"{t['scene_id']}: provenance schema_version mismatch")
        if prov.get("run_id") != apply_run_id:
            raise RuntimeError(
                f"{t['scene_id']}: provenance run_id {prov.get('run_id')} != apply {apply_run_id}"
            )
        if prov.get("repair") is not True:
            raise RuntimeError(f"{t['scene_id']}: provenance repair flag missing")
        if prov.get("repair_commit") != apply_git_head:
            raise RuntimeError(f"{t['scene_id']}: provenance repair_commit mismatch")
        if prov.get("output_bands") != [s.name for s in contract.output_bands]:
            raise RuntimeError(f"{t['scene_id']}: provenance output_bands mismatch")
        if "completed_at" not in prov:
            raise RuntimeError(f"{t['scene_id']}: provenance completed_at missing")
        backup_prov_uri = (
            f"{backup_root}/{t['prov_uri'].removeprefix('gs://berlin-lst-data/')}"
        )
        backup_prov = json.loads(read_gcs_bytes(client, backup_prov_uri))
        if prov.get("source_metadata") != backup_prov.get("source_metadata"):
            raise RuntimeError(
                f"{t['scene_id']}: provenance source_metadata changed vs backup"
            )

    # ── ledger: only allowed fields on the 75 target rows ───────────
    current_rows = _ledger_rows(str(cfg["ard_ledger_uri"]))
    backup_ledger_uri = (
        f"{backup_root}/{str(cfg['ard_ledger_uri']).removeprefix('gs://berlin-lst-data/')}"
    )
    backup_rows = _ledger_rows(backup_ledger_uri)
    if len(current_rows) != len(backup_rows):
        raise RuntimeError("ledger row count changed since backup")
    key_index = {(r["scene_id"], r["source"]): i for i, r in enumerate(current_rows)}
    target_keys = {(t["scene_id"], str(cfg["source"])) for t in targets}
    allowed = {
        "schema_hash",
        "schema_version",
        "run_id",
        "updated_at",
        "aoi_clear_px",
        "aoi_clear_frac",
        "aoi_total_px",
    }
    for b_row in backup_rows:
        key = (b_row["scene_id"], b_row["source"])
        idx = key_index.get(key)
        if idx is None:
            raise RuntimeError(f"ledger row missing in current: {key}")
        c_row = current_rows[idx]
        if key in target_keys:
            for field, c_val in c_row.items():
                if field in allowed:
                    continue
                if c_val != b_row.get(field):
                    raise RuntimeError(
                        f"ledger target row {key} changed outside allowed fields: {field}"
                    )
        elif c_row != b_row:
            raise RuntimeError(f"ledger non-target row changed: {key}")

    # ── mutable object generations + completion payloads ─────────────
    run_id = str(lock_payload["run_id"])
    if frozen is not None:
        if frozen.get("receipt_sha256") != receipt_sha:
            raise RuntimeError("existing prepared.json does not match this receipt")
        comp_by_uri = {t["comp_uri"]: t["comp_payload_sha"] for t in frozen["targets"]}
        for uri, expected_gen in frozen["expected_generations"].items():
            now = snapshot_object(client, uri)
            if uri in comp_by_uri:
                # Completions may already be published from a prior attempt:
                # accept absent or byte-identical to the frozen payload.
                if now is None:
                    continue
                if sha256_bytes(read_gcs_bytes(client, uri)) != comp_by_uri[uri]:
                    raise RuntimeError(
                        f"completion differs from frozen payload: {uri}"
                    )
                continue
            if now is None or now["generation"] != expected_gen:
                raise RuntimeError(f"mutable object generation drifted: {uri}")
        plan_targets = frozen["targets"]
        expected_generations = frozen["expected_generations"]
    else:
        expected_generations: dict[str, int] = {}
        for t in targets:
            for uri_key in ("flag_uri", "stac_uri", "prov_uri", "comp_uri"):
                uri = t[uri_key]
                snap = snapshot_object(client, uri)
                if snap is None:
                    expected_generations[uri] = -1  # absent (completion)
                else:
                    expected_generations[uri] = snap["generation"]
        ledger_snap = snapshot_object(client, str(cfg["ard_ledger_uri"]))
        if ledger_snap is None:
            raise RuntimeError("ledger object missing")
        expected_generations[str(cfg["ard_ledger_uri"])] = ledger_snap["generation"]

        comp_payloads: dict[str, tuple[dict[str, Any], str]] = {}
        for t in targets:
            payload = {
                "published_at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "repair": True,
            }
            payload_bytes = json.dumps(payload, indent=2).encode("utf-8")
            comp_payloads[t["scene_id"]] = (payload, sha256_bytes(payload_bytes))

        plan_targets = []
        for t in targets:
            payload, payload_sha = comp_payloads[t["scene_id"]]
            comp_snap = snapshot_object(client, t["comp_uri"])
            if comp_snap is not None:
                existing = read_gcs_bytes(client, t["comp_uri"])
                existing_payload = json.loads(existing)
                if (
                    existing_payload.get("run_id") == run_id
                    and existing_payload.get("repair") is True
                ):
                    # Completion from the apply's own Phase B (or a prior
                    # finalizer attempt without a frozen plan): adopt it
                    # byte-for-byte instead of rewriting.
                    payload = existing_payload
                    payload_sha = sha256_bytes(existing)
                elif sha256_bytes(existing) != payload_sha:
                    raise RuntimeError(
                        f"{t['scene_id']}: existing completion differs from planned payload"
                    )
            plan_targets.append(
                {**t, "comp_payload": payload, "comp_payload_sha": payload_sha}
            )

    recovery = f"{evidence}/recovery"
    plan = {
        "repair_id": repair_id,
        "run_id": run_id,
        "git_head": str(lock_payload["git_head"]),
        "receipt_sha256": receipt_sha,
        "prepared_at": datetime.now(UTC).isoformat(),
        "backup_root": backup_root,
        "recovery_root": recovery,
        "audit_output_root": f"{recovery}/audit",
        "lock_generation": lock_snap["generation"],
        "expected_generations": expected_generations,
        "protected_snapshot": protected_snapshot,
        "targets": plan_targets,
        "allowed_write_uris": {
            "canonical_completions": [t["comp_uri"] for t in plan_targets],
            "recovery_evidence": [f"{recovery}/prepared.json"],
            "lock_delete": lock_uri,
        },
    }
    return plan


def run_finalize(
    cfg: dict[str, Any],
    repair_id: str,
    receipt_path: str,
    apply: bool,
) -> int:
    """Finalize a failed apply: preflight, then completions + evidence + lock.

    Read-only by default (``apply=False`` prints the frozen plan digest).
    With ``--apply`` the only canonical writes are the 75 missing
    ``complete.json`` markers; recovery evidence goes to
    ``<evidence>/recovery/`` and the owned lock is deleted last.
    """
    from berlin_lst_downscaling.data.io.storage import atomic_write
    from berlin_lst_downscaling.data.qa.s2_snow_ice import run_s2_snow_ice_audit

    evidence = evidence_root(cfg, repair_id)
    client = gcs_client()
    recovery = f"{evidence}/recovery"
    prepared_uri = f"{recovery}/prepared.json"

    existing_prepared = snapshot_object(client, prepared_uri)
    frozen = None
    if existing_prepared is not None:
        frozen = json.loads(read_gcs_bytes(client, prepared_uri))
        print(f"found frozen plan at {prepared_uri}")

    plan = _finalize_preflight(cfg, repair_id, receipt_path, client, evidence, frozen)
    n_completions = len(plan["targets"])

    print(f"\nFinalizer preflight passed for repair {repair_id}")
    print(f"  targets: {n_completions}")
    print(f"  protected objects unchanged: {len(plan['protected_snapshot'])}")
    print(f"  canonical writes (missing complete.json): {n_completions}")
    print(f"  recovery evidence root: {recovery}")
    if not apply:
        print("  read-only preflight — zero GCS writes performed")
        return 0

    # ── freeze the plan (first durable write, immutable) ─────────────
    if existing_prepared is None:
        atomic_write(
            prepared_uri,
            json.dumps(plan, indent=2),
            overwrite=True,
            if_generation_match=0,
        )
        print(f"froze plan at {prepared_uri}")

    # ── fresh full audit (GCS-byte flag reads, no /vsigs/) ───────────
    audit_result = run_s2_snow_ice_audit(
        manifest_uri=str(cfg["manifest_uri"]),
        ard_ledger_uri=str(cfg["ard_ledger_uri"]),
        aoi_mask_uri=str(cfg["aoi_mask_uri"]),
        output_root=plan["audit_output_root"],
        run_id=plan["run_id"],
    )
    summary = audit_result.summary
    finalize = cfg["finalize"]
    gates = {
        "scenes_total": int(finalize["expected_scenes_total"]),
        "scenes_compared": int(finalize["expected_scenes_total"]),
        "scenes_failed": 0,
        "scenes_with_scl11": int(finalize["expected_scenes_with_scl11"]),
        "scenes_with_unflagged_scl11": 0,
        "scl11_px": int(finalize["expected_scl11_px"]),
        "scl11_invalid_px": int(finalize["expected_scl11_invalid_px"]),
        "scl11_unflagged_px": int(finalize["expected_scl11_unflagged_px"]),
    }
    for key, expected in gates.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"finalizer audit gate {key}={summary.get(key)} != {expected}"
            )
    print(
        f"finalizer audit passed: 158/158 compared, 0 failed, "
        f"scl11_unflagged_px={summary['scl11_unflagged_px']}"
    )

    # ── completion markers last (resumable, absent-precondition) ─────
    for t in plan["targets"]:
        uri = t["comp_uri"]
        payload_bytes = json.dumps(t["comp_payload"], indent=2).encode("utf-8")
        comp_snap = snapshot_object(client, uri)
        if comp_snap is not None:
            if sha256_bytes(read_gcs_bytes(client, uri)) != t["comp_payload_sha"]:
                raise RuntimeError(
                    f"{t['scene_id']}: existing completion differs from frozen payload"
                )
            continue
        atomic_write(uri, payload_bytes, overwrite=True, if_generation_match=0)
    for t in plan["targets"]:
        remote = read_gcs_bytes(client, t["comp_uri"])
        if sha256_bytes(remote) != t["comp_payload_sha"]:
            raise RuntimeError(f"{t['scene_id']}: completion verification failed")
    print(f"published + verified {len(plan['targets'])} completion markers")

    # ── recovery evidence ────────────────────────────────────────────
    recovery_payload = {
        "repair_id": repair_id,
        "run_id": plan["run_id"],
        "git_head": plan["git_head"],
        "receipt_sha256": plan["receipt_sha256"],
        "completed_at": datetime.now(UTC).isoformat(),
        "backup_root": plan["backup_root"],
        "audit": summary,
        "completions": [
            {
                "scene_id": t["scene_id"],
                "comp_uri": t["comp_uri"],
                "comp_payload_sha": t["comp_payload_sha"],
            }
            for t in plan["targets"]
        ],
        "protected_unchanged": len(plan["protected_snapshot"]),
    }
    atomic_write(
        f"{recovery}/receipt.json",
        json.dumps(recovery_payload, indent=2),
        overwrite=True,
    )
    atomic_write(
        f"{recovery}/summary.json",
        json.dumps(
            {
                "repair_id": repair_id,
                "run_id": plan["run_id"],
                "completed_at": recovery_payload["completed_at"],
                "audit": summary,
                "completions": len(plan["targets"]),
            },
            indent=2,
        ),
        overwrite=True,
    )
    atomic_write(
        f"{recovery}/complete.json",
        json.dumps(
            {
                "repair_id": repair_id,
                "run_id": plan["run_id"],
                "completed_at": recovery_payload["completed_at"],
            },
            indent=2,
        ),
        overwrite=True,
    )
    print(f"wrote recovery evidence under {recovery}")

    # ── release the owned lock (frozen generation) ───────────────────
    lock_snap = snapshot_object(client, active_lock_uri(cfg))
    if lock_snap is None:
        print("lock already absent at release time")
    elif lock_snap["generation"] != plan["lock_generation"]:
        raise RuntimeError("lock generation changed since preflight")
    else:
        delete_generation(
            client, active_lock_uri(cfg), if_generation_match=lock_snap["generation"]
        )
        print("released owned repair lock")

    print(f"\nFinalized repair {repair_id} — completions published, lock released")
    print(f"  recovery evidence: {recovery}")
    print("  verify with scripts/validate_ard.py and a fresh audit")
    return 0


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.finalize_failed:
        if not args.dry_receipt:
            raise SystemExit("--finalize-failed requires --dry-receipt PATH")
        return run_finalize(cfg, args.finalize_failed, args.dry_receipt, args.apply)
    if args.restore and args.apply:
        return run_restore(cfg, args.restore)
    if args.apply:
        return run_apply(cfg)
    return run_dry_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
