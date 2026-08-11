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

    # Build the ordered restore plan: ledger first, then per scene.
    restore_plan: list[str] = []
    ledger_uri = str(cfg["ard_ledger_uri"])
    restore_plan.append(ledger_uri)
    for cand in receipt["target_scenes"]:
        flag_uri = cand["flag_uri"]
        scene_dir = "/".join(flag_uri.split("/")[:-1])
        name = flag_uri.split("/")[-1].replace(".flag.tif", "")
        restore_plan += [
            flag_uri,
            f"{scene_dir}/{name}.stac.json",
            f"{scene_dir}/provenance.json",
            f"{scene_dir}/complete.json",
        ]
    restore_plan += list(receipt["reflectance_snapshots"].keys())

    print(f"Restoring repair {repair_id} from {backup_root}")
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


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.restore and args.apply:
        return run_restore(cfg, args.restore)
    if args.apply:
        return run_apply(cfg)
    return run_dry_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
