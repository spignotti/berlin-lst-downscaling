#!/usr/bin/env python3
"""One-off ARD schema v7 metadata migration (WB2c-2 follow-up).

Source: published ARD ledger + scene metadata at
gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z.
Purpose: bump the 434 remaining done v6 rows to schema_version 7 in the
ledger, STAC, and provenance so a future run_ard reconcile skips them
(Contract v7). Grain: one target row per scene+source (434 = 345
Landsat + 83 S2 without SCL=11 + 6 ECOSTRESS).

Metadata-only: COG and flag objects are never written; only the 434
STAC + 434 provenance + 434 completion markers and the ledger are
touched, completions last. Dry-run by default; --apply writes with a
fixed global lock, immutable backup, and generation CAS; --restore
rolls a run back byte-identically from its evidence prefix.

Usage
-----
    # Dry-run (default) — full preflight, staged candidates, zero writes
    uv run python scripts/migrate_ard_schema_v7.py \
        --config configs/repair/ard_schema_v7_metadata.yaml

    # Apply (after explicit manual approval of the dry-run evidence)
    GIT_HEAD=$(git rev-parse HEAD) uv run python scripts/migrate_ard_schema_v7.py \
        --config configs/repair/ard_schema_v7_metadata.yaml --apply

    # Restore a migration from its evidence prefix
    uv run python scripts/migrate_ard_schema_v7.py \
        --config configs/repair/ard_schema_v7_metadata.yaml --restore <run-id> --apply
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

# Canonical flag asset title with snow/ice (current product.py contract).
_CANONICAL_FLAG_TITLE = (
    "Quality flag (bitmask: fill, cloudy, shadow, cirrus, saturated, snow/ice)"
)
# The baseline S2 snow/ice audit referenced by the docs (covers all 158 S2 scenes).
_BASELINE_AUDIT_RUN_ID = "cbedd8db"

# decision: completion payload carries `repair: True` and the migration run_id
# to mark metadata maintenance lineage, matching the S2 snow/ice repair
# precedent. Alternative: leave run_id of the original publish (rejected —
# the marker would claim a run that did not write these bytes).


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-off ARD schema v7 metadata migration")
    parser.add_argument(
        "--config",
        default="configs/repair/ard_schema_v7_metadata.yaml",
        help="Migration configuration YAML",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write production metadata (default: read-only dry-run)",
    )
    parser.add_argument(
        "--restore",
        metavar="RUN_ID",
        default=None,
        help="Restore a previously applied migration from its evidence prefix",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    required = [
        "manifest_uri",
        "ard_ledger_uri",
        "ard_root",
        "audit_root",
        "evidence_prefix",
        "lock_uri",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Config {path!r} missing required keys: {missing}")
    return cfg


# ── GCS helpers (retry on transient errors) ───────────────────────────


def _retry(fn, *args, **kwargs):
    from google.api_core.exceptions import (
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    transient = (
        ConnectionError,
        TimeoutError,
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(transient),
        reraise=True,
    )
    def _run():
        return fn(*args, **kwargs)

    return _run()


def gcs_client():
    from google.cloud import storage

    return storage.Client()


def _parse_gs(uri: str) -> tuple[str, str]:
    path = uri.removeprefix("gs://")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return parts[0], parts[1]


def _blob(client, uri: str):
    bucket, key = _parse_gs(uri)
    return client.bucket(bucket).blob(key)


def snapshot_object(client, uri: str) -> dict[str, Any] | None:
    blob = _blob(client, uri)
    if not blob.exists():
        return None
    blob.reload()
    return {
        "uri": uri,
        "generation": blob.generation,
        "metageneration": blob.metageneration,
        "crc32c": blob.crc32c,
        "size": blob.size,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_gcs_bytes(client, uri: str) -> bytes:
    return _blob(client, uri).download_as_bytes()


def copy_backup(client, src_uri: str, src_snapshot: dict[str, Any], dst_uri: str) -> None:
    """Server-side copy of one exact source generation into the backup."""

    def _do():
        bucket, key = _parse_gs(src_uri)
        dst_bucket, dst_key = _parse_gs(dst_uri)
        src = client.bucket(bucket).blob(key)
        client.bucket(bucket).copy_blob(
            src,
            client.bucket(dst_bucket),
            new_name=dst_key,
            source_generation=src_snapshot["generation"],
            if_generation_match=0,
        )

    _retry(_do)


def upload_bytes_generation(
    client, data: bytes, dst_uri: str, if_generation_match: int | None
) -> None:
    def _do():
        blob = _blob(client, dst_uri)
        blob.upload_from_string(
            data,
            if_generation_match=if_generation_match,
            checksum="auto",
        )

    _retry(_do)


def delete_generation(client, uri: str, if_generation_match: int) -> None:
    def _do():
        blob = _blob(client, uri)
        blob.delete(if_generation_match=if_generation_match)

    _retry(_do)


# ── target model ──────────────────────────────────────────────────────


def _derive_targets(cfg: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select done v6 rows and derive their deterministic artifact URIs."""
    from berlin_lst_downscaling.data.ard.paths import (
        completion_path,
        provenance_path,
    )

    root = str(cfg["ard_root"])
    targets: list[dict[str, Any]] = []
    for r in ledger_rows:
        if r["status"] != "done" or int(r["schema_version"]) != 6 or r["schema_hash"] != "6":
            continue
        scene_id = str(r["scene_id"])
        source = str(r["source"])
        year = int(r["year"])
        targets.append(
            {
                "scene_id": scene_id,
                "source": source,
                "year": year,
                "cog_uri": str(r["path_cog"]),
                "flag_uri": str(r["path_flag"]),
                "stac_uri": str(r["path_stac"]),
                "prov_uri": provenance_path(root, source, year, scene_id),
                "comp_uri": completion_path(root, source, year, scene_id),
                "ledger_index": ledger_rows.index(r),
            }
        )
    return targets


# ── staging (semantic patches with diff verification) ────────────────


def _stage_ledger(
    cfg: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    run_id: str,
) -> tuple[str, bytes]:
    """Stage the bumped ledger; return path + MODIFIED bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.io.storage import read_bytes

    raw = read_bytes(str(cfg["ard_ledger_uri"]))
    original = pq.read_table(io.BytesIO(raw))
    rows = [dict(r) for r in ledger_rows]
    allowed = {"schema_hash", "schema_version", "run_id", "updated_at"}
    for t in targets:
        idx = t["ledger_index"]
        old = rows[idx]
        new = dict(old)
        new["schema_hash"] = str(cfg["target_schema_version"])
        new["schema_version"] = int(cfg["target_schema_version"])
        new["run_id"] = run_id
        new["updated_at"] = datetime.now(UTC)
        for key, value in new.items():
            if key in allowed:
                continue
            if value != old[key]:
                raise RuntimeError(
                    f"{t['scene_id']}: unexpected ledger change in {key!r}"
                )
        rows[idx] = new

    table = pa.Table.from_pylist(rows, schema=original.schema)
    local = Path(str(cfg["ard_ledger_uri"]).replace("gs://", "")).name
    # stage to a local temp path in the process working dir temp
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="migrate-schema-v7-")) / f"ledger-{local}"
    pq.write_table(table, tmp)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return str(tmp), buf.getvalue()


def _stage_stac(client, uri: str) -> bytes:
    """Patch ard:schema_version + flag title; require exact semantic diff."""
    original = json.loads(read_gcs_bytes(client, uri))
    staged = json.loads(json.dumps(original))
    props = staged.get("properties", {})
    if props.get("ard:schema_version") != "6":
        raise RuntimeError(f"{uri}: STAC ard:schema_version not '6'")
    props["ard:schema_version"] = "7"
    flag = staged.get("assets", {}).get("flag")
    if flag is not None:
        if flag.get("title") == _CANONICAL_FLAG_TITLE:
            raise RuntimeError(f"{uri}: flag title already canonical")
        flag["title"] = _CANONICAL_FLAG_TITLE
    _assert_semantic_diff(
        original,
        staged,
        uri,
        {".properties.ard:schema_version", ".assets.flag.title"},
    )
    return json.dumps(staged, indent=2).encode("utf-8")


def _stage_provenance(client, uri: str) -> bytes:
    """Patch provenance schema_version 6->7; require exact semantic diff."""
    original = json.loads(read_gcs_bytes(client, uri))
    staged = json.loads(json.dumps(original))
    if staged.get("schema_version") != 6:
        raise RuntimeError(f"{uri}: provenance schema_version not 6")
    staged["schema_version"] = 7
    _assert_semantic_diff(original, staged, uri, {".schema_version"})
    return json.dumps(staged, indent=2).encode("utf-8")


def _assert_semantic_diff(
    original: dict, staged: dict, uri: str, allowed_paths: set[str]
) -> None:
    """Require staged == original except the whitelisted paths.

    Whitelisted transitions are asserted by the caller (``_stage_stac``
    verifies ard:schema_version was "6" and the flag title was not
    already canonical before staging).
    """
    def walk(o, s, path: str):
        if isinstance(o, dict) and isinstance(s, dict):
            if o.keys() != s.keys():
                raise RuntimeError(f"{uri}: key set changed at {path}")
            for k in o:
                walk(o[k], s[k], f"{path}.{k}")
        elif isinstance(o, list) and isinstance(s, list):
            if len(o) != len(s):
                raise RuntimeError(f"{uri}: list length changed at {path}")
            for i, (a, b) in enumerate(zip(o, s, strict=True)):
                walk(a, b, f"{path}[{i}]")
        elif o != s and path not in allowed_paths:
            raise RuntimeError(f"{uri}: value changed at {path}: {o!r} -> {s!r}")

    walk(original, staged, "")


# ── preflight ─────────────────────────────────────────────────────────


def _ledger_rows(uri: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.io.storage import read_bytes

    return pq.read_table(io.BytesIO(read_bytes(uri))).to_pylist()


def _load_audit(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the baseline S2 scene audit keyed by scene_id."""
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.io.storage import read_bytes

    root = str(cfg["audit_root"]).rstrip("/")
    summary = json.loads(read_bytes(f"{root}/summary.json"))
    if summary.get("run_id") != _BASELINE_AUDIT_RUN_ID:
        raise RuntimeError(
            f"audit run mismatch: expected {_BASELINE_AUDIT_RUN_ID}, "
            f"got {summary.get('run_id')!r}"
        )
    if summary.get("scenes_failed", 0) != 0:
        raise RuntimeError(f"audit has failed scenes: {summary.get('scenes_failed')}")
    table = pq.read_table(io.BytesIO(read_bytes(f"{root}/scene_audit.parquet")))
    return {str(r["scene_id"]): r for r in table.to_pylist()}


def _preflight(
    cfg: dict[str, Any],
    client,
    migration_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Verify current state; return the frozen migration plan. Read-only."""
    from berlin_lst_downscaling.data.io.storage import exists

    ledger_uri = str(cfg["ard_ledger_uri"])
    rows = _ledger_rows(ledger_uri)
    if len(rows) != int(cfg["expected_ledger_rows"]):
        raise RuntimeError(
            f"ledger rows {len(rows)} != expected {cfg['expected_ledger_rows']}"
        )
    if sum(1 for r in rows if r["status"] == "done") != int(cfg["expected_ledger_done"]):
        raise RuntimeError("ledger done count != expected")
    keys = {(r["scene_id"], r["source"]) for r in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate ledger keys")
    for r in rows:
        if r["schema_hash"] != str(r["schema_version"]):
            raise RuntimeError(
                f"{r['scene_id']}: schema_hash != str(schema_version)"
            )

    targets = _derive_targets(cfg, rows)
    total = len(targets)
    if total != int(cfg["expected_targets_total"]):
        raise RuntimeError(
            f"target count {total} != expected {cfg['expected_targets_total']}"
        )
    by_source: dict[str, int] = {}
    for t in targets:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1
    for source, expected in cfg["expected_by_source"].items():
        if by_source.get(source, 0) != int(expected):
            raise RuntimeError(
                f"targets {source}: {by_source.get(source, 0)} != {expected}"
            )

    # All target artifacts must exist.
    for t in targets:
        for label, uri in (
            ("cog", t["cog_uri"]),
            ("flag", t["flag_uri"]),
            ("stac", t["stac_uri"]),
            ("prov", t["prov_uri"]),
            ("comp", t["comp_uri"]),
        ):
            if not exists(uri):
                raise RuntimeError(f"{t['scene_id']}: missing {label} at {uri}")

    # The 83 target S2 scenes must have no SCL=11 in the baseline audit.
    audit = _load_audit(cfg)
    for t in targets:
        if t["source"] != "sentinel-2-l2a":
            continue
        row = audit.get(t["scene_id"])
        if row is None or not row.get("ok"):
            raise RuntimeError(f"{t['scene_id']}: not ok in baseline audit")
        if int(row.get("scl11_px") or 0) != 0:
            raise RuntimeError(
                f"{t['scene_id']}: baseline audit scl11_px != 0 — cannot migrate"
            )

    # ── mutable objects: original + staged hashes, expected generations ──
    mutable: list[dict[str, Any]] = []
    for t in targets:
        stac_bytes = _stage_stac(client, t["stac_uri"])
        prov_bytes = _stage_provenance(client, t["prov_uri"])
        for label, uri, staged in (
            ("stac", t["stac_uri"], stac_bytes),
            ("prov", t["prov_uri"], prov_bytes),
            ("comp", t["comp_uri"], None),
        ):
            snap = snapshot_object(client, uri)
            if snap is None:
                raise RuntimeError(f"mutable object missing: {uri}")
            original_bytes = read_gcs_bytes(client, uri) if label != "comp" else b""
            mutable.append(
                {
                    "label": label,
                    "uri": uri,
                    "generation": snap["generation"],
                    "original_sha": (
                        sha256_bytes(original_bytes) if label != "comp" else None
                    ),
                    "staged_sha": sha256_bytes(staged) if staged is not None else None,
                    "staged_bytes": staged if staged is not None else None,
                }
            )

    ledger_staged_path, ledger_staged_bytes = _stage_ledger(
        cfg, rows, targets, run_id
    )
    ledger_snap = snapshot_object(client, ledger_uri)
    if ledger_snap is None:
        raise RuntimeError("ledger object missing")
    ledger_original_sha = sha256_bytes(read_gcs_bytes(client, ledger_uri))
    mutable.append(
        {
            "label": "ledger",
            "uri": ledger_uri,
            "generation": ledger_snap["generation"],
            "original_sha": ledger_original_sha,
            "staged_sha": sha256_bytes(ledger_staged_bytes),
            "staged_bytes": ledger_staged_bytes,
            "staged_path": ledger_staged_path,
        }
    )

    # ── protected objects: 434 COG + 434 flag + 6 manifest/audit ───────
    protected_uris = [t["cog_uri"] for t in targets] + [t["flag_uri"] for t in targets]
    protected_uris += [
        str(cfg["manifest_uri"]),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "pairings.parquet"),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "manifest_report.json"),
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.parquet",
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.csv",
        f"{str(cfg['audit_root']).rstrip('/')}/summary.json",
    ]
    protected_snapshot: dict[str, dict[str, Any]] = {}
    for uri in protected_uris:
        snap = snapshot_object(client, uri)
        if snap is None:
            raise RuntimeError(f"protected object missing: {uri}")
        protected_snapshot[uri] = snap

    evidence = f"{str(cfg['evidence_prefix']).rstrip('/')}/{migration_id}"
    lock_uri = str(cfg["lock_uri"])
    lock_snap = snapshot_object(client, lock_uri)

    plan = {
        "migration_id": migration_id,
        "run_id": run_id,
        "git_head": os.environ.get("GIT_HEAD", ""),
        "prepared_at": datetime.now(UTC).isoformat(),
        "ledger_uri": ledger_uri,
        "ledger_original_sha": ledger_original_sha,
        "ledger_staged_sha": sha256_bytes(ledger_staged_bytes),
        "targets": [
            {k: t[k] for k in ("scene_id", "source", "year")} for t in targets
        ],
        "mutable": [
            {
                "label": m["label"],
                "uri": m["uri"],
                "generation": m["generation"],
                "original_sha": m["original_sha"],
                "staged_sha": m["staged_sha"],
            }
            for m in mutable
        ],
        "protected_snapshot": protected_snapshot,
        "backup_root": f"{evidence}/backup",
        "evidence_root": evidence,
        "lock_uri": lock_uri,
        "lock_generation": lock_snap["generation"] if lock_snap else None,
        "lock_present": lock_snap is not None,
        "target_count": total,
    }
    return plan


# ── apply ─────────────────────────────────────────────────────────────


def run_migrate(cfg: dict[str, Any], apply: bool) -> int:
    """Dry-run (default) or guarded production metadata migration."""
    from berlin_lst_downscaling.data.io.storage import atomic_write

    head = os.environ.get("GIT_HEAD", "")
    if apply and not head:
        raise RuntimeError(
            "GIT_HEAD env is required for --apply — the builder sets it after "
            "verifying the pinned ancestor commits"
        )

    client = gcs_client()
    migration_id = f"ard-schema-v7-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    run_id = uuid4().hex[:8]
    plan = _preflight(cfg, client, migration_id, run_id)
    evidence = plan["evidence_root"]
    backup_root = plan["backup_root"]

    n_stac = sum(1 for m in plan["mutable"] if m["label"] == "stac")
    n_prov = sum(1 for m in plan["mutable"] if m["label"] == "prov")
    n_comp = sum(1 for m in plan["mutable"] if m["label"] == "comp")
    print(f"\nMigration preflight passed ({migration_id})")
    print(f"  targets: {plan['target_count']} (345 Landsat / 83 S2 / 6 ECOSTRESS)")
    print(f"  mutable objects: {len(plan['mutable'])} "
          f"({n_stac} stac + {n_prov} prov + {n_comp} comp + 1 ledger)")
    print(f"  protected objects: {len(plan['protected_snapshot'])} "
          "(434 cog + 434 flag + 6 manifest/audit)")
    print(f"  evidence root: {evidence}")
    print(f"  lock present: {plan['lock_present']}")
    if not apply:
        print("  read-only dry-run — zero GCS writes performed")
        return 0

    lock_uri = str(cfg["lock_uri"])
    if plan["lock_present"]:
        raise RuntimeError(
            f"migration lock {lock_uri} already present — another migration is active"
        )
    lock_payload = json.dumps(
        {
            "migration_id": migration_id,
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
        raise RuntimeError(f"cannot acquire migration lock {lock_uri}: {exc}") from exc
    print(f"acquired migration lock {lock_uri}")

    try:
        # ── immutable backup (1303 objects) ──────────────────────────
        for m in plan["mutable"]:
            src_snap = snapshot_object(client, m["uri"])
            if src_snap is None or src_snap["generation"] != m["generation"]:
                raise RuntimeError(f"mutable object drifted before backup: {m['uri']}")
            dst = f"{backup_root}/{m['uri'].removeprefix('gs://berlin-lst-data/')}"
            copy_backup(client, m["uri"], src_snap, dst)
            verify = snapshot_object(client, dst)
            if (
                verify is None
                or verify["crc32c"] != src_snap["crc32c"]
                or verify["size"] != src_snap["size"]
            ):
                raise RuntimeError(f"backup verification failed for {dst}")
        print(f"backup complete: {len(plan['mutable'])} objects at {backup_root}")

        # ── freeze plan (first durable write, immutable) ─────────────
        plan["lock_generation"] = snapshot_object(client, lock_uri)["generation"]
        plan["lock_present"] = False
        atomic_write(
            f"{evidence}/prepared.json",
            json.dumps(plan, indent=2),
            overwrite=True,
            if_generation_match=0,
        )
        print(f"froze plan at {evidence}/prepared.json")

        # ── phase A0: remove ALL target completion markers ───────────
        for m in plan["mutable"]:
            if m["label"] == "comp":
                delete_generation(client, m["uri"], if_generation_match=m["generation"])
        print(f"removed {n_comp} completion markers")

        # ── phase A: STAC + provenance (CAS) ─────────────────────────
        for m in plan["mutable"]:
            if m["label"] not in ("stac", "prov"):
                continue
            upload_bytes_generation(
                client, m["staged_bytes"], m["uri"], if_generation_match=m["generation"]
            )
        # ── ledger (single CAS write) ────────────────────────────────
        ledger_entry = next(m for m in plan["mutable"] if m["label"] == "ledger")
        upload_bytes_generation(
            client,
            ledger_entry["staged_bytes"],
            ledger_entry["uri"],
            if_generation_match=ledger_entry["generation"],
        )
        print("published 434 stac + 434 provenance + ledger")

        # ── core after-state verification (no completions) ───────────
        for m in plan["mutable"]:
            if m["label"] in ("stac", "prov"):
                live = read_gcs_bytes(client, m["uri"])
                if sha256_bytes(live) != m["staged_sha"]:
                    raise RuntimeError(f"after-state hash mismatch: {m['uri']}")
        ledger_live = read_gcs_bytes(client, str(cfg["ard_ledger_uri"]))
        if sha256_bytes(ledger_live) != ledger_entry["staged_sha"]:
            raise RuntimeError("after-state ledger hash mismatch")
        for uri, snap in plan["protected_snapshot"].items():
            now = snapshot_object(client, uri)
            if (
                now is None
                or now["generation"] != snap["generation"]
                or now["crc32c"] != snap["crc32c"]
                or now["size"] != snap["size"]
            ):
                raise RuntimeError(f"protected object changed during migration: {uri}")
        print("after-state verification passed (868 changed hashes + protected unchanged)")

        # ── phase B: all completion markers LAST ─────────────────────
        comp_payload = {
            "published_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "repair": True,
        }
        comp_bytes = json.dumps(comp_payload, indent=2).encode("utf-8")
        for m in plan["mutable"]:
            if m["label"] == "comp":
                atomic_write(
                    m["uri"], comp_bytes, overwrite=True, if_generation_match=0
                )
        for m in plan["mutable"]:
            if m["label"] == "comp":
                if sha256_bytes(read_gcs_bytes(client, m["uri"])) != sha256_bytes(
                    comp_bytes
                ):
                    raise RuntimeError(f"completion verification failed: {m['uri']}")
        print(f"published + verified {n_comp} completion markers")

        # ── evidence + lock release ──────────────────────────────────
        summary = {
            "migration_id": migration_id,
            "run_id": run_id,
            "git_head": head,
            "completed_at": datetime.now(UTC).isoformat(),
            "target_count": plan["target_count"],
            "mutable": len(plan["mutable"]),
            "protected": len(plan["protected_snapshot"]),
            "backup_root": backup_root,
            "evidence_root": evidence,
        }
        atomic_write(
            f"{evidence}/receipt.json",
            json.dumps(
                {**summary, "prepared": f"{evidence}/prepared.json"}, indent=2
            ),
            overwrite=True,
        )
        atomic_write(
            f"{evidence}/summary.json", json.dumps(summary, indent=2), overwrite=True
        )
        atomic_write(
            f"{evidence}/complete.json",
            json.dumps(
                {
                    "migration_id": migration_id,
                    "run_id": run_id,
                    "completed_at": summary["completed_at"],
                },
                indent=2,
            ),
            overwrite=True,
        )
        lock_snap = snapshot_object(client, lock_uri)
        if lock_snap is not None:
            if lock_snap["generation"] != plan["lock_generation"]:
                raise RuntimeError("lock generation changed since acquisition")
            delete_generation(client, lock_uri, if_generation_match=lock_snap["generation"])
        print("released migration lock")
    except BaseException:
        print(
            f"MIGRATION FAILED — lock and backup retained at {backup_root}; "
            f"restore via --restore {migration_id} --apply",
            file=__import__("sys").stderr,
        )
        raise

    print(f"\nMigration applied — {migration_id}")
    print(f"  evidence: {evidence}")
    print("  verify with scripts/validate_ard.py and a reconcile probe (0 candidates)")
    return 0


# ── restore ───────────────────────────────────────────────────────────


def run_restore(cfg: dict[str, Any], migration_id: str) -> int:
    """Restore a migration byte-identically from its evidence prefix."""
    from berlin_lst_downscaling.data.io.storage import atomic_write

    client = gcs_client()
    evidence = f"{str(cfg['evidence_prefix']).rstrip('/')}/{migration_id}"
    prepared = json.loads(read_gcs_bytes(client, f"{evidence}/prepared.json"))
    backup_root = str(prepared["backup_root"])
    lock_uri = str(cfg["lock_uri"])

    lock_snap = snapshot_object(client, lock_uri)
    if lock_snap is not None:
        payload = json.loads(read_gcs_bytes(client, lock_uri))
        if payload.get("migration_id") != migration_id:
            raise RuntimeError(
                f"active lock belongs to {payload.get('migration_id')!r}; "
                f"refusing to restore {migration_id!r}"
            )

    print(f"Restoring migration {migration_id} from {backup_root}")
    mutable = prepared["mutable"]

    # 1. Remove migrated completions first.
    for m in mutable:
        if m["label"] != "comp":
            continue
        snap = snapshot_object(client, m["uri"])
        if snap is not None:
            delete_generation(client, m["uri"], if_generation_match=snap["generation"])

    # 2. Restore STAC/prov/ledger: only when live == staged, skip when
    #    live == original, STOP on any third state.
    for m in mutable:
        if m["label"] == "comp":
            continue
        live = read_gcs_bytes(client, m["uri"])
        live_sha = sha256_bytes(live)
        if live_sha == m["original_sha"]:
            continue
        if live_sha != m["staged_sha"]:
            raise RuntimeError(
                f"{m['uri']}: live bytes match neither staged nor backup — abort restore"
            )
        backup_uri = f"{backup_root}/{m['uri'].removeprefix('gs://berlin-lst-data/')}"
        original_bytes = read_gcs_bytes(client, backup_uri)
        if sha256_bytes(original_bytes) != m["original_sha"]:
            raise RuntimeError(f"backup byte mismatch for {m['uri']}")
        snap = snapshot_object(client, m["uri"])
        upload_bytes_generation(
            client, original_bytes, m["uri"], if_generation_match=snap["generation"]
        )
        if sha256_bytes(read_gcs_bytes(client, m["uri"])) != m["original_sha"]:
            raise RuntimeError(f"restore verification failed for {m['uri']}")
        print(f"  restored {m['uri']}")

    # 3. Restore original completions last.
    for m in mutable:
        if m["label"] != "comp":
            continue
        backup_uri = f"{backup_root}/{m['uri'].removeprefix('gs://berlin-lst-data/')}"
        original_bytes = read_gcs_bytes(client, backup_uri)
        atomic_write(m["uri"], original_bytes, overwrite=True, if_generation_match=0)
        if sha256_bytes(read_gcs_bytes(client, m["uri"])) != sha256_bytes(original_bytes):
            raise RuntimeError(f"completion restore failed: {m['uri']}")

    if lock_snap is not None:
        delete_generation(client, lock_uri, if_generation_match=lock_snap["generation"])
        print("released owned migration lock")

    print("Restore complete — verify with validate_ard.")
    return 0


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.restore:
        if not args.apply:
            raise SystemExit("--restore requires --apply (explicit operation)")
        return run_restore(cfg, args.restore)
    return run_migrate(cfg, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
