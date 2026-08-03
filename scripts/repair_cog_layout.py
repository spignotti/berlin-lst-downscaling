#!/usr/bin/env python3
"""COG layout repair CLI.

Provides snapshot, prove, apply, verify, and rollback commands for
repairing systemic COG layout issues in GCS.

Usage::

    # Snapshot inventory
    uv run python scripts/repair_cog_layout.py snapshot \\
        --config configs/profiling/full.yaml \\
        --state /tmp/cog-repair/state.parquet

    # Prove Cogger on pilot COGs
    uv run python scripts/repair_cog_layout.py prove \\
        --state /tmp/cog-repair/state.parquet \\
        --engine cogger --cogger-bin /usr/local/bin/cogger \\
        --uri gs://berlin-lst-data/ard/.../LC08_.../LC08_....tif

    # Apply repair to all
    uv run python scripts/repair_cog_layout.py apply \\
        --state /tmp/cog-repair/state.parquet \\
        --engine cogger --cogger-bin /usr/local/bin/cogger \\
        --audit-root gs://berlin-lst-data/_maintenance/cog-layout/20260803

    # Verify repaired COGs
    uv run python scripts/repair_cog_layout.py verify \\
        --state /tmp/cog-repair/state.parquet --all

    # Rollback
    uv run python scripts/repair_cog_layout.py rollback \\
        --state /tmp/cog-repair/state.parquet \\
        --audit-root gs://berlin-lst-data/_maintenance/cog-layout/20260803
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from uuid import uuid4

import pyarrow as pa

from berlin_lst_downscaling.data.ard.cog_repair import (
    apply_repair,
    build_inventory,
    build_inventory_from_ledgers,
    load_table,
    prove_cogger,
    rollback_repair,
    save_table,
    verify_repair,
)

_logger = logging.getLogger(__name__)


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Create frozen inventory."""
    from berlin_lst_downscaling.data.io.run_logging import RunLogSession, log_event
    with RunLogSession(str(Path(args.state).parent), pipeline="cog-repair", run_id=uuid4().hex[:8]):
        log_event(_logger, logging.INFO, "snapshot_started")

        table = build_inventory_from_ledgers(
            manifest_uri="gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet",
            ard_ledger_uri="gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet",
            static_sources_root="gs://berlin-lst-data/static/sources/full",
            static_derived_root="gs://berlin-lst-data/static/derived/full",
            dynamic_full_root="gs://berlin-lst-data/dynamic/full",
            dynamic_inference_root="gs://berlin-lst-data/dynamic/inference/2026",
        )

        if args.expect_count is not None:
            if table.num_rows != args.expect_count:
                print(f"FAIL: Expected {args.expect_count} URIs, got {table.num_rows}")
                return 1

        save_table(table, args.state)
        print(f"Saved inventory: {table.num_rows} URIs → {args.state}")
        print(f"  Data: {sum(1 for r in table.to_pylist() if r['asset_kind'] == 'data')}")
        print(f"  Flags: {sum(1 for r in table.to_pylist() if r['asset_kind'] == 'flag')}")
        return 0


def cmd_prove(args: argparse.Namespace) -> int:
    """Run Cogger proof on pilot COGs."""
    table = load_table(args.state)

    if args.uri:
        table = prove_cogger(table, args.cogger_bin, uri=args.uri)
    elif args.all_layout_signatures:
        table = prove_cogger(table, args.cogger_bin, all_layout_signatures=True)
    else:
        print("Specify --uri or --all-layout-signatures")
        return 1

    save_table(table, args.state)

    proved = sum(1 for r in table.to_pylist() if r["status"] == "proved")
    failed = sum(1 for r in table.to_pylist() if r["status"] == "failed")
    print(f"Proved: {proved}, Failed: {failed}")
    return 0 if failed == 0 else 1


def cmd_apply(args: argparse.Namespace) -> int:
    """Apply repair to selected COGs."""
    table = load_table(args.state)

    table, audit = apply_repair(
        table,
        args.cogger_bin,
        args.audit_root,
        uri=args.uri,
        sources=args.sources,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )

    save_table(table, args.state)
    if args.audit_root and audit.num_rows > 0:
        audit_uri = f"{args.audit_root.rstrip('/')}/audit.parquet"
        save_table(audit, audit_uri)

    repaired = sum(1 for r in table.to_pylist() if r["status"] == "repaired")
    failed = sum(1 for r in table.to_pylist() if r["status"] == "failed")
    print(f"Repaired: {repaired}, Failed: {failed}")
    return 0 if failed == 0 else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify repaired COGs are strict-clean."""
    table = load_table(args.state)

    table, errors = verify_repair(
        table,
        uri=args.uri,
        all=args.all,
    )

    save_table(table, args.state)

    verified = sum(1 for r in table.to_pylist() if r["status"] == "verified")
    failed = sum(1 for r in table.to_pylist() if r["status"] == "failed")

    if errors:
        print("Verification errors:")
        for e in errors:
            print(f"  {e}")

    print(f"Verified: {verified}, Failed: {failed}")
    return 0 if failed == 0 else 1


def cmd_rollback(args: argparse.Namespace) -> int:
    """Rollback repaired COGs."""
    table = load_table(args.state)
    if args.audit_root:
        audit_uri = f"{args.audit_root.rstrip('/')}/audit.parquet"
        audit = load_table(audit_uri)
    else:
        audit = pa.table({
            "uri": [],
            "old_generation": [],
            "new_generation": [],
            "repaired_crc32c": [],
            "verified": [],
            "timestamp": [],
        })

    table, audit = rollback_repair(
        table,
        audit,
        status=args.status,
        workers=args.workers,
    )

    save_table(table, args.state)
    if args.audit_root:
        audit_uri = f"{args.audit_root.rstrip('/')}/audit.parquet"
        save_table(audit, audit_uri)

    rolled_back = sum(1 for r in table.to_pylist() if r["status"] == "rolled_back")
    print(f"Rolled back: {rolled_back}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="COG layout repair CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # snapshot
    p_snap = subparsers.add_parser("snapshot")
    p_snap.add_argument("--config", default="configs/profiling/full.yaml")
    p_snap.add_argument("--state", required=True)
    p_snap.add_argument("--expect-count", type=int, default=None)

    # prove
    p_prove = subparsers.add_parser("prove")
    p_prove.add_argument("--state", required=True)
    p_prove.add_argument("--engine", default="cogger")
    p_prove.add_argument("--cogger-bin", default="cogger")
    p_prove.add_argument("--uri", default=None)
    p_prove.add_argument("--all-layout-signatures", action="store_true")

    # apply
    p_apply = subparsers.add_parser("apply")
    p_apply.add_argument("--state", required=True)
    p_apply.add_argument("--audit-root", default=None)
    p_apply.add_argument("--engine", default="cogger")
    p_apply.add_argument("--cogger-bin", default="cogger")
    p_apply.add_argument("--uri", default=None)
    p_apply.add_argument("--sources", default=None)
    p_apply.add_argument("--workers", type=int, default=1)
    p_apply.add_argument("--checkpoint-every", type=int, default=None)

    # verify
    p_verify = subparsers.add_parser("verify")
    p_verify.add_argument("--state", required=True)
    p_verify.add_argument("--uri", default=None)
    p_verify.add_argument("--all", action="store_true")

    # rollback
    p_rollback = subparsers.add_parser("rollback")
    p_rollback.add_argument("--state", required=True)
    p_rollback.add_argument("--audit-root", default=None)
    p_rollback.add_argument("--status", default="repaired")
    p_rollback.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    commands = {
        "snapshot": cmd_snapshot,
        "prove": cmd_prove,
        "apply": cmd_apply,
        "verify": cmd_verify,
        "rollback": cmd_rollback,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
