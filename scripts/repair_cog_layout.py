#!/usr/bin/env python3
"""COG layout recovery CLI.

Provides fail-closed recovery commands for repairing systemic COG layout
issues in GCS with immutable per-object events, guarded mutations, and
durable evidence.

Usage::

    # Preflight — verify bucket policy, IAM, and inventory
    uv run python scripts/repair_cog_layout.py preflight \\
        --config configs/cog_repair/remediation.yaml

    # Rebaseline — build inventory, Soft Delete catalog, run manifest
    uv run python scripts/repair_cog_layout.py rebaseline \\
        --config configs/cog_repair/remediation.yaml \\
        --run-id <run-id>

    # Capture originals — preserve pre-incident generations
    uv run python scripts/repair_cog_layout.py capture-originals \\
        --config configs/cog_repair/remediation.yaml \\
        --run-id <run-id> [--execute]

    # Stage — generate repair candidates
    uv run python scripts/repair_cog_layout.py stage \\
        --config configs/cog_repair/remediation.yaml \\
        --run-id <run-id> [--execute]

    # Promote — guarded canonical promotion with rollback
    uv run python scripts/repair_cog_layout.py promote \\
        --config configs/cog_repair/remediation.yaml \\
        --run-id <run-id> [--execute]

    # Verify — independent verification of all canonical assets
    uv run python scripts/repair_cog_layout.py verify-recovery \\
        --config configs/cog_repair/remediation.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

_logger = logging.getLogger(__name__)


def _reject_legacy_command(command: str) -> int:
    """Reject legacy commands that lack fail-closed guarantees."""
    print(
        f"ERROR: Command '{command}' has been migrated to the fail-closed "
        f"recovery workflow.\n\n"
        f"Use the new commands:\n"
        f"  preflight         — verify bucket policy and inventory\n"
        f"  rebaseline        — build inventory, catalog, and run manifest\n"
        f"  capture-originals — preserve pre-incident generations\n"
        f"  stage             — generate repair candidates\n"
        f"  promote           — guarded canonical promotion\n"
        f"  verify-recovery   — independent verification\n\n"
        f"See docs/cog-layout-recovery.md for the runbook.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="COG layout recovery CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preflight
    p_preflight = subparsers.add_parser(
        "preflight", help="Verify bucket policy and inventory",
    )
    p_preflight.add_argument("--config", required=True)
    p_preflight.add_argument("--recovery-root", required=True)

    # rebaseline
    p_rebaseline = subparsers.add_parser(
        "rebaseline", help="Build inventory, catalog, and run manifest",
    )
    p_rebaseline.add_argument("--config", required=True)
    p_rebaseline.add_argument("--recovery-root", required=True)
    p_rebaseline.add_argument("--run-id", required=True)

    # capture-originals
    p_capture = subparsers.add_parser(
        "capture-originals", help="Preserve pre-incident generations",
    )
    p_capture.add_argument("--config", required=True)
    p_capture.add_argument("--recovery-root", required=True)
    p_capture.add_argument("--run-id", required=True)
    p_capture.add_argument("--execute", action="store_true", default=False)

    # stage
    p_stage = subparsers.add_parser(
        "stage", help="Generate repair candidates",
    )
    p_stage.add_argument("--config", required=True)
    p_stage.add_argument("--recovery-root", required=True)
    p_stage.add_argument("--run-id", required=True)
    p_stage.add_argument("--cogger-bin", default="cogger")
    p_stage.add_argument("--execute", action="store_true", default=False)

    # promote
    p_promote = subparsers.add_parser(
        "promote", help="Guarded canonical promotion",
    )
    p_promote.add_argument("--config", required=True)
    p_promote.add_argument("--recovery-root", required=True)
    p_promote.add_argument("--run-id", required=True)
    p_promote.add_argument("--execute", action="store_true", default=False)

    # verify-recovery
    p_verify = subparsers.add_parser(
        "verify-recovery", help="Independent verification",
    )
    p_verify.add_argument("--config", required=True)
    p_verify.add_argument("--workers", type=int, default=4)

    # legacy commands
    for cmd in ("snapshot", "prove", "apply", "verify", "rollback"):
        subparsers.add_parser(cmd, help="DEPRECATED")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    # migration guard
    legacy_commands = {"snapshot", "prove", "apply", "verify", "rollback"}
    if args.command in legacy_commands:
        return _reject_legacy_command(args.command)

    from berlin_lst_downscaling.data.ard.cog_recovery import (
        cmd_capture_originals,
        cmd_preflight,
        cmd_promote,
        cmd_rebaseline,
        cmd_stage_candidates,
        cmd_verify_recovery,
    )

    dispatch = {
        "preflight": lambda: cmd_preflight(
            args.config,
            recovery_root=args.recovery_root,
        ),
        "rebaseline": lambda: cmd_rebaseline(
            args.config,
            recovery_root=args.recovery_root,
            run_id=args.run_id,
        ),
        "capture-originals": lambda: cmd_capture_originals(
            args.config,
            recovery_root=args.recovery_root,
            run_id=args.run_id,
            dry_run=not args.execute,
        ),
        "stage": lambda: cmd_stage_candidates(
            args.config,
            recovery_root=args.recovery_root,
            run_id=args.run_id,
            cogger_bin=args.cogger_bin,
            dry_run=not args.execute,
        ),
        "promote": lambda: cmd_promote(
            args.config,
            recovery_root=args.recovery_root,
            run_id=args.run_id,
            dry_run=not args.execute,
        ),
        "verify-recovery": lambda: cmd_verify_recovery(
            args.config,
            workers=args.workers,
        ),
    }

    return dispatch[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
