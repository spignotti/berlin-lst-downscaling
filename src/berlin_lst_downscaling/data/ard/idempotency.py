"""Idempotency reconciliation — decide which scenes to process vs. skip.

``reconcile`` compares a scene list against the ledger and returns
the subset of scenes that actually need processing.
"""

from __future__ import annotations

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.ard.ledger import Ledger

# ── public API ───────────────────────────────────────────────────────


def reconcile(
    scenes: list[tuple[str, str, int]],  # (scene_id, source, year)
    ledger: Ledger,
    contract: Contract,
) -> list[tuple[str, str, int, str]]:
    """Return the subset of scenes that need processing.

    Each entry in the returned list is ``(scene_id, source, year, reason)``
    where ``reason`` is one of ``"new"``, ``"retry"``, ``"interrupted"``,
    ``"schema_changed"``.

    Scenes that already have a matching schema hash and status ``done``
    are excluded (skip).
    """
    result: list[tuple[str, str, int, str]] = []
    contract_hash = contract.schema_hash()

    for scene_id, source, year in scenes:
        row = ledger.get(scene_id, source)

        if row is None:
            # Not in ledger at all → always process
            result.append((scene_id, source, year, "new"))
            continue

        if row.status == "done" and row.schema_hash == contract_hash:
            # Already successfully processed with matching contract → skip
            continue

        if row.status == "done" and row.schema_hash != contract_hash:
            # Contract changed → force reprocessing
            result.append((scene_id, source, year, "schema_changed"))

        elif row.status == "failed":
            # Previous failure → retry
            result.append((scene_id, source, year, "retry"))

        elif row.status == "exporting":
            # Crashed mid-export → retry
            result.append((scene_id, source, year, "interrupted"))

        elif row.status == "pending":
            # Never started → process
            result.append((scene_id, source, year, "new"))

        elif row.status == "skipped":
            # Explicitly skipped — keep as-is
            continue

    return result


def status_summary(ledger: Ledger, source: str) -> dict[str, int]:
    """Return ``{status: count}`` for a given source."""
    return ledger.status_counts(source)


__all__ = [
    "reconcile",
    "status_summary",
]
