"""Idempotency reconciliation — decide which scenes to process vs. skip.

``reconcile`` compares a scene list against the ledger and returns
the subset of scenes that actually need processing.
"""

from __future__ import annotations

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow

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

    Scenes that already have a matching schema hash, status ``done``,
    **and** confirmed file existence are excluded (skip).
    """
    result: list[tuple[str, str, int, str]] = []
    version = contract.schema_version

    for scene_id, source, year in scenes:
        row = ledger.get(scene_id, source)

        if row is None:
            # Not in ledger at all → always process
            result.append((scene_id, source, year, "new"))
            continue

        if row.status == "done" and row.schema_version == version:
            # Verify output files actually exist (T8)
            if _files_exist(row):
                continue
            # Files missing → treat as interrupted (reprocess)
            result.append((scene_id, source, year, "interrupted"))
            continue

        if row.status == "done" and row.schema_version != version:
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




def _files_exist(row: LedgerRow) -> bool:
    """Check that all expected output files exist for a done scene.

    Returns ``True`` if all files are present, ``False`` if any are
    missing (which triggers reprocessing).
    Uses ``exists`` from ``berlin_lst_downscaling.data.io`` which
    dispatches by URI scheme (local, GCS, mounted).
    """
    from berlin_lst_downscaling.data.io import exists

    if row.path_cog and not exists(row.path_cog):
        return False
    if row.path_stac and not exists(row.path_stac):
        return False
    return True


__all__ = [
    "reconcile",
]
