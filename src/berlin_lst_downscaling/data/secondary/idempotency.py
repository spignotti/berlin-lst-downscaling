"""Idempotency reconciliation for secondary-data items.

``reconcile`` compares a work-item list against the secondary ledger and
returns the subset that actually need processing.

A "done" row is considered complete only when its ``completion_uri``
exists and resolves to a real artifact.  This guards against partial
publication: GCS cannot publish multiple blobs atomically, so the
``complete.json`` marker is the final visibility gate.
"""

from __future__ import annotations

from berlin_lst_downscaling.data.io import exists
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger


def reconcile(
    items: list[tuple[str, str, str]],  # (item_id, source, period)
    ledger: SecondaryLedger,
    config_hash: str,
) -> list[tuple[str, str, str, str]]:
    """Return the subset of items that need processing.

    Each entry in the returned list is ``(item_id, source, period, reason)``
    where ``reason`` is one of ``"new"``, ``"retry"``, ``"interrupted"``,
    ``"config_changed"``, or ``"incomplete"``.

    Items that already have ``status='done'``, a matching ``config_hash``,
    a confirmed output file, **and** a confirmed completion marker are
    excluded (skip).
    """
    result: list[tuple[str, str, str, str]] = []

    for item_id, source, period in items:
        row = ledger.get(item_id, source, period)

        if row is None:
            result.append((item_id, source, period, "new"))
            continue

        if row.status == "done" and row.config_hash == config_hash:
            output_ok = row.output_uri and exists(row.output_uri)
            completion_ok = row.completion_uri and exists(row.completion_uri)
            if output_ok and completion_ok:
                continue
            if output_ok and not completion_ok:
                # COG is present but the publication marker is missing —
                # the product was not finalised.  Re-finalise it.
                result.append((item_id, source, period, "incomplete"))
                continue
            result.append((item_id, source, period, "interrupted"))
            continue

        if row.status == "done" and row.config_hash != config_hash:
            result.append((item_id, source, period, "config_changed"))
        elif row.status == "failed":
            result.append((item_id, source, period, "retry"))
        elif row.status == "exporting":
            result.append((item_id, source, period, "interrupted"))
        elif row.status == "pending":
            result.append((item_id, source, period, "new"))
        # ``skipped`` — keep as-is

    return result

__all__ = ["reconcile"]