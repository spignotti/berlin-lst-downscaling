"""QA report generation for secondary-data pipeline runs.

The report is both printed (via :func:`format_secondary_report`) and
persisted to ``qa/secondary/{run_id}/report.json`` (via
:func:`persist_secondary_report`) at the end of each run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from berlin_lst_downscaling.data.io import atomic_write, exists
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
from berlin_lst_downscaling.data.secondary.paths import qa_report_path


def secondary_qa_report(
    ledger: SecondaryLedger,
    run_id: str,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a QA report from the secondary ledger state.

    Returns a dict with per-source counts, artifact completeness, and
    overall status.  Callers should persist the dict via
    :func:`persist_secondary_report`.
    """
    if sources is None:
        sources = list(set(row.source for row in _all_ledger_rows(ledger)))

    per_source: dict[str, dict[str, Any]] = {}
    for src in sources:
        counts = ledger.status_counts(src)
        rows = ledger.items_for_source(src)

        output_ok = sum(
            1 for r in rows if r.status == "done" and r.output_uri and exists(r.output_uri)
        )
        output_missing = sum(
            1 for r in rows if r.status == "done" and (not r.output_uri or not exists(r.output_uri))
        )
        completed = sum(
            1 for r in rows if r.status == "done" and r.completion_uri and exists(r.completion_uri)
        )
        incomplete = sum(
            1
            for r in rows
            if r.status == "done" and (not r.completion_uri or not exists(r.completion_uri))
        )

        per_source[src] = {
            "total": len(rows),
            **counts,
            "output_exists": output_ok,
            "output_missing": output_missing,
            "completed": completed,
            "incomplete": incomplete,
        }

    failed = sum(
        per_source[s].get("failed", 0)
        + per_source[s].get("output_missing", 0)
        + per_source[s].get("incomplete", 0)
        for s in per_source
    )

    return {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "per_source": per_source,
        "total_failed": failed,
        "success": failed == 0,
    }


def persist_secondary_report(
    report: dict[str, Any],
    output_root: str,
) -> str:
    """Persist the QA report as JSON under ``qa/secondary/{run_id}/``."""
    uri = qa_report_path(output_root, report["run_id"])
    atomic_write(uri, json.dumps(report, indent=2), overwrite=True)
    return uri


def _all_ledger_rows(ledger: SecondaryLedger) -> list:
    """Return all rows from the ledger table."""
    from berlin_lst_downscaling.data.secondary.ledger import _rows_from_table

    tbl = ledger.table
    if tbl.num_rows == 0:
        return []
    return _rows_from_table(tbl)


def format_secondary_report(report: dict[str, Any]) -> str:
    """Format a secondary QA report for console output."""
    lines = [
        f"Secondary QA Report — run {report['run_id']}",
        f"  Timestamp : {report['timestamp']}",
        f"  Success   : {'yes' if report['success'] else 'NO'}",
        f"  Failed    : {report['total_failed']}",
        "",
    ]
    for src, data in report.get("per_source", {}).items():
        lines.append(f"  [{src}]")
        lines.append(f"    Total       : {data['total']}")
        for status in ("done", "failed", "exporting", "pending"):
            if data.get(status, 0) > 0:
                lines.append(f"    {status:<12}: {data[status]}")
        lines.append(f"    Output OK   : {data.get('output_exists', 0)}")
        if data.get("output_missing", 0) > 0:
            lines.append(f"    Output MISS : {data['output_missing']}")
        lines.append(f"    Completed   : {data.get('completed', 0)}")
        if data.get("incomplete", 0) > 0:
            lines.append(f"    Incomplete  : {data['incomplete']}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "secondary_qa_report",
    "persist_secondary_report",
    "format_secondary_report",
]
