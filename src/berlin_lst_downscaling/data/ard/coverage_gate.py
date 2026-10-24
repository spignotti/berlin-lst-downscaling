"""Shared Sentinel-2 mosaic AOI coverage-gate exclusion classifier.

The ARD pipeline fails an S2 scene whose pre-cloud SCL footprint misses
more than ``max_missing_aoi_fraction`` of the Berlin AOI (default 0.005)
and records the ledger row as ``status="failed"`` with a
``"Sentinel-2 mosaic AOI coverage failed: missing ..."`` ``last_error``.
No product is written and no ledger schema change is involved.

Validators treat such rows as explicit, documented *coverage-gate
exclusions* — reported with scene id and missing fraction — rather than
as data bugs. Every other non-done row (real processing failures, and any
non-done Landsat/ECOSTRESS row) still fails validation.
"""

from __future__ import annotations

import re

GATE_MARKER = "Sentinel-2 mosaic AOI coverage failed"

_GATE_RE = re.compile(
    r"Sentinel-2 mosaic AOI coverage failed:\s*missing\s*(?P<missing>\d+)\s*/\s*"
    r"(?P<total>\d+)\s*pixels\s*\((?P<fraction>[0-9.eE+-]+)\)\s*>\s*"
    r"max_missing_aoi_fraction=(?P<threshold>[0-9.eE+-]+)",
)


def parse_coverage_gate_error(last_error: str | None) -> dict | None:
    """Parse a coverage-gate ``last_error`` into numbers, or None.

    Returns ``{"aoi_missing_px", "aoi_total_px", "missing_aoi_fraction",
    "max_missing_aoi_fraction", "item_ids"}``. ``item_ids`` is best-effort
    (parsed from the trailing ``item_ids=[...]`` when present, else None).
    """
    if not last_error or GATE_MARKER not in last_error:
        return None
    match = _GATE_RE.search(last_error)
    if match is None:
        return None
    try:
        missing = int(match.group("missing"))
        total = int(match.group("total"))
        fraction = float(match.group("fraction"))
        threshold = float(match.group("threshold"))
    except (TypeError, ValueError):
        return None
    item_ids: list[str] | None = None
    items_match = re.search(r"item_ids=\[(?P<ids>[^\]]*)\]", last_error)
    if items_match is not None:
        raw = items_match.group("ids")
        item_ids = [part.strip().strip("'\"") for part in raw.split(",") if part.strip()]
    return {
        "aoi_missing_px": missing,
        "aoi_total_px": total,
        "missing_aoi_fraction": fraction,
        "max_missing_aoi_fraction": threshold,
        "item_ids": item_ids,
    }


def is_coverage_gate_exclusion(source: str, status: str, last_error: str | None) -> bool:
    """Return True for an S2 non-done row failed by the coverage gate."""
    return (
        source == "sentinel-2-l2a"
        and status != "done"
        and parse_coverage_gate_error(last_error) is not None
    )


def collect_gate_exclusions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ledger rows into ``(exclusions, real_failures)``.

    Each row dict needs ``scene_id``, ``source``, ``status``, and
    ``last_error`` keys. ``exclusions`` carries the parsed gate numbers;
    ``real_failures`` carries every other non-done row.
    """
    exclusions: list[dict] = []
    real_failures: list[dict] = []
    for row in rows:
        status = str(row.get("status"))
        if status == "done":
            continue
        parsed = parse_coverage_gate_error(row.get("last_error"))
        if str(row.get("source")) == "sentinel-2-l2a" and parsed is not None:
            exclusions.append({**row, **parsed})
        else:
            real_failures.append(row)
    return exclusions, real_failures


def excluded_s2_scene_ids(ledger_table) -> list[str]:
    """Return sorted S2 scene ids failed by the coverage gate."""
    cols = ledger_table.to_pydict()
    out: list[str] = []
    for i in range(ledger_table.num_rows):
        if is_coverage_gate_exclusion(
            str(cols["source"][i]),
            str(cols["status"][i]),
            cols.get("last_error", [None])[i],
        ):
            out.append(str(cols["scene_id"][i]))
    return sorted(out)


def excluded_pairings(pairings_table, excluded_s2_ids: set[str]) -> list[dict]:
    """Return pairings whose S2 partner is a gate exclusion.

    Each entry is ``{"landsat_scene_id", "sentinel2_scene_id"}``, sorted by
    Landsat scene id. Conservative rule: excluded pairings drop out; no
    replacement S2 partner is picked.
    """
    cols = pairings_table.to_pydict()
    out: list[dict] = []
    for i in range(pairings_table.num_rows):
        s2_id = str(cols["sentinel2_scene_id"][i])
        if s2_id in excluded_s2_ids:
            out.append(
                {
                    "landsat_scene_id": str(cols["landsat_scene_id"][i]),
                    "sentinel2_scene_id": s2_id,
                }
            )
    out.sort(key=lambda row: row["landsat_scene_id"])
    return out


__all__ = [
    "GATE_MARKER",
    "collect_gate_exclusions",
    "excluded_pairings",
    "excluded_s2_scene_ids",
    "is_coverage_gate_exclusion",
    "parse_coverage_gate_error",
]
