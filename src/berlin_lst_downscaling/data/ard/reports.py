"""QA report — ledger summary + file existence checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.ledger import Ledger


def qa_report(
    ledger: Ledger,
    cfg: DictConfig,
    run_id: str,
) -> dict[str, Any]:
    """Generate a QA report from the ledger state and COG/STAC file checks.

    Returns a dict with per-source counts and overall status. Writes a
    JSON report to ``cfg.qa_dir / f"qa_{run_id}.json"``.
    """
    sources: list[str] = list(cfg.get("sources", []))

    per_source: dict[str, dict[str, Any]] = {}
    for src in sources:
        counts = ledger.status_counts(src)
        rows = ledger.scenes_for_source(src)

        # Check file existence for done scenes
        cog_ok = 0
        stac_ok = 0
        cog_missing = 0
        stac_missing = 0
        for r in rows:
            if r.status != "done":
                continue
            if r.path_cog and Path(r.path_cog).exists():
                cog_ok += 1
            else:
                cog_missing += 1
            if r.path_stac and Path(r.path_stac).exists():
                stac_ok += 1
            else:
                stac_missing += 1

        # Flag COG existence (optional — no ledger field yet)
        flag_missing = 0
        for r in rows:
            if r.status != "done" or not r.path_cog:
                continue
            flag_path = Path(r.path_cog).with_suffix(".flag.tif")
            if not flag_path.exists():
                flag_missing += 1

        per_source[src] = {
            "total": len(rows),
            **counts,
            "cog_exists": cog_ok,
            "cog_missing": cog_missing,
            "flag_missing": flag_missing,
            "stac_exists": stac_ok,
            "stac_missing": stac_missing,
        }

    failed = sum(
        per_source[s].get("failed", 0) + per_source[s].get("cog_missing", 0)
        for s in (sources or per_source)
        if s in per_source
    )

    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "per_source": per_source,
        "total_failed": failed,
        "success": failed == 0,
    }

    # Write to file
    qa_dir = Path(cfg.get("qa_dir", cfg.output_root))
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / f"qa_{run_id}.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


__all__ = [
    "qa_report",
]
