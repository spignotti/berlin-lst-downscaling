"""Training-data run report types and serialisation.

Holds the per-scene and run-level report dataclasses used across the
training-data pipeline (eligibility, index, scaler) and serialises the
run report JSON under ``qa/training/<run_id>/report.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from berlin_lst_downscaling.data.io import atomic_write
from berlin_lst_downscaling.data.training.paths import qa_report_path


@dataclass
class SceneTrainingResult:
    """One per-scene row of the training run report."""

    scene_id: str
    year: int
    s2_scene_id: str
    split: str  # train | validation | test | inference
    status: str  # done | excluded | failed
    sensor: str
    eligible_cells: int | None = None
    target_valid_cells: int | None = None
    exclusion_reason: str | None = None
    error: str | None = None


@dataclass
class TrainingRunReport:
    """Complete training-data run report (serialised to report.json)."""

    run_id: str
    timestamp: str
    inputs: dict
    fingerprints: dict
    grid: dict
    policy_hash: str
    v3_config_hash: str
    total_pairings: int
    assessed: int
    processed: int
    failed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    aggregate: dict
    scenes: list[SceneTrainingResult]
    release_uris: dict[str, str] = field(default_factory=dict)
    readback: dict = field(default_factory=dict)  # publisher-side readback evidence

    @property
    def ok(self) -> bool:
        return self.failed == 0


def new_run_id() -> str:
    """Return a fresh short run ID (uuid4 hex prefix)."""
    return uuid4().hex[:8]


def now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def write_report(report: TrainingRunReport, output_root: str) -> str:
    """Persist the run report JSON under ``qa/training/<run_id>/report.json``."""
    uri = qa_report_path(output_root, report.run_id)
    payload = {
        "pipeline": "training-data",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "ok": report.ok,
        "inputs": report.inputs,
        "fingerprints": report.fingerprints,
        "grid": report.grid,
        "policy_hash": report.policy_hash,
        "v3_config_hash": report.v3_config_hash,
        "scenes": {
            "total_pairings": report.total_pairings,
            "assessed": report.assessed,
            "processed": report.processed,
            "failed": report.failed,
            "excluded": report.excluded,
            "exclusion_reasons": report.exclusion_reasons,
        },
        "aggregate": report.aggregate,
        "readback": report.readback,
        "release_uris": report.release_uris,
        "scene_results": [
            {
                "scene_id": s.scene_id,
                "year": s.year,
                "s2_scene_id": s.s2_scene_id,
                "split": s.split,
                "status": s.status,
                "sensor": s.sensor,
                "eligible_cells": s.eligible_cells,
                "target_valid_cells": s.target_valid_cells,
                "exclusion_reason": s.exclusion_reason,
                "error": s.error,
            }
            for s in report.scenes
        ],
    }
    atomic_write(uri, json.dumps(payload, indent=2, default=str), overwrite=True)
    return uri


__all__ = [
    "SceneTrainingResult",
    "TrainingRunReport",
    "new_run_id",
    "now_iso",
    "write_report",
]
