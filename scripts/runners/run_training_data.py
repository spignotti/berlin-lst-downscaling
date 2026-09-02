# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""WB2c-4 training-data pipeline runner (Hydra-driven).

Usage
-----
    # Smoke: one deterministic scene per temporal split, bounded bbox,
    # local ephemeral output
    uv run python scripts/runners/run_training_data.py --config-name smoke

    # Full: all 324 assessable anchor scenes, canonical release to GCS
    uv run python scripts/runners/run_training_data.py --config-name full

Exits non-zero when any assessable scene fails to publish (fail-closed).
2026 inference scenes are metadata-only (deferred) and never processed.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.training.pipeline import run_training_data
from berlin_lst_downscaling.data.training.report import write_report

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/training", config_name="full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Dispatch to the training-data pipeline and persist the run report."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="training-data", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            output_root=output_root,
            features_root=str(cfg.features_root),
            bbox=list(cfg.get("bbox", []) or []),
            scene_ids=list(cfg.get("scene_ids", []) or []),
        )
        report = run_training_data(cfg, run_id=run_id)
        report_uri = write_report(report, output_root)

        print(f"Training data — run {run_id}")
        print(
            f"  Pairings: {report.total_pairings} | assessed: {report.assessed} | "
            f"processed: {report.processed} | failed: {report.failed} | "
            f"excluded: {report.excluded}"
        )
        print(
            f"  Eligible 100m cells : {report.aggregate['eligible_cells']} "
            f"(target-valid: {report.aggregate['target_valid_cells']})"
        )
        for reason, count in sorted(report.exclusion_reasons.items()):
            print(f"  Excluded [{reason}]: {count}")
        for scene in report.scenes:
            if scene.status == "failed":
                print(f"    ✗ {scene.scene_id}: {scene.error}")
        print(f"Policy hash : {report.policy_hash}")
        print(f"V3 config   : {report.v3_config_hash}")
        for label, uri in sorted(report.release_uris.items()):
            print(f"  {label:<16}: {uri}")
        print(f"Report: {report_uri}")

        # Hydra 1.3.4 discards the decorated task's return value, so
        # ``raise SystemExit(main())`` would exit 0 even on failures.
        # Raise inside the task to make failures propagate as a non-zero
        # process exit (the VM wrapper captures this exit status).
        if not report.ok:
            raise SystemExit(1)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
