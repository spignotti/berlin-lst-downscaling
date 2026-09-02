# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Scene feature-stack pipeline runner (Hydra-driven).

Usage
-----
    # Smoke: one deterministic pair, bounded bbox, local ephemeral output
    uv run python scripts/runners/run_features.py --config-name smoke

    # Full: the published 2017-2025 training universe (324 anchors)
    uv run python scripts/runners/run_features.py --config-name full

Exits non-zero when any assessable scene fails to publish (fail-closed).
Excluded scenes (2026 role=inference) are reported, never processed.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.features.pipeline import run_features, write_report
from berlin_lst_downscaling.data.io import RunLogSession, log_event

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/features", config_name="full", version_base=None)
def main(cfg: DictConfig) -> None:
    """Dispatch to the feature-stack pipeline and persist the run report.

    Hydra 1.3 discards a decorated task's return value in CLI mode, so a
    failed report is signalled with an explicit ``SystemExit(1)`` rather
    than a returned exit code (which would silently become process exit 0).
    """
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="features", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            output_root=output_root,
            manifest_uri=str(cfg.manifest_uri),
            bbox=list(cfg.get("bbox", []) or []),
            scene_ids=list(cfg.get("scene_ids", []) or []),
        )
        report = run_features(cfg, run_id=run_id)
        report_uri = write_report(report, output_root)

        print(f"Feature stacks — run {run_id}")
        print(
            f"  Pairings: {report.total_pairings} | assessed: {report.assessed} | "
            f"processed: {report.processed} | failed: {report.failed} | "
            f"excluded: {report.excluded}"
        )
        print(
            f"  Feature-valid px : {report.aggregate_coverage['feature_valid_px']} "
            f"(inside AOI: {report.aggregate_coverage['inside_aoi_px']}, "
            f"outside: {report.aggregate_coverage['outside_aoi_px']})"
        )
        for result in report.scenes:
            if result.status == "failed":
                print(f"    ✗ {result.scene_id}: {result.error}")
        print(f"Report: {report_uri}")

        if not report.ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
