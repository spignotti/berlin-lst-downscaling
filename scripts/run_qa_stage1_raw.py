# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage-1 raw-input QA gate runner (Hydra-driven).

Usage
-----
    # Smoke: one deterministic pair, bounded bbox, local ephemeral output
    uv run python scripts/run_qa_stage1_raw.py --config-name stage1_raw_smoke

    # Full: the published 2017-2025 training universe, evidence to GCS
    uv run python scripts/run_qa_stage1_raw.py --config-name stage1_raw_full

Exits non-zero when the gate finds contract/range/input faults in the
training universe (fail-closed). Exclusions (e.g. 2026 inference scenes)
are reported, never failures.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.qa.stage1_raw import (
    run_stage1_raw,
    write_report,
)

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/qa", config_name="stage1_raw_full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Dispatch to the Stage-1 raw QA gate and persist the report bundle."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="qa-stage1-raw", run_id=run_id, level=level):
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
        report = run_stage1_raw(cfg, run_id=run_id)
        uris = write_report(report, output_root)

        print(f"Stage-1 raw QA — run {run_id}")
        print(
            f"  Pairings: {report.total_pairings} | assessed: {report.assessed} | "
            f"excluded: {report.excluded}"
        )
        print(f"  Target-valid 100m cells : {report.aggregate['target_valid_cells']}")
        print(f"  All-100 support cells   : {report.aggregate['all_100_cells']}")
        print(f"  Full-support cells      : {report.aggregate['full_support_cells']}")
        print(f"  Findings                : {len(report.findings)}")
        for finding in report.findings:
            print(f"    ✗ {finding}")
        print("Artifacts:")
        for label, uri in uris.items():
            print(f"  {label:<14}: {uri}")

        return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
