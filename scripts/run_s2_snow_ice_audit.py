#!/usr/bin/env python3
"""Read-only S2 snow/ice audit runner (WB2c-2 QA Gate raw, Stufe 1).

Quantifies how many SCL=11 (snow/ice) pixels in the published S2 ARD
data are currently unflagged, by comparing source SCL against the
published flag COG per unique paired scene. Never writes ARD data.

Usage::

    # Full audit against canonical GCS roots
    uv run python scripts/run_s2_snow_ice_audit.py --config-name s2_snow_ice

    # One-scene smoke against the canonical manifest, local output
    uv run python scripts/run_s2_snow_ice_audit.py --config-name s2_snow_ice \
        max_scenes=1 output_root=data/smoke/qa/s2-snow-ice
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io.run_logging import RunLogSession, log_event
from berlin_lst_downscaling.data.qa.s2_snow_ice import run_s2_snow_ice_audit

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/qa", config_name="s2_snow_ice", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to the S2 snow/ice audit."""
    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri:
        raise SystemExit(
            "manifest_uri is required — provide the published bundle, e.g.\n"
            "  manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet"
        )
    ard_ledger_uri = cfg.get("ard_ledger_uri")
    if not ard_ledger_uri:
        raise SystemExit(
            "ard_ledger_uri is required — provide the published ARD ledger, e.g.\n"
            "  ard_ledger_uri=gs://berlin-lst-data/ard/full/.../ledger.parquet"
        )
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)
    max_scenes = cfg.get("max_scenes")
    max_scenes_int = int(max_scenes) if max_scenes is not None else None

    with RunLogSession(output_root, pipeline="qa-s2-snow-ice", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "audit_started",
            manifest_uri=manifest_uri,
            ard_ledger_uri=ard_ledger_uri,
            output_root=output_root,
            max_scenes=max_scenes_int,
        )
        try:
            result = run_s2_snow_ice_audit(
                manifest_uri=manifest_uri,
                ard_ledger_uri=ard_ledger_uri,
                aoi_mask_uri=str(cfg.aoi_mask_uri),
                output_root=output_root,
                run_id=run_id,
                max_scenes=max_scenes_int,
            )
        except Exception as exc:
            log_event(_logger, logging.ERROR, "audit_failed", error=str(exc))
            raise

        summary = result.summary
        log_event(
            _logger,
            logging.INFO,
            "audit_completed",
            scenes_total=summary["scenes_total"],
            scenes_compared=summary["scenes_compared"],
            scenes_failed=summary["scenes_failed"],
            scenes_with_unflagged_scl11=summary["scenes_with_unflagged_scl11"],
            scl11_px=summary["scl11_px"],
            scl11_unflagged_px=summary["scl11_unflagged_px"],
            overall_unflagged_of_scl11_frac=summary["overall_unflagged_of_scl11_frac"],
        )

        if summary["scenes_failed"] > 0:
            log_event(
                _logger,
                logging.WARNING,
                "audit_incomplete",
                scenes_failed=summary["scenes_failed"],
            )
            return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
