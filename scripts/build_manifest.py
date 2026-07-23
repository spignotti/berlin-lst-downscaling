"""Szenen-Selektion & Kopplung — ARD manifest builder (Hydra adapter).

Thin adapter over :func:`berlin_lst_downscaling.data.selection.pipeline.run_couple`.
The library function owns the orchestration (anchors → S2 candidates →
coupling → ECOSTRESS → bundle write) plus checkpointing and telemetry.

Usage
-----
    uv run python scripts/build_manifest.py \
        --config-dir configs/selection \
        --config-name full_2017_2026
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.data.selection.pipeline import run_couple

_logger = logging.getLogger(__name__)


def main(cfg: DictConfig) -> int:
    """Dispatch to the selection library."""
    output_root = cfg.get("output_root")
    if not output_root:
        raise SystemExit(
            "output_root is required (immutable bundle root, override per run)"
        )
    run_id = uuid4().hex[:8]
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)
    with RunLogSession(str(output_root), pipeline="selection", run_id=run_id, level=level):
        return run_couple(cfg)


if __name__ == "__main__":
    @hydra.main(config_path="../configs/selection", config_name="full_2017_2026", version_base=None)
    def _hydra_main(cfg: DictConfig) -> int:
        return main(cfg)

    raise SystemExit(_hydra_main())
