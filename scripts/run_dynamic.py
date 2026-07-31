# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Dynamic scene pipeline entry point — Hydra-driven.

Usage
-----
    # Local smoke test (requires local static smoke products + CDS access)
    uv run python scripts/run_dynamic.py --config-name smoke \
        manifest_uri=data/ard/manifests/v3/...-r2/manifest.parquet

    # Full run on VM
    #   manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet
    uv run python scripts/run_dynamic.py --config-name full \
        output_root=gs://berlin-lst-data/dynamic/full
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.dynamic.pipeline import run_dynamic
from berlin_lst_downscaling.data.io import RunLogSession

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/dynamic", config_name="full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to the Dynamic pipeline."""
    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri:
        raise SystemExit(
            "manifest_uri is required — provide the published bundle, e.g.\n"
            "  manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet"
        )
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    # Acquire GCS run guard
    from berlin_lst_downscaling.data.dynamic.run_guard import (
        acquire_run_guard,
        release_run_guard,
    )

    lease = acquire_run_guard(output_root, run_id)
    if lease is None:
        raise SystemExit(
            f"Cannot acquire run guard for {output_root} — "
            "another Dynamic run is active. Wait for it to finish or "
            "manually remove the lock if the owner is dead."
        )

    try:
        with RunLogSession(output_root, pipeline="dynamic", run_id=run_id, level=level):
            return run_dynamic(cfg, run_id=run_id)
    finally:
        release_run_guard(output_root, lease)


if __name__ == "__main__":
    raise SystemExit(main())
