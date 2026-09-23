# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""WB3 patch-index runner (Hydra-driven).

Usage
-----
    # Smoke: the local training-data smoke release
    uv run python scripts/runners/run_patch_index.py --config-name patch_index_smoke

    # Full: every published training/v1 scene, canonical index to GCS
    uv run python scripts/runners/run_patch_index.py --config-name patch_index_full

Exits non-zero when the publisher-side readback fails. The index is derived
only from the source release manifest and eligibility masks; no feature
pixels are recomputed and no patch rasters are written.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.training.patch_index import (
    build_patch_index,
    publish_patch_index,
)

_logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../configs/training",
    config_name="patch_index_full",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Build and publish the WB3 patch index for the configured source release."""
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="patch-index", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            source_root=source_root,
            output_root=output_root,
        )
        build = build_patch_index(source_root=source_root)
        uris = publish_patch_index(build, output_root=output_root, run_id=run_id)

        qa = build.qa
        print(f"Patch index — run {run_id}")
        print(
            f"  Patches: {qa['patches_total']} | scenes with patches: "
            f"{qa['scenes_with_patches']} | distinct anchors: {qa['distinct_anchors']}"
        )
        print(
            f"  Eligible cells in patches: {qa['n_eligible_total']} | "
            f"accepted frac range: {qa['eligible_frac_min']}–{qa['eligible_frac_max']}"
        )
        for reason, count in sorted(qa["exclusions"].items()):
            print(f"  Excluded [{reason}]: {count}")
        for reason, count in sorted(qa["skipped_reasons"].items()):
            print(f"  Skipped [{reason}]: {count}")
        print(f"  By split: {qa['by_split']}")
        print(f"  By year : {qa['by_year']}")
        print(
            f"  Anchor rows {qa['anchor_row_min']}–{qa['anchor_row_max']}, "
            f"cols {qa['anchor_col_min']}–{qa['anchor_col_max']}"
        )
        print(f"  Source policy hash : {build.source_policy_hash}")
        print(f"  Index policy hash  : {build.index_policy_hash}")
        for label, uri in sorted(uris.items()):
            print(f"  {label:<12}: {uri}")

        # Hydra 1.3.4 discards the decorated task's return value, so failures
        # must raise inside the task to propagate a non-zero exit status.
        if not build.readback.get("ok", False):
            raise SystemExit(1)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
