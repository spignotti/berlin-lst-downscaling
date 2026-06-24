#!/usr/bin/env python3
"""Submit GEE batch export tasks for ARD COG creation.

Usage:
    # Dry run (default): print what would be exported
    uv run python scripts/ard_export.py
    uv run python scripts/ard_export.py source=landsat

    # Single year dry run
    uv run python scripts/ard_export.py year=2023

    # Actually submit exports (use --dry-run=false)
    uv run python scripts/ard_export.py dry_run=false
    uv run python scripts/ard_export.py year=2023 source=landsat dry_run=false

Safety note:
    Defaults to --dry-run=true to prevent accidental mass exports.
"""

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.gee_client import initialize
from berlin_lst_downscaling.data.gee_export import export_scenes_by_year, monitor_tasks


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    """Submit or dry-run export tasks for one or all sources."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print()

    initialize(cfg)

    if cfg.dry_run:
        print("=" * 60)
        print("  DRY RUN — no tasks will be submitted")
        print("  Use `dry_run=false` to submit")
        print("=" * 60)

    sources = ["landsat", "sentinel2"]
    if cfg.source:
        sources = [cfg.source]

    all_tasks = []
    for src in sources:
        tasks = export_scenes_by_year(cfg, source=src, year=cfg.year, dry_run=cfg.dry_run)
        all_tasks.extend(tasks)

    total = len(all_tasks)
    if total > 0:
        print(f"\n{'='*60}")
        print(f"  Submitted {total} export task(s).")
        if not cfg.dry_run:
            print("  Monitoring tasks...")
            completed, failed = monitor_tasks(
                all_tasks,
                poll_interval_sec=cfg.ard.gee.task_poll_interval_sec,
            )
            print(f"  Completed: {len(completed)}, Failed: {len(failed)}")
            if failed:
                sys.exit(1)
    elif not cfg.dry_run:
        print("  No tasks submitted (empty collection?).")


if __name__ == "__main__":
    main()
