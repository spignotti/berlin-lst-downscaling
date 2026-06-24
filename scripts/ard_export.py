#!/usr/bin/env python3
"""Submit ARD export tasks — GEE batch exports or AppEEARS downloads.

Supports:
  * landsat   — GEE batch export to GCS
  * sentinel2 — GEE batch export to GCS
  * ecostress — AppEEARS area task, download, COG conversion, GCS upload

Usage:
    # Dry run (default): print what would be exported
    uv run python scripts/ard_export.py
    uv run python scripts/ard_export.py source=landsat
    uv run python scripts/ard_export.py source=ecostress

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


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    """Submit or dry-run export tasks for one or all sources."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print()

    if cfg.dry_run:
        print("=" * 60)
        print("  DRY RUN — no tasks will be submitted")
        print("  Use `dry_run=false` to submit")
        print("=" * 60)

    sources = ["landsat", "sentinel2", "ecostress"]
    if cfg.source:
        sources = [cfg.source]

    all_tasks = []
    for src in sources:
        if src in ("landsat", "sentinel2"):
            initialize(cfg)
            tasks = _run_gee_export(cfg, src)
            all_tasks.extend(tasks)
        elif src == "ecostress":
            results = _run_ecostress_export(cfg)
            _print_ecostress_results(results)
        else:
            print(f"Unknown source: {src}")

    total = len(all_tasks)
    if total > 0:
        print(f"\n{'='*60}")
        print(f"  Submitted {total} GEE export task(s).")
        if not cfg.dry_run:
            print("  Monitoring tasks...")
            from berlin_lst_downscaling.data.gee_export import monitor_tasks

            completed, failed = monitor_tasks(
                all_tasks,
                poll_interval_sec=cfg.ard.gee.task_poll_interval_sec,
            )
            print(f"  Completed: {len(completed)}, Failed: {len(failed)}")
            if failed:
                sys.exit(1)
    elif not cfg.dry_run:
        # Only print "no tasks" if GEE sources were attempted
        gee_sources_used = [s for s in sources if s in ("landsat", "sentinel2")]
        if gee_sources_used:
            print("  No GEE tasks submitted (empty collection?).")


def _run_gee_export(cfg: DictConfig, source: str) -> list:
    """Run GEE batch export for a given source."""
    from berlin_lst_downscaling.data.gee_export import export_scenes_by_year

    tasks = export_scenes_by_year(cfg, source=source, year=cfg.year, dry_run=cfg.dry_run)
    return tasks


def _run_ecostress_export(cfg: DictConfig) -> list[dict]:
    """Run ECOSTRESS export via AppEEARS."""
    from berlin_lst_downscaling.data.ecostress_export import export_ecostress_by_year

    results = export_ecostress_by_year(cfg, year=cfg.year, dry_run=cfg.dry_run)
    return results


def _print_ecostress_results(results: list[dict]) -> None:
    """Print ECOSTRESS export results."""
    for r in results:
        status = r.get("status", "?")
        year = r.get("year", "?")
        if status == "dry_run":
            n = r.get("granules", 0)
            print(f"  ECOSTRESS {year}: dry-run — would download {n} granules")
        elif status == "error":
            print(f"  ECOSTRESS {year}: ERROR — {r.get('error', 'unknown')}")
        elif status == "success":
            paths = r.get("gcs_paths", [])
            print(f"  ECOSTRESS {year}: {len(paths)} COGs uploaded to GCS")
            for p in paths[:3]:
                print(f"    {p}")
            if len(paths) > 3:
                print(f"    ... and {len(paths) - 3} more")


if __name__ == "__main__":
    main()
