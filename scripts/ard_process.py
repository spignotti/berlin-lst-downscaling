#!/usr/bin/env python3
"""Reproject and regrid ARD exports to the canonical EPSG:25833 grid.

Reads raw GEE/AppEEARS COGs from GCS, reprojects/regrids to the
canonical grid, runs QA checks, and writes validated COGs + JSON
reports back to GCS.

Typical use:
    uv run python scripts/ard_run.py        # orchestrator (preferred)
    uv run python scripts/ard_run.py all    # full pipeline run

Direct use (dev / debugging):
    # Plan (default): show what would be processed
    uv run python scripts/ard_process.py mode=plan source=landsat
    uv run python scripts/ard_process.py mode=plan year=2023

    # Smoke test: 1 scene per source
    uv run python scripts/ard_process.py mode=smoke year=2023

    # Full run: all sources × all years (resume-aware)
    uv run python scripts/ard_process.py mode=all

The ``mode`` flag collapses the old ``dry_run`` + ``smoke`` pair:
  plan  → dry_run=True,  smoke=False
  smoke → dry_run=False, smoke=True
  all   → dry_run=False, smoke=False
"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.ard_modes import apply_mode


@hydra.main(version_base=None, config_path="../configs/ard", config_name="ard_process")
def main(cfg: DictConfig) -> None:
    """Reproject, regrid, QA, and upload ARD scenes."""
    apply_mode(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print()

    if cfg.dry_run:
        print("=" * 60)
        print("  DRY RUN — no files will be downloaded or processed")
        print("  Use `dry_run=false` to start processing")
        print("=" * 60)

    if cfg.smoke:
        print("  SMOKE MODE — processing 1 scene per source, then stopping")

    from berlin_lst_downscaling.data.grid_spec import get_spec

    spec = get_spec()
    print(
        f"\nGrid: {spec.crs}  "
        f"10m: {spec.width_10m}×{spec.height_10m}  "
        f"100m: {spec.width_100m}×{spec.height_100m}"
    )
    print()

    # ── W&B init (optional, graceful fallback) ──
    wandb_run = None
    if not cfg.dry_run:
        try:
            import wandb  # type: ignore[import-untyped]

            wandb_run = wandb.init(
                project="berlin-lst-downscaling",
                config=OmegaConf.to_container(cfg, resolve=True),
                job_type="ard_processing",
                reinit=True,
            )
        except Exception as wb_init_err:
            print(f"  [W&B] Warning: failed to init W&B: {wb_init_err}")
            wandb_run = None

    sources = ["landsat", "sentinel2", "ecostress"]
    if cfg.source:
        sources = [cfg.source]

    all_results: list[dict] = []
    for src in sources:
        print(f"\n{'=' * 60}")
        print(f"  Source: {src}")
        print(f"{'=' * 60}")

        from berlin_lst_downscaling.data.ard_processor import process_source

        results = process_source(
            cfg,
            src,
            spec,
            year=cfg.year,
            dry_run=cfg.dry_run,
            smoke=cfg.smoke,
        )
        all_results.extend(results)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  Summary: {len(all_results)} scene(s) processed")

    if cfg.dry_run:
        for r in all_results:
            print(f"    [{r['source']}] {r.get('scene_id', '?')}  →  {r.get('output_path', '?')}")
    else:
        success = [r for r in all_results if r.get("status") == "success"]
        errors = [r for r in all_results if r.get("status") == "error"]
        print(f"  Success: {len(success)}, Errors: {len(errors)}")
        for r in success:
            sid = r.get("scene_id", "?")
            print(f"    ✅ [{r['source']}] {sid} → {r.get('output_path', '?')}")
        for r in errors:
            sid = r.get("scene_id", "?")
            print(f"    ❌ [{r['source']}] {sid}: {r.get('error', '?')}")

        # Log summary to W&B
        if wandb_run is not None:
            try:
                import wandb  # type: ignore[import-untyped]

                summary = {
                    "processed/scenes_total": len(all_results),
                    "processed/scenes_success": len(success),
                    "processed/scenes_failed": len(errors),
                }

                # Per-source counts
                for src in sources:
                    src_results = [r for r in all_results if r.get("source") == src]
                    src_success = [r for r in src_results if r.get("status") == "success"]
                    summary[f"processed/{src}_total"] = len(src_results)
                    summary[f"processed/{src}_success"] = len(src_success)

                wandb_run.log(summary)

                # Log QA table
                qa_rows = []
                for r in success:
                    qa = r.get("qa_report", {})
                    grid = qa.get("grid_conformity", {})
                    row = {
                        "scene_id": r.get("scene_id", ""),
                        "source": r.get("source", ""),
                        "cloud_fraction": qa.get("cloud_fraction", -1),
                        "aoi_coverage_fraction": qa.get("aoi_coverage_fraction", -1),
                        "crs_match": grid.get("crs_match", False),
                        "resolution_match": grid.get("resolution_match", False),
                        "qa_passed": qa.get("qa_passed", False),
                    }
                    qa_rows.append(row)

                if qa_rows:
                    qa_table = wandb.Table(columns=list(qa_rows[0].keys()))
                    for row in qa_rows:
                        qa_table.add_data(*row.values())
                    wandb_run.log({"qa_summary": qa_table})

            except Exception as wb_err:
                print(f"  [W&B] Warning: failed to log to W&B: {wb_err}")

        if errors:
            sys.exit(1)

    if cfg.smoke and not cfg.dry_run:
        print("\n  SMOKE TEST COMPLETE — inspect output before running full pipeline.")
        print(f"  Run:  rio info {all_results[0]['output_path']}")
        print(f"  View: rio info --indent 2 {all_results[0]['qa_report_path']}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
