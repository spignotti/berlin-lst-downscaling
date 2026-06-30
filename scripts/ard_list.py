#!/usr/bin/env python3
"""List available ARD scenes per source and year.

Supports:
  * landsat   — via GEE
  * sentinel2 — via GEE
  * ecostress — via NASA CMR (earthaccess)

Usage:
    uv run python scripts/ard_list.py
    uv run python scripts/ard_list.py source=landsat
    uv run python scripts/ard_list.py source=ecostress
    uv run python scripts/ard_list.py year=2023
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.gee_client import initialize


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    """List scenes for one or all sources across the configured time window."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print()

    sources = ["landsat", "sentinel2", "ecostress"]
    if cfg.source:
        sources = [cfg.source]

    for src in sources:
        years = (
            [cfg.year]
            if cfg.year
            else list(range(cfg.ard.time.start_year, cfg.ard.time.end_year + 1))
        )

        if src in ("landsat", "sentinel2"):
            initialize(cfg)
            _list_gee_source(cfg, src, years)
        elif src == "ecostress":
            _list_ecostress_source(cfg, years)
        else:
            print(f"Unknown source: {src}")


def _list_gee_source(cfg: DictConfig, src: str, years: list[int]) -> None:
    """List scenes for a GEE-based source."""
    from berlin_lst_downscaling.data.gee_scenes import list_landsat_scenes, list_sentinel2_scenes

    for year in years:
        if src == "landsat":
            col = list_landsat_scenes(cfg, year=year)
        else:
            col = list_sentinel2_scenes(cfg, year=year)

        n = col.size().getInfo()
        cloud_property = (
            "CLOUD_COVER" if src == "landsat" else "CLOUDY_PIXEL_PERCENTAGE"
        )
        cloud_free = col.filterMetadata(cloud_property, "less_than", 20).size().getInfo()

        print(f"{src}: {year:4d} → {n:5d} total,  {cloud_free:5d} <20% cloud")


def _list_ecostress_source(cfg: DictConfig, years: list[int]) -> None:
    """List granules for ECOSTRESS via CMR."""
    from berlin_lst_downscaling.data.boundary import buffered_bbox_wgs84
    from berlin_lst_downscaling.data.ecostress_scenes import (
        list_ecostress_granules,
        summarize_granules,
    )

    wgs84_bbox = list(buffered_bbox_wgs84(cfg.ard.aoi.boundary_file))
    months = list(cfg.ecostress.time.months)

    for year in years:
        granules = list_ecostress_granules(
            wgs84_bbox=wgs84_bbox,
            start_year=year,
            end_year=year,
            months=months,
        )
        summary = summarize_granules(granules)
        landsat_win = summary.get("landsat_window_by_year", {}).get(year, 0)
        print(
            f"ecostress: {year:4d} → {summary['total']:5d} total,  "
            f"{landsat_win:5d} Landsat-adjacent"
        )


if __name__ == "__main__":
    main()
