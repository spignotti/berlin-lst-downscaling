#!/usr/bin/env python3
"""List available GEE scenes per source and year.

Usage:
    uv run python scripts/ard_list.py
    uv run python scripts/ard_list.py source=landsat
    uv run python scripts/ard_list.py year=2023
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.gee_client import initialize
from berlin_lst_downscaling.data.gee_scenes import list_landsat_scenes, list_sentinel2_scenes


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    """List scenes for one or all sources across the configured time window."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print()

    initialize(cfg)

    sources = ["landsat", "sentinel2"]
    if cfg.source:
        sources = [cfg.source]

    for src in sources:
        years = (
            [cfg.year]
            if cfg.year
            else list(range(cfg.ard.time.start_year, cfg.ard.time.end_year + 1))
        )

        for year in years:
            if src == "landsat":
                col = list_landsat_scenes(cfg, year=year)
            else:
                col = list_sentinel2_scenes(cfg, year=year)

            n = col.size().getInfo()
            # Also compute the cloud-free count for reference
            cloud_band = (
                "CLOUD_COVER" if src == "landsat" else "CLOUDY_PIXEL_PERCENTAGE"
            )
            cloud_free = col.filterMetadata(cloud_band, "less_than", 20).size().getInfo()

            print(f"{src}: {year:4d} → {n:5d} total,  {cloud_free:5d} <20% cloud")


if __name__ == "__main__":
    main()
