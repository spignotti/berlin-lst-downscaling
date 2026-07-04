# /// script
# requires-python = ">=3.12"
# dependencies = ["rasterio", "shapely"]
# ///

"""Rasterize Berlin Landesgrenze boundary → AOI COGs.

Produces two AOI masks at the pipeline resolutions:
- ``data/boundaries/aoi_10m.tif``  — 10 m pixels, EPSG:25833
- ``data/boundaries/aoi_100m.tif`` — 100 m pixels, EPSG:25833

Pixels are uint8: 1 = inside Berlin, 0 = outside.
Both outputs are cloud-optimized GeoTIFFs (deflate, overviews).

Usage
-----
    uv run python scripts/build_aoi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from shapely.geometry import shape


def _load_boundary(geojson_path: str) -> dict:
    with open(geojson_path) as f:
        return json.load(f)


def _rasterize(
    geometry: dict,
    out_uri: str,
    resolution: int,
    crs: str = "EPSG:25833",
) -> None:
    """Rasterize a GeoJSON geometry to a COG at the given resolution.

    The GeoJSON is assumed to already be in *crs* (detected from the
    ``crs`` member).  No coordinate transformation is performed.
    """
    geom = shape(geometry)

    # geometry is already in target CRS (EPSG:25833 as declared in the GeoJSON)
    minx, miny, maxx, maxy = geom.bounds
    cx_min, cy_min, cx_max, cy_max = minx, miny, maxx, maxy

    # Pad slightly
    pad = resolution * 2
    xmin = cx_min - pad
    xmax = cx_max + pad
    ymin = cy_min - pad
    ymax = cy_max + pad

    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))

    # Build affine transform: (xmin, resolution, 0, ymax, 0, -resolution)
    from rasterio.transform import from_bounds

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # Rasterize (pixel_is_area → area-type rasterization)
    import numpy as np

    out_data = np.zeros((height, width), dtype=np.uint8)

    # Use rasterio's features.rasterize
    from rasterio import features

    out_data = features.rasterize(
        [(geometry, 1)],
        out_shape=(height, width),
        fill=0,
        out=out_data,
        transform=transform,
        all_touched=True,
        default_value=1,
    )

    # Write COG
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 1,
        "BIGTIFF": "IF_SAFER",
    }

    out_path = Path(out_uri)
    tmp_path = out_path.parent / f".tmp_{out_path.name}"

    try:
        with rasterio.open(tmp_path, "w", **profile) as tmp:
            tmp.write(out_data.astype(np.uint8), 1)

        # Build overviews
        with rasterio.open(tmp_path, "r+") as tmp:
            ov_levels = [2, 4, 8, 16, 32]
            tmp.build_overviews(ov_levels, Resampling.average)
            tmp.update_tags(ns="rio_overview", resampling="nearest")

        tmp_path.replace(out_path)
        print(f"Wrote {out_uri}  ({width}×{height} px, {resolution} m)")

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    root = Path(__file__).parent.parent
    boundary_path = root / "data" / "boundaries" / "berlin_landesgrenze.geojson"
    out_dir = root / "data" / "boundaries"

    feature_collection = _load_boundary(str(boundary_path))

    # Assume single feature (Berlin boundary)
    geometry = feature_collection["features"][0]["geometry"]

    resolutions = [10, 100]
    for res in resolutions:
        out_uri = out_dir / f"aoi_{res}m.tif"
        _rasterize(geometry, str(out_uri), resolution=res)


if __name__ == "__main__":
    main()
