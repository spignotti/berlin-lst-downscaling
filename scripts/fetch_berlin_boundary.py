#!/usr/bin/env python3
"""Fetch the Berlin administrative boundary (Landesgrenze) from Geoportal Berlin
via WFS, apply a 2 km buffer, and save the AOI rectangle as GeoJSON.

Output:
    data/berlin_aoi.geojson — bounding-box rectangle of Berlin + 2 km buffer
    in EPSG:25833. This is the authoritative AOI for all downstream processing.
"""

from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "berlin_aoi.geojson"
LANDESGRENZE_PATH = DATA_DIR / "berlin_landesgrenze.geojson"

WFS_URL = (
    "https://gdi.berlin.de/services/wfs/alkis_land"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeNames=alkis_land:landesgrenze"
    "&srsName=EPSG:25833"
    "&outputFormat=application/json"
)

BUFFER_METERS = 2000


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching Berlin Landesgrenze from Geoportal Berlin WFS...")
    try:
        gdf = gpd.read_file(WFS_URL)
    except Exception as exc:
        print(f"ERROR: Failed to fetch boundary from WFS.\n  URL: {WFS_URL}\n  {exc}")
        raise SystemExit(1) from exc

    if gdf.empty:
        print("ERROR: WFS returned an empty response.")
        raise SystemExit(1)

    print(f"  Received {len(gdf)} feature(s).")
    print(f"  CRS: {gdf.crs}")
    print(f"  Raw bounding box (EPSG:25833): {list(gdf.total_bounds.round(3))}")

    # Save the actual Landesgrenze polygon for visual overlay (COSO feature)
    # Already in EPSG:25833 from the WFS request.
    gdf_25833 = gdf.to_crs("EPSG:25833")
    if len(gdf_25833) > 1:
        # Dissolve multipart features into a single polygon outline
        gdf_25833 = gdf_25833.dissolve()
    gdf_25833.to_file(LANDESGRENZE_PATH, driver="GeoJSON")
    print(f"  Saved Berlin Landesgrenze to {LANDESGRENZE_PATH}")

    # Buffer and compute bounding rectangle
    gdf_buffered = gdf.to_crs("EPSG:25833").buffer(BUFFER_METERS)
    xmin, ymin, xmax, ymax = gdf_buffered.total_bounds

    # Create the bounding rectangle as a GeoDataFrame
    from shapely.geometry import Polygon

    rect = Polygon(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
    )
    aoi_gdf = gpd.GeoDataFrame(
        {"name": ["berlin_aoi_2km_buffer"]},
        geometry=[rect],
        crs="EPSG:25833",
    )
    aoi_gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"  Saved AOI rectangle to {OUTPUT_PATH}")

    # Also compute WGS84 bounding box for GEE
    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")
    wgs84_bounds = aoi_wgs84.total_bounds
    print(f"  AOI bounding box (EPSG:4326): [{wgs84_bounds[0]:.4f}, {wgs84_bounds[1]:.4f}, "
          f"{wgs84_bounds[2]:.4f}, {wgs84_bounds[3]:.4f}]")

    print("\nDone. Use this EPSG:25833 bbox in ard configs:")
    print(f"  aoi_25833: [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]")


if __name__ == "__main__":
    main()
