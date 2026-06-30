#!/usr/bin/env python3
"""Fetch the Berlin administrative boundary (Landesgrenze) from Geoportal Berlin.

Outputs:
    data/boundaries/berlin_landesgrenze.geojson
    data/boundaries/berlin_landesgrenze_2km_buffer.geojson

The buffered polygon is the authoritative AOI source. The export bounding box
is derived from it at runtime; we do not persist a separate rectangle file.
"""

from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "boundaries"
BUFFER_PATH = DATA_DIR / "berlin_landesgrenze_2km_buffer.geojson"
LANDESGRENZE_PATH = DATA_DIR / "berlin_landesgrenze.geojson"

WFS_URL = (
    "https://gdi.berlin.de/services/wfs/alkis_land"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeNames=alkis_land:landesgrenze"
    "&srsName=EPSG:25833"
    "&outputFormat=application/json"
)

BUFFER_METERS = 2000


def fetch_and_save() -> tuple[Path, Path]:
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

    # Buffer and save the buffered polygon
    gdf_buffered = gdf_25833.copy()
    gdf_buffered.geometry = gdf_buffered.buffer(BUFFER_METERS)
    if len(gdf_buffered) > 1:
        gdf_buffered = gdf_buffered.dissolve()
    gdf_buffered.to_file(BUFFER_PATH, driver="GeoJSON")
    print(f"  Saved buffered AOI polygon to {BUFFER_PATH}")

    # Also compute WGS84 bounding box for GEE
    aoi_wgs84 = gdf_buffered.to_crs("EPSG:4326")
    wgs84_bounds = aoi_wgs84.total_bounds
    print(f"  AOI bounding box (EPSG:4326): [{wgs84_bounds[0]:.4f}, {wgs84_bounds[1]:.4f}, "
          f"{wgs84_bounds[2]:.4f}, {wgs84_bounds[3]:.4f}]")

    xmin, ymin, xmax, ymax = gdf_buffered.total_bounds
    print("\nDone. Derived export bbox (EPSG:25833) from buffered polygon:")
    print(f"  [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]")
    return LANDESGRENZE_PATH, BUFFER_PATH


def main() -> None:
    fetch_and_save()


if __name__ == "__main__":
    main()
