## Berlin boundary files

- `berlin_landesgrenze.geojson` — Berlin administrative boundary (MultiPolygon, EPSG:25833)
- `berlin_landesgrenze_2km_buffer.geojson` — Berlin boundary buffered by 2 km in EPSG:25833
- `aoi_10m.tif` — pre-rasterized AOI mask at 10 m resolution (uint8, 1=inside Berlin)
- `aoi_100m.tif` — pre-rasterized AOI mask at 100 m resolution (uint8, 1=inside Berlin)

Source: Geoportal Berlin WFS (`alkis_land:landesgrenze`).

**Regenerate AOI COGs** (after updating the GeoJSON):

```bash
uv run python scripts/build_aoi.py
```

These files are committed because they are the single source of truth for the
AOI used by the ARD pipeline. The buffered polygon is used for QA coverage and
the export bounding box is derived from it at runtime.

The AOI masks (``aoi_10m.tif``, ``aoi_100m.tif``) are COGs pre-rasterized at
the pipeline's native resolutions (10 m for Sentinel-2, 100 m for Landsat).
They are consumed by ``compute_aoi_metrics`` at scene-process time to produce
per-scene pixel-count fields stored in the ledger (schema v3).
