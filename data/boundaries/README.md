## Berlin boundary files

- `berlin_landesgrenze.geojson` — raw Berlin administrative boundary polygon
- `berlin_landesgrenze_2km_buffer.geojson` — Berlin boundary buffered by 2 km in EPSG:25833

Source: Geoportal Berlin WFS (`alkis_land:landesgrenze`).

Refresh both files with:

```bash
uv run python scripts/ard_run.py boundary
```

These files are committed because they are the single source of truth for the
AOI used by the ARD pipeline. The buffered polygon is used for QA coverage and
the export bounding box is derived from it at runtime.
