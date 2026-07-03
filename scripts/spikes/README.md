# ARD Spike — One-Scene Cloud-Native Ingestion

Validates the end-to-end path: PC STAC search → asset signing → `odc.stac.load` → inspected `xr.Dataset` on a common grid.

## Run

```bash
uv run python scripts/spikes/ard_spike.py
```

Optional date:

```bash
uv run python scripts/spikes/ard_spike.py --date 2024-07-15
```

## Expected Output (console)

```
------------------------------------------------------------
Landsat  —  date=2024-06-29  bbox=(13.08, 52.34, 13.76, 52.68)
  Item ID    : LC09_L2SP_193024_20240629_02_T1
  CRS        : EPSG:25833
  Shape      : {'y': 3884, 'x': 4699, 'time': 1}
  Bands      : ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22', 'lwir11', 'qa_pixel']
  Dtypes     : {'red': 'uint16', 'green': 'uint16', 'blue': 'uint16', 'nir08': 'uint16', 'swir16': 'uint16', 'swir22': 'uint16', 'lwir11': 'uint16', 'qa_pixel': 'uint16'}
  Valid frac : 100.0% (18,250,403 / 18,250,916)
------------------------------------------------------------
Sentinel-2  —  date=2024-06-29  bbox=(13.08, 52.34, 13.76, 52.68)
  Item ID    : S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907
  CRS        : EPSG:25833
  Shape      : {'y': 3884, 'x': 4699, 'time': 1}
  Bands      : ['B02', 'B03', 'B04', 'B08', 'SCL']
  Dtypes     : {'B02': 'float32', 'B03': 'float32', 'B04': 'float32', 'B08': 'float32', 'SCL': 'float32'}
  Valid frac : 99.1% (6,243,898 / 6,299,848)
------------------------------------------------------------
Rendering RGB composite …
  Saved: data/tmp/ard_spike_2024-06-29.png (9398×3884 px)
------------------------------------------------------------
Spike OK — both sensors loaded successfully.
  Landsat  : LC09_L2SP_193024_20240629_02_T1
  Sentinel2: S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907
------------------------------------------------------------
```

> **Note:** Both datasets land on an identical pixel grid (3884×4699) because they share the
> same ``crs=EPSG:25833``, ``resolution=10``, and ``bbox``. Landsat uint16 bands are raw
> DN; S2 float32 bands are scaled reflectance. Both are valid for inspection — scaling
> happens in ARD processing.

## Visual Output

Saved to `data/tmp/ard_spike_<date>.png`

| Left panel         | Right panel          |
|--------------------|----------------------|
| Landsat RGB        | Sentinel-2 RGB       |
| (2–98% stretch)    | (2–98% stretch)      |

*(Screenshot slot: insert side-by-side composite here)*

## Acceptance Criteria

- [ ] Exit code 0
- [ ] Both sensor item IDs printed (exact IDs may differ from above)
- [ ] Both datasets show `CRS: EPSG:25833`
- [ ] Both datasets have correct band counts (Landsat: 8, S2: 5)
- [ ] Valid pixel fraction > 0%
- [ ] Both datasets on identical spatial grid (same `x`/`y` dims)
- [ ] PNG saved and viewable
