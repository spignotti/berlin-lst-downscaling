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
  Loaded items: 2
    LC09_L2SP_193024_20240629_02_T1
    LC09_L2SP_193023_20240629_02_T1
  CRS         : EPSG:25833
  Shape       : {'y': 3884, 'x': 4699, 'time': 1}
  Bands       : ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22', 'lwir11', 'qa_pixel']
  Dtypes      : {'red': 'uint16', 'green': 'uint16', 'blue': 'uint16', 'nir08': 'uint16', 'swir16': 'uint16', 'swir22': 'uint16', 'lwir11': 'uint16', 'qa_pixel': 'uint16'}
  Valid pixels: 100.0% (18,250,916 / 18,250,916)
  Load time   : ~0.7s
------------------------------------------------------------
Sentinel-2  —  date=2024-06-29  bbox=(13.08, 52.34, 13.76, 52.68)
  Loaded items: 3
    S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907
    S2A_MSIL2A_20240629T102021_R065_T33UVT_20240629T161848
    S2A_MSIL2A_20240629T102021_R065_T33UUU_20240629T161811
  CRS         : EPSG:25833
  Shape       : {'y': 3884, 'x': 4699, 'time': 1}
  Bands       : ['B02', 'B03', 'B04', 'B08', 'SCL']
  Dtypes      : {'B02': 'float32', 'B03': 'float32', 'B04': 'float32', 'B08': 'float32', 'SCL': 'float32'}
  Valid pixels: 100.0% (18,250,916 / 18,250,916)
  Load time   : ~0.5s
------------------------------------------------------------
Rendering RGB composite …
  Saved: data/tmp/ard_spike_2024-06-29.png (9398×3884 px)
------------------------------------------------------------
Spike OK — ~32s total.
  Landsat  : 2 item(s)
  Sentinel2: 3 item(s)
------------------------------------------------------------
```

> **Note:** Both datasets land on the identical pixel grid (3884×4699 at 10 m, EPSG:25833)
> because they share the same ``crs``, ``resolution``, and ``bbox``. **100% valid coverage
> requires loading multiple items per sensor** (2 Landsat WRS-2 rows or 3 S2 MGRS tiles).
> Landsat uint16 bands are raw DN; S2 float32 bands are scaled reflectance — scaling
> happens in ARD processing.

## Visual Output

Saved to `data/tmp/ard_spike_<date>.png`

| Left panel         | Right panel          |
|--------------------|----------------------|
| Landsat RGB        | Sentinel-2 RGB       |
| (2–98% stretch)    | (2–98% stretch)      |

*(Screenshot slot: insert side-by-side composite here)*

## Data Flow

```
ard_spike.py
  → pystac_client.Client.open(planetarycomputer.microsoft.com/api/stac/v1,
                              modifier=planetary_computer.sign_inplace)
    └─ modifier signs asset URLs in-place after every search()
  → client.search(collections=[...], bbox=..., datetime=..., query={eo:cloud_cover:{lt:20}})
    └─ HTTP POST /search to PC STAC API → returns pystac.Item objects
    └─ sign_inplace mutates each item.assets[*].href to append ?token=<sigv4>
  → odc.stac.load(items, bands, crs=EPSG:25833, resolution=10, bbox=..., chunks={x:2048,y:2048})
    └─ rasterio reads each band's signed CloudFront URL via HTTPS range requests
    └─ GeoTIFF data is reprojected to EPSG:25833 and cropped to bbox
    └─ Dask graph built; .compute() materialises on access
    └─ Multiple items per collection are fused with groupby="solar_day" (last-item-wins)
```

### Performance Baseline (2024-06-29, Berlin bbox)

| Step | Time | Notes |
|------|------|-------|
| `Client.open` | ~0.2s | PC STAC root + signing |
| LS search | ~0.3s | returns 2 items (path 193 rows 23+24) |
| LS load + compute | ~0.7s | 8 bands × uint16 |
| S2 search | ~2.5s | returns 5 items, filters to 3 tiles |
| S2 load + compute | ~0.5s | 5 bands × float32 |
| PNG render + save | ~24s | 9398×3884 px (17 MB) |
| **Total** | **~32s** | |

Data egress: ~150 MB from Azure Blob via CloudFront (GBT cached after first run).

### Coverage Insight

- **Landsat C2-L2**: One WRS-2 row does NOT cover all of Berlin. Path 193 rows 23+24 together give full bbox coverage (two ~185 km × 180 km swaths, overlapping at ~52.6°N).
- **Sentinel-2 L2A**: Berlin sits at the boundary of 3 MGRS tiles (T33UVU, T33UUU, T33UVT). Loading all 3 tiles via `max_items=3` gives 100% valid coverage. The production pipeline will use scene-coupling (`clear_frac - λ·Δt/3`) instead of a simple max_items.
- Both sensors land on an identical pixel grid (3884×4699) when loaded with the same `crs=EPSG:25833`, `resolution=10`, and `bbox`.

## Acceptance Criteria

- [ ] Exit code 0
- [ ] Both sensor item IDs printed (exact IDs may differ from above)
- [ ] Both datasets show `CRS: EPSG:25833`
- [ ] Both datasets have correct band counts (Landsat: 8, S2: 5)
- [ ] **100% valid coverage** (both sensors cover full bbox with their combined items)
- [ ] Both datasets on identical spatial grid (same `y`/`x` dims)
- [ ] `--list-items` mode prints available items without loading data
- [ ] `--verbose` mode shows spatial coverage report (Y range, X range, geo bbox)
- [ ] PNG renders nodata pixels as mid-gray (128) — visually distinct from valid data
- [ ] PNG saved and viewable
