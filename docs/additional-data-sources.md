# Additional Data Sources — Availability & Feature Definitions

Analysis date: 2026-06-18  
Research update: 2026-07-14 (post-rebuild decisions after deleted branch, fresh start)  
Notion context: Sekundärdaten Research Findings + Sekundärdaten Cloud-Workflow  
Data portal: Berlin Open Data / Geoportal Berlin, DWD CDC, Copernicus CDS, NASA CMR  
Reference: `docs/data-availability.md` (Landsat, Sentinel-2, ECOSTRESS)  
Validation scripts: `notebooks/dwd_stations.py`

---

## Research Update — 2026-06-28 (Download Feasibility)

Concrete findings per datasource after detailed feasibility check:

| # | Source | Download Route | Format | CRS | Volume | Library | Status |
|---|--------|---------------|--------|-----|--------|---------|--------|
| 1 | LoD2 CityGML | `https://gdi.berlin.de/data/a_lod2/atom/` (INSPIRE ATOM) | CityGML v2.0 ZIP | EPSG:25833 | ~830 KB/tile, ~1,850 tiles (~1.5 GB) | CityGML parser needed (no PyPI package — use `pycitygml` GitHub, or fiona/XML) | ✅ |
| 2 | Umweltatlas Vegetationshöhe 2020 | `https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip` | GeoTIFF | EPSG:25833 | ~785 MB (single file) | `rasterio` (already in deps) | ✅ |
| 3 | Umweltatlas Versiegelung 2016 & 2021 | ATOM: `https://gdi.berlin.de/data/ua_versiegelung_{2016,2021}/atom/Versiegelung_Raster_{2016,2021}.zip` (raster); WFS: `wfs/ua_versiegelung_2021` (vector) | GeoTIFF uint8 (raster) / GML (WFS) | EPSG:25833 | ~41 MB each (ZIP), native 2.5 m | `rasterio` (already in deps) | ✅ |
| 4 | DGM 1 m | `https://gdi.berlin.de/data/dgm1/atom/0.atom` (INSPIRE ATOM) | XYZ CSV in ZIP | EPSG:25833, DHHN2016 | ~16 MB/tile compressed, 297 tiles (~4.75 GB) | `pandas` → `rasterio` (XYZ→GeoTIFF via `gdal.Grid()`) | ⚠️ Large volume, auxiliary only |
| 5 | ERA5-Land | CDS API (`cdsapi` Python) | **NetCDF** (ZIP-wrapped) | 0.0625° lat/lon | ~1.2 GB/month (Berlin subset) | `cdsapi`, `netcdf4` | ✅ |
| 6 | SVF | **`xarray-spatial.sky_view_factor()`** | Derived from DSM | EPSG:25833 | N/A — compute | `xarray-spatial>=0.10`, `numba>=0.60,<0.61` (macOS x86_64) | ✅ |
| 7 | Shadow masks | Custom ray-casting (no PyPI library) | Derived from DSM + sun position | EPSG:25833 | N/A — compute | Custom numpy + numba + Michalsky-SPA (already implemented), UMEP algorithm as ref | 🔧 |

**Key decision changes from original analysis:**
- Canopy height → Umweltatlas VH 2020 (1 m GeoTIFF, not ETH CHM)
- DGM 1 m → auxiliary for SVF/shadow only (not Copernicus DEM as primary)
- SVF → **`xarray-spatial`** Zakek 2011 (~21s for Berlin AOI, replaces custom impl)
- Shadow → custom ray-casting with pre-computed horizon maps (confirmed: no PyPI library available)
- DWD → **`wetterdienst` v0.90+** (replaces direct URL download)
- ERA5 → **NetCDF format** (ZIP-wrapped from CDS, decoded via netCDF4)
- ERA5 ssrd → ECMWF conversion: ssrd/3600 at 01 UTC, (ssrd[t]-ssrd[t-1])/3600 otherwise

---

## Overview by Ablation Stage

| Stufe | Features | Data Sources (new) | Status |
|-------|----------|-------------------|--------|
| 1 — Spektral | NDVI, NDBI, NDWI, Albedo, Emissivity | Sentinel-2 L2A | ✅ Already in `data-availability.md` |
| 2 — Morphologie | Building height, Canopy height, DEM, Imperviousness, SVF | LoD2, Umweltatlas VH, DGM 1 m, Versiegelung | ✅ See §2 |
| 3 — Verschattung | Shadow masks, Sun geometry | Satellite metadata (derived) | 🔧 See §3 |
| 4 — Meteorologie | Antecedent T, radiation, wind, precip | DWD stations, ERA5-Land, CDS | ✅ See §4 |
| 5 — Loss | — | — | No new source |
| Validation | Independ. reference, scale consistency | ECOSTRESS | ✅ Already in `data-availability.md` |

---

## Stage 2 — Morphologie & Oberflächengeometrie

### 2.1 LoD2-Gebäudemodell (3D Building Model)

| Field | Detail |
|-------|--------|
| Source | **Geoportal Berlin / Berlin Open Data (dl-de-zero-2.0)** |
| Download (ATOM) | `https://gdi.berlin.de/data/a_lod2/atom/0.atom` |
| Format | CityGML v2.0 (ZIP per 1 km × 1 km tile) |
| Tiles | ~1,850 tiles for Berlin (1 km² each), ~830 KB per tile on avg |
| Total volume | ~1.5 GB compressed |
| CRS | EPSG:25833 (confirmed) |
| Reference year | 2024 (published 2024-03-18, revised 2024-04-22) |
| License | **dl-de-zero-2.0** (since 2024 — improved from older "other-closed" on BLC portal) |
| Auth | None (fully open, no registration) |
| Variables extracted | Building height (`measuredHeight`), building footprint, roof geometry |
| CityGML parsing | No dedicated PyPI package. Options: (a) `fiona` with CityGML driver (experimental), (b) XML/SAX parsing of CityGML (reliable, ~100 lines), (c) `pycitygml` via GitHub. Recommend XML-based extraction of `bldg:measuredHeight` + footprint polygon. |
| **Verdict** | ✅ **Available under open license.** Use ATOM feed for bulk download. CityGML parsing approach needed — recommend straightforward XML/SAX extraction. |

**Use case:** Extract `measuredHeight` + footprint → rasterize to 10 m → building height, BCR (building coverage ratio), volume per pixel.

**Note:** The older BLC Downloadportal (`businesslocationcenter.de/berlin3d-downloadportal/`) also offers CityGML but with "other-closed" license. Use the Geoportal Berlin ATOM feed (dl-de-zero-2.0) instead.

### 2.2 Canopy Height / Vegetationshöhe

**Final decision (2026-06-28): Use Umweltatlas Vegetationshöhen 2020.**

| Field | Detail |
|-------|--------|
| Source | **Umweltatlas Berlin** — Vegetationshöhen 2020 |
| Download (ATOM) | `https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip` |
| Format | GeoTIFF |
| Resolution | **1 m** |
| Coverage | Full Berlin |
| CRS | EPSG:25833 |
| Volume | ~785 MB (single file, compressed) |
| License | dl-de-zero-2.0 |
| Python library | `rasterio` (already in deps) — open with `rasterio.open()`, resample to 10 m |
| Gaps | Covers all vegetated areas (forests, parks, street trees). Buildings = NoData. |
| **Verdict** | ✅ **Final selection.** Better native resolution (1 m vs ETH 10 m), open license, and consistent with LoD2 building model (both from Geoportal Berlin). ETM CHM (GEE) dropped as input. |

**Processing:** Open GeoTIFF → resample from 1 m to 10 m (mean aggregation) → clip to AOI.

### 2.3 DEM, DOM & nDOM (Geländehöhe, Oberflächenmodell & Canopy Height)

#### DGM 1 m (auxiliary for SVF/shadow)

| Field | Detail |
|-------|--------|
| Source | **Geoportal Berlin** — INSPIRE Atom feed |
| Download (ATOM) | `https://gdi.berlin.de/data/dgm1/atom/0.atom` |
| Format | XYZ CSV in ZIP, e.g., `dgm1_33_376_5820_2_be.xyz` |
| Tile scheme | 2 km × 2 km, EPSG:25833, DHHN2016 |
| Coverage | Full Berlin + ~250 m Brandenburg buffer |
| Tiles for AOI | 297 tiles |
| Volume per tile | ~16 MB compressed (ZIP), ~120 MB uncompressed (XYZ) |
| Total volume | ~4.75 GB compressed, ~35.6 GB uncompressed |
| Acquisition | ALS flights Feb–Mar 2021 |
| License | dl-de-zero-2.0 |
| Auth | None (fully open) |
| Python library | `pandas` (read XYZ) → `gdal.Grid()` or `scipy.interpolate` (grid to GeoTIFF) |
| **Verdict** | ⚠️ **Available but large.** DGM 1 m is auxiliary for SVF/shadow, not a model input channel. For terrain-only purposes, Copernicus DEM GLO-30 (30 m) via GEE is simpler — use DGM 1 m only where 1 m base surface is needed for urban canyon ray-casting. |

**Processing:** Download ~60–100 tiles covering central Berlin → convert XYZ to 1 m GeoTIFF via `gdal.Grid()` → resample to 10 m for SVF computation.

#### Copernicus DEM GLO-30 (terrain alternative)

| Field | Detail |
|-------|--------|
| Source | **Copernicus Programme** |
| Access | `s3://copernicus-dem-30m/` (AWS eu-central-1, free) or GEE: `COPERNICUS/DEM/GLO30` |
| Resolution | 30 m |
| Coverage | Global (Berlin: fully covered) |
| Temporal | 2021 release |
| License | Free — Copernicus Programme |
| Volume | ~1 MB for Berlin extent |
| **Verdict** | ✅ **Sufficient for terrain.** Berlin is mostly flat (±50 m). Use as default unless 1 m base surface is explicitly needed for SVF/casting. |

### 2.4 Versiegelungsgrad (Imperviousness / Sealing)

| Field | Detail |
|-------|--------|
| Source | **Umweltatlas Berlin** — Versiegelung 2021 |
| Raster download (ATOM) | `https://gdi.berlin.de/data/ua_versiegelung_2021/atom/Versiegelung_Raster_2021.zip` |
| WFS (vector) | `https://gdi.berlin.de/services/wfs/ua_versiegelung_2021` (block-level ISU5) |
| Format | GeoTIFF (raster, uncorrected classification) / GML (WFS, block-level polygons) |
| Resolution | **2.5 m** (Sentinel-2 10 m source → per-pixel sealed class, **not** 10 m as stated in earlier research) |
| Coverage | Full Berlin |
| CRS | EPSG:25833 |
| Volume | ~41 MB compressed (ZIP), ~316 MB uncompressed (GeoTIFF uint8 at 2.5 m) |
| Latest version | **2021** |
| Earlier editions | 2016, 2011, 2005, 2001, 1990 |
| Units | uint8 class codes (0=unsealed, 5/15/…/95=sealing classes, 100=full, 101=building-shadow, 102=building, 103=rail, 110=shadow, 255=nodata) |
| License | dl-de-zero-2.0 |
| Python library | `rasterio` (already in deps) |
| **Pipeline** | Download ZIP → extract GeoTIFF → convert class codes to float32 percent → reproject to canonical 10 m via `Resampling.average` → write COG. Both vintages processed unconditionally; scene-year mapping (≤2020→2016, >2020→2021) applied at training time. |
| **Verdict** | ✅ **Excellent.** Open license, verified 16-code scheme, pipeline-ready. |

**Note:** Two products exist: (a) uncorrected raster (2.5 m, Sentinel-2 classification, pixel-level codes), and (b) block-level WFS (ISU5 statistical blocks, officially corrected). For ML input, the uncorrected raster is more appropriate (pixel-level, no aggregation artifacts). The WFS version is useful for validation.

### 2.5 Sky View Factor (abgeleitet aus LoD2 + DEM)

**Final decision (2026-06-28): Self-implemented SVF in Python.**

| Field | Detail |
|-------|--------|
| Source | **Derived** — computed from LoD2 + DGM 1 m |
| Method | Hemispherical view analysis at 10 m resolution |
| Search radius | 30 m minimum (Scarano 2017) |
| **Library research (update 2026-07-14)** | |
| **`xarray-spatial` v0.10.16 (PyPI)** | **Preferred.** Production-ready `sky_view_factor()` (Zakek et al. 2011). Numba-backed, multi-backend (numpy/CuPy/Dask). ~21s benchmark for Berlin AOI (12M px, max_radius=3, n_directions=16). |
| UMEP (QGIS) | Mature urban climate tool, but requires QGIS Python → heavyweight |
| SAGA GIS `r.skyview` | Available via SAGA Python bindings, but SAGA is Linux-only |
| PyPI alternatives | None other than xarray-spatial — `svf`, `horizon`, `pyviewshed`, `richdem` not available on PyPI |
| **Recommendation** | **`xarray-spatial.sky_view_factor()`** over custom numpy/numba — production-quality, faster, maintained. Pin `numba>=0.60,<0.61` for macOS x86_64 wheel compatibility (llvmlite 0.43.0 has cp312 x86_64 wheel; later versions lack). |
| **Verdict** | ✅ **`xarray-spatial`.** Use LoD2 rasterized to 10 m for buildings + DGM 1 m for terrain → compute DSM heights → SVF via xarray-spatial. |

---

## Stage 3 — Verschattung & Sonnengeometrie

### 3.1 Sun Geometry per Scene

| Field | Detail |
|-------|--------|
| Source | **Satellite metadata** — not a separate download |
| Landsat | Properties `SUN_AZIMUTH`, `SUN_ELEVATION` in each scene |
| Sentinel-2 | Properties `MEAN_SOLAR_AZIMUTH`, `MEAN_SOLAR_ZENITH` in each scene |
| Custom timestamps | Project already has self-contained **Michalsky-SPA** implementation (±0.5° accuracy, in STAC writer). Alternative: `pysolar` (PyPI, v0.8, 405 GitHub stars, GPLv3). |
| **Verdict** | ✅ **Already implemented.** Self-contained SPA in pipeline — no external library needed. `pysolar` available as fallback. |

### 3.2 Shadow Masks (abgeleitet aus LoD2 + Sun Position)

**Final decision (2026-06-28): Custom ray-casting, binary, per scene.**

| Field | Detail |
|-------|--------|
| Source | **Derived** — ray-casting from LoD2-DSM + VH-DSM + sun position |
| **Library research** | |
| GRASS `r.sunmask` | Available via Python bindings (`grass.script`), but requires GRASS GIS installation. |
| UMEP Shadow | Part of UMEP QGIS plugin, not standalone Python. |
| `pvlib` | Solar position only (no shading). PyPI package, MIT license. |
| `pysolar` | Solar position + irradiation, not shadow mapping. |
| PyPI packages for shadow | **None found** — no `shadow`, `pyviewshed`, or similar. |
| **Recommendation** | **Custom numpy + numba approach.** Pre-compute horizon angles per azimuth direction (from DSM) once, then evaluate shadow for each sun position as a pure lookup. Complexity: DSM horizon pre-compute is one-time (~60 min), per-scene shadow lookup is O(pixels × 1) ~ seconds. For 362 cloud-free Landsat scenes: ~2 sec/scene = ~12 min total. |
| **Optimization** | Pre-computed DSM horizon maps reduce per-scene work to O(N). Only compute for acquisitions actually used in training (not all 362). Use ProcessPoolExecutor for parallel scene processing. |
| **Verdict** | 🔧 **Self-implement.** Use pre-computed horizon + ray-casting with numba. One-time DSM horizon pre-compute + fast per-scene shadow lookup. |

---

## Stage 4 — Meteorologie (antezedente Witterung)

### 4.1 DWD Stationsdaten (primary source)

| Field | Detail |
|-------|--------|
| Source | **Deutscher Wetterdienst — CDC Open Data** |
| URL | [https://opendata.dwd.de/climate_environment/CDC/](https://opendata.dwd.de/climate_environment/CDC/) |
| Format | Hourly ZIP files (CSV) per station + parameter |
| License | CC-BY (DWD Open Data) |
| Auth | None (fully open) |
| **Library (update 2026-07-14)** | **`wetterdienst` v0.90+ (PyPI)** — actively maintained DWD client, wraps `Wetterdienst(provider="dwd", network="observation")`. Handles directory listing, file discovery, caching, parsing, unit conversion. Replaces direct URL download. Fallback (no extra dep): `urllib.request` + `zipfile` against pattern `stundenwerte_{PARAM}_{STATION}_*_hist.zip`. |

**Berlin stations active for 2018–2024:**

| Station | Params | Full 2018–2024 | Notes |
|---------|--------|----------------|-------|
| ★ Berlin Brandenburg (BER) / 00427 | **8** | 8/8 | **Best choice** — widest coverage |
| ★ Berlin-Dahlem (FU) / 00403 | **7** | 7/7 | Excellent, at FU Berlin campus |
| ★ Berlin-Tempelhof / 00433 | **8** | 7/8 | SD ends 2022 |
| Berlin-Marzahn / 00420 | **5** | 4/5 | Cloud ends 2011 |

**Available parameters at BER station:**

| Parameter | Unit | DWD Prefix | Available |
|-----------|------|-----------|-----------|
| Air temperature (2m) | °C | TU | ✅ 1973–present |
| Precipitation | mm | RR | ✅ 1995–present |
| Wind speed | m/s | FF | ✅ 1973–present |
| Cloud cover | 1/8 | N | ✅ 1975–present |
| Surface pressure | hPa | P0 | ✅ 1975–present |
| Sunshine duration | min | SD | ✅ 1992–2025 |
| Temperature + Humidity | °C, % | TF | ✅ 1975–present |
| Dew point temp. | °C | TD | ✅ 1975–present |

**Missing:** Direct solar radiation (W/m²) — no Berlin station has it.

**Verdict:** ✅ **Excellent coverage.** DWD provides free hourly data for 6+ meteorological variables across 4 active Berlin stations. BER and Dahlem (FU) are the top choices.

### 4.2 Solar Radiation (missing from DWD Berlin stations)

No Berlin DWD station measures direct solar radiation. Evaluating three complementary sources:

#### 4.2a ERA5-Land (recommended primary)

| Field | Detail |
|-------|--------|
| Source | **Copernicus Climate Data Store** |
| URL | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) |
| Resolution | 0.0625° × 0.0625° (~6.9 km at 52°N) |
| Temporal | Hourly, **1950–present** (real-time through present) |
| Variables | Surface solar radiation downwards (SSRD), 2m temperature (t2m) |
| API | CDS API via `cdsapi` Python package (free registration). `.cdsapirc` needs `url` + `key`. |
| Query | One month per request. Format: `netcdf` (ZIP-wrapped). Decoded via `netcdf4` + `xarray`. |
| Volume estimate | 45 scene-months (2017–2025, May–Sep). Each request ~1.2 GB NetCDF. Total ~54 GB raw, but only 2 months cached locally at a time. |
| Variables for pipeline | `2m_temperature` (t2m, K, instantaneous), `surface_solar_radiation_downwards` (ssrd, J/m², **accumulation** — see below) |
| **ssrd accumulation handling** | ECMWF ERA5-Land rule: at 01 UTC, hourly = ssrd/3600; otherwise hourly = (ssrd[t] - ssrd[t-1])/3600. At 00 UTC this yields the 24th hour's value. See `era5.py:_ssrd_to_hourly`. |
| Antecedent 3-day mean | Rolling 72-hour mean of hourly ssrd before each scene acquisition time |
| Resampling to 10 m | Nearest-neighbor: constant value per ERA5 cell (~6.9 km) → fill 10 m grid |
| Library | `cdsapi`, `netcdf4` (both in pyproject.toml) |
| Account | Copernicus CDS account + API key needed (user has account) |
| Update latency | ~5 days (ERA5-Land-T near-real-time), final product ~2 months |
| **Verdict** | ✅ **Best operational choice.** Tiny volume, hourly resolution, direct API. SSRD needs conversion J/m² → W/m². |

#### 4.2b SARAH-3 (EUMETSAT CM SAF) — validation complement

| Field | Detail |
|-------|--------|
| Source | **EUMETSAT CM SAF** |
| Resolution | 0.05° × 0.05° (~5 km, ~5.6 × 3.4 km at Berlin) |
| Temporal | 30-min instantaneous, **1983–2020** (CDR), extended ICDR to present |
| Variables | Surface incoming shortwave (SIS), direct horizontal (SID), direct normal (DNI), PAR, sunshine duration |
| Access | CM SAF Web UI / EUMETSAT Data Store (free registration, CC BY 4.0) |
| Update latency | ~5 days (ICDR) |
| **Verdict** | ✅ **Best validation source** — higher spatial resolution than ERA5-Land, includes direct irradiance components. Useful for checking ERA5-Land solar radiation biases in cloud-sensitive daytime conditions. |

#### 4.2c CERES SYN1deg — coarse baseline

| Field | Detail |
|-------|--------|
| Source | **NASA ASDC** |
| Resolution | 1° × 1° (~110 × 70 km at Berlin) |
| Temporal | 1-hourly, **2000–present** |
| Variables | All-sky/clear-sky shortwave fluxes, direct/diffuse components, longwave |
| Access | NASA Earthdata Search (free Earthdata login, CC0-like) |
| Update latency | ~2–3 months |
| **Verdict** | ⚠️ Too coarse for Berlin urban analysis (single cell covers most of Brandenburg). Useful only as regional-scale sanity check. |

**Recommendation:** Use **ERA5-Land as the primary solar radiation source** (easy API, hourly, 2018–2024 coverage, same pipeline as other met data). Validate with **SARAH-3** where ERA5-Land biases are suspected. DWD station data remains primary for temperature, precipitation, and wind (higher accuracy at point locations).

---

## Validation Data Sources

### ECOSTRESS (independent thermal reference)

Already fully documented in `docs/data-availability.md` (Section: ECOSTRESS).  
**Product:** `ECO_L2T_LSTE v002` (70 m LST, MGRS-tiled COGs)  
**Access:** NASA CMR via `earthaccess`  
**Coverage:** 6,000 granules (2018–2023), 2,311 summer  
**Status:** ✅ Ready

### Scale-Consistency Check

| Field | Detail |
|-------|--------|
| Method | Compare downscaled 10 m LST against aggregated 100 m Landsat LST |
| Split | 80% training / 20% scale-consistency holdout (random patches) |
| Metric | MAE, bias at 100 m aggregate |
| **Verdict** | 📋 **Methodology** — implemented at eval time, not a data source. |

### Temporal CV — Räumlich + zeitlich getrennte Holdout-Daten

| Field | Detail |
|-------|--------|
| Method | Block years into temporal folds |
| Proposal | Train: **2018–2022** / Val: **2023** / Test: **2024** |
| Rationale | Avoids temporal leakage; 2024 as unseen year |
| Alternative | Leave-one-year-out CV for small-sample robustness (35 summer months → 5-month blocks) |
| **Verdict** | 📋 **Methodology** — implemented at datamodule/eval level, not a data source. |

---

## Consolidated Feature Table

Complete feature-to-source mapping for all 5 ablation stages + validation.

| Feature | Stage | Source / Access | Resolution | Zeitraum Berlin | Status |
|---------|-------|----------------|-----------|----------------|--------|
| Surface Temperature (LST) | 1 (target) | Landsat 8/9 TIRS (GEE) | 100 m | 2013–present | ✅ |
| NDVI / NDBI / NDWI / MNDWI | 1 (predictor) | Sentinel-2 L2A (GEE) | 10 m | 2015–present | ✅ |
| Albedo | 1 (predictor) | Sentinel-2 L2A (GEE, derived from 10 bands) | 10 m | 2015–present | ✅ |
| Emissivity | 1 (predictor) | **ASTER GED v3 (GEE)** — NDVI-threshold rejected | 100 m | static | ✅ |
| Building height | 2 | LoD2 CityGML (Geoportal Berlin ATOM) | building vector (2019) | 2024 | ✅ |
| Building fraction | 2 | LoD2 → rasterized 10 m | 10 m | 2024 | ✅ |
| Canopy height | 2 | **Umweltatlas Vegetationshöhen 2020** (ATOM GeoTIFF) | 1 m → resample 10 m | 2020 | ✅ |
| DEM / terrain | 2 | **Copernicus DEM GLO-30** (GEE/AWS) — DGM 1 m auxiliary only | 30 m | 2021 | ✅ |
| Imperviousness | 2 | Umweltatlas Versiegelung 2021 (ATOM GeoTIFF raster, 10 m) | 10 m | 2021 | ✅ |
| Sky View Factor | 2 | Derived from LoD2+VH DSM (**`xarray-spatial`** Zakek 2011) | 10 m | static | ✅ |
| Sun azimuth / elevation | 3 | Landsat/S2 metadata (GEE) or Michalsky-SPA | per scene | per acquisition | ✅ |
| Shadow mask | 3 | Derived from DSM + sun position (custom ray-casting, UMEP algorithm) | 10 m | dynamic | 🔧 |
| Antecedent T | 4 | DWD BER or Dahlem (`wetterdienst`) | point | 1973–present | ✅ |
| Precip | 4 | DWD BER or Dahlem (`wetterdienst`) | point | 1995–present | ✅ |
| Wind speed | 4 | DWD BER (`wetterdienst`) | point | 1973–present | ✅ |
| Solar radiation | 4 | ERA5-Land (CDS API, `cdsapi`, **NetCDF**) | ~6.9 km grid | 1950–present | ✅ |
| Cloud cover | 4 | DWD BER (hourly) | point | 1975–present | ✅ |
| Surface pressure | 4 | DWD BER (hourly) | point | 1975–present | ✅ |
| Humidity / dew point | 4 | DWD Dahlem (hourly) | point | 1955–present | ✅ |
| ECOSTRESS LST | Validation | NASA CMR / earthaccess | 70 m | 2018–2023 | ✅ |

---

### Emissivity Source Decision

**Verdict: ASTER GED v3 (`NASA/ASTER_GED/AG100_003`)**

Three emissivity sources were evaluated:

| Method | How | NDVI Collinearity | Accuracy | Urban Performance |
|--------|-----|-------------------|----------|-------------------|
| NDVI-threshold (Valor & Caselles 1996, Sobrino 2008) | ε = f(NDVI, fractional vegetation cover) | **High** — deterministic function of NDVI | RMSE ~0.01 over natural surfaces | Poor — roofs, asphalt, concrete misclassified |
| ASTER GED v3 (Hulley et al. 2015) | TIR/TES-derived, 2000–2008 mean, 5 TIR bands | **Lowest** — independent TIR measurement, not current NDVI | ~1% band emissivity error | Good — captures urban surface variability |
| Landsat C2 ST_EMIS | ASTER GED + NDVI/NDSI temporal adjustment | **Medium** — ASTER GED base but NDVI-adjusted per scene | High for Landsat ST consistency | Good but not independent |

**Rationale for ASTER GED v3:**
- NDVI-threshold emissivity creates problematic **collinearity** with the NDVI predictor in Stage 1. The model would see essentially the same information twice, confounding ablation interpretation.
- ASTER GED is an **independent data source** (thermal infrared spectroscopy, not NDVI-based), so it adds genuine new information even with NDVI as a predictor.
- Available in GEE as a single `ee.Image("NASA/ASTER_GED/AG100_003")`. Bands 13/14 cover the Landsat Band 10 wavelength (~10.9 µm). Scale factor 0.001.
- Static (2000–2008 mean) is acceptable for Berlin — emissivity of urban surfaces changes slowly (decadal timescale).
- Landsat C2 `ST_EMIS` is effectively ASTER GED + NDVI adjustment — using it would introduce partial collinearity and dependency on the Landsat C2 processing chain.

**Pipeline note:** Export ASTER GED emissivity bands 13 + 14, plus emissivity standard deviation and `num_obs` for quality masking. Resample from 100 m to model grid only at export/training time.

---

## Open Decisions

- **Landsat 7:** Not queried (SLC-off stripes). Still available 1999–present if additional historical data is needed.
- **Landsat-S2 coupling logic:** Monatsmedian vs. zeitnächster wolkenfreier Pixelwert — deferred to feature engineering phase.
- **Validation split / Temporal CV:** 2018–2022 train, 2023 val, 2024 test (proposed) — to be finalized.
- **Dynamic World land cover (GEE):** Available 2015–present at 10 m (`GOOGLE/DYNAMICWORLD/V1`). Not in ablation plan but could replace/supplement imperviousness.
- **Nighttime ECOSTRESS:** ISS overpass includes ~1,000 night granules over Berlin. Not useful for daytime LST but potentially for diurnal cycle analysis.

---

## Status Update — 2026-07-14 (Post-Rebuild)

Vorheriger Feature-Branch wurde komplett gelöscht. Frischer Start.

### Bucket-Bereinigung

Inhalt von `gs://berlin-lst-data/_raw/secondary/` und `gs://.../ard/static/` wurde am 2026-07-14 vollständig entfernt. Grund: Vorhandene Daten waren unvollständig (3/6 statische COGs hatten schwere Lücken — building: 0,001% valide px, terrain/DSM: 0,2% valide px, ERA5: leeres File 15 Byte).

### Offene Pipeline-Aufgaben

Siehe Notion-Notizen: "Sekundärdaten Cloud-Workflow" + "Sekundärdaten Research Findings". Empfohlene Build-Reihenfolge: Versiegelung+V → ERA5 → DWD → LoD2 → DGM → DSM → SVF → Shadow → Dynamic Meteorology.

---

*Generated by `notebooks/dwd_stations.py` + manual research; complements `docs/data-availability.md`.*
