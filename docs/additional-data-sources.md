# Additional Data Sources — Availability & Feature Definitions

Analysis date: 2026-06-18  
Data portal: Berlin Open Data / FIS-Broker, DWD CDC, Copernicus CDS, NASA CMR  
Reference: `docs/data-availability.md` (Landsat, Sentinel-2, ECOSTRESS)  
Validation scripts: `notebooks/dwd_stations.py`

---

## Overview by Ablation Stage

| Stufe | Features | Data Sources (new) | Status |
|-------|----------|-------------------|--------|
| 1 — Spektral | NDVI, NDBI, NDWI, Albedo, Emissivity | Sentinel-2 L2A | ✅ Already in `data-availability.md` |
| 2 — Morphologie | Building height, Canopy height, DEM, Imperviousness, SVF | LoD2, CHM, Copernicus DEM, Versiegelung | ⬇️ Below |
| 3 — Verschattung | Shadow masks, Sun geometry | Satellite metadata (derived) | ⬇️ Below |
| 4 — Meteorologie | Antecedent T, radiation, wind, precip | DWD stations, ERA5-Land | ⬇️ Below |
| 5 — Loss | — | — | No new source |
| Validation | Independ. reference, scale consistency | ECOSTRESS | ✅ Already in `data-availability.md` |

---

## Stage 2 — Morphologie & Oberflächengeometrie

### 2.1 LoD2-Gebäudemodell (3D Building Model)

| Field | Detail |
|-------|--------|
| Source | **Business Location Center (BLC) Berlin 3D Downloadportal** |
| URL | [https://www.businesslocationcenter.de/berlin3d-downloadportal/](https://www.businesslocationcenter.de/berlin3d-downloadportal/) |
| Format | CityGML (LoD2), also DXF |
| Resolution | Single-building level; roof shapes with dormers, eaves, ridges |
| Coverage | Entire Berlin (890 km²) |
| Reference year | 2019 (based on 2017–2018 aerial imagery), with periodic updates |
| License | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) |
| Access | Downloadportal (bulk download after registration) |
| Variables extracted | Building height (m), building volume (m³), building density per grid cell |
| **Verdict** | ✅ **Available.** Best static morphology layer for urban climate analysis. |

**Use case:** Rasterize to 10 m → building height, building fraction, volume per pixel.

### 2.2 Canopy Height Model (Vegetationshöhe)

Two alternatives evaluated:

#### Option A: ETH Global Canopy Height 2020 (recommended)

| Field | Detail |
|-------|--------|
| Source | **ETH Zurich / Lang et al. 2022** — Sentinel-1 + GEDI fusion |
| GEE asset | `users/nlang/ETH_GlobalCanopyHeight_2020_10m` |
| Resolution | 10 m (matches target resolution) |
| Coverage | Global (Berlin included) |
| Temporal | Single snapshot: **2020** |
| License | CC-BY 4.0 |
| **Verdict** | ✅ **Recommended.** 10 m, complete spatial coverage, directly in GEE. |

#### Option B: Berlin Baumkataster (street trees only)

| Field | Detail |
|-------|--------|
| Source | **Geoportal Berlin / Berlin Open Data** — WFS |
| Format | Point data (GeoJSON, GML) |
| Coverage | ~440,000 street trees — **parks, forests, backyards not included** |
| Attributes | Tree species, planting year, trunk circumference, crown diameter, estimated height |
| License | dl-de/by-2-0 |
| **Verdict** | ⚠️ **Too incomplete** as standalone CHM. Use ETH CHM for complete canopy, supplement with street tree points for urban-specific analysis if needed. |

### 2.3 DEM, DOM & nDOM (Geländehöhe, Oberflächenmodell & Canopy Height)

Two approaches documented here:

#### Option A: Copernicus DEM GLO-30 (recommended for terrain)

| Field | Detail |
|-------|--------|
| Source (recommended) | **Copernicus DEM GLO-30** |
| Access | `s3://copernicus-dem-30m/` (no sign-in required, AWS eu-central-1) |
| Alternative access | GEE: `COPERNICUS/DEM/GLO30` |
| Resolution | 30 m |
| Coverage | Global (Berlin: fully covered) |
| Temporal | 2021 release |
| License | Free — Copernicus Programme |
| **Verdict** | ✅ **Available.** 30 m sufficient for terrain — Berlin is mostly flat (±50 m). |

#### Option B: Berlin DGM1 + DOM1 (1 m, high-res geometry)

| Field | Detail |
|-------|--------|
| Source | **Geoportal Berlin / Berlin Open Data** — INSPIRE Atom feeds |
| DGM1 (terrain model) | `https://gdi.berlin.de/data/dgm1/atom/` → `0.atom` |
| DOM1 (ALS surface model) | `https://gdi.berlin.de/data/dom/atom/` → `0.atom` |
| bDOM1 (image-based surface) | `https://gdi.berlin.de/data/bdom/atom/` → `0.atom` |
| Tile scheme | 2 km × 2 km tiles, EPSG:25833 (UTM 33N), DHHN2016 height |
| Example tile | `DGM1_390_5820.zip` (easting 390000, northing 5820000) |
| WMS view | `https://gdi.berlin.de/services/wms/dgm1` |
| Resolution | **1 m raster** |
| Coverage | Full Berlin + ~250 m Brandenburg buffer |
| Acquisition | ALS flights (Feb–Mar 2021), bDOM from 2024 aerial imagery |
| Format | ZIP of XYZ/CSV (easting, northing, height) |
| License | dl-de/by-2-0 (Datenlizenz Deutschland Zero 2.0) |
| Auth | None (fully open, no registration) |
| Update feed | GeoNetwork RSS: `https://gdi.berlin.de/geonetwork/srv/eng/rss.search?sortBy=changeDate` |
| **Verdict** | ✅ **Available and valuable for morphology validation.** Use DGM1 + DOM1 to compute nDOM = DOM − DGM for 1 m building + canopy height. LoD2 is cleaner for building-only geometry. |

#### nDOM (Normalized Digital Object Model) — Canopy Height

nDOM = DOM − DGM gives the height of surface objects (buildings + vegetation) at 1 m resolution.

**Comparison: Berlin nDOM vs. ETH Global Canopy Height 2020:**

| Criterion | nDOM from Berlin DGM1+DOM1 | ETH CHM 2020 |
|-----------|---------------------------|---------------|
| Resolution | **1 m** | 10 m |
| Coverage | Berlin only | Global (Berlin included) |
| Temporal | 2021 (ALS) | 2020 snapshot |
| What it measures | All surface objects (buildings + vegetation) | Vegetation height only (GEDI-trained) |
| Building separation | Needs building footprint mask (LoD2) | Already vegetation-only |
| Derivation | DOM − DGM (requires both downloads) | Ready-to-use GEE asset |
| Accuracy | ALS point density → cm-level terrain | ~50–70% canopy height RMSE at 10 m |
| License | dl/de/by-2-0 | CC-BY 4.0 |

**Recommendation:**
- **Terrain:** Copernicus DEM GLO-30 (30 m, global, no download hassle)
- **Vegetation height (predictor):** ETH CHM (10 m, ready-to-use, vegetation-only, GEE)
- **Building + canopy geometry for SVF/shadow:** Berlin DOM1 (1 m) or LoD2 for building-only geometry
- **nDOM is not recommended as primary canopy height input** — it requires additional downloads, registration not needed but tile-by-tile assembly required, and contains buildings that must be separated. ETH CHM is simpler and sufficient for 10 m ML features.

### 2.4 Versiegelungsgrad (Imperviousness / Sealing)

| Field | Detail |
|-------|--------|
| Source | **Umweltatlas Berlin** (via FIS-Broker) |
| URL | [https://fbinter.stadt-berlin.de/fb/index.jsp](https://fbinter.stadt-berlin.de/fb/index.jsp) (map: `versiegelung2021@senstadt`) |
| Resolution | **10 m raster** (matches target!) |
| Coverage | Full Berlin |
| Latest version | **2021** |
| Earlier editions | 2016, 2011, 2005, 2001, 1990 |
| Units | Sealing degree in % per cell (0–100) |
| Access | WMS/WFS/Download (GeoTIFF via FIS-Broker) |
| License | dl-de/by-2-0 |
| **Verdict** | ✅ **Excellent.** 10 m, Berlin-wide, regularly updated. Directly usable as predictor. |

### 2.5 Sky View Factor (abgeleitet aus LoD2 + DEM)

| Field | Detail |
|-------|--------|
| Source | **Derived** — computed from LoD2 building model + DEM |
| Method | Hemispherical view analysis at 10 m resolution |
| Tools | UMEP (QGIS), SAGA GIS, or custom `r.skyview` (GRASS GIS) |
| **Verdict** | 🔧 **Derived, not a data source.** Compute during feature engineering from LoD2 height + DEM. |

---

## Stage 3 — Verschattung & Sonnengeometrie

### 3.1 Sun Geometry per Scene

| Field | Detail |
|-------|--------|
| Source | **Satellite metadata** — not a separate download |
| Landsat | Properties `SUN_AZIMUTH`, `SUN_ELEVATION` in each scene |
| Sentinel-2 | Properties `MEAN_SOLAR_AZIMUTH`, `MEAN_SOLAR_ZENITH` in each scene |
| Custom timestamps | `pysolar` or `pvlib` for any (lon, lat, datetime) |
| **Verdict** | ✅ **Trivially available** from GEE metadata per composite/scene. |

### 3.2 Shadow Masks (abgeleitet aus LoD2 + Sun Position)

| Field | Detail |
|-------|--------|
| Source | **Derived** — ray-casting from LoD2 + sun azimuth/elevation + DEM |
| Tools | `r.sunmask` (GRASS GIS), ESA SNAP, UMEP, `shadow` Python package |
| Temporal | Dynamic: shadow pattern shifts with sun position per scene |
| **Verdict** | 🔧 **Derived, not a data source.** Compute per composite/scene during feature engineering. |

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
| Resolution | 0.1° × 0.1° (~9 km, ~11 × 7 km at Berlin) |
| Temporal | Hourly, **1950–present** (real-time through present) |
| Variables | Surface solar radiation downwards (SSRD), surface net solar radiation, surface thermal radiation, plus full meteorology (T2m, wind, precip, humidity) |
| Access | CDS API (`cdsapi`, free registration, CC-BY) |
| Update latency | ~5 days (ERA5-Land-T near-real-time), final product ~2 months |
| **Verdict** | ✅ **Best operational choice** — hourly resolution, 2018–2024 full coverage, easy API automation, same pipeline as reanalysis met variables. No direct/diffuse components. |

**Implementation note:** SSRD is accumulated energy (J/m²). Convert to hourly irradiance (W/m²) by dividing the hourly increment by 3600 s.

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
| Emissivity | 1 (predictor) | **ASTER GED v3 (GEE)** — NDVI-threshold rejected (collinearity with NDVI predictor) | 100 m | static | ✅ |
| Building height | 2 | LoD2 CityGML (BLC Berlin) | building vector | 2019 | ✅ |
| Building fraction | 2 | LoD2 → rasterized 10 m | 10 m | 2019 | ✅ |
| Canopy height | 2 | ETH Global Canopy Height (GEE) | 10 m | 2020 snapshot | ✅ |
| DEM / terrain | 2 | Copernicus DEM GLO-30 (AWS/GEE) | 30 m | 2021 static | ✅ |
| Imperviousness | 2 | Umweltatlas Versiegelung 2021 (FIS-Broker) | 10 m | 2021 | ✅ |
| Sky View Factor | 2 | Derived from LoD2 + DEM | 10 m | static | 🔧 |
| Sun azimuth / elevation | 3 | Landsat/S2 metadata (GEE) | per scene | per acquisition | ✅ |
| Shadow mask | 3 | Derived from LoD2 + sun position + DEM | 10 m | dynamic | 🔧 |
| Antecedent T | 4 | DWD Berlin-Dahlem or BER (hourly) | point | 1973–present | ✅ |
| Precip | 4 | DWD Berlin-Dahlem or BER (hourly) | point | 1995–present | ✅ |
| Wind speed | 4 | DWD Berlin Brandenburg BER (hourly) | point | 1973–present | ✅ |
| Solar radiation | 4 | ERA5-Land (CDS) | ~9 km grid | 1950–present | ✅ |
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
- **Solar radiation:** ERA5-Land is the best option (see §4.2). DWD does not measure direct solar radiation at Berlin stations. Alternatives: SARAH-3 (EUMETSAT CM SAF, 0.05°, hourly) for validation, CERES (1°, monthly) too coarse.
- **Dynamic World land cover (GEE):** Available 2015–present at 10 m (`GOOGLE/DYNAMICWORLD/V1`). Not in ablation plan but could replace/supplement imperviousness.
- **Nighttime ECOSTRESS:** ISS overpass includes ~1,000 night granules over Berlin. Not useful for daytime LST but potentially for diurnal cycle analysis.

---

*Generated by `notebooks/dwd_stations.py` + manual research; complements `docs/data-availability.md`.*
