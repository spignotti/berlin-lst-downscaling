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

### 2.3 DEM (Geländehöhe)

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

**Note:** Berlin DGM1 (1 m) exists via Geoportal but is unnecessary detail for this use case.

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

### 4.2 ERA5-Land (complementary source)

| Field | Detail |
|-------|--------|
| Source | **Copernicus Climate Data Store** |
| URL | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) |
| Resolution | 0.1° (~9 km) — gridded |
| Temporal | Hourly, **1950–present** |
| Variables | 2m temperature, surface solar radiation downwards, 10m wind, total precipitation, 2m dewpoint temperature, surface thermal radiation |
| Access | CDS API (`cdsapi` Python package, requires free registration) |
| License | CC-BY |
| **Verdict** | ✅ **Valuable complement** — fills the solar radiation gap from DWD. Gridded data means no station outages. ~9 km is coarse but acceptable for regional antecedent weather. |

**Recommendation:** Use DWD station data as primary (higher accuracy at point location) and ERA5-Land for solar radiation (not available from Berlin DWD stations).

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
| Emissivity | 1 (predictor) | ASTER GED v3 (GEE) / NDVI-based estimate | 100 m / pixel | static | ✅ |
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

## Open Decisions

- **Landsat 7:** Not queried (SLC-off stripes). Still available 1999–present if additional historical data is needed.
- **Solar radiation:** ERA5-Land is the best option. DWD does not measure direct solar radiation at Berlin stations. Alternatives: SARAH-3 (EUMETSAT CM SAF, 0.05°, hourly), CERES (1°, monthly).
- **Dynamic World land cover (GEE):** Available 2015–present at 10 m. Not in ablation plan but could replace/supplement imperviousness.
- **Nighttime ECOSTRESS:** ISS overpass includes ~1,000 night granules over Berlin. Not useful for daytime LST but potentially for diurnal cycle analysis.

---

*Generated by `notebooks/dwd_stations.py` + manual research; complements `docs/data-availability.md`.*
