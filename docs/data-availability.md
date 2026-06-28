# Data Availability — Berlin LST Downscaling

Analysis date: 2026-06-18  
Query tool: Google Earth Engine (Python API)  
Service account: `masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com`

---

## Berlin Bounding Box

| Corner | Lon | Lat |
|--------|-----|-----|
| SW | 13.08 | 52.34 |
| NE | 13.76 | 52.68 |

WGS84 rectangle, covers Berlin + ~5 km buffer. Landsat WRS-2 paths covered: **192, 193, 194**.

---

## Landsat 8/9 TIRS (Surface Temperature)

**Collections:** `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`  
**Band of interest:** ST_B10 (Surface Temperature, ~100 m)  
**Cloud filter:** `CLOUD_COVER < 20%`

### Landsat 8 (LC08)

- **First scene:** 2013-03-24
- **Last scene:** 2026-06-04
- **Total scenes:** 1,034
- **Cloud-free (<20%):** 269 (26%)
- **Filtered out:** 765 (74%)

| Year | Total | <20% Cloud | Filtered |
|------|-------|------------|----------|
| 2013 | 53 | 12 | 41 |
| 2014 | 75 | 10 | 65 |
| 2015 | 89 | 26 | 63 |
| 2016 | 79 | 23 | 56 |
| 2017 | 66 | 12 | 54 |
| 2018 | 89 | 34 | 55 |
| 2019 | 84 | 23 | 61 |
| 2020 | 73 | 22 | 51 |
| 2021 | 77 | 14 | 63 |
| 2022 | 81 | 15 | 66 |
| 2023 | 75 | 18 | 57 |
| 2024 | 75 | 18 | 57 |
| 2025 | 80 | 23 | 57 |
| 2026 | 38 | 19 | 19 |

| Month | Total | <20% Cloud | % Usable |
|-------|-------|------------|----------|
| Jan | 42 | 5 | 12% |
| Feb | 70 | 18 | 26% |
| Mar | 90 | 35 | 39% |
| Apr | 104 | 33 | 32% |
| May | 113 | 32 | 28% |
| Jun | 106 | 28 | 26% |
| Jul | 116 | 20 | 17% |
| Aug | 113 | 34 | 30% |
| Sep | 103 | 31 | 30% |
| Oct | 98 | 23 | 23% |
| Nov | 68 | 9 | 13% |
| Dec | 11 | 1 | 9% |

### Landsat 9 (LC09)

- **First scene:** 2021-11-03
- **Last scene:** 2026-06-12
- **Total scenes:** 339
- **Cloud-free (<20%):** 93 (27%)

| Year | Total | <20% Cloud | Filtered |
|------|-------|------------|----------|
| 2021 | 7 | 0 | 7 |
| 2022 | 78 | 22 | 56 |
| 2023 | 70 | 17 | 53 |
| 2024 | 64 | 14 | 50 |
| 2025 | 84 | 23 | 61 |
| 2026 | 36 | 17 | 19 |

| Month | Total | <20% Cloud | % Usable |
|-------|-------|------------|----------|
| Jan | 15 | 3 | 20% |
| Feb | 25 | 7 | 28% |
| Mar | 29 | 15 | 52% |
| Apr | 34 | 9 | 26% |
| May | 45 | 12 | 27% |
| Jun | 35 | 10 | 29% |
| Jul | 34 | 7 | 21% |
| Aug | 29 | 10 | 34% |
| Sep | 33 | 11 | 33% |
| Oct | 28 | 5 | 18% |
| Nov | 28 | 4 | 14% |
| Dec | 4 | 0 | 0% |

### Landsat 8+9 Combined

- **Total:** 1,375
- **Cloud-free:** 362 (26%)
- **Effect of L9 join (2022–):** 2× scenes per year vs. L8 alone (~80 → ~160/year after 2021)

| Year | Total | <20% Cloud | Filtered |
|------|-------|------------|----------|
| 2013 | 53 | 12 | 41 |
| 2014 | 75 | 10 | 65 |
| 2015 | 89 | 26 | 63 |
| 2016 | 79 | 23 | 56 |
| 2017 | 66 | 12 | 54 |
| 2018 | 89 | 34 | 55 |
| 2019 | 84 | 23 | 61 |
| 2020 | 73 | 22 | 51 |
| 2021 | 84 | 14 | 70 |
| 2022 | 159 | 37 | 122 |
| 2023 | 145 | 35 | 110 |
| 2024 | 139 | 32 | 107 |
| 2025 | 164 | 46 | 118 |
| 2026 | 76 | 36 | 40 |

### Landsat 8+9 — May–Sep Only (<20% Cloud)

Since the training window is **May–September**, the full-year counts above overstate usable scenes for the target period. Below are Landsat scenes within the seasonal window only.

| Year | Total (May–Sep) | <20% Cloud | Usable |
|------|----------------|------------|--------|
| 2013 | 33 | 9 | 27% |
| 2014 | 37 | 4 | 11% |
| 2015 | 47 | 9 | 19% |
| 2016 | 43 | 15 | 35% |
| 2017 | 41 | 8 | 20% |
| 2018 | 46 | 19 | 41% |
| 2019 | 42 | 8 | 19% |
| 2020 | 44 | 12 | 27% |
| 2021 | 38 | 6 | 16% |
| 2022 | 83 | 22 | 27% |
| 2023 | 88 | 31 | 35% |
| 2024 | 78 | 20 | 26% |
| 2025 | 82 | 21 | 26% |
| 2026 | 27 | 11 | 41% |

**May–Sep cloud-free scenes per year × month** (Landsat 8+9 combined):

| Year | May | Jun | Jul | Aug | Sep | Sum |
|------|-----|-----|-----|-----|-----|-----|
| 2017 | 2 | 3 | 0 | 3 | 0 | 8 |
| 2018 | 7 | 1 | 5 | 4 | 2 | **19** |
| 2019 | 0 | 4 | 2 | 1 | 1 | 8 |
| 2020 | 1 | 4 | 0 | 3 | 4 | 12 |
| 2021 | 0 | 0 | 2 | 0 | 4 | 6 |
| 2022 | 4 | 4 | 3 | 8 | 3 | **22** |
| 2023 | 7 | 9 | 1 | 4 | 10 | **31** |
| 2024 | 5 | 2 | 4 | 4 | 5 | **20** |
| 2025 | 1 | 2 | 4 | 8 | 6 | 21 |

**Window totals (May–Sep, <20% cloud):**

| Window | Total scenes | Cloud-free | Usable |
|--------|-------------|------------|--------|
| 2017–2024 | 460 | 126 | 27% |
| 2018–2024 | 419 | 118 | 28% |

---

## Sentinel-2 L2A (Surface Reflectance)

**Collection:** `COPERNICUS/S2_SR_HARMONIZED`  
**Bands of interest:** B2, B3, B4, B8 (10 m predictors)  
**Cloud filter:** `CLOUDY_PIXEL_PERCENTAGE < 20%`  
**Satellites:** Sentinel-2A (3,295 scenes), Sentinel-2B (3,104), Sentinel-2C (518)

- **First scene:** 2015-07-04
- **Last scene:** 2026-06-16
- **Total scenes:** 6,917
- **Cloud-free (<20%):** 1,336 (19%)
- **Filtered out:** 5,581 (81%)

| Year | Total | <20% Cloud | Filtered |
|------|-------|------------|----------|
| 2015 | 2 | 2 | 0 |
| 2016 | 15 | 15 | 0 |
| 2017 | 293 | 48 | 245 |
| 2018 | 730 | 205 | 525 |
| 2019 | 737 | 164 | 573 |
| 2020 | 728 | 158 | 570 |
| 2021 | 737 | 116 | 621 |
| 2022 | 734 | 130 | 604 |
| 2023 | 738 | 92 | 646 |
| 2024 | 738 | 117 | 621 |
| 2025 | 1,009 | 184 | 825 |
| 2026 | 456 | 105 | 351 |

| Month | Total | <20% Cloud | % Usable |
|-------|-------|------------|----------|
| Jan | 554 | 72 | 13% |
| Feb | 558 | 93 | 17% |
| Mar | 598 | 157 | 26% |
| Apr | 616 | 191 | 31% |
| May | 638 | 142 | 22% |
| Jun | 603 | 120 | 20% |
| Jul | 554 | 87 | 16% |
| Aug | 571 | 125 | 22% |
| Sep | 552 | 150 | 27% |
| Oct | 571 | 79 | 14% |
| Nov | 531 | 54 | 10% |
| Dec | 571 | 66 | 12% |

**Note:** 2015–2016 have very few scenes (S2A launched mid-2015, full operations by 2017). 2017+ is the operational period.

---

## ECOSTRESS

**Product:** `ECO_L2T_LSTE v002` (Gridded Land Surface Temperature & Emissivity, 70 m, MGRS-tiled COGs)  
**Access:** AppEEARS REST API via `appeears_client.py` (primary) / NASA CMR via `earthaccess` (listing)  
**Auth:** `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` in `.env` (AppEEARS), `EARTHDATA_TOKEN` (CMR)  
**Period in catalog:** 2018-07-09 – present (2026+)

**Important — Collection 1 vs Collection 2:**
- The old GEE catalog (`ECOSTRESS/ECO2LSTE.001`) covers only up to 2023-05-05 → this is where the "stops at 2023" impression comes from
- **Collection 2** (`ECO_L2T_LSTE.002`) is actively available through present via AppEEARS/CMR — the LP DAAC continues forward processing in V002
- GEE has only ingested V002 for the Los Angeles area, not globally
- **Berlin has 8,708+ granules through 2026-06-24** in Collection 2

**Access strategy:**
1. **Scene listing:** CMR query via `earthaccess.search_data(short_name="ECO_L2T_LSTE", version="002", ...)`
2. **Download:** AppEEARS API — submit area task with Berlin AOI as GeoJSON, poll for completion, download GeoTIFF bundle
3. **Post-processing:** Convert GeoTIFFs to COGs (float32, NaN-NoData, 512×512 tiles, overviews) via rasterio, upload to GCS as `ard/validation/ecostress/{year}/`

| Year | Granules (Collection 2) |
|------|------------------------|
| 2018 | ~300 |
| 2019 | ~620 |
| 2020 | ~740 |
| 2021 | ~1,200 |
| 2022 | ~1,900 |
| 2023 | ~1,200 |
| 2024 | ~1,700 |
| 2025 | ~1,000 |
| 2026 | ~100 (partial) |
| **Total** | **8,708+** |

**Note:** Berlin at 52.5°N is at the northern edge of the ISS observation corridor. The ISS orbit covers 51.6°S–51.6°N; the sensor swath extends coverage to ~53.6°N. This results in fewer overpasses than equatorial regions and higher view-zenith angles.

**Granules are MGRS-tiled.** A single ISS overpass can produce multiple granules over the Berlin area, so granule counts overstate unique observation times. The ECOSTRESS ISS orbit crosses Berlin at varying local times (no fixed sun-synchronous schedule).

**LST-only validation source.** ECOSTRESS does not carry a high-resolution multispectral imager, so it cannot serve as a predictor — only as an independent thermal check.

**Verdict:** ECOSTRESS Collection 2 via AppEEARS provides strong independent validation across **2018–present**, removing the earlier 2023 limitation. This extends validation coverage to all training window candidates (2018–2024, 2018–2025).

---

## Key Findings & Recommendations

### Cloud reality check

Across both Landsat and Sentinel-2, **only 19–26% of scenes are below 20% cloud cover.** Without cloud filtering, the stated scene counts are severely misleading. The summer months (Jun–Aug) — when LST analysis matters most — are even worse: only **17–22%** of scenes are usable for Landsat, **20–22%** for Sentinel-2.

→ **Individual scene-level analysis is not viable.** A compositing strategy is essential.

### Composite strategy implications

If we use **monthly composites** (one LST map per month):
- **Landsat:** ~2.3 cloud-free scenes/month in summer (27 per Jun–Aug split across 3 months). Barely enough for a median composite.
- **Sentinel-2:** ~6.9 cloud-free scenes/month in summer for predictors. Comfortable for compositing.
- **Together:** 1 Landsat LST scene + ~7 Sentinel-2 scenes per summer month is a realistic training pair volume.

### Temporal coverage

- **Landsat 8 alone:** 2013–present. Provides 10+ years of data.
- **Landsat 8+9:** 2022–present doubles scene density. L9 data is an exact match to L8 (same TIRS sensor).
- **Sentinel-2:** Fully operational from 2017. 2015–2016 are sparse.

### Recommended training period

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Start year | **2018** | First full year of S2 operations + stable L8 data. 2017 is viable for Landsat-only training (41 May–Sep scenes, 8 cloud-free) but ECOSTRESS validation only starts 2018 — including 2017 means no ECOSTRESS validation for that year. |
| End year | **2024** | Full calendar year with L8+L9 at full capacity. 2025 data exists (82 May–Sep scenes, 21 cloud-free) and can be added, but ECOSTRESS validation data stops at 2023 — adding 2024/2025 for training means no ECOSTRESS validation for those years. |
| Seasonal window | **May–September** | Core LST season for urban heat analysis. 5 months × 7 years = 35 monthly composites. |
| Compositing | Monthly median | One composite per site per month. Fill gaps with per-pixel medians from available scenes. |

**Year-window tradeoff summary:**

| Start | End | Landsat scenes | ECOSTRESS validation | Landsat-only years |
|-------|-----|---------------|---------------------|-------------------|
| 2018 | 2024 | 118 cloud-free (7 yrs) | ✅ 2018–2023 (6 yrs) | 0 |
| 2017 | 2024 | 126 (+8 more) | ⚠️ 2018–2023 (no ECOSTRESS for 2017) | 1 |
| 2018 | 2025 | 139 (+21 more) | ⚠️ 2018–2023 (no ECOSTRESS for 2024–2025) | 2 |

**Verdict:** **2018–2024** is the safest default — it maximizes ECOSTRESS validation coverage (6 of 7 years validated). Extending to 2017 (+6%) or 2025 (+18%) adds marginal training data but at the cost of unvalidated years. If more training data is needed (low-shot scenario), include 2025 first (better cloud-free count), then 2017. 2025 also has the benefit of Sentinel-2C joining the constellation, increasing scene density.

### Open decisions

- **Landsat 7:** Not queried (SLC-off stripes introduce artifact handling complexity). Available 1999–present if needed.
- **ECOSTRESS:** Manual access resolution needed if ECOSTRESS coverage turns out to be required.
- **Cloud threshold:** 20% used here. A stricter threshold (10%) would halve the usable scenes. A more lenient one (30%) adds ~15% more.
- **Temporal compositing:** Monthly median assumed here. Alternative: use the single best scene per month + cloud-masking. Decision tbd once data pipeline is built.

---

*Generated by `notebooks/data_availability.py` — re-runnable after service account permissions are updated.*
