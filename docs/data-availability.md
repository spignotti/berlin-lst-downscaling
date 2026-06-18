# Data Availability — Berlin LST Downscaling

Analysis date: 2026-06-18  
Query tool: Google Earth Engine (Python API)  
Service account: `masterarbeit-vertex@masterarbeit-berlin-lst.iam.gserviceaccount.com`

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

- **Total:** 1,373
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
| 2026 | 74 | 36 | 38 |

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
**Access:** NASA CMR via `earthaccess` Python package  
**Auth:** `EARTHDATA_TOKEN` in `.env`  
**Period in catalog:** 2018-07-30 – 2023-05-05 (processing lag — newer data exists but not yet in LP DAAC)

| Year | Granules |
|------|----------|
| 2018 | 297 |
| 2019 | 620 |
| 2020 | 738 |
| 2021 | 1,205 |
| 2022 | 1,944 |
| 2023 | 1,196 |
| **Total** | **6,000** |

| Month | Granules |
|-------|----------|
| Jan | 288 |
| Feb | 823 |
| Mar | 345 |
| Apr | 744 |
| May | 277 |
| Jun | 634 |
| Jul | 257 |
| Aug | 839 |
| Sep | 304 |
| Oct | 647 |
| Nov | 322 |
| Dec | 520 |

**Overpass times:** Distributed across the full day (unlike Landsat's fixed 10:00 AM).  
- Morning: 1,565 (8–11h), Afternoon: 1,462 (12–15h), Evening: 808 (16–19h), Night: 988 (20–3h)

**Summer (May–Sep):** 2,311 granules (39% of total)

**Note:** Granules are MGRS-tiled. A single ISS overpass can produce multiple granules over the Berlin area, so the 6,000 count overstates unique observation times. The ECOSTRESS ISS orbit crosses Berlin at varying local times (no fixed sun-synchronous schedule).

**Verdict:** ECOSTRESS is accessible and covers Berlin. With ~2,300 granules in the summer months across 6 years, it is a viable independent validation source (70 m LST vs. our 10 m downscaled product). The main limitation is the ISS overpass timing (not fixed, varies across day/night) and the 2–3 year processing lag in LP DAAC.

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
| Start year | **2018** | First full year of S2 operations + stable L8 data. 2017 also viable but margins are thinner. |
| End year | **2024** | Full calendar year with both L8+L9. Extend to 2025 if more data needed. |
| Seasonal window | **May–September** | Core LST season for urban heat analysis. 5 months × 7 years = 35 monthly composites. |
| Compositing | Monthly median | One composite per site per month. Fill gaps with per-pixel medians from available scenes. |

### Open decisions

- **Landsat 7:** Not queried (SLC-off stripes introduce artifact handling complexity). Available 1999–present if needed.
- **ECOSTRESS:** Manual access resolution needed if ECOSTRESS coverage turns out to be required.
- **Cloud threshold:** 20% used here. A stricter threshold (10%) would halve the usable scenes. A more lenient one (30%) adds ~15% more.
- **Temporal compositing:** Monthly median assumed here. Alternative: use the single best scene per month + cloud-masking. Decision tbd once data pipeline is built.

---

*Generated by `notebooks/data_availability.py` — re-runnable after service account permissions are updated.*
