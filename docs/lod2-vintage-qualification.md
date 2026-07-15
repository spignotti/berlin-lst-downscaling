# LoD2 Vintage Qualification & Temporal Policy

Status: Draft — 2026-07-15  
Context: Notion task `a699efb8-92b4-46d7-98b8-b5bc3d9f8dce`

## Temporal Policy for All Static/Quasi-Static Sources

Every static geometry source (DGM, LoD2, vegetation height, sealing) follows
the same strict mapping rule:

```text
selected_vintage = max(vintage in catalog
                       where effective_date <= scene_acquisition_date
                       AND coverage = full_AOI
                       AND licence = dl-de/zero-2.0)
```

If no vintage satisfies the conditions for a given scene year, that scene
**cannot** be used in the morphology ablation stage. The catalog records
every obtainable version, but only past-or-present versions are mapped
to scenes.

### Why not use the latest vintage retroactively?

- The ablation tests whether morphological features help downscaling.
  Retroactive data (e.g. 2024 LoD2 for a 2018 scene) would leak future
  knowledge about urban geometry into the training set.
- The study period is 2017–2025 (May–Sep, warm season). Any source
  published after the scene date is a future data point.
- The conservative approach is honest: a narrower but temporally valid
  mapping is preferable to a broader mapping with uncontrolled future leakage.

### Fallback if a source has no valid vintage for some years

Scenes without a qualifying static source vintage are excluded from the
morphology ablation stages (2–4) but may still be used in stage 1
(spectral only). The vintage mapping report must list all unmapped years
explicitly.

---

## Source Qualification Criteria

Every accepted source vintage must satisfy:

| Criterion | Minimum |
|-----------|---------|
| **Coverage** | Full Berlin AOI (4699×3884 pixels at 10 m) |
| **CRS** | EPSG:25833 (Berlin standard) |
| **Licence** | dl-de/zero-2.0 |
| **Effective date** | Documented publication or release date |
| **Semantic comparability** | Consistent height semantics, coordinate reference, and processing level across vintages |

If a vintage fails any criterion, it is rejected and recorded in the
qualification document with the rejection reason.

---

## Source 1: LoD2 CityGML (Geoportal Berlin)

### Current ATOM feed (verified 2026-07-15)

- URL: `https://gdi.berlin.de/data/a_lod2/atom/0.atom`
- Updated: 2026-03-26
- Format: CityGML v2.0 ZIP per 1 km × 1 km tile
- CRS: EPSG:25833
- Tiles: ~925 entries in feed (coverage varies)
- License: dl-de/zero-2.0

### Vintage discovery status

| Version | Effective date | Coverage | Status | Notes |
|---------|---------------|----------|--------|-------|
| ATOM feed (2026-03-26) | 2026-03-26 | Full Berlin | ❌ Future | Published after all scene dates (2017–2025). Cannot be used. |
| Berlin Open Data archive (2019) | 2019 | Berlin | ⚠️ To verify | Businesslocationcenter / Berlin Partner download portal. Historical CityGML. |
| Berlin 3D 2015 | ~2015 | Berlin districts | ⚠️ To verify | Earlier CityGML 1.0 extraction. Check CRS and height semantics. |
| BKG LoD2-DE | ~2022 | Germany-wide | ⚠️ To verify | Federal gazette data. Berlin coverage uncertain. |

### Qualification status

**No vintage has been fully qualified yet.** The current 2026 feed is
rejected as a future source. Historical versions must be located, downloaded,
and checked for CRS, CityGML version, height semantics, and AOI coverage
before any LoD2 morphology products can be produced.

### What to check for each candidate

1. **Download and hash** — archive SHA-256 for provenance
2. **CityGML version** — 1.0 vs 2.0 (different namespace for `bldg:measuredHeight`)
3. **Height semantics** — `measuredHeight` = ground-to-roof-top above ground
4. **Effective date** — publication date from metadata, not file modification date
5. **Coverage** — rasterized footprints must cover the full AOI with reasonable completeness
6. **License** — confirm dl-de/zero-2.0

---

## Source 2: DGM 1 m (Geoportal Berlin)

### Current ATOM feed (verified 2026-07-15)

- URL: `https://gdi.berlin.de/data/dgm1/atom/0.atom`
- Updated: 2025-12-18
- Format: XYZ CSV in ZIP (regular 1 m grid)
- CRS: EPSG:25833, DHHN2016
- Tiles: ~297 entries
- License: dl-de/zero-2.0
- Acquisition: ALS flights Feb–Mar 2021

### Vintage status

| Version | Effective date | Coverage | Status |
|---------|---------------|----------|--------|
| 2021 ALS | 2021-02/03 | Full Berlin | ⚠️ Future for 2017–2020 scenes |

The DGM is a terrain model (ground surface only), and terrain changes
very slowly in Berlin. However, to follow the strict temporal rule,
the 2021 DGM is a future source for 2017–2020 scenes. If no older
DGM is obtainable, scenes before 2021 will be excluded from
morphology-dependent stages or the DGM vintage will be documented as
a controlled exception (decision pending).

---

## Source 3: Vegetationshöhe 2020 (Umweltatlas Berlin)

### Current source (verified 2026-07-14)

- URL: `https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip`
- Format: GeoTIFF, 1 m, float32 with NoData
- CRS: EPSG:25833
- Coverage: Full Berlin
- License: dl-de/zero-2.0
- Pipeline adapter exists and is operational.

### Vintage status

| Version | Effective date | Coverage | Status |
|---------|---------------|----------|--------|
| 2020 | 2020 | Full Berlin | ⚠️ Future for 2017–2019 scenes |

For 2017–2019 scenes, the 2020 vegetation height is technically a
future source. Trees grow slowly; the 2020 product is likely a
reasonable proxy for 2017–2019, but the strict policy excludes it
from those years. Earlier Baumkataster/CHM versions may exist.

---

## Source 4: Versiegelung 2016/2021 (Umweltatlas Berlin)

### Current sources (verified 2026-07-14)

- 2016: `https://gdi.berlin.de/data/ua_versiegelung_2016/atom/Versiegelung_Raster_2016.zip`
- 2021: `https://gdi.berlin.de/data/ua_versiegelung_2021/atom/Versiegelung_Raster_2021.zip`
- Format: GeoTIFF, 2.5 m, uint8 class codes
- CRS: EPSG:25833
- Coverage: Full Berlin
- License: dl-de/zero-2.0
- Pipeline adapter exists and is operational.

### Vintage status

| Version | Effective date | Coverage | Status |
|---------|---------------|----------|--------|
| 2016 | 2016 | Full Berlin | ✅ Past for all scenes |
| 2021 | 2021 | Full Berlin | ⚠️ Future for 2017–2020 scenes |

The existing `vintage_for_scene_year()` maps ≤2019→2016, >2019→2021.
This is correct under the strict policy. Scenes from 2017–2019 get
the 2016 vintage (past). Scenes from 2020 would need a 2020 vintage,
but none exists—only 2016 and 2021. The 2021 vintage is technically
future for 2020. Decision: use 2016 for 2020 as well, or accept the
2021 vintage as a controlled exception.

---

## Next Steps

1. **Obtain historical LoD2 archives** — contact Berlin Partner/SenStadt
   for the 2015 and/or 2019 CityGML exports.
2. **Verify DGM history** — check whether any pre-2021 DGM version is
   available from Geoportal Berlin.
3. **Implement catalog** — create source manifest JSON files per vintage
   with checksums, effective dates, and coverage.
4. **Run qualification** — download, validate, and accept/reject each
   candidate.
5. **Emit mapping report** — for each scene year, list the qualifying
   vintage or state "unmapped".
