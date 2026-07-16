# LoD2 Vintage Qualification & Temporal Policy

Status: Updated — 2026-07-16  
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
**cannot** be used in the morphology ablation stage.

**Decision:** Scene-year-to-vintage mapping is deferred to feature assembly,
not to the source publication pipeline. Pipeline A publishes all available
revisions unconditionally; the mapping logic lives in the feature engineering
stage.

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

## Accepted Source Revisions

The following source revisions have been validated and published by
Pipeline A. They are used unconditionally for source product publication;
temporal assignment to scenes happens at feature-assembly time.

| Source | Vintage | Effective date | Coverage | Status | Notes |
|--------|---------|---------------|----------|--------|-------|
| imperviousness | 2016 | 2016 | Full Berlin | ✅ Published | Past for all scenes |
| imperviousness | 2021 | 2021 | Full Berlin | ✅ Published | Future for 2017–2020 scenes |
| vegetation_height | 2020 | 2020 | Full Berlin | ✅ Published | Future for 2017–2019 scenes |
| terrain_height | 2021 | 2021 (ALS Feb–Mar) | Full Berlin | ✅ Published | Future for 2017–2020 scenes |
| lod2_morphology | 2024 | 2024-04-22 (data revision) | Full Berlin | ✅ Published | Future for all scenes |

---

## Source 1: LoD2 CityGML (Geoportal Berlin)

### Current ATOM feed (verified 2026-07-15)

- URL: `https://gdi.berlin.de/data/a_lod2/atom/0.atom`
- Updated: 2026-03-26
- Format: CityGML v2.0 ZIP per 1 km × 1 km tile
- CRS: EPSG:25833
- Tiles: ~925 entries in feed (coverage varies)
- License: dl-de/zero-2.0

### Qualification status

**Revision 2024-04-22 accepted for source publication.** The feed is
technically future for all study scenes (2017–2025), but is the only
available LoD2 dataset covering full Berlin. Scene-year mapping is
deferred to feature assembly.

### Parser notes

- CityGML v1.0 and v2.0 are both supported (different namespaces).
- `Building` elements are parsed; `BuildingPart` children are ignored.
- `measuredHeight` = ground-to-roof-top above ground.
- Rasterized bands: mean height, std, BCR, max height.

---

## Source 2: DGM 1 m (Geoportal Berlin)

### Current ATOM feed (verified 2026-07-15)

- URL: `https://gdi.berlin.de/data/dgm1/atom/0.atom`
- Updated: 2025-12-18
- Format: XYZ CSV in ZIP (variable-size grid at 1 m spacing)
- CRS: EPSG:25833, DHHN2016
- Tiles: ~297 entries
- License: dl-de/zero-2.0
- Acquisition: ALS flights Feb–Mar 2021

### Vintage status

| Version | Effective date | Coverage | Status |
|---------|---------------|----------|--------|
| 2021 ALS | 2021-02/03 | Full Berlin | ✅ Published (future for 2017–2020 scenes) |

Edge tiles may have incomplete point coverage (fewer than 2000×2000 points).
The parser handles this via coordinate-lookup accumulation.

---

## Source 3: Vegetationshöhe 2020 (Umweltatlas Berlin)

- URL: `https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip`
- Format: GeoTIFF, 1 m, float32 with NoData
- CRS: EPSG:25833
- Coverage: Full Berlin
- License: dl-de/zero-2.0
- ✅ Published (future for 2017–2019 scenes)

---

## Source 4: Versiegelung 2016/2021 (Umweltatlas Berlin)

- 2016: `https://gdi.berlin.de/data/ua_versiegelung_2016/atom/Versiegelung_Raster_2016.zip`
- 2021: `https://gdi.berlin.de/data/ua_versiegelung_2021/atom/Versiegelung_Raster_2021.zip`
- Format: GeoTIFF, 2.5 m, uint8 class codes
- CRS: EPSG:25833
- Coverage: Full Berlin
- License: dl-de/zero-2.0
- ✅ Both vintages published

---

## Next Steps

1. **Implement vintage mapping** — in the feature engineering stage, map each
   scene year to the correct source vintage(s) using the strict temporal rule.
2. **Emit mapping report** — for each scene year, list the qualifying vintage
   or state "unmapped".
3. **Exclude unmapped scenes** from morphology ablation stages (2–4).
