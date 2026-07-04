# ARD Output Contracts

## Overview

Each sensor produces:
- One main COG per scene (float32 bands) — ``{scene_id}.tif``
- One flag COG per scene (uint8 bitmask) — ``{scene_id}.flag.tif``
- One STAC item — ``{scene_id}.stac.json``

All artefacts live under a deterministic path:

```
<output_root>/{source}/{year}/{scene_id}/
    {scene_id}.tif        ← Main COG (float32 data bands)
    {scene_id}.flag.tif   ← Flag COG (uint8 bitmask, LZ4)
    {scene_id}.stac.json  ← STAC item
```

## Common Parameters

| Property | Value |
|---|---|
| Target CRS | EPSG:25833 (ETRS89 / UTM zone 33N) |
| Float nodata | NaN (IEEE 754) |
| Flag nodata | — (fill is encoded as bit 0 of the flag byte) |
| COG tiling | 512 × 512 px internal tiles |
| Overviews | 2, 4, 8, 16 |
| Main COG compression | **deflate** + predictor 2 (horizontal differencing) |
| Flag COG compression | **ZSTD** (fast, good for uint8 bitmask data) |

> **Note:** Main COG compression changed from LZ4 → **deflate** after the Phase A audit.
> LZ4 was the initial choice but is unsupported by the local GDAL build.
> Deflate is lossless with better compression ratios for float data.
> Flag COG uses ZSTD (also unavailable in LZ4, ZSTD is fast and effective on sparse uint8 data).

## Flag Band (shared across sensors)

Stored as a **separate single-band uint8 COG** (``.flag.tif``) to avoid
promoting uint8 to float32 in the main multi-band COG. Bitmask semantics:

| Bit | Value | Name | Sensor support |
|---|---|---|---|
| 0 | 1 | fill / nodata / dilated cloud buffer | LS, S2, EC |
| 1 | 2 | cloudy (high confidence) | LS (qa_pixel bit 3 + conf ≥ med), S2 (SCL class 8 or 9) |
| 2 | 4 | cloud shadow | LS (qa_pixel bit 4), S2 (directional-offset projection, see below) |
| 3 | 8 | cirrus / thin cirrus | LS (cirrus bit 2), S2 (SCL class 10) |
| 4 | 16 | saturated / invalid | S2 only (SCL class 1) |
| 5–7 | — | reserved (snow, terrain occlusion, future Stage 3) | — |

`clear_pixel = (flag & 0b00111) == 0`.

## Landsat C2 L2 — `landsat-c2-l2`

**Main COG bands:**

| Band | dtype | Units | Source scaling |
|---|---|---|---|
| `st` | float32 | Kelvin | Raw `lwir11` DN × 0.00341802 + 149.0 (USGS Collection 2 Level-2 ST scale) |

Flag band is a separate ``.flag.tif`` COG (see §Flag Band above).

- **Spatial resolution:** 100 m (no upsampling — anti-leakage, per task body).
- **Cloud mask:** `qa_pixel` bit 3 (cloud) with confidence ≥ medium, dilated by 2 px. Cloud shadow: bit 4 (USGS-geometric, best available).
- **Only ST kept** — spectral/SR bands removed per task body.

## Sentinel-2 L2A — `sentinel-2-l2a`

**Main COG bands:**

| Band | dtype | Units | Source scaling |
|---|---|---|---|
| `B02` | float32 | reflectance 0–1 | DN / 10000 (Baseline‑04.00) |
| `B03` | float32 | reflectance 0–1 | DN / 10000 |
| `B04` | float32 | reflectance 0–1 | DN / 10000 |
| `B08` | float32 | reflectance 0–1 | DN / 10000 |

Flag band is a separate ``.flag.tif`` COG (see §Flag Band above).

- **Spatial resolution:** 10 m.
- **Cloud mask:** SCL classes 8 (medium probability) and 9 (high probability).
  ``s2cloudless`` is **not available** on PC Sentinel-2 L2A items.
- **Cloud-shadow mask:** SCL class 3 (lower bound) augmented by directional-offset
  projection (Option B, see §Cloud-Shadow Projection below). **Not** ray-cast —
  DSM-occluded shadows are deferred to Stage 3 (Sekundärdaten-Pipeline, SVF/Shadow module).
- **Cirrus:** SCL class 10.
- **Saturated:** SCL class 1.

### Cloud-Shadow Projection (S2 directional offset)

Geometric directional-offset method, not ray-cast:

```
shadow_xy = cloud_xy + (Δx, Δy)

Δx = -h × tan(zenith) × sin(azimuth)
Δy = -h × tan(zenith) × cos(azimuth)
```

Where:

| Parameter | Default | Source |
|---|---|---|---|
| `h` (cloud base height) | 1000 m | Hydra `cloud_base_height_m` |
| `sun_elevation` | per scene | NOAA computation from datetime + lat (S2 items on PC lack ``view:sun_elevation``) |
| `sun_azimuth` | per scene | NOAA computation from datetime + lat |

- `zenith = 90° − elevation`.
- Shadow is cast from each cloud pixel via nearest-neighbour shift (0-order).
- At solar zenith zenith < 0.5° (sun near zenith): no shadow cast.
- **Limitation:** shadows behind tall buildings (DSM-occluded) are not caught — that requires ray-cast through DSM (Stage 3).
- For Berlin's 3 MGRS tiles, shadows are projected per-tile and flagged per-scene. Inter-tile shadow stitching is not performed in Phase A.

## ECOSTRESS — `ecostress`

**Main COG bands:**

| Band | dtype | Units | Source |
|---|---|---|---|
| `lst` | float32 | Kelvin | ECO_L2T_LSTE.002 native |

Flag band is a separate ``.flag.tif`` COG (bitmask; see below).

**Flag bitmask (shared across all sources):**

| Bit | Constant | Meaning |
|---|---|---|
| 0 | ``FLAG_FILL`` | fill / outside granule / water body / QC not-produced |
| 1 | ``FLAG_CLOUDY`` | high-confidence cloud OR TES degraded quality |
| 2 | ``FLAG_SHADOW`` | not used for ECOSTRESS (no shadow projection in L2T) |
| 3 | ``FLAG_CIRRUS`` | not used for ECOSTRESS |
| 4 | ``FLAG_SATURATED`` | not used for ECOSTRESS |

**Masking logic (Collection 2 L2T semantics):**

| Condition | Flag bit set |
|---|---|
| ``cloud == 255`` (fill) | ``FLAG_FILL`` |
| ``cloud == 1`` (cloudy) | ``FLAG_CLOUDY`` |
| ``water == 1`` (water body) | ``FLAG_FILL`` |
| ``water == 255`` (fill) | ``FLAG_FILL`` |
| ``(QC & 0b11) == 3`` (pixel not produced) | ``FLAG_FILL`` |
| ``(QC & 0b11) == 1`` (TES degraded) | ``FLAG_CLOUDY`` |

- **Spatial resolution:** ~70 m native, reprojected to EPSG:25833 at 70 m.
- **Native grid:** MGRS UTM tiles, 1568×1568 px per granule.
- **Acquisition:** Raw L2T data is **staged** (not permanently stored): NASA Earthdata S3 → local tmp via ``earthaccess`` → upload to stage URI (``local`` or ``gs://``) → pipeline reads → stage deleted after processing.
- **Granule discovery:** CMR query via ``earthaccess`` (``scripts/build_manifest.py --config-name full_2017_2025``).
- **Staging model:** ``scripts/download_ecostress_fixture.py`` stages raw L2T COGs to a per-run URI (``gs://berlin-lst-data/_staging/ecostress/<run_id>`` or ``data/tmp/ecostress_stage/<run_id>``). The pipeline's ``run_ard_ecostress.py`` cleans up the stage in its ``finally`` block (``persist_stage=false``). Set ``persist_stage=true`` to keep the stage for debugging.
- **GCS staging ops note:** The ``gs://berlin-lst-data/_staging/`` prefix should have a bucket-level lifecycle policy that auto-deletes objects after 7 days as a safety net.
- **Local smoke test:** ``nox -s smoke-ecostress`` — stages to ``data/tmp/ecostress_stage/<run_id>/``.
- **Cloud smoke test:** ``nox -s smoke-ecostress-cloud`` — stages to ``gs://berlin-lst-data/_staging/ecostress/<run_id>/``.

## Schema Hash

Each contract provides a deterministic BLAKE3 hash computed over:

```
{source}::{target_crs}::{band_name}|{dtype}|{nodata}::…::blocksize={v}::overviews={…}::compress={c}::predictor={p}::flag_mode={fm}::v{schema_version}
```

Fields:
- ``flag_mode`` — ``separate`` (own COG), ``inline`` (mixed into main COG, deprecated), or ``none``

This hash is stored in:
- STAC `properties.ard:schema_hash`
- Ledger `schema_hash` column

Idempotency check: if the stored hash matches the current contract's hash, the COG is considered valid and skipped. A mismatch forces reprocessing (e.g., after a contract change).

## Schema Version

Current: `4` (AOI overlap field added).

| Version | Change |
|---|---|---|
| 1 | Initial Phase A. Flag band inline in main COG (uint8→float32 promotion). Compression LZ4. |
| 2 | Flag band split into separate ``.flag.tif`` (uint8, ZSTD). Main COG uses deflate. Schema hash includes ``flag_mode`` field. |
| 3 | AOI metrics added to ledger: ``aoi_clear_px``, ``aoi_cloudy_px``, ``aoi_shadow_px``, ``aoi_cirrus_px``, ``aoi_saturated_px``, ``aoi_fill_px``, ``aoi_total_px``, ``aoi_clear_frac``. Flag COG compression ZSTD. |
| 4 | ``aoi_overlap_px`` added: count of all pixels (including fill) in the COG∩AOI intersection. Enables detection of off-target swaths where the COG covers the AOI bbox but LST data is absent. |

The `schema_version` column in the ledger is `int` and fields start at `1`. Never mutate in place — increment for breaking changes and document the delta in this file.

## Output Storage

The pipeline writes to ``output_root`` which accepts three URI schemes:

| Scheme | Example | Notes |
|---|---|---|
| **Local POSIX** | ``data/ard`` | ``os.replace`` atomic rename. ``.tmp`` sibling dir for staging. |
| **GCS** | ``gs://berlin-lst-data/ard/`` | ``copy_blob`` + ``delete`` (eventually consistent; tolerated by reconcile). |
| **FUSE mount** | ``~/.mnt/berlin-lst/ard/`` | Best-effort atomic via ``os.replace``. GDAL ZSTD support required. |

All writers (``write_cog_atomic``, ``write_flag_cog_atomic``, ``write_stac_atomic``) and the ledger use ``atomic_write`` from ``berlin_lst_downscaling.data.io.storage``, which dispatches by URI prefix.

Smoke tests always write locally (``data/tmp/smoke_ard_<date>``). Production runs targeting GCS must set ``output_root: gs://berlin-lst-data/ard/`` in the config.

## AOI Metrics (Schema v4)

Per-scene AOI pixel counts are computed by intersecting the flag COG with a pre-rasterized Berlin Landesgrenze mask (``aoi_10m.tif``, ``aoi_100m.tif``). Fields stored in the ledger:

| Field | Type | Description |
|---|---|---|
| ``aoi_clear_px`` | int | Clear (non-cloud/shadow/cirrus/saturated/fill) pixels inside Berlin |
| ``aoi_cloudy_px`` | int | Cloudy pixels inside Berlin |
| ``aoi_shadow_px`` | int | Cloud-shadow pixels inside Berlin |
| ``aoi_cirrus_px`` | int | Cirrus pixels inside Berlin |
| ``aoi_saturated_px`` | int | Saturated pixels inside Berlin |
| ``aoi_fill_px`` | int | Fill (no-data / dilated buffer) pixels inside Berlin |
| ``aoi_total_px`` | int | All non-fill pixels inside Berlin (sum of above six categories) |
| ``aoi_overlap_px`` | int | **All** pixels in COG∩AOI intersection (including fill) — detects off-target swaths |
| ``aoi_clear_frac`` | float | ``aoi_clear_px / aoi_total_px`` (NaN if ``aoi_total_px == 0``) |

The mask COGs (``data/boundaries/aoi_10m.tif``, ``data/boundaries/aoi_100m.tif``) are pre-baked by ``scripts/build_aoi.py`` from ``berlin_landesgrenze.geojson`` (EPSG:25833). They must be in the same CRS as the flag COG; reprojection to the scene grid is performed at metrics time if needed.
