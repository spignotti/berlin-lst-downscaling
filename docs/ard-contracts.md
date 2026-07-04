# ARD Output Contracts

## Overview

Each sensor produces one COG per scene per source plus one STAC item.
All artefacts live under a deterministic path:

```
<output_root>/{source}/{year}/{scene_id}/
    {scene_id}.tif        ← COG
    {scene_id}.stac.json  ← STAC item
```

## Common Parameters

| Property | Value |
|---|---|
| Target CRS | EPSG:25833 (ETRS89 / UTM zone 33N) |
| Float nodata | NaN (IEEE 754) |
| Integer nodata (flag) | — (fill is encoded as bit 0 of the flag byte) |
| COG tiling | 512 × 512 px internal tiles |
| Overviews | 2, 4, 8, 16 |
| Compression | LZ4 (lossless, fast) |
| Predictor | 2 (horizontal differencing for float bands) |

## Flag Band (shared across sensors)

Single `uint8` band with bitmask semantics:

| Bit | Value | Name | Sensor support |
|---|---|---|---|
| 0 | 1 | fill / nodata / dilated cloud buffer | LS, S2, EC |
| 1 | 2 | cloudy (high confidence) | LS (qa_pixel bit 3 + conf ≥ med), S2 (s2cloudless ≥ 40) |
| 2 | 4 | cloud shadow | LS (qa_pixel bit 4), S2 (directional-offset projection, see below) |
| 3 | 8 | cirrus / thin cirrus | LS (cirrus bit 2), S2 (SCL class 10) |
| 4 | 16 | saturated / invalid | S2 only (SCL class 1) |
| 5–7 | — | reserved (snow, terrain occlusion, future Stage 3) | — |

`clear_pixel = (flag & 0b00111) == 0`.

## Landsat C2 L2 — `landsat-c2-l2`

| Band | dtype | Units | Source scaling |
|---|---|---|---|
| `st` | float32 | Kelvin | Raw `lwir11` DN × 0.00341802 + 149.0 (USGS Collection 2 Level-2 ST scale) |
| `flag` | uint8 | — | See flag-band spec above |

- **Spatial resolution:** 100 m (no upsampling — anti-leakage, per task body).
- **Cloud mask:** `qa_pixel` bit 3 (cloud) with confidence ≥ medium, dilated by 2 px. Cloud shadow: bit 4 (USGS-geometric, best available).
- **Only ST kept** — spectral/SR bands removed per task body.

## Sentinel-2 L2A — `sentinel-2-l2a`

| Band | dtype | Units | Source scaling |
|---|---|---|---|
| `B02` | float32 | reflectance 0–1 | DN / 10000 (Baseline‑04.00) |
| `B03` | float32 | reflectance 0–1 | DN / 10000 |
| `B04` | float32 | reflectance 0–1 | DN / 10000 |
| `B08` | float32 | reflectance 0–1 | DN / 10000 |
| `flag` | uint8 | — | See flag-band spec above |

- **Spatial resolution:** 10 m.
- **Cloud mask:** `s2cloudless` (PC asset) with threshold 40; fallback to SCL classes 8/9/10.
- **Cloud-shadow mask:** directional-offset projection (Option B, see §Shadow projection below). **Not** ray-cast — DSM-occluded shadows are deferred to Stage 3 (Sekundärdaten-Pipeline, SVF/Shadow module).
- **SCL** is used for cirrus (class 10) and saturated (class 1), but **not** as the primary cloud/shadow source.

### Cloud-Shadow Projection (S2 directional offset)

Geometric directional-offset method, not ray-cast:

```
shadow_xy = cloud_xy + (Δx, Δy)

Δx = -h × tan(zenith) × sin(azimuth)
Δy = -h × tan(zenith) × cos(azimuth)
```

Where:

| Parameter | Default | Source |
|---|---|---|
| `h` (cloud base height) | 1000 m | Hydra `cloud_base_height_m` |
| `sun_elevation` | per scene | STAC `view:sun_elevation` or computed from datetime + lat |
| `sun_azimuth` | per scene | STAC `view:sun_azimuth` or computed from datetime + lat |

- `zenith = 90° − elevation`.
- Shadow is cast from each cloud pixel via nearest-neighbour shift (0-order).
- At solar zenith zenith < 0.5° (sun near zenith): no shadow cast.
- **Limitation:** shadows behind tall buildings (DSM-occluded) are not caught — that requires ray-cast through DSM (Stage 3).
- For Berlin's 3 MGRS tiles, shadows are projected per-tile and flagged per-scene. Inter-tile shadow stitching is not performed in Phase A.

## ECOSTRESS — `ecostress` (Phase B, forward-declared)

| Band | dtype | Units | Source |
|---|---|---|---|
| `lst` | float32 | Kelvin | ECO_L2T_LSTE.002 native |
| `flag` | uint8 | — | fill bits only; no cloud/shadow derived |

- **Spatial resolution:** ~70 m native, reprojected to EPSG:25833.
- **Acquisition:** AppEEARS Collection 2 (Phase B). Phase A: stubbed (fixture fallback via envar).

## Schema Hash

Each contract provides a deterministic BLAKE3 hash computed over:

```
{source}::{target_crs}::{band_name}|{dtype}|{nodata}::…::blocksize={v}::overviews={…}::compress={c}::predictor={p}::v{schema_version}
```

This hash is stored in:
- STAC `properties.ard:schema_hash`
- Ledger `schema_hash` column

Idempotency check: if the stored hash matches the current contract's hash, the COG is considered valid and skipped. A mismatch forces reprocessing (e.g., after a contract change).

## Schema Version

Current: `1`.

The `schema_version` column in the ledger is `int` and fields start at `1`. Never mutate in place — increment for breaking changes and document the delta in this file.
