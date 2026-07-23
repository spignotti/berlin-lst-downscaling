# ARD Manifest Schema

The manifest bundle describes every scene that the ARD pipeline may
process. It consists of three Parquet files published together.

## Bundle artifacts

| File | Purpose |
|------|---------|
| `manifest.parquet` | One unique executable scene per row |
| `pairings.parquet` | One Landsat→Sentinel-2 relation per anchor |
| `manifest_report.json` | Publication gate with hashes, counts, policy |

The bundle is the only accepted manifest contract. v1/v2 single-file
manifests are retired — the readers fail fast on non-current schemas
or on the old single-file layout.

## Manifest — schema_version 3

| Field | Type | Rule |
|-------|------|------|
| `scene_id` | string, non-null | Primary key with `source` |
| `source` | string, non-null | `landsat-c2-l2`, `sentinel-2-l2a`, `ecostress` |
| `role` | string, non-null | `anchor`, `predictor`, `validation` |
| `platform` | string, non-null | Normalized mission; Landsat restricted to `landsat-8/9` |
| `year` | int32, non-null | Must equal acquisition year |
| `acquisition_datetime` | timestamp UTC, non-null | Temporal authority |
| `item_href` | string, nullable | Required for PC STAC rows; must resolve to `scene_id` |
| `aoi_clear_px` | int64, nullable | Required for Landsat/Sentinel-2 |
| `aoi_total_px` | int64, nullable | Full Berlin AOI cells; required for Landsat/Sentinel-2 |
| `aoi_clear_frac` | float32, nullable | Must be >= 0.05 for anchor/predictor rows |
| `cloud_cover` | float32, nullable | Diagnostic only; never used as gate |
| `solar_azimuth` | float32, nullable | Provenance/scene geometry |
| `solar_elevation` | float32, nullable | Provenance/scene geometry |

### Parquet metadata keys

| Key | Description |
|-----|-------------|
| `schema_name` | `berlin-lst-manifest` |
| `schema_version` | `3` |
| `policy_sha256` | SHA-256 fingerprint of selection policy |
| `cutoff_utc` | ISO timestamp; 2026 data up to this instant |
| `generated_at` | ISO timestamp of bundle creation |

Note: File hashes are NOT embedded in Parquet metadata. They appear
exclusively in `manifest_report.json` to avoid a circular hash contract.

## Pairings — schema_version 1 (current)

| Field | Type | Rule |
|-------|------|------|
| `landsat_scene_id` | string, non-null | FK to anchor manifest row; unique |
| `sentinel2_scene_id` | string, non-null | FK to predictor manifest row; may repeat across anchors |
| `dt_seconds` | int64, non-null | Absolute acquisition delta |
| `landsat_clear_px` | int64, non-null | Must be > 0 |
| `joint_clear_px` | int64, non-null | Must satisfy `0 ≤ joint_clear_px ≤ landsat_clear_px` |
| `joint_clear_frac` | float32, non-null | Must equal `joint_clear_px / landsat_clear_px` exactly through the float32 round-trip; must be in `[0, 1]` |
| `score` | float32, non-null | `joint_clear_frac − λ · (Δt / 3)` |

`pairings.parquet` is the *current* schema (v1), not a legacy
contract. The validator rejects rows that violate the count/fraction
invariant — a tautological mismatch is treated as a hard error.

## ECOSTRESS

Exactly six unique granules from 2018-08-25 are allowed. They are
stored in `manifest.parquet` with `role=validation` and are not part
of Landsat→S2 pairings.

## Selection policy

- **Landsat:** L8/L9 only; no L7. No metadata `eo:cloud_cover` gate.
- **AOI gate:** `aoi_clear_frac >= 0.05` (pixel-level QA_PIXEL ∩ Berlin AOI).
- **Temporal:** May–September. Full seasons 2017–2025; 2026 through explicit `cutoff_utc`.

## Publication and immutability

- Published bundle prefixes are immutable. `scripts/publish_manifest.py`
  rejects upload if the destination objects already exist (via GCS
  `if_generation_match=0` precondition).
- Repairs or regenerated pairings go to a new bundle prefix (e.g. the
  current canonical bundle is `…-r2/`); the original prefix is
  retained as a historical artifact and continues to fail the
  corrected validator.
- The active bundle for the ARD pipeline is
  `gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/`.
