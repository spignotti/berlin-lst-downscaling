# ARD Manifest Schema

## Purpose

The manifest is the **scene list** that drives a full pipeline run
(``mode=full``).  It is produced by the Szenen-Selektion & Kopplung
module (``scripts/build_manifest.py``) and consumed by the ARD orchestrator.

## Relation to Ledger

| | Manifest | Ledger |
|---|---|---|
| **Producer** | Szenen-Selektion (Task 3) | ARD pipeline (this task) |
| **Consumer** | ARD orchestrator (``mode=full``) | ARD pipeline + QA |
| **Contents** | Which scenes to process | What happened during processing |
| **Mutation** | Write-once per run | Upsert per scene |

The manifest and the ledger share the same ``(scene_id, source)``
composite key, making it easy for the orchestrator to join them:

1. Read manifest → get scene list
2. ``reconcile(scenes, ledger, contract)`` → get subset to process
3. Process each scene → update ledger

## Manifest Schema

Stored as a single PyArrow Parquet file at ``data/ard/manifest.parquet``.

### Core Columns (v1 — always present)

| Column | Type | Nullable | Description |
|---|---|---|---|
| ``scene_id`` | string | no | Unique scene identifier (e.g. ``LC09_L2SP_193024_20240629_02_T1``) |
| ``source`` | string | no | One of ``landsat-c2-l2``, ``sentinel-2-l2a``, ``ecostress`` |
| ``year`` | int32 | no | Acquisition year |
| ``status`` | string | no | ``coupled``, ``orphaned``, ``validated``, ``dropped`` |
| ``coupled_s2_id`` | string | yes | For Landsat anchors: matched S2 scene ID |
| ``ecostress_id`` | string | yes | For Landsat anchors: matched ECOSTRESS granule ID (if any) |
| ``paired_at`` | timestamp[us] UTC | yes | Datetime of the partner scene, not the anchor. |
| ``clear_frac`` | float32 | yes | Fraction of clear overlapping pixels (0–1) |
| ``dt_days`` | float32 | yes | Time delta in days between coupled scenes (0–3) |

### Optional Pipeline Columns (v2 — forward-compatible)

These columns are **not required**; the pipeline falls back to querying
Planetary Computer when they are absent.

| Column | Type | Nullable | Description |
|---|---|---|---|
| ``date`` | string | yes | Acquisition date in ISO format (``"2024-06-29"``). Prevents date-based re-query. |
| ``item_href`` | string | yes | Direct asset URL (PC signed or GCS). Pipeline can bypass STAC search entirely. |
| ``acquisition_datetime`` | timestamp[us] UTC | yes | UTC datetime of the acquisition. Used for solar position computation. |
| ``cloud_cover`` | float32 | yes | Scene cloud cover percentage (0–100) |
| ``solar_azimuth`` | float32 | yes | Sun azimuth in degrees clockwise from North. From STAC ``view:sun_azimuth``. |
| ``solar_elevation`` | float32 | yes | Sun elevation in degrees above horizon. From STAC ``view:sun_elevation``. |

## Interoperability Notes

- The ``scene_id`` for Landsat is the item ID from the PC STAC search
  (e.g. ``LC09_L2SP_193024_20240629_02_T1``).
- The ``scene_id`` for S2 is the tile-level item ID
  (e.g. ``S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907``).
- For ECOSTRESS (Phase B), the ``scene_id`` is the granule ID from
  AppEEARS.
- ``paired_at`` is the UTC datetime of the partner scene, not the anchor.
- When ``item_href`` is populated, the pipeline passes it directly to
  ``odc.stac.load`` instead of searching by date + bbox.

## Pipeline Behaviour

In ``mode=full``:

1. Read manifest → get scene list with ``scene_id`` + ``source`` + ``year``.
2. If the manifest has a ``date`` column, use it per-scene instead of
   ``cfg.scene_date`` (which is ``null`` for ``mode=full``).
3. If ``solar_azimuth`` / ``solar_elevation`` are present, use them
   directly (skip NOAA computation).
4. ``reconcile(scenes, ledger, contract)`` → get subset to process.
5. Process each scene → update ledger.

## Implementation Status

The manifest is produced by ``scripts/build_manifest.py`` and consumed
by the ARD orchestrator (``scripts/run_ard.py --mode=full``).
Smoke tests use a hard-coded 3-row manifest (see ``nox -s smoke-primary``).
columns are parsed if present; the pipeline continues to work if missing.
