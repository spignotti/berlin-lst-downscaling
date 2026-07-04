# ARD Manifest Schema

## Purpose

The manifest is the **scene list** that drives a full pipeline run
(``mode=full``).  It is produced by a future Szenen-Selektion &
Kopplung (Task 3) module and consumed by the ARD orchestrator.

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

| Column | Type | Nullable | Description |
|---|---|---|---|
| ``scene_id`` | string | no | Unique scene identifier (e.g. ``LC09_L2SP_193024_20240629_02_T1``) |
| ``source`` | string | no | One of ``landsat-c2-l2``, ``sentinel-2-l2a``, ``ecostress`` |
| ``year`` | int32 | no | Acquisition year |
| ``status`` | string | no | ``coupled``, ``orphaned``, ``validated``, ``dropped`` |
| ``coupled_s2_id`` | string | yes | For Landsat anchors: matched S2 scene ID |
| ``ecostress_id`` | string | yes | For Landsat anchors: matched ECOSTRESS granule ID (if any) |
| ``paired_at`` | timestamp[us] UTC | yes | Datetime of the paired S2 scene (or ECOSTRESS) |
| ``clear_frac`` | float32 | yes | Fraction of clear overlapping pixels (0–1) |
| ``dt_days`` | float32 | yes | Time delta in days between coupled scenes (0–3) |

## Interoperability Notes

- The ``scene_id`` for Landsat is the item ID from the PC STAC search
  (e.g. ``LC09_L2SP_193024_20240629_02_T1``).
- The ``scene_id`` for S2 is the tile-level item ID
  (e.g. ``S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907``).
- For ECOSTRESS (Phase B), the ``scene_id`` is the granule ID from
  AppEEARS.
- ``paired_at`` is the UTC datetime of the partner scene, not the anchor.

## Implementation Status (Phase A)

In Phase A, the manifest does not exist yet — ``mode=full`` emits a
clear error asking the user to run the Szenen-Selektion task first.
``smoke`` mode constructs a synthetic one-scene manifest internally.
