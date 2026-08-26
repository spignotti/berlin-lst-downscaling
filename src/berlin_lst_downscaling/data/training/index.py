"""Scene manifest and eligible-cell index builders.

The scene manifest is the per-scene release table: every pairing in the
universe (2017-2026) with its temporal split, sensor, eligible-cell count,
status, and asset references. The cell index is one row per eligible
100 m cell per scene, carrying the stable spatial cell ID, global
canonical row/column, center coordinates, and the owning mask URI.

Cell identity is derived from the **global canonical** EPSG:25833 100 m
grid (origin ``(369190, 5838410)``): a mask written on any canonical-
aligned analysis grid is mapped back to global row/col via its affine
origin, so the same spatial cell carries the same ID across every scene.
"""

from __future__ import annotations

import numpy as np
import rasterio

from berlin_lst_downscaling.data.features.paths import (
    feature_cog,
    feature_mask_cog,
)
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
    NO_ELIGIBLE_CELLS_REASON,
    cell_id,
)
from berlin_lst_downscaling.data.training.paths import eligibility_cog
from berlin_lst_downscaling.data.training.report import SceneTrainingResult

# ── manifest ──────────────────────────────────────────────────────────

MANIFEST_FIELDNAMES = [
    "scene_id",
    "year",
    "sensor",
    "s2_scene_id",
    "split",
    "status",
    "eligible_cells",
    "exclusion_reason",
    "feature_stack",
    "feature_valid",
    "eligibility_mask",
]


def build_manifest_rows(
    results: list[SceneTrainingResult],
    *,
    features_root: str,
    output_root: str,
) -> list[dict]:
    """Build one manifest row per scene (published or excluded).

    Published scenes carry all three asset references; ``no_eligible_cells``
    scenes keep their (all-zero) mask reference; 2026 inference scenes
    carry no references (no V3 stack exists yet — deferred).
    """
    rows = []
    for s in results:
        mask_published = s.status == "done" or s.exclusion_reason == NO_ELIGIBLE_CELLS_REASON
        status = "published" if s.status == "done" else "excluded"
        rows.append(
            {
                "scene_id": s.scene_id,
                "year": s.year,
                "sensor": s.sensor,
                "s2_scene_id": s.s2_scene_id,
                "split": s.split,
                "status": status,
                "eligible_cells": s.eligible_cells or 0,
                "exclusion_reason": s.exclusion_reason or "",
                "feature_stack": feature_cog(features_root, s.scene_id) if mask_published else "",
                "feature_valid": (
                    feature_mask_cog(features_root, s.scene_id) if mask_published else ""
                ),
                "eligibility_mask": (
                    eligibility_cog(output_root, s.scene_id) if mask_published else ""
                ),
            }
        )
    return rows


# ── eligible-cell index ───────────────────────────────────────────────

CELLS_FIELDNAMES = [
    "scene_id",
    "year",
    "split",
    "cell_id",
    "row",
    "col",
    "center_x",
    "center_y",
    "eligibility_mask",
]


def _global_row_col(src: rasterio.DatasetReader, row: int, col: int) -> tuple[int, int]:
    """Map a local mask cell to the global canonical 100 m row/col."""
    gcol = round((src.transform.xoff - CANON_GRID_ORIGIN_X) / CELL_SIZE_M) + col
    grow = round((CANON_GRID_ORIGIN_Y - src.transform.yoff) / CELL_SIZE_M) + row
    return grow, gcol


def build_cells_rows(
    results: list[SceneTrainingResult],
    *,
    output_root: str,
) -> list[dict]:
    """Build one row per eligible 100 m cell by reading back the masks.

    Reads the published per-scene eligibility COG (100 m, tiny) and emits a
    row for every eligible cell with its stable spatial cell ID, global
    canonical row/column, and center coordinates (EPSG:25833).
    """
    rows = []
    for s in results:
        if s.status != "done" or not s.eligible_cells:
            continue
        mask_uri = eligibility_cog(output_root, s.scene_id)
        with rasterio.open(mask_uri) as src:
            mask = src.read(1) == 1
            for row, col in zip(*np.where(mask), strict=False):
                grow, gcol = _global_row_col(src, int(row), int(col))
                rows.append(
                    {
                        "scene_id": s.scene_id,
                        "year": s.year,
                        "split": s.split,
                        "cell_id": cell_id(grow, gcol),
                        "row": grow,
                        "col": gcol,
                        "center_x": src.transform.xoff + (int(col) + 0.5) * CELL_SIZE_M,
                        "center_y": src.transform.yoff - (int(row) + 0.5) * CELL_SIZE_M,
                        "eligibility_mask": mask_uri,
                    }
                )
    return rows


__all__ = [
    "CELLS_FIELDNAMES",
    "MANIFEST_FIELDNAMES",
    "build_cells_rows",
    "build_manifest_rows",
]
