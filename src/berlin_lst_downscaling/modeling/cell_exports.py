"""Per-cell validation export (Parquet) for the spatial error analysis.

One row per valid 100 m cell of an evaluated validation patch: the global
canonical 100 m ``row``/``col`` (``common/grid.py``), the native target and the
pooled prediction in Kelvin. The prediction is clamped once with
``clamp_eval_prediction``, exactly as the per-patch export scores it, so the
absolute cell errors of a patch sum to that patch's ``abs_error_sum``.

The export is validation-only by construction: ``write_cell_export`` refuses any
other split, and a row whose metadata names another split is rejected. Run
identity lives in the Parquet footer; rows carry no metric and change no
report field.

Importing this module does not import torch; the tensor helper imports the
clamp lazily.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from torch import Tensor

    from berlin_lst_downscaling.modeling.contracts import RealSampleMeta
    from berlin_lst_downscaling.modeling.evaluation_exports import EvaluationIdentity

CELL_EXPORT_SCHEMA_VERSION = 1
CELL_EXPORT_METADATA_KEY = b"cell_export"
CELL_EXPORT_SPLIT = "validation"
CELL_EXPORT_FILENAME = "eval_validation_cells.parquet"

CELL_EXPORT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("patch_id", pa.string(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),
        pa.field("col", pa.int32(), nullable=False),
        pa.field("target_k", pa.float32(), nullable=False),
        pa.field("prediction_k", pa.float32(), nullable=False),
    ]
)
CELL_EXPORT_COLUMNS: tuple[str, ...] = tuple(CELL_EXPORT_SCHEMA.names)


def cell_columns_from_batch(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    metadata: Sequence[RealSampleMeta],
) -> dict[str, np.ndarray]:
    """Return the valid cells of an already pooled 100 m batch as column arrays.

    ``row``/``col`` are the patch anchor (top-left cell, ``RealSampleMeta.row``/
    ``col``) plus the cell's offset inside the patch.
    """
    from berlin_lst_downscaling.modeling.metrics import clamp_eval_prediction

    if prediction_100m.shape != target_100m.shape or prediction_100m.shape != mask_100m.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction_100m.shape)}, "
            f"target {tuple(target_100m.shape)}, mask {tuple(mask_100m.shape)}"
        )
    if prediction_100m.ndim != 4 or prediction_100m.shape[1] != 1:
        raise ValueError(f"expected (B, 1, H, W) tensors, got {tuple(prediction_100m.shape)}")
    if prediction_100m.shape[0] != len(metadata):
        raise ValueError(f"batch size {prediction_100m.shape[0]} != metadata {len(metadata)}")
    prediction = clamp_eval_prediction(prediction_100m).detach().cpu().numpy()[:, 0]
    target = target_100m.detach().cpu().numpy()[:, 0]
    mask = mask_100m.detach().cpu().numpy()[:, 0].astype(bool)
    out: dict[str, list[np.ndarray]] = {name: [] for name in CELL_EXPORT_COLUMNS}
    for i, meta in enumerate(metadata):
        if meta.split != CELL_EXPORT_SPLIT:
            raise ValueError(f"{meta.patch_id}: split {meta.split!r} is not {CELL_EXPORT_SPLIT!r}")
        local_row, local_col = np.nonzero(mask[i])
        n = int(local_row.size)
        out["scene_id"].append(np.full(n, meta.scene_id, dtype=object))
        out["patch_id"].append(np.full(n, meta.patch_id, dtype=object))
        out["row"].append((meta.row + local_row).astype(np.int32))
        out["col"].append((meta.col + local_col).astype(np.int32))
        out["target_k"].append(target[i][mask[i]].astype(np.float32))
        out["prediction_k"].append(prediction[i][mask[i]].astype(np.float32))
    return {name: np.concatenate(parts) for name, parts in out.items()}


def concat_cell_columns(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-batch column dicts in order."""
    if not parts:
        raise ValueError("no cell columns to concatenate")
    return {name: np.concatenate([part[name] for part in parts]) for name in CELL_EXPORT_COLUMNS}


def write_cell_export(
    columns: Mapping[str, np.ndarray],
    path: Path,
    *,
    identity: EvaluationIdentity,
    split: str,
    aggregation: str,
) -> Path:
    """Validate ``columns`` and write them atomically as the cell-export Parquet."""
    from berlin_lst_downscaling.modeling.metrics import EVAL_PREDICTION_CLAMP_K

    if split != CELL_EXPORT_SPLIT:
        raise ValueError(f"cell exports are {CELL_EXPORT_SPLIT}-only, got split {split!r}")
    if not identity.arm.strip() or not identity.run_id.strip():
        raise ValueError("identity.arm and identity.run_id must be nonempty")
    missing = [name for name in CELL_EXPORT_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"cell columns are missing {missing}")
    n = len(columns["row"])
    if n == 0:
        raise ValueError("no cells to export")
    if any(len(columns[name]) != n for name in CELL_EXPORT_COLUMNS):
        raise ValueError("cell columns differ in length")
    for name in ("target_k", "prediction_k"):
        if not np.isfinite(np.asarray(columns[name], dtype=np.float64)).all():
            raise ValueError(f"{name} holds a non-finite value")
    low, high = EVAL_PREDICTION_CLAMP_K
    prediction = np.asarray(columns["prediction_k"], dtype=np.float64)
    if prediction.min() < low or prediction.max() > high:
        raise ValueError(f"prediction_k lies outside the evaluation clamp [{low}, {high}]")
    arrays = {
        name: pa.array(columns[name], type=CELL_EXPORT_SCHEMA.field(name).type)
        for name in CELL_EXPORT_COLUMNS
    }
    table = pa.table(arrays).sort_by(
        [("scene_id", "ascending"), ("row", "ascending"), ("col", "ascending")]
    )
    keys = table.group_by(["scene_id", "row", "col"]).aggregate([([], "count_all")])
    if keys.num_rows != table.num_rows:
        raise ValueError("a (scene_id, row, col) cell occurs more than once: patches overlap")
    footer = json.dumps(
        {
            "schema_version": CELL_EXPORT_SCHEMA_VERSION,
            "split": split,
            "arm": identity.arm,
            "seed": identity.seed,
            "run_id": identity.run_id,
            "checkpoint": identity.checkpoint,
            "aggregation": aggregation,
            "prediction_clamp_k": [float(low), float(high)],
            "grid": "canonical EPSG:25833 100 m (common/grid.py), row/col from the grid origin",
        },
        sort_keys=True,
    ).encode("utf-8")
    table = table.cast(CELL_EXPORT_SCHEMA.with_metadata({CELL_EXPORT_METADATA_KEY: footer}))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "CELL_EXPORT_COLUMNS",
    "CELL_EXPORT_FILENAME",
    "CELL_EXPORT_METADATA_KEY",
    "CELL_EXPORT_SCHEMA",
    "CELL_EXPORT_SCHEMA_VERSION",
    "CELL_EXPORT_SPLIT",
    "cell_columns_from_batch",
    "concat_cell_columns",
    "write_cell_export",
]
