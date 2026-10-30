"""Per-patch evaluation export (Parquet sidecar) for validation and test.

One row per evaluated patch with the additive sums the scene bootstrap needs
(absolute-error sum, valid cells, SSIM sum and windows). The export is a pure
sidecar: it never changes a metric, a checkpoint choice, or a report field.

Importing this module does not import torch or the training orchestration;
the tensor scoring helper imports the metric functions lazily.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from torch import Tensor

    from berlin_lst_downscaling.modeling.contracts import RealSampleMeta

PATCH_EXPORT_SCHEMA_VERSION = 1
PATCH_EXPORT_METADATA_KEY = b"patch_export"

PATCH_EXPORT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=True),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("checkpoint", pa.string(), nullable=True),
        pa.field("patch_id", pa.string(), nullable=False),
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("scene_date", pa.string(), nullable=True),
        pa.field("split", pa.string(), nullable=False),
        pa.field("abs_error_sum", pa.float64(), nullable=False),
        pa.field("valid_cells", pa.int64(), nullable=False),
        pa.field("ssim_sum", pa.float64(), nullable=False),
        pa.field("ssim_windows", pa.int64(), nullable=False),
        pa.field("filled_feature_pixels", pa.int64(), nullable=True),
    ]
)

_SPLITS = frozenset({"train", "validation", "test"})
_REQUIRED_KEYS = (
    "patch_id",
    "scene_id",
    "split",
    "abs_error_sum",
    "valid_cells",
    "ssim_sum",
    "ssim_windows",
)


@dataclass(frozen=True)
class EvaluationIdentity:
    """Who produced the export: arm, seed, run, and the evaluated checkpoint."""

    arm: str
    seed: int | None
    run_id: str
    checkpoint: str | None


def patch_export_filename(split: str, *, prefix: str = "eval") -> str:
    """Return ``<prefix>_<split>_patches.parquet``."""
    return f"{prefix}_{split}_patches.parquet"


def score_batch_patches(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    metadata: Sequence[RealSampleMeta],
    *,
    clamp_predictions: bool = False,
) -> list[dict[str, object]]:
    """Return one additive-sum row per sample of an already pooled 100 m batch."""
    from berlin_lst_downscaling.modeling.metrics import masked_abs_error_sums, masked_ssim_stats

    batch = int(prediction_100m.shape[0])
    if batch != len(metadata):
        raise ValueError(f"batch size {batch} != metadata length {len(metadata)}")
    rows: list[dict[str, object]] = []
    for i, meta in enumerate(metadata):
        sl = slice(i, i + 1)
        pred, target, mask = prediction_100m[sl], target_100m[sl], mask_100m[sl]
        error_sum, count = masked_abs_error_sums(pred, target, mask)
        ssim_sum, ssim_count = masked_ssim_stats(
            pred, target, mask, clamp_predictions=clamp_predictions
        )
        rows.append(
            {
                "patch_id": meta.patch_id,
                "scene_id": meta.scene_id,
                "scene_date": meta.scene_date,
                "split": meta.split,
                "abs_error_sum": float(error_sum),
                "valid_cells": int(count),
                "ssim_sum": float(ssim_sum),
                "ssim_windows": int(ssim_count),
                "filled_feature_pixels": meta.filled_feature_pixels,
            }
        )
    return rows


def _count(value: object, name: str, patch_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{patch_id}: {name} must be an integer, got bool")
    if isinstance(value, int):
        out = value
    elif isinstance(value, float) and math.isfinite(value) and value == int(value):
        out = int(value)
    else:
        # numpy integer scalars expose __index__
        try:
            out = value.__index__()  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            raise ValueError(f"{patch_id}: {name} must be an integer, got {value!r}") from None
        if isinstance(out, bool):
            raise ValueError(f"{patch_id}: {name} must be an integer, got bool")
    if out < 0:
        raise ValueError(f"{patch_id}: {name} must be >= 0, got {out}")
    return out


def _finite(value: object, name: str, patch_id: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{patch_id}: {name} must be a number, got bool")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{patch_id}: {name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{patch_id}: {name} must be finite, got {out}")
    return out


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def write_patch_export(
    records: Sequence[Mapping[str, object]],
    path: Path,
    *,
    identity: EvaluationIdentity,
    split: str,
) -> Path:
    """Validate ``records`` and write them atomically as the patch-export Parquet."""
    _nonempty(identity.arm, "identity.arm")
    _nonempty(identity.run_id, "identity.run_id")
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got {split!r}")

    if not records:
        raise ValueError("no patch records to export")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        missing = [key for key in _REQUIRED_KEYS if key not in record]
        if missing:
            raise ValueError(f"patch record is missing required keys {missing}")
        patch_id = _nonempty(record["patch_id"], "patch_id")
        if patch_id in seen:
            raise ValueError(f"duplicate patch_id {patch_id!r}")
        seen.add(patch_id)
        scene_id = record["scene_id"]
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(f"{patch_id}: scene_id must be a nonempty string")
        if record["split"] != split:
            raise ValueError(f"{patch_id}: row split {record['split']!r} != export split {split!r}")
        abs_error_sum = _finite(record["abs_error_sum"], "abs_error_sum", patch_id)
        if abs_error_sum < 0:
            raise ValueError(f"{patch_id}: abs_error_sum must be >= 0")
        ssim_sum = _finite(record["ssim_sum"], "ssim_sum", patch_id)
        valid_cells = _count(record["valid_cells"], "valid_cells", patch_id)
        ssim_windows = _count(record["ssim_windows"], "ssim_windows", patch_id)
        if valid_cells <= 0:
            raise ValueError(f"{patch_id}: valid_cells must be > 0")
        if ssim_windows == 0 and ssim_sum != 0.0:
            raise ValueError(f"{patch_id}: ssim_windows == 0 requires ssim_sum == 0")
        filled_raw = record.get("filled_feature_pixels")
        filled = (
            None if filled_raw is None else _count(filled_raw, "filled_feature_pixels", patch_id)
        )
        scene_date = record.get("scene_date")
        if scene_date is not None and not isinstance(scene_date, str):
            raise ValueError(f"{patch_id}: scene_date must be a string or None")
        rows.append(
            {
                "schema_version": PATCH_EXPORT_SCHEMA_VERSION,
                "arm": identity.arm,
                "seed": identity.seed,
                "run_id": identity.run_id,
                "checkpoint": identity.checkpoint,
                "patch_id": patch_id,
                "scene_id": scene_id,
                "scene_date": scene_date,
                "split": split,
                "abs_error_sum": abs_error_sum,
                "valid_cells": valid_cells,
                "ssim_sum": ssim_sum,
                "ssim_windows": ssim_windows,
                "filled_feature_pixels": filled,
            }
        )
    rows.sort(key=lambda row: (str(row["scene_id"]), str(row["patch_id"])))

    footer = json.dumps(
        {
            "schema_version": PATCH_EXPORT_SCHEMA_VERSION,
            "split": split,
            "arm": identity.arm,
            "seed": identity.seed,
            "run_id": identity.run_id,
            "checkpoint": identity.checkpoint,
        },
        sort_keys=True,
    ).encode("utf-8")
    schema = PATCH_EXPORT_SCHEMA.with_metadata({PATCH_EXPORT_METADATA_KEY: footer})
    table = pa.Table.from_pylist(rows, schema=schema)

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
    "PATCH_EXPORT_METADATA_KEY",
    "PATCH_EXPORT_SCHEMA",
    "PATCH_EXPORT_SCHEMA_VERSION",
    "EvaluationIdentity",
    "patch_export_filename",
    "score_batch_patches",
    "write_patch_export",
]
