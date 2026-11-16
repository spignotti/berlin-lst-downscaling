"""Validation-only per-cell export for selected frozen v6 matrix checkpoints.

Scope
-----
For a list of matrix run labels this scores the retained ``best.ckpt`` of each
run on the 2024 validation split and writes one row per valid 100 m cell
(``modeling.cell_exports``). It is the input the spatial error analysis needs
and the per-patch rescore export does not carry: signed errors on the grid.

No training step runs and no checkpoint is written. No test (2025) ref, metric
or export is loaded: every run is composed with ``evaluate_test=false`` and the
shared cache is built with ``splits == ("validation",)``, both through the
rescore module's own functions. Checkpoint identity, strict loading, the
hyperparameter comparison and the best-score recheck are the rescore's.

Outputs
-------
Local files only, create-only (an existing output file stops the job before any
work)::

    <output_root>/<run-label>/eval_validation_cells.parquet
    <output_root>/cell_export_manifest.json

Nothing is uploaded. The retained evidence prefixes in GCS are not written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.modeling import v6_val_rescore as rescore
from berlin_lst_downscaling.modeling.aggregation import (
    AggregationMethod,
    resolve_aggregation_method,
)
from berlin_lst_downscaling.modeling.cell_exports import (
    CELL_EXPORT_FILENAME,
    CELL_EXPORT_SCHEMA_VERSION,
    CELL_EXPORT_SPLIT,
    cell_columns_from_batch,
    concat_cell_columns,
    write_cell_export,
)
from berlin_lst_downscaling.modeling.contracts import RealBatch, validate_real_batch
from berlin_lst_downscaling.modeling.evaluation_exports import EvaluationIdentity
from berlin_lst_downscaling.modeling.metrics import MaskedMAE, pool_10m_to_100m
from berlin_lst_downscaling.modeling.real_task import RealLSTTask
from berlin_lst_downscaling.modeling.ssim_range import SsimDataRange
from berlin_lst_downscaling.modeling.v6_val_rescore import RescoreRun

_logger = logging.getLogger(__name__)

MANIFEST_NAME = "cell_export_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_SELECTION_TOLERANCE = 1e-3


def select_runs(run_labels: Sequence[str]) -> tuple[RescoreRun, ...]:
    """Return the matrix runs named by ``run_labels``, in the given order."""
    if not run_labels:
        raise ValueError("no run labels given")
    if len(set(run_labels)) != len(run_labels):
        raise ValueError("duplicate run label")
    by_label = {run.run_label: run for run in rescore.rescore_inventory()}
    unknown = [label for label in run_labels if label not in by_label]
    if unknown:
        raise ValueError(f"run labels outside the frozen v6 matrix: {unknown}")
    return tuple(by_label[label] for label in run_labels)


def score_validation_cells(
    task: RealLSTTask,
    loader: DataLoader[RealBatch],
    *,
    aggregation: AggregationMethod,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    """Score every validation batch; return the cell columns and the unclamped MAE recheck."""
    recheck = MaskedMAE()
    parts: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for batch in loader:
            validate_real_batch(batch, n_active_channels=task.n_active_channels)
            dev_batch = dataclasses.replace(
                batch,
                features=batch.features.to(device),
                lst_prior=batch.lst_prior.to(device),
                target_100m=batch.target_100m.to(device),
                mask_100m=batch.mask_100m.to(device),
            )
            prediction = task(dev_batch).float().cpu()
            if not torch.isfinite(prediction).all():
                raise RuntimeError("checkpoint produced non-finite predictions")
            pooled = pool_10m_to_100m(prediction, aggregation, batch.mask_100m)
            recheck.update(pooled, batch.target_100m, batch.mask_100m)
            try:
                parts.append(
                    cell_columns_from_batch(
                        pooled, batch.target_100m, batch.mask_100m, batch.metadata
                    )
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
    if not parts:
        raise RuntimeError("validation scoring produced no cells")
    return concat_cell_columns(parts), float(recheck.compute())


def ssim_range_for_checkpoint(path: Path, ssim_range: SsimDataRange) -> SsimDataRange:
    """Rebind the SSIM artifact path to the one the checkpoint recorded, if the content agrees.

    The rescore compares the checkpoint's ``ssim_data_range`` hyperparameter with the
    resolved artifact, and that payload holds the artifact's absolute path: ``/app/...``
    in the training image, a checkout path anywhere else. The path is replaced only when
    every other field, including ``artifact_sha256``, is equal; otherwise the range is
    returned unchanged and the rescore's comparison fails as it should.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") if isinstance(ckpt, dict) else None
    stored = hparams.get("ssim_data_range") if isinstance(hparams, Mapping) else None
    if not isinstance(stored, Mapping) or stored.get("artifact_sha256") is None:
        return ssim_range

    def without_path(payload: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in payload.items() if key != "artifact_path"}

    if without_path(stored) != without_path(ssim_range.payload()):
        return ssim_range
    return dataclasses.replace(ssim_range, artifact_path=str(stored.get("artifact_path")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cell_export(
    *, run_labels: Sequence[str], output_root: Path, device: str
) -> dict[str, object]:
    """Write one validation cell export per run label and a local manifest."""
    runs = select_runs(run_labels)
    export_paths = {
        run.run_label: output_root / run.run_label / CELL_EXPORT_FILENAME for run in runs
    }
    manifest_path = output_root / MANIFEST_NAME
    for path in (manifest_path, *export_paths.values()):
        if path.exists():
            raise FileExistsError(f"create-only output already exists: {path}")
    cfgs = {
        run.run_label: rescore.compose_run_config(
            run, output_root=str(output_root / "runs" / run.run_label)
        )
        for run in runs
    }
    rescore.assert_shared_data_identity(list(cfgs.values()))
    ref_cfg = max(cfgs.values(), key=lambda c: int(c.data.cache_max_bytes))
    source = rescore.real_source_config(ref_cfg)
    fingerprints = rescore.patch_index_fingerprints(source.patch_index_root)
    ssim_range = rescore.resolve_run_ssim_data_range(ref_cfg, expected_fingerprints=fingerprints)
    output_root.mkdir(parents=True, exist_ok=True)

    log_event(_logger, logging.INFO, "cell_export_cache_build_start", runs=len(runs))
    cache = rescore.build_validation_cache(ref_cfg, output_root / "cache")

    entries: list[dict[str, object]] = []
    for run in runs:
        cfg = cfgs[run.run_label]
        seed_everything(run.seed)
        dest = output_root / "checkpoints" / f"{run.run_label}.ckpt"
        aggregation = resolve_aggregation_method(cfg.get("aggregation"))
        try:
            ref_ckpt = rescore.fetch_checkpoint(run.checkpoint_uri, dest)
            selection = rescore.checkpoint_selection_state(dest)
            task = rescore.load_rescore_task(
                dest, cfg, ssim_range_for_checkpoint(dest, ssim_range)
            ).to(device)
            columns, recomputed = score_validation_cells(
                task,
                rescore.validation_loader(cache, cfg),
                aggregation=aggregation,
                device=torch.device(device),
            )
        finally:
            dest.unlink(missing_ok=True)
        best_score = selection["best_model_score"]
        if not isinstance(best_score, float):
            raise RuntimeError(f"{run.run_label}: checkpoint carries no best_model_score")
        tolerance = _SELECTION_TOLERANCE * max(1.0, abs(best_score))
        if abs(recomputed - best_score) > tolerance:
            raise RuntimeError(
                f"{run.run_label}: recomputed validation MAE {recomputed:.6f} != stored "
                f"best_model_score {best_score:.6f} (tolerance {tolerance:.6f})"
            )
        patches = len(set(columns["patch_id"].tolist()))
        if patches != len(cache.patch_ids):
            raise RuntimeError(
                f"{run.run_label}: cells cover {patches} patches, cache holds "
                f"{len(cache.patch_ids)}"
            )
        export_path = write_cell_export(
            columns,
            export_paths[run.run_label],
            identity=EvaluationIdentity(
                arm=run.arm_id, seed=run.seed, run_id=run.run_label, checkpoint=run.checkpoint_uri
            ),
            split=CELL_EXPORT_SPLIT,
            aggregation=str(aggregation),
        )
        error = columns["prediction_k"].astype(np.float64) - columns["target_k"].astype(np.float64)
        entries.append(
            {
                "arm": run.arm_id,
                "seed": run.seed,
                "config_name": run.config_name,
                "run_label": run.run_label,
                "hydra_overrides": list(run.hydra_overrides),
                "checkpoint": {
                    "uri": ref_ckpt.uri,
                    "generation": ref_ckpt.generation,
                    "bytes": ref_ckpt.bytes,
                    "sha256": ref_ckpt.sha256,
                    "best_model_score": best_score,
                },
                "recomputed_validation_mae_100m": recomputed,
                "clamped_validation_mae_100m": float(np.abs(error).mean()),
                "cells": int(error.size),
                "patches": patches,
                "scenes": len(set(columns["scene_id"].tolist())),
                "export": {
                    "path": export_path.relative_to(output_root).as_posix(),
                    "sha256": _sha256_file(export_path),
                    "bytes": export_path.stat().st_size,
                    "schema_version": CELL_EXPORT_SCHEMA_VERSION,
                },
                "split": CELL_EXPORT_SPLIT,
                "evaluate_test": False,
            }
        )
        log_event(_logger, logging.INFO, "cell_export_run_done", run_label=run.run_label)

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "v6_validation_cell_export",
        "evaluate_test": False,
        "split": CELL_EXPORT_SPLIT,
        "splits_read": list(rescore.RESCORE_SPLITS),
        "test_split": "not_read",
        "matrix_source_sha": rescore.MATRIX_SOURCE_SHA,
        "device": device,
        "uploaded": False,
        "patch_cache": {
            "patches": len(cache.patch_ids),
            "provenance_sha256": cache.provenance_sha256,
            "build_seconds": cache.build_seconds,
        },
        "source_roots": cache.provenance.get("source_roots"),
        "patch_index_fingerprints": fingerprints,
        "runs": entries,
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
