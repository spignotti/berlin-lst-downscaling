"""One-time test-2025 per-patch export for the frozen v6 matrix checkpoints.

Scope
-----
For a list of matrix run labels (the 27 runs from ``rescore_inventory``, or a
subset) this scores each retained ``best.ckpt`` on the 2025 test split and
writes one schema-version-2 per-patch export. It is the input
``analysis.scene_bootstrap`` reads with ``split="test"``.

No training step runs and no checkpoint is written. Nothing is uploaded. The
retained evidence prefixes in GCS are not written.

What is read
------------
One shared cache is built from the test split only (``splits == ("test",)``).
Train patches are never read.

The stored ``best_model_score`` is the training-time unclamped validation MAE.
Checking it still requires scoring the validation split: that is the rescore's
identity recheck (``assert_recomputed_best_score``), reused rather than copied.
A second shared cache, validation only, exists for that check and for nothing
else. Its rows are not exported. The composed arm config stays
``evaluate_test=false``; the cache builder overrides the read.

Scoring
-------
Test rows go through ``score_validation(..., split="test")``, the same clamping
and aggregation as the validation exports (100 m MAE and the SSIM variants), so
the numbers are comparable. ``recomputed_test_mae_100m`` in the manifest is the
unclamped ``MaskedMAE``, the same definition as
``recomputed_validation_mae_100m``. ``test_report`` is ``report_metrics`` of the
clamped per-patch rows; its ``mae_100m`` is what the scene bootstrap pools.

Outputs
-------
Local files only, create-only (an existing output file, or a staged checkpoint
or cache directory, stops the job before any cache build or checkpoint
download)::

    <output_root>/<run-label>/eval_test_patches.parquet
    <output_root>/test_export_manifest.json

A failed job can leave earlier exports and the cache directories. Delete the
output root before a retry; nothing here overwrites.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch
from lightning.pytorch import seed_everything

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.modeling import v6_val_rescore as rescore
from berlin_lst_downscaling.modeling.aggregation import resolve_aggregation_method
from berlin_lst_downscaling.modeling.evaluation_exports import (
    PATCH_EXPORT_SCHEMA_VERSION,
    EvaluationIdentity,
    patch_export_filename,
    report_metrics,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.v6_val_cell_export import (
    _sha256_file,
    select_runs,
    ssim_range_for_checkpoint,
)
from berlin_lst_downscaling.modeling.v6_val_rescore import RescoreRun

_logger = logging.getLogger(__name__)

MANIFEST_NAME = "test_export_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
TEST_SPLIT = "test"
TEST_SPLITS: tuple[str, ...] = ("test",)


def code_commit() -> str:
    """Commit of the running code, or ``unknown`` when it cannot be read.

    ``VERTEX_SOURCE_SHA`` wins when set (the image), otherwise local HEAD.
    """
    from berlin_lst_downscaling.modeling.prebuilt_cache import running_code_commit

    commit = running_code_commit()
    return commit if commit else "unknown"


def _export_path(output_root: Path, run: RescoreRun) -> Path:
    return output_root / run.run_label / patch_export_filename(TEST_SPLIT)


def run_test_export(
    *, run_labels: Sequence[str], output_root: Path, device: str
) -> dict[str, object]:
    """Write one test patch export per run label and a local manifest.

    Create-only: an existing export or manifest raises ``FileExistsError``
    before a cache is built or a checkpoint is downloaded.
    """
    runs = select_runs(run_labels)
    export_paths = {run.run_label: _export_path(output_root, run) for run in runs}
    manifest_path = output_root / MANIFEST_NAME
    staged = (
        output_root / "checkpoints",
        output_root / "cache-test",
        output_root / "cache-validation",
    )
    for path in (manifest_path, *export_paths.values(), *staged):
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

    log_event(_logger, logging.INFO, "test_export_cache_build_start", runs=len(runs))
    test_cache = rescore.build_validation_cache(
        ref_cfg, output_root / "cache-test", splits=TEST_SPLITS
    )
    # Identity only: the stored best score is the unclamped validation MAE.
    validation_cache = rescore.build_validation_cache(ref_cfg, output_root / "cache-validation")

    entries: list[dict[str, object]] = []
    for run in runs:
        cfg = cfgs[run.run_label]
        seed_everything(run.seed)
        dest = output_root / "checkpoints" / f"{run.run_label}.ckpt"
        aggregation = resolve_aggregation_method(cfg.get("aggregation"))
        dev = torch.device(device)
        scored: tuple[rescore.CheckpointRef, list[dict[str, object]], float, float] | None = None
        try:
            ref_ckpt = rescore.fetch_checkpoint(run.checkpoint_uri, dest)
            selection = rescore.checkpoint_selection_state(dest)
            task = rescore.load_rescore_task(dest, cfg, ssim_range_for_checkpoint(dest, ssim_range))
            task = task.to(dev)
            _, recomputed_val = rescore.score_validation(
                task,
                rescore.validation_loader(validation_cache, cfg),
                aggregation=aggregation,
                ssim_range=ssim_range,
                device=dev,
            )
            best_score = selection["best_model_score"]
            if not isinstance(best_score, float):
                raise RuntimeError(f"{run.run_label}: checkpoint carries no best_model_score")
            rescore.assert_recomputed_best_score(run.run_label, recomputed_val, best_score)
            rows, recomputed_test = rescore.score_validation(
                task,
                rescore.validation_loader(test_cache, cfg),
                aggregation=aggregation,
                ssim_range=ssim_range,
                device=dev,
                split=TEST_SPLIT,
            )
            scored = (ref_ckpt, rows, recomputed_val, recomputed_test)
        finally:
            dest.unlink(missing_ok=True)
        if scored is None:
            raise RuntimeError(f"{run.run_label}: test scoring did not finish")
        ref_ckpt, rows, recomputed_val, recomputed_test = scored
        if len(rows) != len(test_cache.patch_ids):
            raise RuntimeError(
                f"{run.run_label}: scored {len(rows)} test patches, cache holds "
                f"{len(test_cache.patch_ids)}"
            )
        report = report_metrics(rows)
        export_path = write_patch_export(
            rows,
            export_paths[run.run_label],
            identity=EvaluationIdentity(
                arm=run.arm_id,
                seed=run.seed,
                run_id=run.run_label,
                checkpoint=run.checkpoint_uri,
            ),
            split=TEST_SPLIT,
            ssim_range=ssim_range,
        )
        entries.append(
            {
                "arm": run.arm_id,
                "seed": run.seed,
                "config_name": run.config_name,
                "run_label": run.run_label,
                "checkpoint": {
                    "uri": ref_ckpt.uri,
                    "generation": ref_ckpt.generation,
                    "bytes": ref_ckpt.bytes,
                    "sha256": ref_ckpt.sha256,
                },
                "recomputed_validation_mae_100m": recomputed_val,
                "recomputed_test_mae_100m": recomputed_test,
                "test_report": report,
                "patches": len(rows),
                "scenes": len({str(row["scene_id"]) for row in rows}),
                "export": {
                    "path": export_path.relative_to(output_root).as_posix(),
                    "sha256": _sha256_file(export_path),
                    "bytes": export_path.stat().st_size,
                    "schema_version": PATCH_EXPORT_SCHEMA_VERSION,
                },
                "split": TEST_SPLIT,
            }
        )
        log_event(_logger, logging.INFO, "test_export_run_done", run_label=run.run_label)

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "v6_test_patch_export",
        "split": TEST_SPLIT,
        "splits_read": [TEST_SPLIT, rescore.RESCORE_SPLIT],
        "uploaded": False,
        "code_commit": code_commit(),
        "matrix_source_sha": rescore.MATRIX_SOURCE_SHA,
        "device": device,
        "run_labels": [run.run_label for run in runs],
        "patch_cache": {
            "split": TEST_SPLIT,
            "patches": len(test_cache.patch_ids),
            "provenance_sha256": test_cache.provenance_sha256,
            "build_seconds": test_cache.build_seconds,
        },
        "validation_recheck": {
            "purpose": "identity",
            "split": rescore.RESCORE_SPLIT,
            "patches": len(validation_cache.patch_ids),
            "note": (
                "Unclamped validation MAE compared with the stored best_model_score. Not exported."
            ),
        },
        "source_roots": test_cache.provenance.get("source_roots"),
        "patch_index_fingerprints": fingerprints,
        "runs": entries,
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
