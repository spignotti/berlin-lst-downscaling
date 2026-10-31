"""WB3 modeling run orchestration — the configured training lifecycle.

Builds the synthetic data module, the fixed U-Net, the Lightning task,
the local checkpoint callback, and the W&B logger from the resolved
Hydra config, runs ``trainer.fit``, then reloads the best checkpoint and
validates it as the selected model state.

Reproducibility posture
-----------------------
- ``seed`` is applied once and passed to the data module; the Trainer runs
  with ``deterministic=True`` and ``benchmark=False``.
- Run metadata carries the resolved config, the data-release identifier,
  the seed, the Git revision (read from the run-context JSON written by
  the runner's :class:`~berlin_lst_downscaling.data.io.RunLogSession`),
  and a reproducibility note.
- Limitations are documented at run level: deterministic kernels do **not**
  guarantee byte-identical results across environments/hardware/library
  versions; the synthetic data is a lifecycle fixture, not a model of LST.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6
from berlin_lst_downscaling.data.io import CANONICAL_BUCKET, log_event, run_context_path
from berlin_lst_downscaling.modeling.aggregation import (
    AggregationMethod,
    resolve_aggregation_method,
)
from berlin_lst_downscaling.modeling.channels import (
    schema_version_from_config,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.contracts import validate_real_batch
from berlin_lst_downscaling.modeling.evaluation_exports import (
    EvaluationIdentity,
    patch_export_filename,
    report_metrics,
    score_batch_patches,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.guards import (
    STAGE1_PROBE_CONFIG_NAME,
    assert_ablation_lock,
    assert_probe_minima,
    assert_stage1_efficiency,
    assert_stage1_efficiency_scope,
    assert_stage1_full_bounds,
    assert_stage1_lock,
    assert_stage1_probe,
    assert_stage1_probe_lr3,
    assert_stage1_runtime,
    assert_vertex_smoke_bounds,
    cache_build_workers_from_config,
    contract_invariants,
    early_stopping_from_config,
    evaluate_test_from_config,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.lr_schedule import lr_schedule_from_config
from berlin_lst_downscaling.modeling.metrics import (
    EVAL_PREDICTION_CLAMP_K,
    MaskedMAE,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patches import (
    RealSourceConfig,
    assert_scaler_matches_config,
    load_scaler,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.real_task import (
    _REAL_SPLITS,
    ProbeScope,
    RealLSTTask,
    RealPatchDataModule,
    reconstruct_prior_kelvin,
)
from berlin_lst_downscaling.modeling.run_identity import (
    enforce_run_lock,
    persist_run_identity,
    resolve_run_identity,
)
from berlin_lst_downscaling.modeling.ssim_range import (
    SsimDataRange,
    load_ssim_data_range,
    test_override_ssim_data_range,
)
from berlin_lst_downscaling.modeling.synthetic import (
    ContractSyntheticDataModule,
    SyntheticDataModule,
)
from berlin_lst_downscaling.modeling.task import LSTRegressionTask
from berlin_lst_downscaling.modeling.timing import EpochTimingRecorder, timed_phase

_logger = logging.getLogger(__name__)

# Extra free disk required beyond the planned cache/checkpoint scratch, so a
# full-size job cannot fill its worker disk mid-run.
_FULL_CACHE_DISK_RESERVE_BYTES = 10 * 1024**3


def _resolve_cache_budget(cfg: DictConfig) -> int:
    """Return the job-local cache budget in bytes.

    Reads the normal ``data.cache_max_bytes`` field with fallback to the
    legacy ``stage1_efficiency_cache_max_bytes`` root key, so historical
    efficiency configs keep their meaning. Both missing means no cache.
    """
    data_cfg = cfg.get("data", {})
    raw = data_cfg.get("cache_max_bytes", 0)
    if isinstance(raw, bool):
        raise ValueError(f"data.cache_max_bytes must be an int, got {raw!r}")
    try:
        budget = int(raw or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.cache_max_bytes must be an int, got {raw!r}") from exc
    if budget < 0:
        raise ValueError(f"data.cache_max_bytes must be >= 0, got {budget}")
    if budget > 0:
        return budget
    legacy = cfg.get("stage1_efficiency_cache_max_bytes", 0)
    if isinstance(legacy, bool):
        raise ValueError("stage1_efficiency_cache_max_bytes must be an int")
    return int(legacy or 0)


_PATCH_CACHE_KEYS = frozenset(
    {"source", "uri_root", "allow_job_local_fallback", "download_workers", "download_retries"}
)
_DEFAULT_PATCH_CACHE_URI_ROOT = f"{CANONICAL_BUCKET}patch-cache"


def _resolve_patch_cache(cfg: DictConfig) -> dict[str, object]:
    """Return the strictly validated ``data.patch_cache`` settings.

    Absent means a job-local build, the behavior before the prebuilt cache.
    """
    data_cfg = cfg.get("data", {})
    raw = data_cfg.get("patch_cache")
    resolved: dict[str, object] = {
        "source": "job_local",
        "uri_root": _DEFAULT_PATCH_CACHE_URI_ROOT,
        "allow_job_local_fallback": False,
        "download_workers": 8,
        "download_retries": 3,
    }
    if raw is None:
        return resolved
    container = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else raw
    if not isinstance(container, dict):
        raise ValueError("data.patch_cache must be a mapping")
    values = {str(key): value for key, value in container.items()}
    unknown = sorted(set(values) - _PATCH_CACHE_KEYS)
    if unknown:
        raise ValueError(f"unknown data.patch_cache key(s) {unknown}")
    for key, value in values.items():
        if key == "source":
            if value not in ("job_local", "prebuilt"):
                raise ValueError(
                    f"data.patch_cache.source must be 'job_local' or 'prebuilt', got {value!r}"
                )
        elif key == "uri_root":
            if not isinstance(value, str) or not value.startswith(("gs://", "/")):
                raise ValueError(
                    f"data.patch_cache.uri_root must be a gs:// URI or absolute path, got {value!r}"
                )
        elif key == "allow_job_local_fallback":
            if not isinstance(value, bool):
                raise ValueError(
                    f"data.patch_cache.allow_job_local_fallback must be a bool, got {value!r}"
                )
        elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"data.patch_cache.{key} must be an int >= 1, got {value!r}")
        resolved[key] = value
    return resolved


def _resolve_cache_root(cfg: DictConfig, *, output_root: Path) -> Path | None:
    """Return the job-local cache path when a positive budget is configured.

    An explicit ``data.cache_root`` always wins. Otherwise a positive
    ``data.cache_max_bytes`` (or the legacy efficiency budget key) composes
    ``<output_root>/cache``, so smoke and full runs keep a job-local cache
    without a static path colliding across jobs.
    """
    data_cfg = cfg.get("data", {})
    raw = data_cfg.get("cache_root")
    if raw is not None:
        return Path(str(raw))
    if _resolve_cache_budget(cfg) > 0:
        return output_root / "cache"
    return None


def write_checkpoint_manifest(checkpoint_dir: Path, manifest_path: Path) -> dict[str, object]:
    """Record SHA-256/bytes for the retained best/last checkpoints.

    Pure inventory: hashes every ``*.ckpt`` in ``checkpoint_dir`` and writes
    the manifest. The future full run calls this before uploading each epoch
    package; synthetic calibration exercises the same helper. Never deletes
    or overwrites a checkpoint.
    """
    entries: dict[str, dict[str, object]] = {}
    for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
        digest = hashlib.sha256()
        with ckpt.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        entries[ckpt.name] = {"sha256": digest.hexdigest(), "bytes": ckpt.stat().st_size}
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    return {"checkpoints": entries, "manifest": str(manifest_path)}


@dataclass(frozen=True)
class ModelingRunResult:
    """Outcome of a configured training run."""

    run_ok: bool
    best_checkpoint: str | None
    validation_loss: float | None
    selection_metric: str = "validation/loss"


def run_training(cfg: DictConfig, run_id: str) -> ModelingRunResult:
    """Execute the configured synthetic training lifecycle.

    ``run_id`` is the runner's run identifier; its context JSON (written by
    the runner's ``RunLogSession``) provides the recorded Git revision.

    Fails closed: any lifecycle failure (fit error, missing checkpoint,
    reload mismatch) raises — the runner translates that into a non-zero
    process exit. The W&B run is finalized as ``"failed"`` on every error
    path and only as ``"success"`` after the reload validation passes.
    """
    seed = int(cfg.seed)
    lr_spec = lr_schedule_from_config(cfg)
    seed_everything(seed)

    data_module = SyntheticDataModule(
        n_active_channels=selection_from_config(cfg).n_active,
        batch_size=int(cfg.data.batch_size),
        patch_size=int(cfg.data.patch_size),
        n_train=int(cfg.data.n_train),
        n_val=int(cfg.data.n_val),
        n_test=int(cfg.data.n_test),
        seed=seed,
    )

    task = LSTRegressionTask(
        n_active_channels=selection_from_config(cfg).n_active,
        base_width=int(cfg.model.base_width),
        depth=int(cfg.model.depth),
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        lr_schedule=lr_spec.payload(),
    )

    output_root = Path(str(cfg.output_root))
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = _logger_for(cfg, output_root)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{validation/loss:.4f}",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        deterministic=True,
        benchmark=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    # Run-level metadata: resolved config, release identifier, seed,
    # reproducibility note, and Git revision from the runner's context.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    metadata = {
        "resolved_config": resolved,
        "data_release_id": str(cfg.data_release_id),
        "features_root": str(cfg.features_root),
        "seed": seed,
        "git_revision": _git_revision_from_context(
            run_context_path(str(cfg.output_root), "modeling", run_id)
        ),
        "reproducibility": (
            "deterministic kernels requested (Trainer deterministic=True); "
            "byte identity is environment/hardware/library dependent; "
            "synthetic data is a lifecycle fixture, not a model of LST"
        ),
        "lr_schedule": lr_spec.payload(),
    }
    log_event(_logger, logging.INFO, "run_start", **metadata)

    # The W&B run status is decided after the complete lifecycle, so the
    # config update + finalize live in one ``finally`` that sees the final
    # status (success only after reload validation passed).
    success = False
    best_checkpoint = ""
    validation_loss = 0.0
    try:
        trainer.fit(task, datamodule=data_module)

        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint or not Path(best_checkpoint).is_file():
            raise RuntimeError(f"no best checkpoint produced at {checkpoint_dir}")
        if checkpoint_callback.best_model_score is None:
            raise RuntimeError("best checkpoint produced no monitored score")
        validation_loss = float(checkpoint_callback.best_model_score)

        # Recoverable selected model state: reload the best checkpoint and
        # verify it predicts (finite, matching extent).
        reloaded = LSTRegressionTask.load_from_checkpoint(best_checkpoint, map_location="cpu")
        sample = next(iter(data_module.val_dataloader()))
        with torch.inference_mode():
            prediction = reloaded(sample)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("reloaded best checkpoint produced non-finite predictions")
        expected_shape = sample.features.shape[:1] + (1,) + tuple(sample.features.shape[2:])
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"reloaded checkpoint prediction shape mismatch: {tuple(prediction.shape)}"
            )

        success = True
    except BaseException:
        raise
    finally:
        # Record run metadata on every path; a failed run is still a
        # recorded run. Only a fully validated lifecycle is "success".
        with timed_phase("wandb_finalize", logger=_logger):
            _finalize_wandb(wandb_logger, metadata, success)

    log_event(
        _logger,
        logging.INFO,
        "run_done",
        best_checkpoint=best_checkpoint,
        validation_loss=validation_loss,
    )
    return ModelingRunResult(
        run_ok=True, best_checkpoint=best_checkpoint, validation_loss=validation_loss
    )


def _git_revision_from_context(context_uri: str) -> str:
    """Return the recorded Git revision from the run-context JSON.

    The context file is written by the runner's ``RunLogSession`` before
    training starts; a missing file or missing field falls back to
    ``"unknown"`` (the run still records).
    """
    if not Path(context_uri).is_file():
        return "unknown"
    with open(context_uri, encoding="utf-8") as fh:
        return str(json.load(fh).get("git_commit", "unknown"))


def _best_epoch_from_path(path: str) -> int | None:
    """Recover the selected epoch (1-based) from a ``best-<epoch>-<metric>.ckpt``.

    # decision: add one, because Lightning writes ``epoch`` into the checkpoint
    # filename as ``trainer.current_epoch`` (0-based), while the recorded curve
    # and the validator use 1-based epoch numbers. Alternative: store 0-based
    # everywhere (rejected — worse for a human readout).
    """
    parts = Path(path).stem.split("-")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1]) + 1


def _logger_for(cfg: DictConfig, output_root: Path) -> WandbLogger:
    """Build the W&B logger for the configured mode."""
    wandb_mode = str(cfg.wandb.mode)
    if wandb_mode not in ("online", "offline"):
        raise ValueError(f"wandb.mode must be 'online' or 'offline', got {wandb_mode!r}")
    return WandbLogger(
        project=str(cfg.wandb.project),
        save_dir=str(output_root),
        offline=wandb_mode == "offline",
    )


def real_source_config(cfg: DictConfig) -> RealSourceConfig:
    """Build the published-source config for the contract-conforming path."""
    required = ("patch_index_root", "training_root", "features_root", "ard_root")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"real path requires {missing} in the config")
    return RealSourceConfig(
        patch_index_root=str(cfg.patch_index_root),
        training_root=str(cfg.training_root),
        features_root=str(cfg.features_root),
        ard_root=str(cfg.ard_root),
        aggregation=resolve_aggregation_method(cfg.get("aggregation")),
    )


class EpochMetricsRecorder(Callback):
    """Write the per-epoch train/validation metrics for a bounded probe.

    The W&B run holds the curve, but a retained, self-contained evidence record
    needs the numbers locally. Validation metrics are read at
    ``on_validation_epoch_end``; the epoch-aggregated ``train/mae_100m`` is read
    at ``on_train_epoch_end``, because Lightning commits that reduced metric at
    the end of the *training* epoch — which runs after the in-epoch validation —
    so reading it at validation end would record the previous epoch (or nothing
    on epoch one). The file is rewritten on every hook so a run that dies
    mid-fit still leaves the completed epochs for the operator to read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._epochs: dict[int, dict[str, object]] = {}
        self._last_amp_scale: float | None = None

    def _entry(self, epoch: int) -> dict[str, object]:
        return self._epochs.setdefault(
            epoch,
            {
                "epoch": epoch,
                "train_mae_100m": None,
                "validation_mae_100m": None,
                "validation_valid_cells": None,
                "validation_ssim_100m": None,
                "validation_ssim_100m_L250": None,
                "validation_ssim_100m_Lpatch": None,
                "validation_ssim_Lpatch_zero_range_patches": None,
                "validation_ssim_windows": None,
                "validation_ssim_supported": None,
                "validation_pred_outside_lst_range_frac": None,
                "train_ssim_supported": None,
                "train_thermal_l1_unweighted": None,
                "train_thermal_ssim_unweighted": None,
                "train_thermal_grad_unweighted": None,
                "validation_thermal_l1_unweighted": None,
                "validation_thermal_ssim_unweighted": None,
                "validation_thermal_grad_unweighted": None,
                "validation_ssim_coverage": None,
                "validation_grad_coverage": None,
                "amp_scale": None,
                "amp_scale_decreased": False,
            },
        )

    def _read(self, trainer: Trainer, key: str) -> float | None:
        value = trainer.callback_metrics.get(key)
        return None if value is None else float(value.detach().cpu())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        entry = self._entry(int(trainer.current_epoch) + 1)
        entry["validation_mae_100m"] = self._read(trainer, "validation/mae_100m")
        entry["validation_valid_cells"] = self._read(trainer, "validation/valid_cells")
        entry["validation_ssim_100m"] = self._read(trainer, "validation/ssim_100m")
        entry["validation_ssim_windows"] = self._read(trainer, "validation/ssim_windows")
        entry["validation_ssim_100m_L250"] = self._read(trainer, "validation/ssim_100m_L250")
        entry["validation_ssim_100m_Lpatch"] = self._read(trainer, "validation/ssim_100m_Lpatch")
        entry["validation_ssim_Lpatch_zero_range_patches"] = self._read(
            trainer, "validation/ssim_Lpatch_zero_range_patches"
        )
        entry["validation_ssim_supported"] = self._read(trainer, "validation/ssim_supported")
        entry["validation_pred_outside_lst_range_frac"] = self._read(
            trainer, "validation/pred_outside_lst_range_frac"
        )
        for key, field in (
            ("validation/thermal_l1_unweighted", "validation_thermal_l1_unweighted"),
            ("validation/thermal_ssim_unweighted", "validation_thermal_ssim_unweighted"),
            ("validation/thermal_grad_unweighted", "validation_thermal_grad_unweighted"),
            ("validation/ssim_coverage", "validation_ssim_coverage"),
            ("validation/grad_coverage", "validation_grad_coverage"),
        ):
            entry[field] = self._read(trainer, key)
        self._write()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        entry = self._entry(int(trainer.current_epoch) + 1)
        entry["train_mae_100m"] = self._read(trainer, "train/mae_100m")
        entry["train_ssim_supported"] = self._read(trainer, "train/ssim_supported")
        for key, field in (
            ("train/thermal_l1_unweighted", "train_thermal_l1_unweighted"),
            ("train/thermal_ssim_unweighted", "train_thermal_ssim_unweighted"),
            ("train/thermal_grad_unweighted", "train_thermal_grad_unweighted"),
        ):
            entry[field] = self._read(trainer, key)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            # One scaler read per epoch; per-batch get_scale() would introduce
            # a CPU/GPU synchronization into the measured training hot path.
            scale = float(scaler.get_scale())
            entry["amp_scale"] = scale
            entry["amp_scale_decreased"] = (
                self._last_amp_scale is not None and scale < self._last_amp_scale
            )
            self._last_amp_scale = scale
        self._write()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.epochs, indent=2, sort_keys=True), encoding="utf-8")

    @property
    def epochs(self) -> list[dict[str, object]]:
        return [self._epochs[key] for key in sorted(self._epochs)]


# Pre-registered identity tolerance for the residual probe (issue #47): the
# zero-initialized correction head must reproduce the pooled prior arm to within
# this before any paid epoch (docs/stage1-debug-results.md).
RESIDUAL_IDENTITY_TOLERANCE_K = 1e-3


def _residual_identity_mae(
    task: RealLSTTask, val_loader, *, aggregation: AggregationMethod
) -> float:
    """Masked MAE between the zero-init residual output and the pooled prior arm.

    Both sides use the shared 10x10 pooling and the same validation mask, on the
    validation cohort the run reads. A non-zero-initialized or mis-wired residual
    head shows up here, before any paid epoch.
    """
    metric = MaskedMAE()
    with torch.inference_mode():
        for batch in val_loader:
            validate_real_batch(batch, n_active_channels=task.n_active_channels)
            prior_k = reconstruct_prior_kelvin(batch.lst_prior)
            metric.update(
                pool_10m_to_100m(task(batch), aggregation, batch.mask_100m),
                pool_10m_to_100m(prior_k, aggregation, batch.mask_100m),
                batch.mask_100m,
            )
    return float(metric.compute())


def _score_test_once(
    cfg: DictConfig,
    source: RealSourceConfig,
    reloaded: RealLSTTask,
    output_root: Path,
    *,
    export_identity: EvaluationIdentity,
    ssim_range: SsimDataRange,
) -> dict[str, object]:
    """Score the 2025 test split once with the frozen selected checkpoint.

    A test-only streamed data module reads the full published test split and
    scores it through the same nested 10x10 pooling and cell-weighted masked MAE
    used for validation. ``mae_100m``, its sums and ``test_report`` pool the clamped
    rows of the test export; ``mae_100m_unclamped`` is the raw MaskedMAE diagnostic.
    Callers invoke this only after the best checkpoint and its validation reload
    have succeeded, so the test set cannot influence checkpoint choice. The read
    scope is retained as ``test_scope.json``.
    """
    selection = selection_from_config(cfg)
    test_module = RealPatchDataModule(
        source,
        batch_size=int(cfg.data.batch_size),
        max_patches_per_split=None,
        scene_ids=None,
        mode=str(cfg.data.get("mode", "eager")),
        num_workers=int(cfg.data.get("num_workers", 0)),
        shuffle_train=False,
        seed=int(cfg.seed),
        n_active_channels=selection.n_active,
        channel_indices=selection.indices,
        splits=("test",),
    )
    test_module.setup()
    scope = test_module.stats()
    (output_root / "test_scope.json").write_text(
        json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8"
    )

    metric = MaskedMAE()
    method = resolve_aggregation_method(cfg.get("aggregation"))
    patch_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in test_module.test_dataloader():
            validate_real_batch(batch, n_active_channels=reloaded.n_active_channels)
            prediction = reloaded(batch)
            expected = (batch.features.shape[0], 1) + tuple(batch.features.shape[2:])
            if tuple(prediction.shape) != expected:
                raise RuntimeError(
                    f"test pass prediction shape mismatch: {tuple(prediction.shape)} != {expected}"
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("test pass produced non-finite predictions")
            pooled = pool_10m_to_100m(prediction, method, batch.mask_100m)
            metric.update(pooled, batch.target_100m, batch.mask_100m)
            patch_rows.extend(
                score_batch_patches(
                    pooled,
                    batch.target_100m,
                    batch.mask_100m,
                    batch.metadata,
                    data_range_k=ssim_range.data_range_k,
                )
            )
    write_patch_export(
        patch_rows,
        output_root / patch_export_filename("test"),
        identity=export_identity,
        split="test",
        ssim_range=ssim_range,
    )

    report = report_metrics(patch_rows)
    patches = scope.get("patches_per_split", {})
    return {
        "mae_100m": report["mae_100m"],
        "mae_100m_unclamped": float(metric.compute()),
        "test_report": report,
        "prediction_clamp_k": [float(v) for v in EVAL_PREDICTION_CLAMP_K],
        "abs_error_sum": report["abs_error_sum"],
        "valid_cells": report["valid_cells"],
        "patches": int(patches.get("test", 0)) if isinstance(patches, dict) else 0,
        "scope_uri": str(output_root / "test_scope.json"),
    }


def _export_arm(cfg: DictConfig, tessera_enabled: bool) -> str:
    """Name the evaluated arm for the per-patch export identity."""
    arm = OmegaConf.select(cfg, "ablation_arm")
    if isinstance(arm, str) and arm.strip():
        return arm
    try:
        from hydra.core.hydra_config import HydraConfig

        name = HydraConfig.get().job.config_name if HydraConfig.initialized() else None
    except (ImportError, ValueError):  # no Hydra job context; fall through to the default
        name = None
    if name:
        return str(name)
    return "tessera" if tessera_enabled else "real_lst"


def _relative_checkpoint(best_checkpoint: str, output_root: Path) -> str:
    """Return the checkpoint path relative to ``output_root``, else its file name."""
    path = Path(best_checkpoint)
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _early_stop_epoch(
    early_stop: EarlyStopping | None,
    callback_metrics: Mapping[str, object],
    monitored: str,
    current_epoch: int,
) -> int | None:
    """Return the 1-based patience-stop epoch, or None; raise on a non-finite monitor."""
    if early_stop is None:
        return None
    reason = early_stop.stopping_reason
    raw = callback_metrics.get(monitored)
    value = float(cast(float, raw)) if raw is not None else None
    if reason == EarlyStoppingReason.NON_FINITE_METRIC or (
        value is not None and not math.isfinite(value)
    ):
        epoch = (
            early_stop.stopped_epoch + 1
            if reason != EarlyStoppingReason.NOT_STOPPED
            else current_epoch
        )
        raise RuntimeError(
            f"non-finite {monitored} at epoch {epoch} (value={value}); "
            "early stopping must not record it as a stop"
        )
    if reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:
        return early_stop.stopped_epoch + 1
    if reason == EarlyStoppingReason.NOT_STOPPED:
        return None
    raise RuntimeError(f"unexpected early stopping reason {reason!r} for {monitored}")


def resolve_run_ssim_data_range(
    cfg: DictConfig, *, expected_fingerprints: Mapping[str, str] | None
) -> SsimDataRange:
    """Resolve the run's fixed SSIM data range (artifact, or smoke-only override)."""
    override = OmegaConf.select(cfg, "ssim_data_range_test_override_k")
    if override is not None:
        if str(cfg.data.get("kind", "synthetic")) == "real":
            raise RuntimeError(
                "ssim_data_range_test_override_k is test/smoke-only and is rejected "
                "for data.kind: real"
            )
        return test_override_ssim_data_range(float(override))
    if str(cfg.data.get("kind", "synthetic")) == "real" and expected_fingerprints is None:
        raise RuntimeError(
            "real runs must check the SSIM data-range artifact against the patch-index "
            "fingerprints; none were given"
        )
    return load_ssim_data_range(expected_fingerprints=expected_fingerprints)


def _fit_contract_lifecycle(
    cfg: DictConfig,
    run_id: str,
    *,
    data_module: RealPatchDataModule | ContractSyntheticDataModule,
    source_metadata: dict,
    reproducibility: str,
) -> ModelingRunResult:
    """Fit, select on ``validation/mae_100m``, and reload-verify.

    Shared by the contract-shaped synthetic and real paths so the fit loop,
    checkpoint selection, reload check, and W&B lifecycle cannot drift apart.
    Only the data module and the source-specific metadata differ.
    """
    seed = int(cfg.seed)
    lr_spec = lr_schedule_from_config(cfg)
    seed_everything(seed)
    aggregation = resolve_aggregation_method(cfg.get("aggregation"))

    is_probe = bool(cfg.get("stage1_probe", False))
    is_full = bool(cfg.get("stage1_full", False)) or bool(cfg.get("ablation_lock", False))
    is_v6 = cfg.get("ablation_release") == "v6"
    es = early_stopping_from_config(cfg)
    evaluate_test = evaluate_test_from_config(cfg)
    residual_prior = bool(cfg.get("stage1_residual_prior", False))
    loss_name = str(OmegaConf.select(cfg, "contract.loss") or "masked_l1")
    ssim_range = resolve_run_ssim_data_range(
        cfg, expected_fingerprints=source_metadata.get("patch_index_fingerprints")
    )
    tessera_enabled = bool(OmegaConf.select(cfg, "data.tessera_enabled") or False)
    tessera_embeddings_root = str(OmegaConf.select(cfg, "data.tessera_embeddings_root") or "")
    if tessera_enabled:
        from berlin_lst_downscaling.modeling.tessera_adapter import (
            TESSERA_ADAPTER_WIDTH_DEFAULT,
            TesseraRealLSTTask,
            tessera_kept_channel_indices,
        )

        if not tessera_embeddings_root:
            raise ValueError(
                "tessera arm requires data.tessera_embeddings_root "
                "(canonical per-year product prefix)"
            )
        raw_width = OmegaConf.select(cfg, "model.tessera_adapter_width")
        adapter_width = TESSERA_ADAPTER_WIDTH_DEFAULT if raw_width is None else int(raw_width)
        task = TesseraRealLSTTask(
            n_kept_channels=len(tessera_kept_channel_indices()),
            tessera_adapter_width=adapter_width,
            base_width=int(cfg.model.base_width),
            depth=int(cfg.model.depth),
            learning_rate=float(cfg.trainer.learning_rate),
            weight_decay=float(cfg.trainer.weight_decay),
            record_probe_metrics=is_probe or is_full,
            residual_prior=residual_prior,
            loss_name=loss_name,
            aggregation=aggregation,
            ssim_data_range_k=ssim_range.data_range_k,
            ssim_data_range=ssim_range.payload(),
            lr_schedule=lr_spec.payload(),
        )
    else:
        task = RealLSTTask(
            n_active_channels=selection_from_config(cfg).n_active,
            base_width=int(cfg.model.base_width),
            depth=int(cfg.model.depth),
            learning_rate=float(cfg.trainer.learning_rate),
            weight_decay=float(cfg.trainer.weight_decay),
            record_probe_metrics=is_probe or is_full,
            residual_prior=residual_prior,
            loss_name=loss_name,
            aggregation=aggregation,
            ssim_data_range_k=ssim_range.data_range_k,
            ssim_data_range=ssim_range.payload(),
            lr_schedule=lr_spec.payload(),
        )

    output_root = Path(str(cfg.output_root))
    source_for_identity = (
        data_module.source if isinstance(data_module, RealPatchDataModule) else None
    )
    run_identity = resolve_run_identity(
        cfg,
        run_id=run_id,
        output_root=str(output_root),
        source=source_for_identity,
    )
    enforce_run_lock(cfg, run_identity)
    persist_run_identity(output_root, run_identity)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = _logger_for(cfg, output_root)
    if wandb_logger.experiment is not None:
        wandb_logger.experiment.config.update(
            {
                "aggregation": aggregation,
                "run_identity": run_identity.to_payload(),
            },
            allow_val_change=True,
        )

    # Record the read scope before fitting: which patches entered each split and
    # which were dropped, with reasons. Retained beside the run.
    with timed_phase("data_prep", logger=_logger):
        data_module.setup()
        scope = data_module.stats()
    # Fail closed on an empty split: a run with no admitted train or validation
    # patch cannot train or select a checkpoint, so it must not proceed.
    raw_counts = scope["patches_per_split"]
    per_split = (
        {str(k): int(v) for k, v in raw_counts.items()} if isinstance(raw_counts, dict) else {}
    )
    empty_splits = [s for s in ("train", "validation") if per_split.get(s, 0) <= 0]
    if empty_splits:
        raise RuntimeError(f"no patches admitted for split(s) {empty_splits}; refusing to train")
    if is_probe:
        assert_probe_minima(cfg, scope)

    # Residual wiring check (issue #47): a zero-initialized correction head must
    # reproduce the pooled prior arm before any paid epoch. The value is retained
    # in the probe summary so the validator can screen it. Run it in eval mode so
    # the validation pass does not pollute the BatchNorm running statistics the
    # fit then starts from.
    residual_identity_mae: float | None = None
    if residual_prior:
        with timed_phase("residual_identity", logger=_logger):
            was_training = task.training
            task.eval()
            try:
                residual_identity_mae = _residual_identity_mae(
                    task, data_module.val_dataloader(), aggregation=aggregation
                )
            finally:
                task.train(was_training)
            if (
                not math.isfinite(residual_identity_mae)
                or residual_identity_mae > RESIDUAL_IDENTITY_TOLERANCE_K
            ):
                raise RuntimeError(
                    f"residual identity check failed: pooled zero-init output vs prior arm "
                    f"MAE {residual_identity_mae:.6f} K exceeds "
                    f"{RESIDUAL_IDENTITY_TOLERANCE_K} K"
                )

    scope_uri = output_root / "data_scope.json"
    scope_uri.write_text(json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8")
    log_event(
        _logger,
        logging.INFO,
        "data_read",
        mode=scope.get("mode", ""),
        batches_per_split=scope["batches_per_split"],
        patches_per_split=scope["patches_per_split"],
        skipped_per_split=scope.get("skipped_per_split", {}),
        exclusions=scope["exclusions"],
        patch_ids_uri=str(scope_uri),
    )

    monitored = "validation/mae_100m"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{" + monitored + ":.4f}",
        monitor=monitored,
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # The epoch recorder is a probe/full artifact: the full Stage-1 run keeps
    # the same retained curve so its readout does not depend on W&B alone. A
    # plain local `real_full` invocation keeps its on-disk surface unchanged.
    recorder = (
        EpochMetricsRecorder(output_root / "epoch_metrics.json") if is_probe or is_full else None
    )
    callbacks: list[Callback] = [checkpoint_callback, EpochTimingRecorder(logger=_logger)]
    if recorder is not None:
        callbacks.append(recorder)
    early_stop: EarlyStopping | None = None
    if es.enabled:
        early_stop = EarlyStopping(
            monitor=monitored,
            mode="min",
            patience=es.patience,
            min_delta=es.min_delta,
            check_on_train_epoch_end=False,
        )
        callbacks.append(early_stop)

    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cast(
            Literal["32-true", "16-mixed"],
            str(cfg.trainer.get("precision", "32-true")),
        ),
        deterministic=True,
        benchmark=False,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    resolved = OmegaConf.to_container(cfg, resolve=True)
    metadata = {
        "resolved_config": resolved,
        "data_release_id": str(cfg.data_release_id),
        "seed": seed,
        "selection_metric": monitored,
        "data_scope": {
            "mode": scope.get("mode", ""),
            "batches_per_split": scope["batches_per_split"],
            "patches_per_split": scope["patches_per_split"],
            "skipped_per_split": scope.get("skipped_per_split", {}),
            "exclusions": scope["exclusions"],
            "patch_ids_uri": str(scope_uri),
            "cache_build_workers": scope.get("cache_build_workers"),
            "patch_cache_source": scope.get("patch_cache_source"),
            "patch_cache_key": scope.get("patch_cache_key"),
        },
        "git_revision": _git_revision_from_context(
            run_context_path(str(cfg.output_root), "modeling", run_id)
        ),
        "reproducibility": reproducibility,
        "run_identity": run_identity.to_payload(),
        "ssim_data_range": ssim_range.payload(),
        "lr_schedule": lr_spec.payload(),
        **source_metadata,
    }
    if is_v6:
        metadata["early_stopping"] = es.payload()
        metadata["evaluate_test"] = evaluate_test
    log_event(_logger, logging.INFO, "run_start", **metadata)

    success = False
    best_checkpoint = ""
    best_metric = 0.0
    try:
        with timed_phase("fit", logger=_logger):
            trainer.fit(task, datamodule=data_module)
        with timed_phase("post_fit", logger=_logger):
            stopped_epoch = _early_stop_epoch(
                early_stop, trainer.callback_metrics, monitored, trainer.current_epoch
            )

            best_checkpoint = checkpoint_callback.best_model_path
            if not best_checkpoint or not Path(best_checkpoint).is_file():
                raise RuntimeError(f"no best checkpoint produced at {checkpoint_dir}")
            if checkpoint_callback.best_model_score is None:
                raise RuntimeError("best checkpoint produced no monitored score")
            best_metric = float(checkpoint_callback.best_model_score)
            export_identity = EvaluationIdentity(
                arm=_export_arm(cfg, tessera_enabled),
                seed=seed,
                run_id=run_id,
                checkpoint=_relative_checkpoint(best_checkpoint, output_root),
            )
            validation_rows: list[dict[str, object]] = []

            # Recoverable selected model state: reload the best checkpoint, verify
            # it predicts on the contract shape, and recompute the selection metric
            # over the validation batches in eval mode. The saved score is checked,
            # not assumed.
            # Verify on CPU: the validation batches below are CPU tensors, so the
            # reloaded module must match them regardless of the training device.
            with timed_phase("post_fit_checkpoint_load", logger=_logger):
                if tessera_enabled:
                    from berlin_lst_downscaling.modeling.tessera_adapter import TesseraRealLSTTask

                    reloaded = TesseraRealLSTTask.load_from_checkpoint(
                        best_checkpoint, map_location="cpu"
                    )
                else:
                    reloaded = RealLSTTask.load_from_checkpoint(best_checkpoint, map_location="cpu")
                reloaded.eval()
            with timed_phase("post_fit_validation_verify", logger=_logger):
                recheck = MaskedMAE()
                correction_abs_sum = 0.0
                correction_cells = 0.0
                with torch.inference_mode():
                    for batch in data_module.val_dataloader():
                        validate_real_batch(batch, n_active_channels=reloaded.n_active_channels)
                        prediction = reloaded(batch)
                        expected = (batch.features.shape[0], 1) + tuple(batch.features.shape[2:])
                        if tuple(prediction.shape) != expected:
                            raise RuntimeError(
                                f"reloaded checkpoint prediction shape mismatch: "
                                f"{tuple(prediction.shape)} != {expected}"
                            )
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError(
                                "reloaded best checkpoint produced non-finite predictions"
                            )
                        pooled_prediction = pool_10m_to_100m(
                            prediction, aggregation, batch.mask_100m
                        )
                        recheck.update(pooled_prediction, batch.target_100m, batch.mask_100m)
                        validation_rows.extend(
                            score_batch_patches(
                                pooled_prediction,
                                batch.target_100m,
                                batch.mask_100m,
                                batch.metadata,
                                data_range_k=ssim_range.data_range_k,
                            )
                        )
                        if residual_prior:
                            prior_k = reconstruct_prior_kelvin(batch.lst_prior)
                            pooled_prior = pool_10m_to_100m(prior_k, aggregation, batch.mask_100m)
                            correction = pooled_prediction - pooled_prior
                            valid = batch.mask_100m.bool()
                            correction_abs_sum += float(correction[valid].abs().sum())
                            correction_cells += float(valid.sum())
                recomputed = float(recheck.compute())
                tolerance = 1e-3 * max(1.0, abs(best_metric))
                if abs(recomputed - best_metric) > tolerance:
                    raise RuntimeError(
                        f"reloaded checkpoint validation MAE {recomputed:.6f} != selected "
                        f"{best_metric:.6f} (tolerance {tolerance:.6f})"
                    )
                correction_mean = None
                if residual_prior:
                    if correction_cells <= 0:
                        raise RuntimeError(
                            "no valid validation cell for the residual correction check"
                        )
                    correction_mean = correction_abs_sum / correction_cells
                write_patch_export(
                    validation_rows,
                    output_root / patch_export_filename("validation"),
                    identity=export_identity,
                    split="validation",
                    ssim_range=ssim_range,
                )

            # Full Stage-1 run (issue #53): score the 2025 test split exactly once,
            # only after the selected checkpoint and its validation reload are
            # frozen. The test read is a separate test-only module; its score is
            # never fed to the fit, callbacks, or checkpoint selection.
            test_result: dict[str, object] | None = None
            if is_full and evaluate_test and isinstance(data_module, RealPatchDataModule):
                with timed_phase("post_fit_test_eval", logger=_logger):
                    test_result = _score_test_once(
                        cfg,
                        data_module.source,
                        reloaded,
                        output_root,
                        export_identity=export_identity,
                        ssim_range=ssim_range,
                    )
            elif is_full and not evaluate_test:
                test_result = {"skipped": True}

            with timed_phase("post_fit_summary", logger=_logger):
                if recorder is not None:
                    summary = {
                        "profile": (
                            "full"
                            if is_full
                            else ("probe-residual" if residual_prior else STAGE1_PROBE_CONFIG_NAME)
                        ),
                        "epochs_completed": len(recorder.epochs),
                        "max_epochs": int(cfg.trainer.max_epochs),
                        "selection_metric": monitored,
                        "best_epoch": _best_epoch_from_path(best_checkpoint),
                        "best_metric": best_metric,
                        "reload_recomputed": recomputed,
                        "validation_abs_error_sum": float(recheck.abs_error_sum),
                        "validation_valid_cells": float(recheck.valid_cells),
                        "validation_report": report_metrics(validation_rows),
                        "residual_prior": residual_prior,
                        "residual_identity_mae_k": residual_identity_mae,
                        "residual_correction_mean_abs_k": correction_mean,
                        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
                        "ssim_data_range": ssim_range.payload(),
                        "lr_schedule": lr_spec.payload(),
                    }
                    if is_probe or is_full:
                        experiment = wandb_logger.experiment
                        summary["wandb"] = {
                            "project": str(cfg.wandb.project),
                            "run_id": getattr(experiment, "id", None),
                            "url": getattr(experiment, "url", None),
                        }
                    if is_v6:
                        summary["early_stopping"] = es.payload()
                        summary["evaluate_test"] = evaluate_test
                        summary["stopped_epoch"] = stopped_epoch
                    if test_result is not None:
                        summary["test"] = test_result
                    summary_name = "full_summary.json" if is_full else "probe_summary.json"
                    (output_root / summary_name).write_text(
                        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
                    )

        success = True
    except BaseException:
        raise
    finally:
        with timed_phase("wandb_finalize", logger=_logger):
            _finalize_wandb(wandb_logger, metadata, success)

    log_event(
        _logger,
        logging.INFO,
        "run_done",
        best_checkpoint=best_checkpoint,
        selection_metric=monitored,
        selection_value=best_metric,
    )
    return ModelingRunResult(
        run_ok=True,
        best_checkpoint=best_checkpoint,
        validation_loss=best_metric,
        selection_metric=monitored,
    )


def run_contract_synthetic_training(cfg: DictConfig, run_id: str) -> ModelingRunResult:
    """Execute the contract-shaped synthetic lifecycle (no GCS, no credentials).

    Runs deterministic fixture tensors matching the pseudo-pair contract
    through the same :class:`RealLSTTask`, masked L1 loss, and masked-MAE
    checkpoint selection as the real path, so the masked lifecycle is proven
    without reading published sources.
    """
    contract_invariants(cfg)
    data_module = ContractSyntheticDataModule(
        n_active_channels=selection_from_config(cfg).n_active,
        batch_size=int(cfg.data.batch_size),
        n_train=int(cfg.data.n_train),
        n_val=int(cfg.data.n_val),
        n_test=int(cfg.data.n_test),
        seed=int(cfg.seed),
    )
    return _fit_contract_lifecycle(
        cfg,
        run_id,
        data_module=data_module,
        source_metadata={"synthetic_source": "contract_fixture"},
        reproducibility=(
            "deterministic kernels requested (Trainer deterministic=True); "
            "byte identity is environment/hardware/library dependent; the data is "
            "a contract-shaped synthetic fixture, not a model of LST, so a finite "
            "loss proves tensor/loss/lifecycle wiring, not training quality"
        ),
    )


def build_real_datamodule(
    cfg: DictConfig,
    *,
    cache_root: Path | None,
    cache_max_bytes: int,
    **overrides: Any,
) -> RealPatchDataModule:
    """Build the real patch data module exactly as a training run does.

    The prebuilt-cache builder reuses this so its reader, scaler, channel
    selection and cache build are the ones a run would use. ``overrides`` replace
    constructor keywords (the builder forces ``cache_source="job_local"``).
    """
    source = real_source_config(cfg)
    scene_ids = [str(s) for s in (cfg.data.get("scene_ids") or [])]
    # ``null`` means unbounded (a full run); ``0`` would otherwise be falsy and
    # silently become unbounded, so reject it explicitly.
    max_patches = cfg.data.get("max_patches_per_split")
    if max_patches is not None:
        if isinstance(max_patches, bool) or int(max_patches) <= 0:
            raise ValueError(
                "data.max_patches_per_split must be a positive int or null "
                f"(unbounded), got {max_patches!r}"
            )
        max_patches = int(max_patches)
    splits = tuple(str(s) for s in (cfg.data.get("splits") or _REAL_SPLITS))
    probe_cfg = cfg.data.get("probe")
    probe_scope = (
        ProbeScope(
            max_refs_per_split={str(k): int(v) for k, v in probe_cfg.max_refs_per_split.items()},
            max_refs_per_scene=int(probe_cfg.max_refs_per_scene),
            require_partial_masks=bool(probe_cfg.get("require_partial_masks", False)),
        )
        if probe_cfg is not None
        else None
    )
    selection = selection_from_config(cfg)
    if schema_version_from_config(cfg) == FEATURE_SCHEMA_VERSION_V6:
        assert_scaler_matches_config(cfg, load_scaler(source.training_root))
    patch_cache = _resolve_patch_cache(cfg)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.data.batch_size),
        "max_patches_per_split": max_patches,
        "scene_ids": scene_ids or None,
        "mode": str(cfg.data.get("mode", "eager")),
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "shuffle_train": bool(cfg.data.get("shuffle_train", False)),
        "seed": int(cfg.seed),
        "n_active_channels": selection.n_active,
        "channel_indices": selection.indices,
        "splits": splits,
        "probe_scope": probe_scope,
        "pin_memory": bool(cfg.data.get("pin_memory", False)),
        "cache_root": cache_root,
        "cache_max_bytes": cache_max_bytes,
        "cache_build_workers": cache_build_workers_from_config(cfg),
        "cache_source": patch_cache["source"],
        "prebuilt_uri_root": patch_cache["uri_root"],
        "allow_job_local_fallback": patch_cache["allow_job_local_fallback"],
        "prebuilt_download_workers": patch_cache["download_workers"],
        "prebuilt_download_retries": patch_cache["download_retries"],
    }
    kwargs.update(overrides)
    return RealPatchDataModule(source, **kwargs)


def run_real_training(cfg: DictConfig, run_id: str) -> ModelingRunResult:
    """Execute the contract-conforming real training lifecycle.

    Selects the checkpoint on ``validation/mae_100m`` (cell-weighted masked MAE
    at 100 m) and logs SSIM alongside it. Fails closed exactly like the
    synthetic lifecycle: any fit error, missing checkpoint, or reload mismatch
    raises and the runner translates it into a non-zero exit.

    The scope of the data read is whatever the config bounds it to; a full
    train/validation run over every published patch is an explicit invocation,
    not something this function decides. ``data.mode`` reads that scope either
    eagerly (a bounded smoke) or lazily through loader workers (a full run).
    """
    contract_invariants(cfg)
    if bool(cfg.get("stage1_efficiency", False)) or bool(cfg.get("stage1_efficiency_fit", False)):
        raise RuntimeError(
            "Stage-1 efficiency profiles are closed; retained results are in "
            "docs/stage1-efficiency.md"
        )
    if bool(cfg.get("stage1_verification", False)):
        raise RuntimeError(
            "Stage-1 verification profile is cancelled; runtime is frozen from "
            "the J1–J4 efficiency results in docs/stage1-efficiency.md"
        )
    # Full Stage-1 (issue #53) and Tag-11 ablations (issue #57): the fit admits
    # train/validation only. The 2025 test split is scored once after checkpoint
    # freeze via _score_test_once.
    if bool(cfg.get("stage1_full", False)) or bool(cfg.get("ablation_lock", False)):
        declared = tuple(str(s) for s in (cfg.data.get("splits") or ()))
        if declared != ("train", "validation"):
            raise RuntimeError(
                "full Stage-1 / ablation fit splits must be exactly "
                f"('train', 'validation'); got {declared!r}"
            )
    output_root_hint = Path(str(cfg.get("output_root", "data/runs/modeling")))
    cache_budget = _resolve_cache_budget(cfg)
    cache_root = _resolve_cache_root(cfg, output_root=output_root_hint)
    if cache_root is not None:
        # Fail closed when the worker disk cannot hold the cache budget plus
        # the reserve; build_patch_cache would abort mid-run otherwise.
        free = shutil.disk_usage(output_root_hint.parent).free
        if free < cache_budget + _FULL_CACHE_DISK_RESERVE_BYTES:
            raise RuntimeError(
                f"cache needs budget {cache_budget} plus "
                f"{_FULL_CACHE_DISK_RESERVE_BYTES} reserve but only {free} are free"
            )
        if cache_budget <= 0:
            raise RuntimeError("a cache path requires a positive cache budget")
    data_module = build_real_datamodule(cfg, cache_root=cache_root, cache_max_bytes=cache_budget)
    source = real_source_config(cfg)
    return _fit_contract_lifecycle(
        cfg,
        run_id,
        data_module=data_module,
        source_metadata={
            "patch_index_root": source.patch_index_root,
            "features_root": source.features_root,
            "ard_root": source.ard_root,
            "patch_index_fingerprints": patch_index_fingerprints(source.patch_index_root),
        },
        reproducibility=(
            "deterministic kernels requested (Trainer deterministic=True); "
            "byte identity is environment/hardware/library dependent; the prior "
            "for train/validation/test is the 1000 m block-expanded native LST, "
            "so the scores are a comparison under the pseudo-pair construction"
        ),
    )


def run_modeling(
    cfg: DictConfig, run_id: str, *, config_name: str | None = None
) -> ModelingRunResult:
    """Dispatch on ``data.kind`` to the matching training lifecycle.

    ``synthetic`` is the all-valid MSE fixture (the CI lifecycle smoke),
    ``synthetic_contract`` is the GCS-free contract-shaped fixture, and
    ``real`` is the WB3 patch-index path. An unknown value raises rather than
    silently training on the wrong data.

    ``config_name`` is the Hydra-selected config identity. It triggers the
    fail-closed Stage-1 lock and probe guards before any source read, so the
    named profile cannot be silently run off-contract.
    """
    guard_modeling_config(cfg, config_name)
    kind = str(cfg.data.get("kind", "synthetic"))
    if kind == "synthetic":
        return run_training(cfg, run_id=run_id)
    if kind == "synthetic_contract":
        return run_contract_synthetic_training(cfg, run_id=run_id)
    if kind == "real":
        return run_real_training(cfg, run_id=run_id)
    raise ValueError(
        f"data.kind must be 'synthetic', 'synthetic_contract', or 'real', got {kind!r}"
    )


def _finalize_wandb(wandb_logger: WandbLogger, metadata: dict, success: bool) -> None:
    """Record run metadata on every path and finalize the W&B run status."""
    if wandb_logger.experiment is not None:
        wandb_logger.experiment.config.update(metadata, allow_val_change=True)
        wandb_logger.finalize("success" if success else "failed")


__all__ = [
    "EpochMetricsRecorder",
    "ModelingRunResult",
    "assert_ablation_lock",
    "assert_probe_minima",
    "assert_stage1_efficiency",
    "assert_stage1_efficiency_scope",
    "assert_stage1_full_bounds",
    "assert_stage1_lock",
    "assert_stage1_probe",
    "assert_stage1_probe_lr3",
    "assert_stage1_runtime",
    "assert_vertex_smoke_bounds",
    "contract_invariants",
    "guard_modeling_config",
    "real_source_config",
    "run_contract_synthetic_training",
    "run_modeling",
    "run_real_training",
    "run_training",
    "write_checkpoint_manifest",
]
