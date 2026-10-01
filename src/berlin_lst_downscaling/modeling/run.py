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

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.io import log_event, run_context_path
from berlin_lst_downscaling.modeling.contracts import validate_real_batch
from berlin_lst_downscaling.modeling.guards import (
    STAGE1_PROBE_CONFIG_NAME,
    assert_probe_minima,
    assert_stage1_lock,
    assert_stage1_probe,
    assert_stage1_probe_lr3,
    assert_vertex_smoke_bounds,
    contract_invariants,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.metrics import MaskedMAE, pool_10m_to_100m
from berlin_lst_downscaling.modeling.patches import (
    RealSourceConfig,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.real_task import (
    _REAL_SPLITS,
    ProbeScope,
    RealLSTTask,
    RealPatchDataModule,
    reconstruct_prior_kelvin,
)
from berlin_lst_downscaling.modeling.synthetic import (
    ContractSyntheticDataModule,
    SyntheticDataModule,
)
from berlin_lst_downscaling.modeling.task import LSTRegressionTask

_logger = logging.getLogger(__name__)


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
    seed_everything(seed)

    data_module = SyntheticDataModule(
        n_active_channels=int(cfg.data.n_active_channels),
        batch_size=int(cfg.data.batch_size),
        patch_size=int(cfg.data.patch_size),
        n_train=int(cfg.data.n_train),
        n_val=int(cfg.data.n_val),
        n_test=int(cfg.data.n_test),
        seed=seed,
    )

    task = LSTRegressionTask(
        n_active_channels=int(cfg.data.n_active_channels),
        base_width=int(cfg.model.base_width),
        depth=int(cfg.model.depth),
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
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

    def _entry(self, epoch: int) -> dict[str, object]:
        return self._epochs.setdefault(
            epoch,
            {
                "epoch": epoch,
                "train_mae_100m": None,
                "validation_mae_100m": None,
                "validation_valid_cells": None,
                "validation_ssim_100m": None,
                "validation_ssim_windows": None,
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
        self._write()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        entry = self._entry(int(trainer.current_epoch) + 1)
        entry["train_mae_100m"] = self._read(trainer, "train/mae_100m")
        self._write()

    def _write(self) -> None:
        self.path.write_text(
            json.dumps(self.epochs, indent=2, sort_keys=True), encoding="utf-8"
        )

    @property
    def epochs(self) -> list[dict[str, object]]:
        return [self._epochs[key] for key in sorted(self._epochs)]


# Pre-registered identity tolerance for the residual probe (issue #47): the
# zero-initialized correction head must reproduce the pooled prior arm to within
# this before any paid epoch (docs/stage1-debug-results.md).
RESIDUAL_IDENTITY_TOLERANCE_K = 1e-3


def _residual_identity_mae(task: RealLSTTask, val_loader) -> float:
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
                pool_10m_to_100m(task(batch)),
                pool_10m_to_100m(prior_k),
                batch.mask_100m,
            )
    return float(metric.compute())


def _residual_correction_mean_abs(reloaded: RealLSTTask, val_loader) -> float:
    """Mean absolute *pooled* Kelvin correction over valid validation cells.

    The correction is the model's own delta (prediction minus reconstructed
    prior), pooled and scored with the shared path. Near zero means the selected
    checkpoint is still a prior passthrough.
    """
    total = 0.0
    cells = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            validate_real_batch(batch, n_active_channels=reloaded.n_active_channels)
            delta = reloaded(batch) - reconstruct_prior_kelvin(batch.lst_prior)
            pooled = pool_10m_to_100m(delta)
            valid = batch.mask_100m.bool()
            total += float(pooled[valid].abs().sum())
            cells += float(valid.sum())
    if cells <= 0.0:
        raise RuntimeError("no valid validation cell for the residual correction check")
    return total / cells


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
    seed_everything(seed)

    is_probe = bool(cfg.get("stage1_probe", False))
    residual_prior = bool(cfg.get("stage1_residual_prior", False))
    task = RealLSTTask(
        n_active_channels=int(cfg.data.n_active_channels),
        base_width=int(cfg.model.base_width),
        depth=int(cfg.model.depth),
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        record_probe_metrics=is_probe,
        residual_prior=residual_prior,
    )

    output_root = Path(str(cfg.output_root))
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = _logger_for(cfg, output_root)

    # Record the read scope before fitting: which patches entered each split and
    # which were dropped, with reasons. Retained beside the run.
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
        raise RuntimeError(
            f"no patches admitted for split(s) {empty_splits}; refusing to train"
        )
    if is_probe:
        assert_probe_minima(cfg, scope)

    # Residual wiring check (issue #47): a zero-initialized correction head must
    # reproduce the pooled prior arm before any paid epoch. The value is retained
    # in the probe summary so the validator can screen it. Run it in eval mode so
    # the validation pass does not pollute the BatchNorm running statistics the
    # fit then starts from.
    residual_identity_mae: float | None = None
    if residual_prior:
        was_training = task.training
        task.eval()
        try:
            residual_identity_mae = _residual_identity_mae(task, data_module.val_dataloader())
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

    # The epoch recorder is a probe-only artifact: a full run keeps its previous
    # on-disk surface unchanged.
    recorder = (
        EpochMetricsRecorder(output_root / "epoch_metrics.json") if is_probe else None
    )
    callbacks: list[Callback] = [checkpoint_callback]
    if recorder is not None:
        callbacks.append(recorder)

    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
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
        },
        "git_revision": _git_revision_from_context(
            run_context_path(str(cfg.output_root), "modeling", run_id)
        ),
        "reproducibility": reproducibility,
        **source_metadata,
    }
    log_event(_logger, logging.INFO, "run_start", **metadata)

    success = False
    best_checkpoint = ""
    best_metric = 0.0
    try:
        trainer.fit(task, datamodule=data_module)

        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint or not Path(best_checkpoint).is_file():
            raise RuntimeError(f"no best checkpoint produced at {checkpoint_dir}")
        if checkpoint_callback.best_model_score is None:
            raise RuntimeError("best checkpoint produced no monitored score")
        best_metric = float(checkpoint_callback.best_model_score)

        # Recoverable selected model state: reload the best checkpoint, verify
        # it predicts on the contract shape, and recompute the selection metric
        # over the validation batches in eval mode. The saved score is checked,
        # not assumed.
        # Verify on CPU: the validation batches below are CPU tensors, so the
        # reloaded module must match them regardless of the training device.
        reloaded = RealLSTTask.load_from_checkpoint(best_checkpoint, map_location="cpu")
        reloaded.eval()
        recheck = MaskedMAE()
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
                    raise RuntimeError("reloaded best checkpoint produced non-finite predictions")
                recheck.update(
                    pool_10m_to_100m(prediction), batch.target_100m, batch.mask_100m
                )
        recomputed = float(recheck.compute())
        tolerance = 1e-3 * max(1.0, abs(best_metric))
        if abs(recomputed - best_metric) > tolerance:
            raise RuntimeError(
                f"reloaded checkpoint validation MAE {recomputed:.6f} != selected "
                f"{best_metric:.6f} (tolerance {tolerance:.6f})"
            )

        if recorder is not None:
            summary = {
                "profile": "probe-residual" if residual_prior else STAGE1_PROBE_CONFIG_NAME,
                "epochs_completed": len(recorder.epochs),
                "max_epochs": int(cfg.trainer.max_epochs),
                "selection_metric": monitored,
                "best_epoch": _best_epoch_from_path(best_checkpoint),
                "best_metric": best_metric,
                "reload_recomputed": recomputed,
                "residual_prior": residual_prior,
                "residual_identity_mae_k": residual_identity_mae,
                "residual_correction_mean_abs_k": (
                    _residual_correction_mean_abs(reloaded, data_module.val_dataloader())
                    if residual_prior
                    else None
                ),
                "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            }
            (output_root / "probe_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )

        success = True
    except BaseException:
        raise
    finally:
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
        n_active_channels=int(cfg.data.n_active_channels),
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
            max_refs_per_split={
                str(k): int(v) for k, v in probe_cfg.max_refs_per_split.items()
            },
            max_refs_per_scene=int(probe_cfg.max_refs_per_scene),
        )
        if probe_cfg is not None
        else None
    )
    data_module = RealPatchDataModule(
        source,
        batch_size=int(cfg.data.batch_size),
        max_patches_per_split=max_patches,
        scene_ids=scene_ids or None,
        mode=str(cfg.data.get("mode", "eager")),
        num_workers=int(cfg.data.get("num_workers", 0)),
        shuffle_train=bool(cfg.data.get("shuffle_train", False)),
        seed=int(cfg.seed),
        n_active_channels=int(cfg.data.n_active_channels),
        splits=splits,
        probe_scope=probe_scope,
    )
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
    "assert_probe_minima",
    "assert_stage1_lock",
    "assert_stage1_probe",
    "assert_stage1_probe_lr3",
    "assert_vertex_smoke_bounds",
    "contract_invariants",
    "guard_modeling_config",
    "real_source_config",
    "run_contract_synthetic_training",
    "run_modeling",
    "run_real_training",
    "run_training",
]