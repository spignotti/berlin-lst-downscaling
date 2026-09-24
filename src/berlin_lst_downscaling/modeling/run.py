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
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.io import log_event, run_context_path
from berlin_lst_downscaling.modeling.contracts import validate_real_batch
from berlin_lst_downscaling.modeling.metrics import MaskedMAE, pool_10m_to_100m
from berlin_lst_downscaling.modeling.patches import RealSourceConfig
from berlin_lst_downscaling.modeling.real_task import RealLSTTask, RealPatchDataModule
from berlin_lst_downscaling.modeling.synthetic import SyntheticDataModule
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
    torch.manual_seed(seed)

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
        reloaded = LSTRegressionTask.load_from_checkpoint(best_checkpoint)
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


def run_real_training(cfg: DictConfig, run_id: str) -> ModelingRunResult:
    """Execute the contract-conforming real training lifecycle.

    Selects the checkpoint on ``validation/mae_100m`` (cell-weighted masked MAE
    at 100 m) and logs SSIM alongside it. Fails closed exactly like the
    synthetic lifecycle: any fit error, missing checkpoint, or reload mismatch
    raises and the runner translates it into a non-zero exit.

    The scope of the data read is whatever the config bounds it to; a full
    train/validation run over every published patch is an explicit invocation,
    not something this function decides.
    """
    seed = int(cfg.seed)
    torch.manual_seed(seed)

    source = real_source_config(cfg)
    scene_ids = [str(s) for s in (cfg.data.get("scene_ids") or [])]
    max_batches = cfg.data.get("max_batches_per_split") or None
    data_module = RealPatchDataModule(
        source,
        batch_size=int(cfg.data.batch_size),
        max_batches_per_split=None if max_batches is None else int(max_batches),
        scene_ids=scene_ids or None,
    )

    task = RealLSTTask(
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

    # Record the read scope before fitting: which patches entered each split and
    # which were dropped, with reasons. Retained beside the run.
    data_module.setup()
    scope = data_module.stats()
    scope_uri = output_root / "data_scope.json"
    scope_uri.write_text(json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8")
    log_event(
        _logger,
        logging.INFO,
        "data_read",
        batches_per_split=scope["batches_per_split"],
        patches_per_split=scope["patches_per_split"],
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

    trainer = Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        deterministic=True,
        benchmark=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    resolved = OmegaConf.to_container(cfg, resolve=True)
    metadata = {
        "resolved_config": resolved,
        "data_release_id": str(cfg.data_release_id),
        "patch_index_root": source.patch_index_root,
        "features_root": source.features_root,
        "ard_root": source.ard_root,
        "seed": seed,
        "selection_metric": monitored,
        "data_scope": {
            "batches_per_split": scope["batches_per_split"],
            "patches_per_split": scope["patches_per_split"],
            "exclusions": scope["exclusions"],
            "patch_ids_uri": str(scope_uri),
        },
        "git_revision": _git_revision_from_context(
            run_context_path(str(cfg.output_root), "modeling", run_id)
        ),
        "reproducibility": (
            "deterministic kernels requested (Trainer deterministic=True); "
            "byte identity is environment/hardware/library dependent; the prior "
            "for train/validation/test is the 1000 m block-expanded native LST, "
            "so the scores are a comparison under the pseudo-pair construction"
        ),
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
        reloaded = RealLSTTask.load_from_checkpoint(best_checkpoint)
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


def run_modeling(cfg: DictConfig, run_id: str) -> ModelingRunResult:
    """Dispatch to the synthetic or the contract-conforming training lifecycle.

    ``data.kind`` selects the path: ``synthetic`` (the CI lifecycle fixture) or
    ``real`` (the WB3 patch-index path). An unknown value raises rather than
    silently training on the wrong data.
    """
    kind = str(cfg.data.get("kind", "synthetic"))
    if kind == "synthetic":
        return run_training(cfg, run_id=run_id)
    if kind == "real":
        return run_real_training(cfg, run_id=run_id)
    raise ValueError(f"data.kind must be 'synthetic' or 'real', got {kind!r}")


def _finalize_wandb(wandb_logger: WandbLogger, metadata: dict, success: bool) -> None:
    """Record run metadata on every path and finalize the W&B run status."""
    if wandb_logger.experiment is not None:
        wandb_logger.experiment.config.update(metadata, allow_val_change=True)
        wandb_logger.finalize("success" if success else "failed")


__all__ = [
    "ModelingRunResult",
    "real_source_config",
    "run_modeling",
    "run_real_training",
    "run_training",
]