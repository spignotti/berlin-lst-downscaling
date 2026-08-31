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
from berlin_lst_downscaling.modeling.synthetic import SyntheticDataModule
from berlin_lst_downscaling.modeling.task import LSTRegressionTask

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelingRunResult:
    """Outcome of a configured training run."""

    run_ok: bool
    best_checkpoint: str | None
    validation_loss: float | None


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

    wandb_mode = str(cfg.wandb.mode)
    if wandb_mode not in ("online", "offline"):
        raise ValueError(f"wandb.mode must be 'online' or 'offline', got {wandb_mode!r}")
    wandb_logger = WandbLogger(
        project=str(cfg.wandb.project),
        save_dir=str(output_root),
        offline=wandb_mode == "offline",
    )

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
        if wandb_logger.experiment is not None:
            wandb_logger.experiment.config.update(metadata, allow_val_change=True)
            wandb_logger.finalize("success" if success else "failed")

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


__all__ = ["ModelingRunResult", "run_training"]