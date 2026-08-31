"""WB3 modeling run orchestration — the configured training lifecycle.

Builds the synthetic data module, the fixed U-Net, the Lightning task,
the local checkpoint callback, and the W&B logger from the resolved
Hydra config, runs ``trainer.fit``, then reloads the best checkpoint and
validates it as the selected model state.

Reproducibility posture
-----------------------
- ``seed`` is applied once and passed to the data module; the Trainer runs
  with ``deterministic=True`` and ``benchmark=False``.
- ``run_context`` carries the resolved config, the data-release identifier,
  the seed, and a reproducibility note; the Git revision and dirty state
  are recorded by :class:`~berlin_lst_downscaling.data.io.RunLogSession`
  (``write_run_context``) at the runner level.
- Limitations are documented at run level: deterministic kernels do **not**
  guarantee byte-identical results across environments/hardware/library
  versions; the synthetic data is a lifecycle fixture, not a model of LST.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.modeling.synthetic import SyntheticDataModule
from berlin_lst_downscaling.modeling.task import LSTRegressionTask

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelingRunResult:
    """Outcome of a configured training run."""

    run_ok: bool
    best_checkpoint: str | None
    validation_loss: float | None


def run_training(cfg: DictConfig) -> ModelingRunResult:
    """Execute the configured synthetic training lifecycle.

    Fails closed: any lifecycle failure (fit error, missing checkpoint,
    reload mismatch) raises or returns ``run_ok=False`` — the runner
    translates that into a non-zero process exit.
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
    # reproducibility note, and Git revision (best-effort subprocess).
    resolved = OmegaConf.to_container(cfg, resolve=True)
    metadata = {
        "resolved_config": resolved,
        "data_release_id": str(cfg.data_release_id),
        "features_root": str(cfg.features_root),
        "seed": seed,
        "git_revision": _git_revision(),
        "reproducibility": (
            "deterministic kernels requested (Trainer deterministic=True); "
            "byte identity is environment/hardware/library dependent; "
            "synthetic data is a lifecycle fixture, not a model of LST"
        ),
    }
    log_event(_logger, logging.INFO, "run_start", **metadata)

    fit_error: BaseException | None = None
    try:
        trainer.fit(task, datamodule=data_module)
    except BaseException as exc:
        fit_error = exc
    finally:
        # Record run metadata on every path; a failed run is still a
        # recorded run.
        if wandb_logger.experiment is not None:
            wandb_logger.experiment.config.update(metadata, allow_val_change=True)
            wandb_logger.finalize("success" if fit_error is None else "failed")

    if fit_error is not None:
        raise fit_error

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
    if prediction.shape != sample.features.shape[:1] + (1,) + tuple(sample.features.shape[2:]):
        raise RuntimeError(
            f"reloaded checkpoint prediction shape mismatch: {tuple(prediction.shape)}"
        )

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


def _git_revision() -> str:
    """Return the short HEAD revision (best-effort, non-blocking)."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S607
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: S110 — best-effort git context
        pass
    return "unknown"


__all__ = ["ModelingRunResult", "run_training"]