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
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.io import log_event, run_context_path
from berlin_lst_downscaling.data.training.contracts import split_for_year
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    validate_real_batch,
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


def contract_invariants(cfg: DictConfig) -> None:
    """Assert the declared frozen contract invariants; fail closed on drift.

    ``configs/modeling/_base.yaml`` declares the frozen geometry, temporal
    split, Stage-1 loss, and feature order so they are visible in every
    resolved config. They are not free parameters: a mismatch with the
    contract constants raises instead of silently training on a drifted
    geometry or loss. The split mapping itself stays a policy constant and is
    only probed here for the two holdout years.
    """
    declared = cfg.get("contract")
    if declared is None:
        raise ValueError("config is missing the frozen 'contract' block")
    expected: dict[str, object] = {
        "split": "temporal",
        "patch_px": REAL_PATCH_PX,
        "patch_cells": REAL_PATCH_CELLS,
        "loss": "masked_l1",
        "feature_order": "v3_first_c",
    }
    for key, value in expected.items():
        actual = declared.get(key)
        if actual != value:
            raise ValueError(
                f"contract.{key} = {actual!r} contradicts the frozen contract "
                f"value {value!r}"
            )
    if split_for_year(2024) != "validation" or split_for_year(2025) != "test":
        raise RuntimeError("temporal split contract no longer maps 2024/2025 as frozen")


def assert_vertex_smoke_bounds(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the bounded Vertex GPU smoke.

    The Vertex acceptance run is a paid, one-shot GPU job. The worker must
    refuse to start if the resolved config drifted from the bounded smoke
    contract — a silent fall back to CPU, an unbounded split, or a full-length
    run would waste the job. Called before any source read when the resolved
    config sets ``vertex_smoke_bounds: true``; the guarded launcher enforces the
    same bounds before submitting.
    """
    problems: list[str] = []
    if str(cfg.data.get("kind")) != "real":
        problems.append(f"data.kind={cfg.data.get('kind')!r} (expected 'real')")
    if str(cfg.data.get("mode")) != "stream":
        problems.append(f"data.mode={cfg.data.get('mode')!r} (expected 'stream')")

    bound = cfg.data.get("max_patches_per_split")
    if isinstance(bound, bool) or not isinstance(bound, int) or not 1 <= bound <= 4:
        problems.append(
            f"data.max_patches_per_split={bound!r} (expected an int in [1, 4])"
        )

    epochs = cfg.trainer.get("max_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 1:
        problems.append(f"trainer.max_epochs={epochs!r} (expected 1)")

    if str(cfg.trainer.get("accelerator")) != "gpu":
        problems.append(
            f"trainer.accelerator={cfg.trainer.get('accelerator')!r} (expected 'gpu')"
        )

    devices = cfg.trainer.get("devices")
    if isinstance(devices, bool) or not isinstance(devices, int) or devices != 1:
        problems.append(f"trainer.devices={devices!r} (expected 1)")

    if str(cfg.wandb.get("mode")) != "online":
        problems.append(f"wandb.mode={cfg.wandb.get('mode')!r} (expected 'online')")

    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://"):
        problems.append(
            f"output_root={output_root!r} (expected a non-empty local path)"
        )

    if problems:
        raise ValueError("vertex smoke config is out of bounds: " + "; ".join(problems))


# The selected config name that carries the frozen Stage-1 experiment contract.
STAGE1_LOCKED_CONFIG_NAME = "stage1_locked"

# Frozen Stage-1 scientific values (issue #40). Batch size, AMP precision,
# dataloader workers, and early-stopping patience are deliberately absent: they
# are operational retunes, not method choices.
_STAGE1_LOCK_EXPECTED: dict[str, object] = {
    "data.kind": "real",
    "data.mode": "stream",
    "data.n_active_channels": 10,
    "data.max_patches_per_split": None,
    "data.shuffle_train": True,
    "model.depth": 4,
    "model.base_width": 32,
    "trainer.learning_rate": 1.0e-3,
    "trainer.weight_decay": 0.0,
    "trainer.max_epochs": 20,
    "seed": 0,
}


def assert_stage1_lock(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the frozen Stage-1 profile.

    The Stage-1 full run is the locked experiment contract (issue #40): a
    silent drift in the spectral block, backbone, or optimisation defaults
    would invalidate a later comparison. Called before any source read when the
    selected config name is ``stage1_locked``; the lock marker is required too,
    so overriding it away cannot disable the guard. The separate ``lst_prior``
    input is not counted among the ten feature channels.
    """
    problems: list[str] = []
    if cfg.get("stage1_lock") is not True:
        problems.append("stage1_lock is not true (the lock marker was removed or overridden)")
    for key, expected in _STAGE1_LOCK_EXPECTED.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")
    scene_ids = cfg.data.get("scene_ids")
    if scene_ids is None or len(scene_ids) != 0:
        problems.append(
            f"data.scene_ids={scene_ids!r} (expected empty = all published scenes)"
        )
    if problems:
        raise ValueError("stage1_locked config is off-contract: " + "; ".join(problems))


# The selected config name that carries the bounded Stage-1 learning probe.
STAGE1_PROBE_CONFIG_NAME = "stage1_probe"

_STAGE1_PROBE_SPLITS = ("train", "validation")

# Frozen Stage-1 scientific values the probe shares with the full profile. The
# probe's own scope keys (epochs, per-split ref target, per-scene cap, minima)
# are asserted separately below: they are the probe's bounds, not the method.
_STAGE1_PROBE_FROZEN: dict[str, object] = {
    "data.kind": "real",
    "data.mode": "stream",
    "data.n_active_channels": 10,
    "data.shuffle_train": True,
    "data.max_patches_per_split": None,
    "model.depth": 4,
    "model.base_width": 32,
    "trainer.learning_rate": 1.0e-3,
    "trainer.weight_decay": 0.0,
    "trainer.accelerator": "gpu",
    "trainer.devices": 1,
    "seed": 0,
}

_STAGE1_PROBE_SCOPE: dict[str, object] = {
    "max_refs_per_scene": 8,
    "max_refs_per_split": {"train": 128, "validation": 64},
    "min_admitted_per_split": {"train": 120, "validation": 60},
    "min_scenes_per_split": {"train": 16, "validation": 8},
    "min_train_years": 3,
}


def assert_stage1_probe(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the bounded Stage-1 probe.

    The probe must not become a full run, read the 2025 test split, or quietly
    drop the frozen Stage-1 method. Checked before ``RunLogSession`` opens the
    output path and before any source read, so a drifted invocation cannot
    create a run directory, touch GCS, or start a paid GPU job.
    """
    problems: list[str] = []
    if cfg.get("stage1_probe") is not True:
        problems.append("stage1_probe is not true (the probe marker was removed or overridden)")
    for key, expected in _STAGE1_PROBE_FROZEN.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")

    if list(cfg.data.get("splits") or []) != list(_STAGE1_PROBE_SPLITS):
        problems.append(
            f"data.splits={cfg.data.get('splits')!r} (expected {list(_STAGE1_PROBE_SPLITS)!r})"
        )
    if cfg.data.get("scene_ids"):
        problems.append(
            f"data.scene_ids={cfg.data.get('scene_ids')!r} (expected empty; the probe "
            "selects its cohort deterministically)"
        )

    probe = cfg.data.get("probe")
    if probe is None:
        problems.append("data.probe block is missing")
    else:
        per_scene = probe.get("max_refs_per_scene")
        if per_scene != _STAGE1_PROBE_SCOPE["max_refs_per_scene"]:
            problems.append(
                f"data.probe.max_refs_per_scene={per_scene!r} "
                f"(expected {_STAGE1_PROBE_SCOPE['max_refs_per_scene']!r})"
            )
        for scope_key in ("max_refs_per_split", "min_admitted_per_split", "min_scenes_per_split"):
            actual_scope = dict(probe.get(scope_key) or {})
            expected_scope = _STAGE1_PROBE_SCOPE[scope_key]
            if actual_scope != expected_scope:
                problems.append(
                    f"data.probe.{scope_key}={actual_scope!r} (expected {expected_scope!r})"
                )
        min_years = probe.get("min_train_years")
        if min_years != _STAGE1_PROBE_SCOPE["min_train_years"]:
            problems.append(
                f"data.probe.min_train_years={min_years!r} "
                f"(expected {_STAGE1_PROBE_SCOPE['min_train_years']!r})"
            )

    epochs = cfg.trainer.get("max_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 6:
        problems.append(f"trainer.max_epochs={epochs!r} (expected 6 for the probe)")

    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append(
            f"output_root={output_root!r} (expected a job-local path under data/runs/)"
        )

    if str(cfg.wandb.get("mode")) != "online":
        problems.append(f"wandb.mode={cfg.wandb.get('mode')!r} (expected 'online')")

    if problems:
        raise ValueError("stage1_probe config is off-contract: " + "; ".join(problems))


def assert_probe_minima(cfg: DictConfig, scope: dict) -> None:
    """Fail closed unless the realized probe cohort meets the declared minima."""
    probe = cfg.data.probe
    admitted_raw = scope.get("patches_per_split")
    scenes_raw = scope.get("scenes_per_split")
    years_raw = scope.get("years_per_split")
    admitted = admitted_raw if isinstance(admitted_raw, dict) else {}
    scenes = scenes_raw if isinstance(scenes_raw, dict) else {}
    years = years_raw if isinstance(years_raw, dict) else {}

    problems: list[str] = []
    extra = sorted(set(admitted) - set(_STAGE1_PROBE_SPLITS))
    if extra:
        problems.append(f"unexpected split(s) loaded: {extra} (probe is train/validation only)")
    for split in _STAGE1_PROBE_SPLITS:
        need_admitted = int(probe.min_admitted_per_split[split])
        got_admitted = int(admitted.get(split, 0))
        if got_admitted < need_admitted:
            problems.append(f"{split} admitted {got_admitted} < required {need_admitted}")
        need_scenes = int(probe.min_scenes_per_split[split])
        got_scenes = len(scenes.get(split, []))
        if got_scenes < need_scenes:
            problems.append(f"{split} covers {got_scenes} scenes < required {need_scenes}")
    need_years = int(probe.min_train_years)
    got_years = len(years.get("train", []))
    if got_years < need_years:
        problems.append(f"train covers {got_years} years < required {need_years}")

    if problems:
        raise ValueError("stage1_probe cohort is out of bounds: " + "; ".join(problems))


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
    task = RealLSTTask(
        n_active_channels=int(cfg.data.n_active_channels),
        base_width=int(cfg.model.base_width),
        depth=int(cfg.model.depth),
        learning_rate=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
        record_probe_metrics=is_probe,
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
                "profile": STAGE1_PROBE_CONFIG_NAME,
                "epochs_completed": len(recorder.epochs),
                "max_epochs": int(cfg.trainer.max_epochs),
                "selection_metric": monitored,
                "best_epoch": _best_epoch_from_path(best_checkpoint),
                "best_metric": best_metric,
                "reload_recomputed": recomputed,
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


def guard_modeling_config(cfg: DictConfig, config_name: str | None) -> None:
    """Run the config-identity guards before any run directory or source read.

    The runner calls this before ``RunLogSession`` opens the output path, so a
    drifted invocation cannot create a run directory or begin a paid GPU run.
    Re-invoked by :func:`run_modeling` for programmatic callers; the guards are
    pure assertions and idempotent.
    """
    if bool(cfg.get("stage1_probe", False)) and config_name != STAGE1_PROBE_CONFIG_NAME:
        raise ValueError(
            "stage1_probe marker is set on a non-probe config "
            f"({config_name!r}); the probe guard would not run"
        )
    if config_name == STAGE1_LOCKED_CONFIG_NAME:
        assert_stage1_lock(cfg)
    if config_name == STAGE1_PROBE_CONFIG_NAME:
        assert_stage1_probe(cfg)
    if bool(cfg.get("vertex_smoke_bounds", False)):
        assert_vertex_smoke_bounds(cfg)


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
    "assert_vertex_smoke_bounds",
    "contract_invariants",
    "guard_modeling_config",
    "real_source_config",
    "run_contract_synthetic_training",
    "run_modeling",
    "run_real_training",
    "run_training",
]