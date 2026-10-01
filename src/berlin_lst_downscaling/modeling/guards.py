"""Fail-closed config-identity guards for the WB3 modeling runs.

The runner calls :func:`guard_modeling_config` before ``RunLogSession``
opens the output path and before any source read, so a drifted invocation
cannot create a run directory, touch GCS, or start a paid GPU job. The
guards read only the resolved config and raise on any deviation from the
frozen Stage-1 contract, the bounded recovery probe, or the Vertex smoke
bounds.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.training.contracts import split_for_year
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
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
    # The residual representation is frozen into the full method after the
    # issue #47 recovery GO (docs/stage1-debug-results.md §6/§7).
    "stage1_residual_prior": True,
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


# The selected config names that carry the bounded Stage-1 recovery probe
# (issue #47). Both share one frozen method; only the learning rate differs.
STAGE1_PROBE_CONFIG_NAME = "stage1_probe"
STAGE1_PROBE_LR3_CONFIG_NAME = "stage1_probe_lr3"

_STAGE1_PROBE_SPLITS = ("train", "validation")

# Frozen recovery-probe method (issues #45/#47): the spectral block, backbone,
# residual representation, seed, and probe bounds the two trials share. The
# probe's own scope keys (epochs, per-split ref target, per-scene cap, minima)
# are asserted separately below: they are the probe's bounds, not the method.
# The learning rate is the single deliberate variable between the two trials and
# is asserted per config name via ``_STAGE1_PROBE_LR``.
_STAGE1_PROBE_FROZEN: dict[str, object] = {
    "data.kind": "real",
    "data.mode": "stream",
    "data.n_active_channels": 10,
    "data.shuffle_train": True,
    "data.max_patches_per_split": None,
    "model.depth": 4,
    "model.base_width": 32,
    "trainer.weight_decay": 0.0,
    "trainer.accelerator": "gpu",
    "trainer.devices": 1,
    "seed": 0,
    "stage1_residual_prior": True,
}

# Learning rate per trial config. Trial 2 is only ever run when trial 1 is
# finite and stable but flat (docs/stage1-debug-results.md pre-registration).
_STAGE1_PROBE_LR: dict[str, float] = {
    STAGE1_PROBE_CONFIG_NAME: 1.0e-3,
    STAGE1_PROBE_LR3_CONFIG_NAME: 3.0e-3,
}

_STAGE1_PROBE_SCOPE: dict[str, object] = {
    "max_refs_per_scene": 8,
    "max_refs_per_split": {"train": 128, "validation": 64},
    "min_admitted_per_split": {"train": 120, "validation": 60},
    "min_scenes_per_split": {"train": 16, "validation": 8},
    "min_train_years": 3,
}


def _assert_stage1_probe(cfg: DictConfig, *, config_name: str) -> None:
    """Fail closed unless the resolved config is a bounded residual probe.

    The recovery probe must not become a full run, read the 2025 test split,
    drop the residual representation, or drift the frozen method. Checked before
    ``RunLogSession`` opens the output path and before any source read, so a
    drifted invocation cannot create a run directory, touch GCS, or start a paid
    GPU job.
    """
    problems: list[str] = []
    if cfg.get("stage1_probe") is not True:
        problems.append("stage1_probe is not true (the probe marker was removed or overridden)")
    for key, expected in _STAGE1_PROBE_FROZEN.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")

    expected_lr = _STAGE1_PROBE_LR[config_name]
    actual_lr = OmegaConf.select(cfg, "trainer.learning_rate")
    if actual_lr != expected_lr:
        problems.append(
            f"trainer.learning_rate={actual_lr!r} (expected {expected_lr!r} for {config_name})"
        )

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


def assert_stage1_probe(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the residual probe at LR 1e-3."""
    _assert_stage1_probe(cfg, config_name=STAGE1_PROBE_CONFIG_NAME)


def assert_stage1_probe_lr3(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the residual probe at LR 3e-3."""
    _assert_stage1_probe(cfg, config_name=STAGE1_PROBE_LR3_CONFIG_NAME)


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


def guard_modeling_config(cfg: DictConfig, config_name: str | None) -> None:
    """Run the config-identity guards before any run directory or source read.

    The runner calls this before ``RunLogSession`` opens the output path, so a
    drifted invocation cannot create a run directory or begin a paid GPU run.
    Re-invoked by :func:`run_modeling` for programmatic callers; the guards are
    pure assertions and idempotent.
    """
    if bool(cfg.get("stage1_probe", False)) and config_name not in (
        STAGE1_PROBE_CONFIG_NAME,
        STAGE1_PROBE_LR3_CONFIG_NAME,
    ):
        raise ValueError(
            "stage1_probe marker is set on a non-probe config "
            f"({config_name!r}); the probe guard would not run"
        )
    if bool(cfg.get("stage1_residual_prior", False)) and config_name not in (
        STAGE1_LOCKED_CONFIG_NAME,
        STAGE1_PROBE_CONFIG_NAME,
        STAGE1_PROBE_LR3_CONFIG_NAME,
    ):
        raise ValueError(
            "stage1_residual_prior is the Stage-1 recovery representation "
            f"(issues #40/#47) but is set on config {config_name!r}; refusing to run"
        )
    if config_name == STAGE1_LOCKED_CONFIG_NAME:
        assert_stage1_lock(cfg)
    if config_name == STAGE1_PROBE_CONFIG_NAME:
        assert_stage1_probe(cfg)
    if config_name == STAGE1_PROBE_LR3_CONFIG_NAME:
        assert_stage1_probe_lr3(cfg)
    if bool(cfg.get("vertex_smoke_bounds", False)):
        assert_vertex_smoke_bounds(cfg)


__all__ = [
    "assert_probe_minima",
    "assert_stage1_lock",
    "assert_stage1_probe",
    "assert_stage1_probe_lr3",
    "assert_vertex_smoke_bounds",
    "contract_invariants",
    "guard_modeling_config",
]
