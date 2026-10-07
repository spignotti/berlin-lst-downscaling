"""Fail-closed config-identity guards for the WB3 modeling runs.

The runner calls :func:`guard_modeling_config` before ``RunLogSession``
opens the output path and before any source read, so a drifted invocation
cannot create a run directory, touch GCS, or start a paid GPU job. The
guards read only the resolved config and raise on any deviation from the
frozen Stage-1 contract, the bounded recovery probe, or the Vertex smoke
bounds.
"""

from __future__ import annotations

import os

from omegaconf import DictConfig, ListConfig, OmegaConf

from berlin_lst_downscaling.data.training.contracts import split_for_year
from berlin_lst_downscaling.modeling.channels import (
    ABLATION_ISOLATION_CHANNELS,
    ABLATION_LOCKED_CONFIG_NAMES,
    ABLATION_STAGE_CHANNELS,
    feature_order_for,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
)
from berlin_lst_downscaling.modeling.efficiency_protocol import EFFICIENCY_SESSION_ID

_ALLOWED_CONTRACT_LOSSES = frozenset({"masked_l1", "thermal_aware"})
_ALLOWED_FEATURE_ORDERS = frozenset({"v3_first_c", "v3_named_subset"})


def contract_invariants(cfg: DictConfig) -> None:
    """Assert the declared frozen contract invariants; fail closed on drift.

    ``configs/modeling/_base.yaml`` declares the frozen geometry, temporal
    split, Stage-1 loss, and feature order so they are visible in every
    resolved config. They are not free parameters: a mismatch with the
    contract constants raises instead of silently training on a drifted
    geometry or loss. Tag-11 ablation configs may declare ``v3_named_subset``
    and Stage 5 may declare ``thermal_aware``; both are still checked against
    the resolved channel selection. The split mapping itself stays a policy
    constant and is only probed here for the two holdout years.
    """
    declared = cfg.get("contract")
    if declared is None:
        raise ValueError("config is missing the frozen 'contract' block")
    expected_geometry: dict[str, object] = {
        "split": "temporal",
        "patch_px": REAL_PATCH_PX,
        "patch_cells": REAL_PATCH_CELLS,
    }
    for key, value in expected_geometry.items():
        actual = declared.get(key)
        if actual != value:
            raise ValueError(
                f"contract.{key} = {actual!r} contradicts the frozen contract value {value!r}"
            )
    loss = declared.get("loss")
    if loss not in _ALLOWED_CONTRACT_LOSSES:
        raise ValueError(
            f"contract.loss = {loss!r} is not one of {sorted(_ALLOWED_CONTRACT_LOSSES)}"
        )
    feature_order = declared.get("feature_order")
    if feature_order not in _ALLOWED_FEATURE_ORDERS:
        raise ValueError(
            "contract.feature_order = "
            f"{feature_order!r} is not one of {sorted(_ALLOWED_FEATURE_ORDERS)}"
        )
    selection = selection_from_config(cfg)
    expected_order = feature_order_for(selection)
    if feature_order != expected_order:
        raise ValueError(
            f"contract.feature_order = {feature_order!r} does not match the resolved "
            f"channel selection ({expected_order!r}, C={selection.n_active})"
        )
    if not bool(cfg.get("ablation_lock", False)) and loss != "masked_l1":
        raise ValueError("thermal_aware loss is only admitted on ablation_lock configs")
    if (
        not bool(cfg.get("ablation_lock", False))
        and feature_order != "v3_first_c"
    ):
        raise ValueError("v3_named_subset is only admitted on ablation_lock configs")
    if split_for_year(2024) != "validation" or split_for_year(2025) != "test":
        raise RuntimeError("temporal split contract no longer maps 2024/2025 as frozen")


# Frozen scientific method + Stage-1 cheap runtime shared by Tag-11 ablations
# (issue #57). Channel count / names, Stage-5 loss, and the scaled cache budget
# are the only intentional deltas from stage1_locked.
_ABLATION_LOCK_EXPECTED: dict[str, object] = {
    "data.kind": "real",
    "data.mode": "stream",
    "data.max_patches_per_split": None,
    "data.shuffle_train": True,
    "data.batch_size": 4,
    "data.num_workers": 0,
    "data.pin_memory": False,
    "model.depth": 4,
    "model.base_width": 32,
    "trainer.learning_rate": 1.0e-3,
    "trainer.weight_decay": 0.0,
    "trainer.max_epochs": 20,
    "trainer.precision": "16-mixed",
    "seed": 0,
}

# Stage-1 used 12 GiB for C=10. Scale the floor linearly with channel count so
# larger ablation subsets cannot silently reuse an undersized job-local cache.
_STAGE1_CACHE_BYTES = 12_884_901_888
_STAGE1_CACHE_CHANNELS = 10


def _check_ablation_channels(
    problems: list[str],
    cfg: DictConfig,
    *,
    label: str,
    expected_names: list[str],
    expected_loss: str,
) -> None:
    selection = selection_from_config(cfg)
    if list(selection.names) != expected_names:
        problems.append(
            f"active channels for {label} must be {expected_names}; "
            f"got {list(selection.names)}"
        )
    actual_loss = OmegaConf.select(cfg, "contract.loss")
    if actual_loss != expected_loss:
        problems.append(
            f"contract.loss={actual_loss!r} (expected {expected_loss!r} for {label})"
        )
    cache_bytes = OmegaConf.select(cfg, "data.cache_max_bytes")
    min_cache = int(_STAGE1_CACHE_BYTES * (selection.n_active / _STAGE1_CACHE_CHANNELS))
    if (
        isinstance(cache_bytes, bool)
        or not isinstance(cache_bytes, int)
        or cache_bytes < min_cache
    ):
        problems.append(
            f"data.cache_max_bytes={cache_bytes!r} "
            f"(expected an int >= {min_cache} for C={selection.n_active})"
        )


def assert_ablation_lock(cfg: DictConfig) -> None:
    """Fail closed unless the config is a Tag-11 ablation under the Stage-1 lock."""
    problems: list[str] = []
    if cfg.get("ablation_lock") is not True:
        problems.append("ablation_lock marker is not true")
    stage = cfg.get("ablation_stage")
    isolation = cfg.get("ablation_isolation")
    stage_ok = (
        not isinstance(stage, bool) and isinstance(stage, int) and stage in (2, 3, 4, 5)
    )
    isolation_ok = isinstance(isolation, str) and isolation in ABLATION_ISOLATION_CHANNELS
    if stage_ok and isolation_ok:
        problems.append("ablation_stage and ablation_isolation are mutually exclusive")
    elif stage_ok:
        if isolation is not None:
            problems.append(
                f"ablation_isolation={isolation!r} (expected null on a cumulative stage)"
            )
    elif isolation_ok:
        if stage is not None:
            problems.append(
                f"ablation_stage={stage!r} (expected null on an isolation run)"
            )
    else:
        problems.append(
            f"ablation_stage={stage!r} ablation_isolation={isolation!r} "
            "(expected stage in {2, 3, 4, 5} or isolation in "
            f"{sorted(ABLATION_ISOLATION_CHANNELS)})"
        )
    if cfg.get("stage1_residual_prior") is not True:
        problems.append("ablation configs require the frozen residual prior")
    for marker in ("stage1_lock", "stage1_full", "stage1_probe"):
        if bool(cfg.get(marker, False)):
            problems.append(f"ablation config must not carry {marker}")
    for key, expected in _ABLATION_LOCK_EXPECTED.items():
        actual = OmegaConf.select(cfg, key)
        if key == "trainer.precision" and actual == "32-true":
            continue
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")
    if list(cfg.data.get("splits") or []) != ["train", "validation"]:
        problems.append("ablation data.splits must be exactly ['train', 'validation']")
    if stage_ok and not isolation_ok and isinstance(stage, int):
        _check_ablation_channels(
            problems,
            cfg,
            label=f"stage {stage}",
            expected_names=list(ABLATION_STAGE_CHANNELS[stage]),
            expected_loss="thermal_aware" if stage == 5 else "masked_l1",
        )
    elif isolation_ok and not stage_ok:
        _check_ablation_channels(
            problems,
            cfg,
            label=f"isolation {isolation}",
            expected_names=list(ABLATION_ISOLATION_CHANNELS[str(isolation)]),
            expected_loss="masked_l1",
        )
    # Operational bounds shared with the Stage-1 full Vertex path: one GPU,
    # W&B online, job-local output root. Checked here so the launcher and the
    # worker refuse a drifted ablation submit without a separate marker.
    if str(OmegaConf.select(cfg, "trainer.accelerator")) != "gpu":
        problems.append(
            f"trainer.accelerator={OmegaConf.select(cfg, 'trainer.accelerator')!r} "
            "(expected 'gpu')"
        )
    devices = OmegaConf.select(cfg, "trainer.devices")
    if isinstance(devices, bool) or not isinstance(devices, int) or devices != 1:
        problems.append(f"trainer.devices={devices!r} (expected 1)")
    if str(OmegaConf.select(cfg, "wandb.mode")) != "online":
        problems.append(
            f"wandb.mode={OmegaConf.select(cfg, 'wandb.mode')!r} (expected 'online')"
        )
    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append(
            f"output_root={output_root!r} (expected a job-local path under data/runs/)"
        )
    if problems:
        raise ValueError("ablation lock is off-contract: " + "; ".join(problems))


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
        problems.append(f"data.max_patches_per_split={bound!r} (expected an int in [1, 4])")

    epochs = cfg.trainer.get("max_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 1:
        problems.append(f"trainer.max_epochs={epochs!r} (expected 1)")

    if str(cfg.trainer.get("accelerator")) != "gpu":
        problems.append(f"trainer.accelerator={cfg.trainer.get('accelerator')!r} (expected 'gpu')")

    devices = cfg.trainer.get("devices")
    if isinstance(devices, bool) or not isinstance(devices, int) or devices != 1:
        problems.append(f"trainer.devices={devices!r} (expected 1)")

    if str(cfg.wandb.get("mode")) != "online":
        problems.append(f"wandb.mode={cfg.wandb.get('mode')!r} (expected 'online')")

    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://"):
        problems.append(f"output_root={output_root!r} (expected a non-empty local path)")

    if problems:
        raise ValueError("vertex smoke config is out of bounds: " + "; ".join(problems))


# The selected config name that carries the frozen Stage-1 experiment contract.
STAGE1_LOCKED_CONFIG_NAME = "stage1_locked"

# The full Stage-1 fit reads train/validation only; the 2025 test split is
# scored once, after the selected checkpoint is frozen (issue #53).
_STAGE1_FULL_SPLITS = ("train", "validation")

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
        problems.append(f"data.scene_ids={scene_ids!r} (expected empty = all published scenes)")
    if list(cfg.data.get("splits") or []) != list(_STAGE1_FULL_SPLITS):
        problems.append(
            f"data.splits={cfg.data.get('splits')!r} (expected {list(_STAGE1_FULL_SPLITS)!r}; "
            "the fit must not admit the 2025 test split)"
        )
    if problems:
        raise ValueError("stage1_locked config is off-contract: " + "; ".join(problems))


def assert_stage1_full_bounds(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config is the full Stage-1 Vertex run.

    Guards the operational bounds the full run shares with the smoke/probe: one
    GPU, W&B online, and a job-local output root. The locked method and the
    train/validation-only admission are asserted separately by
    :func:`assert_stage1_lock`. Called by the Vertex launcher before submitting
    and by the worker before ``RunLogSession`` opens the output path.
    """
    problems: list[str] = []
    if cfg.get("stage1_full") is not True:
        problems.append("stage1_full is not true (the full-run marker is missing)")
    if str(cfg.trainer.get("accelerator")) != "gpu":
        problems.append(f"trainer.accelerator={cfg.trainer.get('accelerator')!r} (expected 'gpu')")
    devices = cfg.trainer.get("devices")
    if isinstance(devices, bool) or not isinstance(devices, int) or devices != 1:
        problems.append(f"trainer.devices={devices!r} (expected 1)")
    if str(cfg.wandb.get("mode")) != "online":
        problems.append(f"wandb.mode={cfg.wandb.get('mode')!r} (expected 'online')")
    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append(f"output_root={output_root!r} (expected a job-local path under data/runs/)")
    if problems:
        raise ValueError("stage1 full config is out of bounds: " + "; ".join(problems))


# The selected config names that carry the bounded Stage-1 recovery probe
# (issue #47). Both share one frozen method; only the learning rate differs.
STAGE1_PROBE_CONFIG_NAME = "stage1_probe"
STAGE1_PROBE_LR3_CONFIG_NAME = "stage1_probe_lr3"
STAGE1_EFFICIENCY_CONFIG_NAME = "stage1_efficiency"
STAGE1_VERIFICATION_CONFIG_NAME = "stage1_verification"

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
    "trainer.max_epochs": 6,
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

_STAGE1_EFFICIENCY_FROZEN: dict[str, object] = {
    "data.kind": "real",
    "data.mode": "stream",
    "data.n_active_channels": 10,
    "data.shuffle_train": True,
    "model.depth": 4,
    "model.base_width": 32,
    "trainer.learning_rate": 1.0e-3,
    "trainer.weight_decay": 0.0,
    "trainer.accelerator": "gpu",
    "trainer.devices": 1,
    "seed": 0,
    "stage1_residual_prior": True,
}

_STAGE1_EFFICIENCY_ROLES = {"baseline", "cache", "diagnostic", "final"}


def assert_stage1_efficiency_fit(cfg: DictConfig) -> None:
    """Guard the matched six-epoch fit with only declared operational retunes."""
    problems: list[str] = []
    if (
        cfg.get("stage1_efficiency") is not True
        or cfg.get("stage1_efficiency_fit") is not True
        or cfg.get("stage1_probe") is not True
    ):
        problems.append("efficiency fit requires both efficiency and probe markers")
    if cfg.get("stage1_full") is not False or cfg.get("stage1_lock") is not False:
        problems.append("efficiency fit must not carry full-run or full-lock markers")
    for key, expected in _STAGE1_EFFICIENCY_FROZEN.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")
    if cfg.trainer.get("max_epochs") != 6:
        problems.append("trainer.max_epochs must remain the recovery probe's six epochs")
    if list(cfg.data.get("splits") or []) != ["train", "validation"]:
        problems.append("efficiency fit splits must be train/validation only")
    if cfg.data.get("scene_ids") not in ([], None):
        problems.append(
            "efficiency fit must use all published scenes selected by the recovery cohort"
        )
    probe = cfg.data.get("probe")
    if probe is None:
        problems.append("efficiency fit requires the recovery cohort")
    else:
        expected = {
            "max_refs_per_split": {"train": 128, "validation": 64},
            "max_refs_per_scene": 8,
            "min_admitted_per_split": {"train": 120, "validation": 60},
            "min_scenes_per_split": {"train": 16, "validation": 8},
            "min_train_years": 3,
        }
        for key, value in expected.items():
            actual_node = probe.get(key)
            actual = (
                OmegaConf.to_container(actual_node, resolve=True)
                if isinstance(actual_node, (DictConfig, ListConfig))
                else actual_node
            )
            if actual != value:
                problems.append(f"data.probe.{key}={actual!r} (expected {value!r})")
    if str(cfg.trainer.get("precision")) not in ("32-true", "16-mixed"):
        problems.append("efficiency fit precision must be 32-true or 16-mixed")
    if cfg.data.get("num_workers") not in (0, 2, 4):
        problems.append("efficiency fit workers must be 0, 2, or 4")
    if not isinstance(cfg.data.get("pin_memory", False), bool):
        problems.append("efficiency fit pin_memory must be boolean")
    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append("efficiency fit output_root must be job-local under data/runs/")
    if str(cfg.wandb.get("mode")) != "online":
        problems.append("efficiency fit requires online W&B")
    if problems:
        raise ValueError("stage1 efficiency fit is off-contract: " + "; ".join(problems))


def assert_efficiency_fit_runtime() -> None:
    """Require the launcher-authorized J1/J4 worker context for a learning fit."""
    role = os.environ.get("VERTEX_EFFICIENCY_ROLE")
    slot = os.environ.get("VERTEX_EFFICIENCY_SLOT")
    if (
        os.environ.get("VERTEX_PROFILE") != "efficiency"
        or os.environ.get("VERTEX_EFFICIENCY_SESSION") != EFFICIENCY_SESSION_ID
        or (role, slot) not in (("baseline", "1"), ("final", "4"))
    ):
        raise ValueError("efficiency learning fit requires a reserved J1/J4 Vertex worker context")


def assert_stage1_efficiency(cfg: DictConfig) -> None:
    """Fail closed unless the config is a bounded, train/validation-only profile."""
    problems: list[str] = []
    if cfg.get("stage1_efficiency") is not True:
        problems.append("stage1_efficiency marker is not true")
    if cfg.get("stage1_full") is not False or cfg.get("stage1_lock") is not False:
        problems.append("efficiency config must not carry full-run or full-lock markers")
    if bool(cfg.get("stage1_probe", False)):
        problems.append("efficiency config must not carry the learning-probe marker")
    for key, expected in _STAGE1_EFFICIENCY_FROZEN.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")

    if list(cfg.data.get("splits") or []) != ["train", "validation"]:
        problems.append("efficiency data.splits must be exactly ['train', 'validation']")
    limit = cfg.data.get("max_patches_per_split")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 512:
        problems.append(f"data.max_patches_per_split={limit!r} (expected integer in [1, 512])")
    role = str(cfg.get("stage1_efficiency_role", ""))
    if role not in _STAGE1_EFFICIENCY_ROLES:
        problems.append(f"stage1_efficiency_role={role!r} is not an allowed bounded role")
    precision = str(cfg.trainer.get("precision", "32-true"))
    if precision not in ("32-true", "16-mixed"):
        problems.append(f"trainer.precision={precision!r} is not an efficiency candidate")
    workers = cfg.data.get("num_workers")
    if isinstance(workers, bool) or workers not in (0, 2, 4):
        problems.append(f"data.num_workers={workers!r} is not an allowed measured candidate")
    if not isinstance(cfg.data.get("pin_memory", False), bool):
        problems.append("data.pin_memory must be a boolean")
    data_budget = cfg.data.get("cache_max_bytes", 0)
    legacy_budget = cfg.get("stage1_efficiency_cache_max_bytes", 0)
    cache_budget = data_budget if data_budget else legacy_budget
    if isinstance(cache_budget, bool) or not isinstance(cache_budget, int) or cache_budget <= 0:
        problems.append(
            "cache budget must be a positive integer "
            "(data.cache_max_bytes, legacy stage1_efficiency_cache_max_bytes)"
        )

    probe = cfg.data.get("probe")
    if probe is None:
        problems.append("bounded scene-spread data.probe selection is required")
    else:
        refs = dict(probe.get("max_refs_per_split") or {})
        if refs != {"train": 384, "validation": 128}:
            problems.append(
                f"data.probe.max_refs_per_split={refs!r} is not the preregistered cohort"
            )
        if int(probe.get("max_refs_per_scene", 0)) != 16:
            problems.append("data.probe.max_refs_per_scene must be 16")
        if int(probe.get("min_train_years", 0)) < 3:
            problems.append("data.probe.min_train_years must be at least 3")
        if probe.get("require_partial_masks") is not True:
            problems.append("data.probe.require_partial_masks must be true")
        minima = dict(probe.get("min_admitted_per_split") or {})
        if minima != {"train": 384, "validation": 128}:
            problems.append(f"data.probe.min_admitted_per_split={minima!r} is not preregistered")

    epochs = cfg.trainer.get("max_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 1:
        problems.append(
            f"trainer.max_epochs={epochs!r} (efficiency role must not fit multiple epochs)"
        )
    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append("output_root must be a job-local path under data/runs/")
    if str(cfg.wandb.get("mode")) != "online":
        problems.append("wandb.mode must be online for a Vertex efficiency job")
    if problems:
        raise ValueError("stage1 efficiency config is off-contract: " + "; ".join(problems))


def assert_stage1_efficiency_scope(cfg: DictConfig, scope: dict) -> None:
    """Require the fixed admitted cohort to include years and partial masks."""
    probe = cfg.data.probe
    admitted_raw = scope.get("patches_per_split")
    years_raw = scope.get("years_per_split")
    partial_raw = scope.get("partial_mask_patches_per_split")
    admitted = admitted_raw if isinstance(admitted_raw, dict) else {}
    years = years_raw if isinstance(years_raw, dict) else {}
    partial = partial_raw if isinstance(partial_raw, dict) else {}
    minima = dict(probe.min_admitted_per_split)
    problems: list[str] = []
    if set(admitted) != {"train", "validation"}:
        problems.append(f"admitted splits {sorted(admitted)} are not train/validation only")
    for split in ("train", "validation"):
        if int(admitted.get(split, 0)) < int(minima[split]):
            problems.append(f"{split} admitted {admitted.get(split, 0)} < required {minima[split]}")
        if int(partial.get(split, 0)) <= 0:
            problems.append(f"{split} cohort has no partially eligible mask")
    required_years = int(probe.min_train_years)
    if len(years.get("train", [])) < required_years:
        problems.append(
            f"train covers {len(years.get('train', []))} years < required {required_years}"
        )
    if problems:
        raise ValueError("stage1 efficiency cohort is out of bounds: " + "; ".join(problems))


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
        problems.append(f"output_root={output_root!r} (expected a job-local path under data/runs/)")

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


# Verified full-training runtime (docs/stage1-training-readiness.md). Operational
# retunes only — the frozen method in ``_STAGE1_LOCK_EXPECTED`` is untouched.
# Operational retunes frozen from J1–J4; this guard resolves offline and in the
# readiness validator, so the later execution plan cannot silently drift.
_STAGE1_RUNTIME_EXPECTED: dict[str, object] = {
    "data.batch_size": 4,
    "data.mode": "stream",
    "data.num_workers": 0,
    "data.pin_memory": False,
    "data.cache_max_bytes": 12884901888,
    "trainer.precision": "16-mixed",
}


def assert_stage1_runtime(cfg: DictConfig) -> None:
    """Fail closed unless the resolved config carries the verified runtime.

    Checks only the operational retunes the readiness plan froze; the frozen
    scientific method is asserted separately by :func:`assert_stage1_lock`.
    The ``32-true`` fallback is a documented launcher override, not a second
    runtime: the freeze receipt records which precision the verification
    selected.
    """
    problems: list[str] = []
    for key, expected in _STAGE1_RUNTIME_EXPECTED.items():
        actual = OmegaConf.select(cfg, key)
        if key == "trainer.precision" and actual == "32-true":
            continue
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected!r})")
    if problems:
        raise ValueError("stage1 runtime is off-contract: " + "; ".join(problems))


def assert_stage1_verification(cfg: DictConfig) -> None:
    """Fail closed unless the config is the single bounded verification profile.

    Identity/scope carrier for the verification harness: train/validation
    only, bounded outer cohort, GPU, online W&B, job-local output. The harness
    freezes smaller probe scopes inside this bound; this guard only holds the
    outer walls.
    """
    problems: list[str] = []
    if cfg.get("stage1_verification") is not True:
        problems.append("stage1_verification marker is not true")
    for marker in ("stage1_full", "stage1_lock", "stage1_probe", "stage1_efficiency"):
        if bool(cfg.get(marker, False)):
            problems.append(f"verification config must not carry {marker}")
    # The verification inherits the locked science; re-assert it here so drift
    # in stage1_locked.yaml cannot pass the verification guard unnoticed.
    for key, expected in {
        "data.n_active_channels": 10,
        "model.depth": 4,
        "model.base_width": 32,
        "trainer.learning_rate": 1.0e-3,
        "trainer.weight_decay": 0.0,
        "seed": 0,
    }.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected frozen {expected!r})")
    if cfg.get("stage1_residual_prior") is not True:
        problems.append("verification requires the frozen residual prior")
    if list(cfg.data.get("splits") or []) != ["train", "validation"]:
        problems.append("verification data.splits must be exactly ['train', 'validation']")
    bound = cfg.data.get("max_patches_per_split")
    if isinstance(bound, bool) or not isinstance(bound, int) or bound != 512:
        problems.append(f"data.max_patches_per_split={bound!r} (expected 512)")
    if str(cfg.data.get("mode")) != "stream":
        problems.append("verification data.mode must be stream")
    if cfg.data.get("num_workers") != 0:
        problems.append("verification data.num_workers must be 0")
    if str(cfg.trainer.get("precision", "16-mixed")) not in ("32-true", "16-mixed"):
        problems.append("verification trainer.precision must be 32-true or 16-mixed")
    if str(cfg.trainer.get("accelerator")) != "gpu":
        problems.append("verification trainer.accelerator must be gpu")
    output_root = str(cfg.get("output_root", ""))
    if not output_root or output_root.startswith("gs://") or "/runs/" not in output_root:
        problems.append("verification output_root must be job-local under data/runs/")
    if str(cfg.wandb.get("mode")) != "online":
        problems.append("verification wandb.mode must be online")
    if problems:
        raise ValueError("stage1 verification config is off-contract: " + "; ".join(problems))


def guard_modeling_config(cfg: DictConfig, config_name: str | None) -> None:
    """Run the config-identity guards before any run directory or source read.

    The runner calls this before ``RunLogSession`` opens the output path, so a
    drifted invocation cannot create a run directory or begin a paid GPU run.
    Re-invoked by :func:`run_modeling` for programmatic callers; the guards are
    pure assertions and idempotent.

    The bounded efficiency and verification jobs are closed. Their retained
    results live in ``docs/stage1-efficiency.md``; the modeling runner admits
    smoke, recovery probes, and the full Stage-1 path (issue #53).
    """
    if bool(cfg.get("stage1_probe", False)) and config_name not in (
        STAGE1_PROBE_CONFIG_NAME,
        STAGE1_PROBE_LR3_CONFIG_NAME,
    ):
        raise ValueError(
            "stage1_probe marker is set on a non-probe config "
            f"({config_name!r}); the probe guard would not run"
        )
    if bool(cfg.get("stage1_efficiency", False)) or bool(cfg.get("stage1_efficiency_fit", False)):
        raise ValueError(
            "Stage-1 efficiency profiles are closed; retained results are in "
            "docs/stage1-efficiency.md"
        )
    if bool(cfg.get("stage1_verification", False)):
        raise ValueError(
            "Stage-1 verification profile is cancelled; runtime is frozen from "
            "the J1–J4 efficiency results in docs/stage1-efficiency.md"
        )
    _residual_prior_configs = {
        STAGE1_LOCKED_CONFIG_NAME,
        STAGE1_PROBE_CONFIG_NAME,
        STAGE1_PROBE_LR3_CONFIG_NAME,
        *ABLATION_LOCKED_CONFIG_NAMES,
    }
    if bool(cfg.get("stage1_residual_prior", False)) and config_name not in _residual_prior_configs:
        raise ValueError(
            "stage1_residual_prior is the Stage-1 recovery representation "
            f"(issues #40/#47) but is set on config {config_name!r}; refusing to run"
        )
    if bool(cfg.get("stage1_full", False)) and config_name != STAGE1_LOCKED_CONFIG_NAME:
        raise ValueError(
            "stage1_full is the unbounded Stage-1 Vertex run marker "
            f"(issue #53) but is set on config {config_name!r}; refusing to run"
        )
    if bool(cfg.get("ablation_lock", False)) and config_name not in ABLATION_LOCKED_CONFIG_NAMES:
        raise ValueError(
            "ablation_lock is the Tag-11 ablation marker (issue #57) but is set on "
            f"config {config_name!r}; refusing to run"
        )
    if config_name == STAGE1_LOCKED_CONFIG_NAME:
        assert_stage1_lock(cfg)
        assert_stage1_full_bounds(cfg)
        assert_stage1_runtime(cfg)
    if config_name == STAGE1_PROBE_CONFIG_NAME:
        assert_stage1_probe(cfg)
    if config_name == STAGE1_PROBE_LR3_CONFIG_NAME:
        assert_stage1_probe_lr3(cfg)
    if config_name in ABLATION_LOCKED_CONFIG_NAMES:
        assert_ablation_lock(cfg)
    if bool(cfg.get("vertex_smoke_bounds", False)):
        assert_vertex_smoke_bounds(cfg)


__all__ = [
    "assert_ablation_lock",
    "assert_efficiency_fit_runtime",
    "assert_probe_minima",
    "assert_stage1_efficiency",
    "assert_stage1_efficiency_fit",
    "assert_stage1_efficiency_scope",
    "assert_stage1_full_bounds",
    "assert_stage1_lock",
    "assert_stage1_probe",
    "assert_stage1_probe_lr3",
    "assert_stage1_runtime",
    "assert_stage1_verification",
    "STAGE1_EFFICIENCY_CONFIG_NAME",
    "STAGE1_VERIFICATION_CONFIG_NAME",
    "assert_vertex_smoke_bounds",
    "contract_invariants",
    "guard_modeling_config",
]
