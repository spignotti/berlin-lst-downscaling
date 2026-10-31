"""v6 ablation lock: arm registry, configs, guards, run identity, matrix, plan helper."""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling.channels import (
    ABLATION_LOCKED_CONFIG_NAMES,
    V6_ABLATION_ARMS,
    V6_ABLATION_LOCKED_CONFIG_NAMES,
    V6_FAMILIES,
    feature_order_for,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.guards import (
    assert_ablation_lock,
    contract_invariants,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.run_identity import resolve_run_identity
from berlin_lst_downscaling.modeling.v6_ablation_matrix import (
    load_v6_matrix,
    parse_v6_matrix,
    plan_runs,
    plan_totals,
    run_hours_estimate,
)

_ROOT = Path(__file__).resolve().parents[1]
_MATRIX = _ROOT / "configs" / "ablation" / "v6_matrix.yaml"
_COUNTS = {
    "L1": 10,
    "L2": 16,
    "L3": 18,
    "L4": 22,
    "L5": 22,
    "LOO-M": 16,
    "LOO-S": 20,
    "NO-NDVI-NDWI": 20,
    "NO-RED-NIR-IDX": 16,
}
_ARMS = list(V6_ABLATION_ARMS.values())


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_ROOT / "configs" / "modeling"), version_base=None):

        def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
            return compose(config_name=name, overrides=overrides or [])

        yield _compose
    GlobalHydra.instance().clear()


def _mutable(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _check(cfg: DictConfig, name: str) -> None:
    guard_modeling_config(cfg, name)
    contract_invariants(cfg)
    assert_ablation_lock(cfg)


def test_families_partition_the_v6_roster() -> None:
    flat = [n for family in V6_FAMILIES.values() for n in family]
    assert len(flat) == 22
    assert len(set(flat)) == 22


def test_arm_registry_counts_and_losses() -> None:
    assert {a: len(arm.channels) for a, arm in V6_ABLATION_ARMS.items()} == _COUNTS
    assert {a for a, arm in V6_ABLATION_ARMS.items() if arm.loss == "thermal_aware"} == {"L5"}
    assert V6_ABLATION_LOCKED_CONFIG_NAMES == tuple(a.config_name for a in _ARMS)
    assert not set(V6_ABLATION_LOCKED_CONFIG_NAMES) & set(ABLATION_LOCKED_CONFIG_NAMES)


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: a.arm_id)
def test_arm_config_composes_and_passes(compose_modeling: Callable[..., DictConfig], arm) -> None:
    cfg = compose_modeling(arm.config_name)
    _check(cfg, arm.config_name)
    selection = selection_from_config(cfg)
    assert selection.names == arm.channels
    expected = "v6_registry" if arm.arm_id in ("L4", "L5") else "v6_named_subset"
    assert feature_order_for(selection, 5) == expected
    assert cfg.contract.feature_order == expected
    assert cfg.output_root.endswith("/seed-0")


@pytest.mark.parametrize(
    "overrides", [["seed=1"], ["seed=2"], ["trainer.max_epochs=2"], ["trainer.max_epochs=40"]]
)
def test_parameter_overrides_pass(
    compose_modeling: Callable[..., DictConfig], overrides: list[str]
) -> None:
    cfg = compose_modeling("v6_l4_locked", overrides)
    _check(cfg, "v6_l4_locked")
    if overrides[0] == "seed=1":
        assert cfg.output_root.endswith("/seed-1")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["seed=-1"], "seed"),
        (["seed=true"], "seed"),
        (["trainer.max_epochs=0"], "max_epochs"),
        (["data.batch_size=8"], "batch_size"),
        (["trainer.learning_rate=0.003"], "learning_rate"),
        (["model.depth=3"], "model.depth"),
        (["data_release_id=training/v1"], "data_release_id"),
        (["patch_index_root=gs://x/y"], "patch_index_root"),
        (["training_root=gs://x/y"], "training_root"),
        (["features_root=gs://x/y"], "features_root"),
        (["aggregation=t4"], "aggregation"),
        (["data.splits=[train,validation,test]"], "splits"),
        (["data.cache_max_bytes=1000"], "cache_max_bytes"),
        (["contract.loss=thermal_aware"], "contract.loss"),
        (["output_root=data/runs/x/shared"], "output_root"),
    ],
)
def test_drift_is_rejected(
    compose_modeling: Callable[..., DictConfig], overrides: list[str], match: str
) -> None:
    cfg = compose_modeling("v6_l4_locked", overrides)
    with pytest.raises(ValueError, match=match):
        _check(cfg, "v6_l4_locked")


def test_wrong_channel_list_is_rejected(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = _mutable(compose_modeling("v6_l2_locked"))
    cfg.data.active_channel_names = cfg.data.active_channel_names[:-1]
    cfg.data.n_active_channels = 15
    with pytest.raises(ValueError):
        _check(cfg, "v6_l2_locked")


def test_out_of_registry_order_is_rejected(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = _mutable(compose_modeling("v6_l2_locked"))
    names = list(cfg.data.active_channel_names)
    names[0], names[1] = names[1], names[0]
    cfg.data.active_channel_names = names
    with pytest.raises(ValueError, match="registry order"):
        _check(cfg, "v6_l2_locked")


def test_thermal_loss_on_l4_is_rejected(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = _mutable(compose_modeling("v6_l4_locked"))
    cfg.contract.loss = "thermal_aware"
    with pytest.raises(ValueError, match="masked_l1"):
        assert_ablation_lock(cfg)


def test_seeds_get_distinct_output_roots_and_identities(
    compose_modeling: Callable[..., DictConfig], tmp_path: Path
) -> None:
    roots, seeds = set(), set()
    for seed in (0, 1, 2):
        cfg = _mutable(compose_modeling("v6_l4_locked", [f"seed={seed}"]))
        roots.add(cfg.output_root)
        cfg.training_root = None
        cfg.patch_index_root = None
        identity = resolve_run_identity(cfg, run_id="r", output_root=str(tmp_path))
        seeds.add(identity.seed)
    assert len(roots) == 3
    assert seeds == {0, 1, 2}


def test_base_is_not_runnable(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = compose_modeling("_v6_ablation_base")
    with pytest.raises(ValueError, match="refusing to run"):
        guard_modeling_config(cfg, "_v6_ablation_base")


@pytest.mark.parametrize("overrides", [["seed=1"], ["trainer.max_epochs=40"]])
def test_v1_lock_rejects_parameter_overrides(
    compose_modeling: Callable[..., DictConfig], overrides: list[str]
) -> None:
    cfg = compose_modeling("stage4_locked", overrides)
    with pytest.raises(ValueError):
        guard_modeling_config(cfg, "stage4_locked")


def test_v1_lock_rejects_release_marker(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = _mutable(compose_modeling("stage4_locked"))
    cfg.ablation_release = "v6"
    with pytest.raises(ValueError, match="ablation_release"):
        guard_modeling_config(cfg, "stage4_locked")


def test_v6_config_with_wrong_arm_is_rejected(
    compose_modeling: Callable[..., DictConfig],
) -> None:
    cfg = _mutable(compose_modeling("v6_l4_locked"))
    cfg.ablation_arm = "L5"
    with pytest.raises(ValueError, match="ablation_arm"):
        guard_modeling_config(cfg, "v6_l4_locked")


@pytest.mark.parametrize("name", ["v6", "tessera_arm"])
def test_non_lock_v6_profiles_still_pass(
    compose_modeling: Callable[..., DictConfig], name: str
) -> None:
    cfg = compose_modeling(name)
    guard_modeling_config(cfg, name)
    contract_invariants(cfg)


def test_non_lock_v6_proper_subset_is_rejected(
    compose_modeling: Callable[..., DictConfig],
) -> None:
    cfg = _mutable(compose_modeling("v6"))
    arm = V6_ABLATION_ARMS["L1"]
    cfg.data.n_active_channels = 10
    cfg.data.active_channel_names = list(arm.channels)
    cfg.contract.feature_order = "v6_named_subset"
    with pytest.raises(ValueError, match="only admitted on the v6 ablation lock"):
        contract_invariants(cfg)


# ── matrix and plan helper ───────────────────────────────────────────


def test_default_matrix_totals() -> None:
    matrix = load_v6_matrix(_MATRIX)
    planned = plan_runs(matrix)
    assert len(planned) == 27
    assert {a: sorted(r.seed for r in planned if r.arm_id == a) for a in V6_ABLATION_ARMS} == {
        a: [0, 1, 2] for a in V6_ABLATION_ARMS
    }
    runs, hours, usd = plan_totals(matrix)
    # 27 x (0.1 + 40 x 0.12)
    assert (runs, math.isclose(hours, 132.3), math.isclose(usd, 132.3)) == (27, True, True)


def test_epoch_aware_totals() -> None:
    raw = _raw()
    raw["max_epochs"] = 10
    raw["arms"]["L1"] = {"seeds": [0, 1]}
    raw["hours_per_epoch_estimate"] = 0.5
    raw["fixed_overhead_hours_estimate"] = 1.0
    matrix = parse_v6_matrix(raw)
    assert {run_hours_estimate(matrix, r) for r in plan_runs(matrix)} == {1.0 + 10 * 0.5}
    # 2 x L1 + 8 other arms x 1 seed, each 6.0 h
    runs, hours, usd = plan_totals(matrix)
    assert runs == 10
    assert math.isclose(hours, 10 * 6.0)
    assert math.isclose(usd, hours)


def _raw() -> dict:
    arms = {a: {"seeds": [0]} for a in V6_ABLATION_ARMS}
    return {
        "max_epochs": 20,
        "arms": arms,
        "hours_per_epoch_estimate": 0.12,
        "fixed_overhead_hours_estimate": 0.1,
        "usd_per_hour": 1.0,
    }


def test_valid_raw_matrix_parses() -> None:
    assert len(plan_runs(parse_v6_matrix(_raw()))) == 9


def test_matrix_lr_schedule_defaults_to_constant_and_reaches_runs() -> None:
    assert parse_v6_matrix(_raw()).lr_schedule == "constant"
    raw = _raw()
    raw["lr_schedule"] = "cosine"
    runs = plan_runs(parse_v6_matrix(raw))
    assert {r.lr_schedule for r in runs} == {"cosine"}


@pytest.mark.parametrize("value", ["linear", "", None, True])
def test_bad_matrix_lr_schedule_rejected(value: object) -> None:
    raw = _raw()
    raw["lr_schedule"] = value
    with pytest.raises(ValueError, match="lr_schedule"):
        parse_v6_matrix(raw)


def test_per_arm_lr_schedule_rejected() -> None:
    raw = _raw()
    raw["arms"]["L2"]["lr_schedule"] = "cosine"
    with pytest.raises(ValueError, match="matrix-level"):
        parse_v6_matrix(raw)


def test_shipped_matrix_is_40_epoch_cosine_prebuilt_without_early_stopping() -> None:
    matrix = load_v6_matrix(_MATRIX)
    planned = plan_runs(matrix)
    assert len(planned) == 27
    assert {(r.arm_id, r.seed) for r in planned} == {
        (a, s) for a in V6_ABLATION_ARMS for s in (0, 1, 2)
    }
    assert {(r.max_epochs, r.lr_schedule, r.early_stopping, r.patch_cache) for r in planned} == {
        (40, "cosine", False, "prebuilt")
    }


def test_shipped_matrix_has_no_per_arm_schema_keys() -> None:
    raw = OmegaConf.to_container(OmegaConf.load(_MATRIX))
    assert isinstance(raw, dict)
    assert {k: raw[k] for k in ("max_epochs", "lr_schedule", "early_stopping", "patch_cache")} == {
        "max_epochs": 40,
        "lr_schedule": "cosine",
        "early_stopping": False,
        "patch_cache": "prebuilt",
    }
    assert all(entry == {"seeds": [0, 1, 2]} for entry in raw["arms"].values())


def test_matrix_defaults_when_absent() -> None:
    matrix = parse_v6_matrix(_raw())
    assert (matrix.early_stopping, matrix.patch_cache) == (False, "job_local")
    assert {(r.early_stopping, r.patch_cache) for r in plan_runs(matrix)} == {(False, "job_local")}


def test_matrix_level_switches_reach_runs() -> None:
    raw = _raw()
    raw["early_stopping"] = True
    raw["patch_cache"] = "prebuilt"
    raw["max_epochs"] = 3
    raw["lr_schedule"] = "cosine"
    runs = plan_runs(parse_v6_matrix(raw))
    assert {(r.max_epochs, r.early_stopping, r.patch_cache) for r in runs} == {
        (3, True, "prebuilt")
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [("max_epochs", 40), ("early_stopping", False), ("patch_cache", "prebuilt")],
)
def test_per_arm_matrix_level_keys_rejected(key: str, value: object) -> None:
    raw = _raw()
    raw["arms"]["L3"][key] = value
    with pytest.raises(ValueError, match="matrix-level"):
        parse_v6_matrix(raw)


@pytest.mark.parametrize("value", [0, 1, "false", None, "no"])
def test_non_bool_early_stopping_rejected(value: object) -> None:
    raw = _raw()
    raw["early_stopping"] = value
    with pytest.raises(ValueError, match="early_stopping"):
        parse_v6_matrix(raw)


@pytest.mark.parametrize("value", ["gcs", "", None, True, "PREBUILT"])
def test_unknown_patch_cache_rejected(value: object) -> None:
    raw = _raw()
    raw["patch_cache"] = value
    with pytest.raises(ValueError, match="patch_cache"):
        parse_v6_matrix(raw)


@pytest.mark.parametrize("epochs", [1, 2])
def test_cosine_needs_epochs_after_warmup(epochs: int) -> None:
    raw = _raw()
    raw["lr_schedule"] = "cosine"
    raw["max_epochs"] = epochs
    with pytest.raises(ValueError, match="warmup"):
        parse_v6_matrix(raw)
    raw["lr_schedule"] = "constant"
    assert parse_v6_matrix(raw).max_epochs == epochs


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "40", None])
def test_bad_matrix_max_epochs_rejected(value: object) -> None:
    raw = _raw()
    raw["max_epochs"] = value
    with pytest.raises(ValueError, match="max_epochs"):
        parse_v6_matrix(raw)


def test_missing_matrix_max_epochs_rejected() -> None:
    raw = _raw()
    del raw["max_epochs"]
    with pytest.raises(ValueError, match="max_epochs"):
        parse_v6_matrix(raw)


def test_shipped_runs_compose_to_the_planned_schema(
    compose_modeling: Callable[..., DictConfig],
) -> None:
    # Same overrides as validate_ablation_configs._check_v6_matrix (that session runs it too).
    for run in plan_runs(load_v6_matrix(_MATRIX)):
        cfg = compose_modeling(
            run.config_name,
            [
                f"seed={run.seed}",
                f"trainer.max_epochs={run.max_epochs}",
                f"trainer.lr_schedule.name={run.lr_schedule}",
                f"early_stopping.enabled={str(run.early_stopping).lower()}",
                "+data.patch_cache.source=prebuilt",
            ],
        )
        _check(cfg, run.config_name)
        assert cfg.trainer.max_epochs == 40
        assert cfg.trainer.lr_schedule.name == "cosine"
        assert cfg.trainer.lr_schedule.warmup_epochs == 2
        assert cfg.early_stopping.enabled is False
        assert cfg.data.patch_cache.source == "prebuilt"


def test_plan_arm_modes_match_the_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _load_script("validators", "plan_v6_ablation")
    if importlib.util.find_spec("google.cloud.aiplatform_v1") is None:
        stub = MagicMock()
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1", stub)
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1.types", stub.types)
    launcher = _load_script("operators", "launch_vertex_modeling")
    assert set(plan.ARM_LAUNCHER_MODE) == set(V6_ABLATION_ARMS)
    assert sorted(plan.ARM_LAUNCHER_MODE.values()) == sorted(launcher._V6_ABLATION_MODES)
    for arm_id, mode in plan.ARM_LAUNCHER_MODE.items():
        assert launcher.MODE_CONFIG_NAME[mode] == V6_ABLATION_ARMS[arm_id].config_name


def test_missing_arm_rejected() -> None:
    raw = _raw()
    del raw["arms"]["L1"]
    with pytest.raises(ValueError, match="missing"):
        parse_v6_matrix(raw)


def test_extra_arm_rejected() -> None:
    raw = _raw()
    raw["arms"]["LOO-ERA5"] = {"seeds": [0]}
    with pytest.raises(ValueError, match="extra"):
        parse_v6_matrix(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", [0, 0]),
        ("seeds", [-1]),
        ("seeds", [True]),
        ("seeds", []),
    ],
)
def test_malformed_arm_entry_rejected(field: str, value: object) -> None:
    raw = _raw()
    raw["arms"]["L4"][field] = value
    with pytest.raises(ValueError):
        parse_v6_matrix(raw)


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan"), "2"])
def test_bad_estimates_rejected(value: object) -> None:
    raw = _raw()
    raw["usd_per_hour"] = value
    with pytest.raises(ValueError):
        parse_v6_matrix(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fixed_overhead_hours_estimate", -0.1),
        ("fixed_overhead_hours_estimate", float("nan")),
        ("hours_per_epoch_estimate", 0),
    ],
)
def test_bad_epoch_estimates_rejected(key: str, value: float) -> None:
    raw = _raw()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        parse_v6_matrix(raw)


@pytest.mark.parametrize("key", ["hours_per_epoch_estimate", "fixed_overhead_hours_estimate"])
def test_missing_epoch_estimate_key_rejected(key: str) -> None:
    raw = _raw()
    del raw[key]
    with pytest.raises(ValueError, match=key):
        parse_v6_matrix(raw)


def test_zero_overhead_allowed() -> None:
    raw = _raw()
    raw["fixed_overhead_hours_estimate"] = 0
    assert parse_v6_matrix(raw).fixed_overhead_hours_estimate == 0.0


def _load_script(folder: str, name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", _ROOT / "scripts" / folder / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_helper_lists_27_runs(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script("validators", "plan_v6_ablation")
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert (
        "TOTAL runs=27 est_gpu_hours=132.3 est_usd=132.3 (placeholder estimates: "
        "hours_per_epoch=0.12 fixed_overhead=0.10; nothing submitted)"
    ) in out
    run_lines = [line for line in out.splitlines() if line.startswith("arm=")]
    assert len(run_lines) == 27
    for line in run_lines:
        assert (
            "epochs=40 lr_schedule=cosine early_stopping=false patch_cache=prebuilt "
            "est_hours=4.90 launcher_args: --mode v6-"
        ) in line
        assert line.endswith("--max-epochs 40 --lr-schedule cosine --patch-cache prebuilt")
    assert "--skip-test" not in out
    assert "test" not in out.replace("est_", "")
    assert (
        "arm=LOO-S config=v6_loo_shadow_locked seed=2 epochs=40 lr_schedule=cosine "
        "early_stopping=false patch_cache=prebuilt est_hours=4.90 launcher_args: "
        "--mode v6-loo-shadow --seed 2 --max-epochs 40 --lr-schedule cosine "
        "--patch-cache prebuilt"
    ) in out


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_cache_build_workers_accepted_without_changing_the_lock(
    compose_modeling: Callable[..., DictConfig], workers: int
) -> None:
    cfg = _mutable(compose_modeling("v6_l4_locked"))
    cfg.data.cache_build_workers = workers
    _check(cfg, "v6_l4_locked")
    assert cfg.data.num_workers == 0


@pytest.mark.parametrize("bad", [True, 2.0, "4", None, 0, 5, -1])
def test_cache_build_workers_malformed_rejected(
    compose_modeling: Callable[..., DictConfig], bad: object
) -> None:
    cfg = _mutable(compose_modeling("v6_l4_locked"))
    cfg.data.cache_build_workers = bad
    with pytest.raises(ValueError, match="cache_build_workers"):
        guard_modeling_config(cfg, "v6_l4_locked")


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: a.arm_id)
def test_all_arms_keep_zero_loader_workers_and_parallel_cache_build(
    compose_modeling: Callable[..., DictConfig], arm: object
) -> None:
    name = arm.config_name  # type: ignore[attr-defined]
    cfg = compose_modeling(name)
    assert cfg.data.num_workers == 0
    assert cfg.data.cache_build_workers == 4
    _check(cfg, name)


def test_nonzero_loader_workers_override_is_rejected(
    compose_modeling: Callable[..., DictConfig],
) -> None:
    cfg = _mutable(compose_modeling("v6_l4_locked"))
    cfg.data.num_workers = 2
    with pytest.raises(ValueError):
        _check(cfg, "v6_l4_locked")
