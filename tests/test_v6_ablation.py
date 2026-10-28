"""v6 ablation lock: arm registry, configs, guards, run identity, matrix, plan helper."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path

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
    assert len(plan_runs(matrix)) == 17
    runs, hours, usd = plan_totals(matrix)
    assert (runs, math.isclose(hours, 42.5), math.isclose(usd, 42.5)) == (17, True, True)


def _raw() -> dict:
    arms = {a: {"seeds": [0], "max_epochs": 20} for a in V6_ABLATION_ARMS}
    return {"arms": arms, "hours_per_run_estimate": 2.5, "usd_per_hour": 1.0}


def test_valid_raw_matrix_parses() -> None:
    assert len(plan_runs(parse_v6_matrix(_raw()))) == 9


def test_missing_arm_rejected() -> None:
    raw = _raw()
    del raw["arms"]["L1"]
    with pytest.raises(ValueError, match="missing"):
        parse_v6_matrix(raw)


def test_extra_arm_rejected() -> None:
    raw = _raw()
    raw["arms"]["LOO-ERA5"] = {"seeds": [0], "max_epochs": 20}
    with pytest.raises(ValueError, match="extra"):
        parse_v6_matrix(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", [0, 0]),
        ("seeds", [-1]),
        ("seeds", [True]),
        ("seeds", []),
        ("max_epochs", 0),
        ("max_epochs", True),
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


def test_plan_helper_lists_17_runs(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "plan_v6_ablation", _ROOT / "scripts" / "validators" / "plan_v6_ablation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert "TOTAL runs=17 est_gpu_hours=42.5 est_usd=42.5" in out
    assert out.count("arm=") == 17
