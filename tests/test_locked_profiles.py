"""Locked modeling profiles fail closed when a frozen value drifts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.training.contracts import SPLIT_BY_YEAR
from berlin_lst_downscaling.modeling.guards import (
    assert_ablation_lock,
    assert_stage1_lock,
    contract_invariants,
    guard_modeling_config,
)

_LOCKED_PROFILES = (
    "stage1_locked",
    "stage2_locked",
    "stage3_locked",
    "stage4_locked",
    "stage5_locked",
    "isolate_shadows_locked",
    "isolate_era5_locked",
)


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[[str], DictConfig]:
    """Compose a modeling config from the repository's config directory."""
    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "modeling")
    with initialize_config_dir(config_dir=config_dir, version_base=None):

        def _compose(name: str) -> DictConfig:
            return compose(config_name=name)

        yield _compose
    GlobalHydra.instance().clear()


def _mutable_copy(cfg: DictConfig) -> DictConfig:
    """Return a mutable copy of a resolved Hydra config."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


@pytest.mark.parametrize("name", _LOCKED_PROFILES)
def test_shipped_locked_profiles_pass_and_omit_the_test_split(
    compose_modeling: Callable[[str], DictConfig],
    name: str,
) -> None:
    """Each shipped lock is accepted, and the fit splits exclude 2025."""
    cfg = compose_modeling(name)
    guard_modeling_config(cfg, name)
    contract_invariants(cfg)
    assert list(cfg.data.splits) == ["train", "validation"]


def test_stage1_lock_rejects_seed_and_test_split_drift(
    compose_modeling: Callable[[str], DictConfig],
) -> None:
    """A seed edit or admitting the 2025 test split fails the Stage-1 lock."""
    cfg = compose_modeling("stage1_locked")
    assert_stage1_lock(cfg)

    drifted_seed = _mutable_copy(cfg)
    drifted_seed.seed = 1
    with pytest.raises(ValueError, match="seed"):
        assert_stage1_lock(drifted_seed)

    drifted_splits = _mutable_copy(cfg)
    drifted_splits.data.splits = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="test split"):
        assert_stage1_lock(drifted_splits)


def test_ablation_lock_rejects_a_learning_rate_edit(
    compose_modeling: Callable[[str], DictConfig],
) -> None:
    """Stage 2 stays on the frozen learning rate. An edit fails the ablation lock."""
    cfg = compose_modeling("stage2_locked")
    assert_ablation_lock(cfg)

    drifted = _mutable_copy(cfg)
    drifted.trainer.learning_rate = 1.0e-4
    with pytest.raises(ValueError, match="learning_rate"):
        assert_ablation_lock(drifted)


def test_contract_invariants_reject_a_moved_holdout_year(
    compose_modeling: Callable[[str], DictConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remapping 2025 away from the test split fails the contract guard."""
    cfg = compose_modeling("stage1_locked")
    contract_invariants(cfg)
    monkeypatch.setitem(SPLIT_BY_YEAR, 2025, "validation")
    with pytest.raises(RuntimeError, match="2024/2025"):
        contract_invariants(cfg)
