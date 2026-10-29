"""Early-stopping / evaluate_test config, guards, run identity, and callback wiring."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling.channels import V6_ABLATION_LOCKED_CONFIG_NAMES
from berlin_lst_downscaling.modeling.guards import (
    EarlyStoppingSpec,
    assert_ablation_lock,
    early_stopping_from_config,
    evaluate_test_from_config,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.run_identity import (
    RunIdentity,
    resolve_run_identity,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_ALL_CONFIGS = sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):

        def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
            return compose(config_name=name, overrides=overrides or [])

        yield _compose
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_defaults_are_off_everywhere(
    compose_modeling: Callable[..., DictConfig], name: str
) -> None:
    cfg = compose_modeling(name)
    assert cfg.early_stopping.enabled is False
    assert cfg.evaluate_test is True


def test_v6_arm_accepts_early_stopping(compose_modeling: Callable[..., DictConfig]) -> None:
    name = "v6_l4_locked"
    cfg = compose_modeling(
        name,
        [
            "early_stopping.enabled=true",
            "early_stopping.patience=5",
            "early_stopping.min_delta=0.01",
        ],
    )
    guard_modeling_config(cfg, name)
    assert_ablation_lock(cfg)
    assert early_stopping_from_config(cfg) == EarlyStoppingSpec(True, 5, 0.01)


def test_v6_arm_accepts_skip_test(compose_modeling: Callable[..., DictConfig]) -> None:
    name = "v6_l4_locked"
    cfg = compose_modeling(name, ["evaluate_test=false"])
    guard_modeling_config(cfg, name)
    assert_ablation_lock(cfg)
    assert evaluate_test_from_config(cfg) is False


@pytest.mark.parametrize(
    "override",
    [
        "early_stopping.patience=0",
        "early_stopping.patience=1.5",
        "early_stopping.patience=true",
        "early_stopping.min_delta=-0.1",
        "early_stopping.min_delta=x",
        "early_stopping.min_delta=.nan",
        "early_stopping.enabled=yes",
        "evaluate_test=maybe",
    ],
)
def test_malformed_values_are_rejected(
    compose_modeling: Callable[..., DictConfig], override: str
) -> None:
    name = "v6_l4_locked"
    cfg = compose_modeling(name, [override])
    with pytest.raises(ValueError, match=r"early_stopping|evaluate_test"):
        guard_modeling_config(cfg, name)
    with pytest.raises(ValueError, match=r"early_stopping|evaluate_test"):
        assert_ablation_lock(cfg)


@pytest.mark.parametrize("name", ["stage1_locked", "stage2_locked", "isolate_era5_locked"])
def test_v1_locks_reject_early_stopping_and_skip_test(
    compose_modeling: Callable[..., DictConfig], name: str
) -> None:
    with pytest.raises(ValueError, match="only admitted on the v6"):
        guard_modeling_config(compose_modeling(name, ["early_stopping.enabled=true"]), name)
    with pytest.raises(ValueError, match="only admitted on the v6"):
        guard_modeling_config(compose_modeling(name, ["evaluate_test=false"]), name)


@pytest.mark.parametrize("name", ["stage2_locked", "isolate_era5_locked"])
def test_v1_ablation_lock_itself_rejects_them(
    compose_modeling: Callable[..., DictConfig], name: str
) -> None:
    with pytest.raises(ValueError, match="early_stopping.enabled=true"):
        assert_ablation_lock(compose_modeling(name, ["early_stopping.enabled=true"]))
    with pytest.raises(ValueError, match="evaluate_test=false"):
        assert_ablation_lock(compose_modeling(name, ["evaluate_test=false"]))


def test_missing_blocks_default_to_off() -> None:
    cfg = OmegaConf.create({})
    assert early_stopping_from_config(cfg) == EarlyStoppingSpec(False, 10, 0.0)
    assert evaluate_test_from_config(cfg) is True
    assert early_stopping_from_config(cfg).payload() == {
        "enabled": False,
        "patience": 10,
        "min_delta": 0.0,
    }


def test_v6_names_are_the_admitted_set() -> None:
    assert "v6_l4_locked" in V6_ABLATION_LOCKED_CONFIG_NAMES
    assert "stage2_locked" not in V6_ABLATION_LOCKED_CONFIG_NAMES


_V1_KEYS = {
    "git_commit",
    "vertex_source_sha",
    "image_digest",
    "data_release_id",
    "training_root",
    "patch_index_root",
    "training_policy_hash",
    "patch_index_fingerprints",
    "training_content_fingerprint",
    "patch_index_content_fingerprint",
    "patch_index_source_release_root",
    "aggregation",
}


def test_v1_identity_payload_is_byte_identical_to_main() -> None:
    identity = RunIdentity(
        git_commit="abc",
        vertex_source_sha=None,
        image_digest=None,
        data_release_id="rel",
        training_root=None,
        patch_index_root=None,
        training_policy_hash="h",
        patch_index_fingerprints={},
        training_content_fingerprint=None,
        patch_index_content_fingerprint=None,
        patch_index_source_release_root=None,
        aggregation="arithmetic",
    )
    payload = identity.to_payload()
    assert set(payload) == _V1_KEYS
    expected = """{
  "aggregation": "arithmetic",
  "data_release_id": "rel",
  "git_commit": "abc",
  "image_digest": null,
  "patch_index_content_fingerprint": null,
  "patch_index_fingerprints": {},
  "patch_index_root": null,
  "patch_index_source_release_root": null,
  "training_content_fingerprint": null,
  "training_policy_hash": "h",
  "training_root": null,
  "vertex_source_sha": null
}"""
    assert json.dumps(payload, indent=2, sort_keys=True) == expected


def test_resolved_identity_adds_keys_only_for_v6(tmp_path: Path) -> None:
    base = {"seed": 0, "aggregation": "arithmetic", "data_release_id": "rel"}
    v1 = resolve_run_identity(OmegaConf.create(base), run_id="r", output_root=str(tmp_path))
    assert set(v1.to_payload()) == _V1_KEYS
    v6_cfg = OmegaConf.create(
        {**base, "ablation_release": "v6", "early_stopping": {"enabled": True, "patience": 3}}
    )
    v6 = resolve_run_identity(v6_cfg, run_id="r", output_root=str(tmp_path)).to_payload()
    assert v6["seed"] == 0
    assert v6["early_stopping"] == {"enabled": True, "patience": 3, "min_delta": 0.0}


def test_early_stopping_callback_monitors_validation_mae(
    compose_modeling: Callable[..., DictConfig],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    real = run_module.EarlyStopping

    def _spy(**kwargs: object) -> object:
        captured.update(kwargs)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_module, "EarlyStopping", _spy)
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            compose_modeling(
                "contract_smoke",
                [
                    "trainer.max_epochs=1",
                    "early_stopping.enabled=true",
                    "early_stopping.patience=3",
                    "early_stopping.min_delta=0.5",
                ],
            ),
            resolve=True,
        )
    )
    cfg.output_root = str(tmp_path / "data" / "runs" / "es")
    run_module.run_contract_synthetic_training(cfg, run_id="es-test")
    assert captured["monitor"] == "validation/mae_100m"
    assert captured["mode"] == "min"
    assert captured["patience"] == 3
    assert captured["min_delta"] == 0.5
    assert captured["check_on_train_epoch_end"] is False


def _stopped(reason: EarlyStoppingReason, epoch: int = 0) -> EarlyStopping:
    callback = EarlyStopping(monitor="validation/mae_100m", mode="min")
    callback.stopping_reason = reason
    callback.stopped_epoch = epoch
    return callback


_MONITORED = "validation/mae_100m"


def test_early_stop_epoch_patience_is_one_based() -> None:
    patience = EarlyStoppingReason.PATIENCE_EXHAUSTED
    metrics = {_MONITORED: torch.tensor(1.0)}
    assert run_module._early_stop_epoch(_stopped(patience, 0), metrics, _MONITORED, 0) == 1
    assert run_module._early_stop_epoch(_stopped(patience, 3), metrics, _MONITORED, 3) == 4


def test_early_stop_epoch_not_stopped_and_disabled() -> None:
    metrics = {_MONITORED: torch.tensor(1.0)}
    not_stopped = _stopped(EarlyStoppingReason.NOT_STOPPED)
    assert run_module._early_stop_epoch(not_stopped, metrics, _MONITORED, 5) is None
    assert (
        run_module._early_stop_epoch(None, {_MONITORED: torch.tensor(float("nan"))}, "m", 0) is None
    )


def test_early_stop_epoch_non_finite_reason_raises() -> None:
    callback = _stopped(EarlyStoppingReason.NON_FINITE_METRIC, 1)
    with pytest.raises(RuntimeError, match=r"non-finite validation/mae_100m at epoch 2"):
        run_module._early_stop_epoch(
            callback, {_MONITORED: torch.tensor(float("nan"))}, _MONITORED, 1
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_early_stop_epoch_non_finite_value_raises(bad: float) -> None:
    callback = _stopped(EarlyStoppingReason.NOT_STOPPED)
    with pytest.raises(RuntimeError, match=r"epoch 4 \(value="):
        run_module._early_stop_epoch(callback, {_MONITORED: torch.tensor(bad)}, _MONITORED, 4)


def test_early_stop_epoch_unexpected_reason_raises() -> None:
    callback = _stopped(EarlyStoppingReason.STOPPING_THRESHOLD, 1)
    with pytest.raises(RuntimeError, match="unexpected early stopping reason"):
        run_module._early_stop_epoch(callback, {_MONITORED: torch.tensor(1.0)}, _MONITORED, 1)


def _v6_cfg(
    compose_modeling: Callable[..., DictConfig], tmp_path: Path, overrides: list[str]
) -> DictConfig:
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            compose_modeling("contract_smoke", ["data.batch_size=1", *overrides]), resolve=True
        )
    )
    cfg.output_root = str(tmp_path / "data" / "runs" / "es")
    cfg.ablation_release = "v6"
    cfg.stage1_full = True
    return cfg


@pytest.fixture
def _no_run_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # The synthetic fixture is not a v6 arm; bypass only the v6 lock/feature-order guards.
    monkeypatch.setattr(run_module, "enforce_run_lock", lambda cfg, identity: None)
    monkeypatch.setattr(run_module, "contract_invariants", lambda cfg: None)


@pytest.mark.usefixtures("_no_run_lock")
def test_summary_records_patience_stop(
    compose_modeling: Callable[..., DictConfig], tmp_path: Path
) -> None:
    # min_delta far above any change: epoch 1 sets the best, epoch 2 cannot beat it.
    cfg = _v6_cfg(
        compose_modeling,
        tmp_path,
        [
            "trainer.max_epochs=4",
            "early_stopping.enabled=true",
            "early_stopping.patience=1",
            "early_stopping.min_delta=1000000.0",
        ],
    )
    run_module.run_contract_synthetic_training(cfg, run_id="es-patience")
    summary = json.loads((Path(cfg.output_root) / "full_summary.json").read_text())
    assert summary["stopped_epoch"] == 2
    assert summary["epochs_completed"] == 2


@pytest.mark.usefixtures("_no_run_lock")
def test_summary_has_no_stop_when_training_runs_to_max_epochs(
    compose_modeling: Callable[..., DictConfig], tmp_path: Path
) -> None:
    cfg = _v6_cfg(
        compose_modeling,
        tmp_path,
        ["trainer.max_epochs=2", "early_stopping.enabled=true", "early_stopping.patience=50"],
    )
    run_module.run_contract_synthetic_training(cfg, run_id="es-none")
    summary = json.loads((Path(cfg.output_root) / "full_summary.json").read_text())
    assert summary["stopped_epoch"] is None
    assert summary["epochs_completed"] == 2


@pytest.mark.usefixtures("_no_run_lock")
def test_non_finite_validation_mae_fails_the_run(
    compose_modeling: Callable[..., DictConfig],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(run_module.MaskedMAE, "compute", lambda self: torch.tensor(float("nan")))
    cfg = _v6_cfg(
        compose_modeling,
        tmp_path,
        ["trainer.max_epochs=3", "early_stopping.enabled=true", "early_stopping.patience=5"],
    )
    with pytest.raises(RuntimeError, match=r"non-finite validation/mae_100m at epoch 1 .*nan"):
        run_module.run_contract_synthetic_training(cfg, run_id="es-nan")
    assert not (Path(cfg.output_root) / "full_summary.json").exists()
