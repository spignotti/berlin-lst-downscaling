"""Phase timing: timed_phase, epoch timing callback, and the post-fit lifecycle (offline)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling.synthetic import ContractSyntheticDataModule
from berlin_lst_downscaling.modeling.timing import EpochTimingRecorder, timed_phase

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_LOG = logging.getLogger("test_vertex_phase_timings")


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict]:
    return [getattr(r, "fields", {}) for r in caplog.records if r.getMessage() == event]


def test_timed_phase_success(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with timed_phase("demo", logger=_LOG, patches=3):
        pass
    (start,) = _events(caplog, "phase_start")
    (end,) = _events(caplog, "phase_end")
    assert start["phase"] == end["phase"] == "demo"
    assert end["status"] == "ok"
    assert end["patches"] == 3
    assert end["duration_seconds"] >= 0.0
    assert end["started_at"] == start["started_at"]


def test_timed_phase_failure_reraises_and_marks_failed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError, match="boom"), timed_phase("demo", logger=_LOG):
        raise ValueError("boom")
    (end,) = _events(caplog, "phase_end")
    assert end["status"] == "failed"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _trainer(epoch: int, *, sanity: bool = False) -> SimpleNamespace:
    return SimpleNamespace(current_epoch=epoch, sanity_checking=sanity)


def test_epoch_timing_separates_train_and_validation(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    clock = _Clock()
    recorder = EpochTimingRecorder(logger=_LOG, clock=clock)
    module = object()
    for epoch, (train, val) in enumerate([(5.0, 2.0), (4.0, 3.0)]):
        trainer = _trainer(epoch)
        clock.now += 100.0
        recorder.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
        clock.now += train
        recorder.on_validation_epoch_start(trainer, module)  # type: ignore[arg-type]
        clock.now += val
        recorder.on_validation_epoch_end(trainer, module)  # type: ignore[arg-type]
        recorder.on_train_epoch_end(trainer, module)  # type: ignore[arg-type]
    assert recorder.records == [
        {"epoch": 0, "train_seconds": 5.0, "validation_seconds": 2.0, "epoch_seconds": 7.0},
        {"epoch": 1, "train_seconds": 4.0, "validation_seconds": 3.0, "epoch_seconds": 7.0},
    ]
    assert _events(caplog, "epoch_timing") == recorder.records


def test_epoch_timing_ignores_sanity_and_incomplete_epochs() -> None:
    clock = _Clock()
    recorder = EpochTimingRecorder(logger=_LOG, clock=clock)
    module = object()
    sanity = _trainer(0, sanity=True)
    recorder.on_validation_epoch_start(sanity, module)  # type: ignore[arg-type]
    clock.now += 50.0
    recorder.on_validation_epoch_end(sanity, module)  # type: ignore[arg-type]
    trainer = _trainer(0)
    # Epoch end without a start (interrupted/incomplete) emits nothing.
    recorder.on_train_epoch_end(trainer, module)  # type: ignore[arg-type]
    # Validation started but never finished: incomplete epoch, nothing emitted.
    recorder.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    recorder.on_validation_epoch_start(trainer, module)  # type: ignore[arg-type]
    recorder.on_train_epoch_end(trainer, module)  # type: ignore[arg-type]
    assert recorder.records == []


class _CountingModule(ContractSyntheticDataModule):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.setup_calls = 0
        self.val_loader_calls = 0

    def setup(self, stage: str | None = None) -> None:
        self.setup_calls += 1
        super().setup(stage)

    def val_dataloader(self):  # noqa: ANN201
        self.val_loader_calls += 1
        return super().val_dataloader()


@pytest.fixture
def cfg_for_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    monkeypatch.setattr(run_module, "enforce_run_lock", lambda cfg, identity: None)
    monkeypatch.setattr(run_module, "contract_invariants", lambda cfg: None)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        composed = compose(
            config_name="contract_smoke", overrides=["data.batch_size=1", "trainer.max_epochs=2"]
        )
    GlobalHydra.instance().clear()
    cfg = OmegaConf.create(OmegaConf.to_container(composed, resolve=True))
    cfg.output_root = str(tmp_path / "data" / "runs" / "timing")
    cfg.ablation_release = "v6"
    cfg.stage1_full = True
    return cfg


def test_post_fit_reuses_validation_cache_and_keeps_verification(
    cfg_for_lifecycle: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = cfg_for_lifecycle
    caplog.set_level(logging.INFO)
    module = _CountingModule(
        n_active_channels=run_module.selection_from_config(cfg).n_active,
        batch_size=1,
        n_train=int(cfg.data.n_train),
        n_val=int(cfg.data.n_val),
        n_test=int(cfg.data.n_test),
        seed=int(cfg.seed),
    )
    after_fit: dict[str, int] = {}
    real_fit = run_module.Trainer.fit

    def _fit(self: run_module.Trainer, *args: object, **kwargs: object) -> None:
        real_fit(self, *args, **kwargs)  # type: ignore[arg-type]
        after_fit["setup"] = module.setup_calls

    monkeypatch.setattr(run_module.Trainer, "fit", _fit)
    result = run_module._fit_contract_lifecycle(
        cfg, "timing", data_module=module, source_metadata={}, reproducibility="test"
    )
    assert result.run_ok
    # No data preparation (source read/admission) recurs after the fit returns.
    assert module.setup_calls == after_fit["setup"]
    assert module.val_loader_calls >= 1

    summary = json.loads((Path(cfg.output_root) / "full_summary.json").read_text())
    for key in (
        "best_metric",
        "reload_recomputed",
        "validation_abs_error_sum",
        "validation_valid_cells",
        "epochs_completed",
        "stopped_epoch",
    ):
        assert key in summary
    assert summary["epochs_completed"] == 2
    assert abs(summary["reload_recomputed"] - summary["best_metric"]) <= 1e-3 * max(
        1.0, abs(summary["best_metric"])
    )

    ended = {f["phase"]: f for f in _events(caplog, "phase_end")}
    for phase in (
        "data_prep",
        "fit",
        "post_fit",
        "post_fit_checkpoint_load",
        "post_fit_validation_verify",
        "post_fit_summary",
        "wandb_finalize",
    ):
        assert ended[phase]["status"] == "ok", phase
    assert "residual_identity" not in ended
    assert len(_events(caplog, "epoch_timing")) == 2
    assert [r.getMessage() for r in caplog.records].count("run_done") == 1
