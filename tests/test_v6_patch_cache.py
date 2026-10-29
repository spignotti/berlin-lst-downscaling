"""Schema-width patch cache (v6 22-channel stack) and the real_task wiring."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from berlin_lst_downscaling.data.features.contracts import feature_schema
from berlin_lst_downscaling.data.training.contracts import scaler_channel_index
from berlin_lst_downscaling.modeling import real_task, run
from berlin_lst_downscaling.modeling.aggregation import resolve_aggregation_method
from berlin_lst_downscaling.modeling.channels import selection_from_config
from berlin_lst_downscaling.modeling.contracts import RealSampleMeta
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    build_patch_cache,
    estimate_cache_bytes,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealSample,
    RealSourceConfig,
    ScalerChannel,
    ScalerSpec,
)
from berlin_lst_downscaling.modeling.real_task import RealLSTTask, RealPatchDataModule

_ROOT = Path(__file__).resolve().parents[1]
_STACK_V6 = {"schema_version": 5, "stack_channels": 22}
_MAX_BYTES = 256 * 1024 * 1024
_V3_MANIFEST_KEYS = {
    "version",
    "patch_ids",
    "patch_ids_sha256",
    "metadata",
    "provenance",
    "provenance_sha256",
    "active_channels",
    "channel_indices",
    "shapes",
    "dtypes",
    "array_bytes",
    "estimated_bytes",
}


def _sample(patch_id: str, channels: int, split: str = "train", year: int = 2020) -> RealSample:
    rng = np.random.default_rng(abs(hash(patch_id)) % (2**32))
    features = np.stack(
        [np.full((160, 160), float(c), dtype=np.float32) for c in range(channels)]
    ) + rng.random((channels, 160, 160), dtype=np.float32)
    mask = np.zeros((1, 16, 16), dtype=bool)
    mask[:, :8] = True
    return RealSample(
        meta=RealSampleMeta(
            patch_id=patch_id,
            scene_id="s",
            split=split,
            year=year,
            row=0,
            col=0,
            filled_feature_pixels=0,
        ),
        features=features.astype(np.float32),
        lst_prior_k=(295.0 + rng.random((1, 160, 160))).astype(np.float32),
        target_100m=(295.0 + rng.random((1, 16, 16))).astype(np.float32),
        mask_100m=mask,
    )


def _provenance(n_active: int, *, stack: dict[str, int] | None) -> dict[str, object]:
    provenance: dict[str, object] = {
        "channel_order": [f"c{i}" for i in range(n_active)],
        "aggregation": "arithmetic",
    }
    if stack is not None:
        provenance["feature_stack"] = dict(stack)
    return provenance


def _build(
    root: Path,
    *,
    channels: int,
    indices: list[int] | None,
    n_active: int,
    stack: dict[str, int] | None,
    **kwargs: Any,
) -> dict[str, object]:
    return build_patch_cache(
        root,
        [_sample("p0", channels)],
        patch_ids=["p0"],
        active_channels=n_active,
        provenance=_provenance(n_active, stack=stack),
        max_bytes=_MAX_BYTES,
        channel_indices=indices,
        **kwargs,
    )


def test_22_stack_accepts_full_and_subset_and_reopens(tmp_path: Path) -> None:
    _build(
        tmp_path / "full",
        channels=22,
        indices=list(range(22)),
        n_active=22,
        stack=_STACK_V6,
        stack_channels=22,
    )
    PatchCache(tmp_path / "full", _provenance(22, stack=_STACK_V6))
    _build(
        tmp_path / "subset",
        channels=22,
        indices=[0, 5, 21],
        n_active=3,
        stack=_STACK_V6,
        stack_channels=22,
    )
    cache = PatchCache(tmp_path / "subset", _provenance(3, stack=_STACK_V6))
    source = _sample("p0", 22).features
    assert np.array_equal(cache[0].features, source[[0, 5, 21]])


def test_22_stack_rejects_index_22(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="out of range"):
        _build(
            tmp_path / "c",
            channels=22,
            indices=[0, 22],
            n_active=2,
            stack=_STACK_V6,
            stack_channels=22,
        )


def test_22_stack_requires_feature_stack_in_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature_stack"):
        _build(
            tmp_path / "c",
            channels=22,
            indices=[0],
            n_active=1,
            stack=None,
            stack_channels=22,
        )


def test_22_channel_samples_into_default_builder_fail(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="source feature shape mismatch"):
        _build(tmp_path / "c", channels=22, indices=[0], n_active=1, stack=None)


def test_28_stack_unchanged(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path / "ok", channels=28, indices=list(range(28)), n_active=28, stack=None
    )
    assert set(manifest) == _V3_MANIFEST_KEYS
    assert "feature_stack" not in manifest["provenance"]  # type: ignore[operator]
    with pytest.raises(ValueError, match="out of range"):
        _build(tmp_path / "bad", channels=28, indices=[28], n_active=1, stack=None)


def test_cross_schema_reuse_fails_closed(tmp_path: Path) -> None:
    _build(
        tmp_path / "c",
        channels=22,
        indices=[0, 21],
        n_active=2,
        stack=_STACK_V6,
        stack_channels=22,
    )
    with pytest.raises(RuntimeError):
        PatchCache(tmp_path / "c", _provenance(2, stack=None))
    with pytest.raises(RuntimeError):
        PatchCache(
            tmp_path / "c", _provenance(2, stack={"schema_version": 4, "stack_channels": 28})
        )


def test_cache_rejects_out_of_range_manifest_indices(tmp_path: Path) -> None:
    _build(
        tmp_path / "c",
        channels=22,
        indices=[0, 21],
        n_active=2,
        stack=_STACK_V6,
        stack_channels=22,
    )
    path = tmp_path / "c" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["channel_indices"] = [0, 22]
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="channel_indices"):
        PatchCache(tmp_path / "c", _provenance(2, stack=_STACK_V6))


def test_estimate_cache_bytes_width() -> None:
    with pytest.raises(ValueError):
        estimate_cache_bytes(1, 23, stack_channels=22)
    assert estimate_cache_bytes(1, 28) > 0


def test_data_module_rejects_index_outside_v6_stack() -> None:
    source = RealSourceConfig("pi", "tr", "ft", "ard")
    with pytest.raises(ValueError):
        RealPatchDataModule(
            source, feature_schema_version=5, n_active_channels=1, channel_indices=[22]
        )


# ── end to end through run_real_training ──────────────────────────────


class _FakeReader:
    scaler: ScalerSpec

    def __init__(self, source: object) -> None:
        self.exclusions: dict[str, int] = {}

    def preload(self, scene_ids: object) -> None:
        return None

    def admission_reason(self, ref: PatchRef) -> None:
        return None

    def read_patch(self, ref: PatchRef) -> RealSample:
        return _sample(ref.patch_id, 22, ref.split, ref.year)


def _fake_scaler() -> ScalerSpec:
    names = tuple(feature_schema(5).names)
    channels = tuple(
        ScalerChannel(
            index=scaler_channel_index(i), name=name, transform="identity", mean=None, std=None
        )
        for i, name in enumerate(names)
    )
    return ScalerSpec(channel_order=names, channels=channels, policy_hash="test")


def _refs() -> list[PatchRef]:
    rows = [("a", "train", 2017), ("b", "train", 2018), ("c", "validation", 2024)]
    return [
        PatchRef(
            patch_id=f"{sid}:0",
            scene_id=sid,
            s2_scene_id="s2",
            split=split,
            year=year,
            row=i * 16,
            col=0,
            n_eligible=128,
            eligibility_mask="m",
        )
        for i, (sid, split, year) in enumerate(rows)
    ]


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_ROOT / "configs" / "modeling"), version_base=None):

        def _compose(name: str) -> DictConfig:
            return compose(config_name=name)

        yield _compose
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("config_name", ["v6_l4_locked", "v6_loo_shadow_locked"])
def test_run_real_training_wiring_v6(
    config_name: str,
    compose_modeling: Callable[..., DictConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = compose_modeling(config_name)
    cfg.output_root = str(tmp_path / "out")
    selection = selection_from_config(cfg)
    scaler = _fake_scaler()
    captured: dict[str, Any] = {}

    def _lifecycle(*_args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "dummy"

    fingerprints = {"marker": "abc"}
    monkeypatch.setattr(real_task, "RealPatchReader", lambda source: _setup_reader(scaler))
    monkeypatch.setattr(real_task, "load_patch_refs", lambda *a, **k: _refs())
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: fingerprints)
    monkeypatch.setattr(run, "patch_index_fingerprints", lambda _root: fingerprints)
    monkeypatch.setattr(run, "load_scaler", lambda _root: scaler)
    monkeypatch.setattr(run, "_resolve_cache_budget", lambda _cfg: _MAX_BYTES)
    monkeypatch.setattr(run, "_resolve_cache_root", lambda _cfg, **_k: tmp_path / "cache")
    monkeypatch.setattr(run, "_FULL_CACHE_DISK_RESERVE_BYTES", 0)
    monkeypatch.setattr(run, "_fit_contract_lifecycle", _lifecycle)

    run.run_real_training(cfg, "test-run")
    data_module = captured["data_module"]
    data_module.setup()

    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["provenance"]["feature_stack"] == _STACK_V6
    assert manifest["channel_indices"] == list(selection.indices)
    cache = PatchCache(tmp_path / "cache", manifest["provenance"])
    assert cache.arrays["features"].shape == (3, selection.n_active, 160, 160)
    for row, ref in enumerate(_refs()):
        expected = _sample(ref.patch_id, 22, ref.split, ref.year).features[list(selection.indices)]
        assert np.array_equal(cache[row].features, expected)
    assert all(
        ref.split != "test" for ref in [r for refs in data_module._admitted.values() for r in refs]
    )

    task = RealLSTTask(
        n_active_channels=selection.n_active,
        base_width=8,
        depth=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        residual_prior=bool(cfg.get("stage1_residual_prior", False)),
        loss_name=str(cfg.contract.loss),
        aggregation=resolve_aggregation_method(cfg.get("aggregation")),
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(task, datamodule=data_module)
    mae = float(trainer.callback_metrics["validation/mae_100m"])
    assert math.isfinite(mae)


def _setup_reader(scaler: ScalerSpec) -> _FakeReader:
    reader = _FakeReader(None)
    reader.scaler = scaler
    return reader
