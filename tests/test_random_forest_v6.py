"""Offline tests for the RF/naive baselines on the v3 chain (feature schema 5)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
)
from berlin_lst_downscaling.modeling.channels import (
    ABLATION_STAGE_CHANNELS,
    V6_LADDER_STAGE_CHANNELS,
    schema_version_from_config,
)
from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
)
from berlin_lst_downscaling.modeling.patches import PatchRef, RealSample, RealSampleMeta
from berlin_lst_downscaling.modeling.random_forest import (
    ForestSpec,
    collect_train_pool,
    predict_10m,
    resolve_stage_selection,
    stage_matrix,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
V3_CHAIN = {
    "data_release_id": "training/v3",
    "features_root": "gs://berlin-lst-training-data/features/v6",
    "training_root": "gs://berlin-lst-training-data/training/v3",
    "patch_index_root": "gs://berlin-lst-training-data/training/patch-index/v3",
    "ard_root": "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10",
}
N_V6 = 22


def _compose(name: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


@pytest.mark.parametrize(
    "name", ["random_forest_v6_full", "random_forest_v6_smoke", "baseline_v6_full"]
)
def test_v6_configs_point_at_v3_chain(name: str) -> None:
    cfg = _compose(name)
    for key, expected in V3_CHAIN.items():
        assert cfg[key] == expected
    assert cfg.data.feature_schema_version == 5
    assert schema_version_from_config(cfg) == FEATURE_SCHEMA_VERSION_V6


def test_v6_rf_protocol_constants() -> None:
    v6 = _compose("random_forest_v6_full")
    v1 = _compose("random_forest_full")
    assert OmegaConf.to_container(v6.forest) == OmegaConf.to_container(v1.forest)
    assert (v6.forest.n_estimators, v6.forest.max_depth) == (200, 20)
    assert (v6.forest.min_samples_leaf, v6.forest.max_features) == (5, 1.0)
    assert v6.seed == 0 and v6.max_cells == 500_000
    assert list(v6.stages) == [1, 2, 3, 4]
    assert v6.output_root == "data/runs/random-forest-v6"
    smoke = _compose("random_forest_v6_smoke")
    assert smoke.seed == 0 and smoke.profile == "smoke"
    ForestSpec.from_config(v6)
    ForestSpec.from_config(smoke)
    assert _compose("baseline_v6_full").output_root == "data/runs/baseline-real-v6"


@pytest.mark.parametrize("name", ["random_forest_full", "random_forest_smoke", "baseline_full"])
def test_v1_configs_unchanged(name: str) -> None:
    cfg = _compose(name)
    assert cfg.training_root.endswith("/training/v1")
    assert cfg.patch_index_root.endswith("/training/patch-index/v1")
    assert cfg.features_root.endswith("/features/v3")
    assert schema_version_from_config(cfg) == FEATURE_SCHEMA_VERSION


def test_v6_stage_selection() -> None:
    counts = {s: resolve_stage_selection(s, FEATURE_SCHEMA_VERSION_V6) for s in (1, 2, 3, 4)}
    assert [sel.n_active_channels for sel in counts.values()] == [10, 16, 18, 22]
    for stage, sel in counts.items():
        assert sel.names == V6_LADDER_STAGE_CHANNELS[stage]
    assert [counts[s].feature_order for s in (1, 2, 3, 4)] == [
        "v6_named_subset",
        "v6_named_subset",
        "v6_named_subset",
        "v6_registry",
    ]
    with pytest.raises(ValueError):
        resolve_stage_selection(5, FEATURE_SCHEMA_VERSION_V6)
    with pytest.raises(ValueError):
        resolve_stage_selection(1, 3)


def test_v3_stage_selection_unchanged() -> None:
    sels = [resolve_stage_selection(s) for s in (1, 2, 3, 4)]
    assert [sel.n_active_channels for sel in sels] == [10, 18, 20, 28]
    for stage, sel in zip((1, 2, 3, 4), sels, strict=True):
        assert sel.names == ABLATION_STAGE_CHANNELS[stage]
    assert resolve_stage_selection(2, FEATURE_SCHEMA_VERSION).names == sels[1].names


def _sample(patch_id: str, n_channels: int, seed: int, row: int = 0) -> RealSample:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_channels, REAL_PATCH_PX, REAL_PATCH_PX)).astype(np.float32)
    prior = np.full((1, REAL_PATCH_PX, REAL_PATCH_PX), 290.0, dtype=np.float32)
    target = (290.0 + rng.normal(size=(1, REAL_PATCH_CELLS, REAL_PATCH_CELLS))).astype(np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    mask[:, 0, :] = False
    meta = RealSampleMeta(
        patch_id=patch_id,
        scene_id="LC08_X",
        split="train",
        year=2018,
        row=row,
        col=0,
        filled_feature_pixels=0,
    )
    return RealSample(meta, features, prior, target, mask)


def _ref(patch_id: str, row: int) -> PatchRef:
    return PatchRef(patch_id, "LC08_X", "S2_X", "train", 2018, row, 0, 255, "m")


class _FakeReader:
    def __init__(self, samples: dict[str, RealSample]) -> None:
        self.samples = samples
        self.exclusions: Counter[str] = Counter()

    def read_patch(self, ref: PatchRef) -> RealSample:
        return self.samples[ref.patch_id]


def _refs() -> list[PatchRef]:
    return [_ref(f"p{i}", i * REAL_PATCH_CELLS) for i in range(3)]


def test_v6_pool_fit_predict() -> None:
    reader = _FakeReader({f"p{i}": _sample(f"p{i}", N_V6, i) for i in range(3)})
    pool = collect_train_pool(
        reader,  # type: ignore[arg-type]
        _refs(),
        max_cells=1000,
        seed=0,
        n_feature_channels=N_V6,
    )
    assert pool.features.shape == (3 * 240, N_V6)
    spec = ForestSpec(n_estimators=2, max_depth=3, n_jobs=1)
    sample = reader.samples["p0"]
    prior_100m = np.full((REAL_PATCH_CELLS, REAL_PATCH_CELLS), 290.0, dtype=np.float32)
    for stage_no, width in ((1, 11), (4, 23)):
        stage = resolve_stage_selection(stage_no, FEATURE_SCHEMA_VERSION_V6)
        matrix = stage_matrix(pool.features, pool.prior_100m, stage, n_feature_channels=N_V6)
        assert matrix.shape == (pool.selected_cells, width)
        model = spec.build().fit(matrix, pool.residual_k)
        residual = predict_10m(model, sample.features, prior_100m, stage, n_feature_channels=N_V6)
        assert residual.shape == (REAL_PATCH_PX, REAL_PATCH_PX)
        assert np.isfinite(residual).all()


def test_v6_path_rejects_28_channel_sample() -> None:
    reader = _FakeReader({"p0": _sample("p0", N_FEATURE_CHANNELS, 0)})
    with pytest.raises(ValueError, match="22 feature channels"):
        collect_train_pool(
            reader,  # type: ignore[arg-type]
            [_ref("p0", 0)],
            max_cells=10,
            seed=0,
            n_feature_channels=N_V6,
        )
    stage = resolve_stage_selection(4, FEATURE_SCHEMA_VERSION_V6)
    model_features = np.zeros((REAL_PATCH_PX * REAL_PATCH_PX, N_FEATURE_CHANNELS), np.float32)
    with pytest.raises(ValueError):
        stage_matrix(model_features, np.zeros(len(model_features)), stage, n_feature_channels=N_V6)


def _load_rf_validator():
    import importlib.util
    import sys

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "validators" / "validate_random_forest.py"
    )
    spec = importlib.util.spec_from_file_location("validate_random_forest_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validator_layout_selection() -> None:
    mod = _load_rf_validator()
    assert mod._layout_for({}) is mod._V3_LAYOUT
    assert mod._layout_for({"feature_schema_version": FEATURE_SCHEMA_VERSION}) is mod._V3_LAYOUT
    assert mod._V3_LAYOUT.width == 28
    assert mod._V3_LAYOUT.stage_indices[3] == (*range(18), 26, 27)
    layout = mod._layout_for({"feature_schema_version": FEATURE_SCHEMA_VERSION_V6})
    assert layout.width == N_V6
    assert [len(layout.stage_indices[s]) for s in (1, 2, 3, 4)] == [10, 16, 18, 22]
    assert layout.stage_indices[4] == tuple(range(N_V6))
    assert [layout.feature_order[s] for s in (1, 2, 3, 4)] == [
        "v6_named_subset",
        "v6_named_subset",
        "v6_named_subset",
        "v6_registry",
    ]
    for stage, channels in V6_LADDER_STAGE_CHANNELS.items():
        assert tuple(layout.names[i] for i in layout.stage_indices[stage]) == channels
    with pytest.raises(ValueError):
        mod._layout_for({"feature_schema_version": 3})


def test_validator_native_inputs_v6_and_v3() -> None:
    mod = _load_rf_validator()
    layout = mod._layout_for({"feature_schema_version": FEATURE_SCHEMA_VERSION_V6})
    features = np.random.default_rng(0).normal(size=(N_V6, 160, 160)).astype(np.float32)
    prior = np.full((16, 16), 290.0, dtype=np.float32)
    out = mod._native_inputs(features, prior, 1, layout)
    assert out.shape == (160 * 160, 11)
    assert np.all(out[:, -1] == 290.0)
    assert mod._native_inputs(features, prior, 4, layout).shape == (160 * 160, 23)
    with pytest.raises(ValueError, match="22"):
        mod._native_inputs(np.zeros((28, 160, 160), np.float32), prior, 1, layout)
    # v1 default unchanged
    v3 = mod._native_inputs(np.zeros((28, 160, 160), np.float32), prior, 4)
    assert v3.shape == (160 * 160, 29)
    with pytest.raises(ValueError, match="28"):
        mod._native_inputs(features, prior, 4)
