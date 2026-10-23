"""TESSERA training path: loader replaces morphology/shadows; task uses the adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.data.embeddings.loader import (
    build_tessera_model_inputs,
    load_scene_year_embedding,
    select_kept_features,
)
from berlin_lst_downscaling.modeling.contracts import RealBatch, RealSampleMeta
from berlin_lst_downscaling.modeling.tessera_adapter import (
    TesseraRealLSTTask,
    tessera_kept_channel_indices,
)

_REPO = Path(__file__).resolve().parents[1]


def _patch_meta(year: int = 2024) -> RealSampleMeta:
    return RealSampleMeta(
        patch_id=f"scene:{year}",
        scene_id="LC09_L2SP_000000",
        split="train",
        year=year,
        row=0,
        col=0,
        filled_feature_pixels=0,
    )


def test_loader_keeps_14_and_drops_replaced() -> None:
    features = np.arange(22 * 4 * 4, dtype=np.float32).reshape(22, 4, 4)
    kept = select_kept_features(features)
    assert kept.shape == (14, 4, 4)
    # Kept indices exclude morphology+shadow positions.
    assert tuple(tessera_kept_channel_indices()) == tuple(sorted(tessera_kept_channel_indices()))
    assert len(tessera_kept_channel_indices()) == 14


def test_load_scene_year_embedding_uses_int8_scale_product(tmp_path) -> None:
    from affine import Affine

    from berlin_lst_downscaling.data.embeddings.product import write_tessera_year_product

    rng = np.random.default_rng(2)
    quantized = rng.integers(-50, 50, size=(8, 8, 128)).astype(np.int8)
    scale = np.full((8, 8), 0.5, dtype=np.float32)
    scale[0, 0] = np.nan
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0)
    write_tessera_year_product(
        quantized=quantized,
        scale=scale,
        valid=np.isfinite(scale),
        transform=transform,
        crs="EPSG:25833",
        output_dir=str(tmp_path / "2024"),
        year=2024,
        provenance={"dataset_version": "v1.1-dclimate", "year": 2024},
    )
    embedding, valid = load_scene_year_embedding(str(tmp_path), 2024)
    assert embedding.shape == (8, 8, 128)
    assert valid.tolist()[0][0] is False
    assert bool(np.all(np.isnan(embedding[0, 0])))


def test_build_inputs_and_tessera_task_forward_cpu() -> None:
    torch.manual_seed(0)
    full = np.arange(22 * 16 * 16, dtype=np.float32).reshape(22, 16, 16)
    emb_hw = np.random.default_rng(0).standard_normal((16, 16, 128)).astype(np.float32)
    kept_np2, emb_np2 = build_tessera_model_inputs(full, emb_hw)
    assert kept_np2.shape == (14, 16, 16)
    assert emb_np2.shape == (128, 16, 16)

    task = TesseraRealLSTTask(n_kept_channels=14, tessera_adapter_width=8, base_width=8, depth=2)
    task.eval()
    kept = torch.randn(2, 14, 160, 160)
    embedding = torch.randn(2, 128, 160, 160)
    batch = RealBatch(
        features=kept,
        lst_prior=torch.zeros(2, 1, 160, 160),
        target_100m=torch.full((2, 1, 16, 16), 290.0),
        mask_100m=torch.ones(2, 1, 16, 16, dtype=torch.bool),
        metadata=[_patch_meta(), _patch_meta()],
        tessera_embedding=embedding,
    )
    with torch.inference_mode():
        out = task(batch)
    # Tessera arm predicts at 10 m like the main arm (160x160 contract patch).
    assert tuple(out.shape) == (2, 1, 160, 160)


def test_modeling_run_reads_tessera_enabled() -> None:
    with initialize_config_dir(config_dir=str(_REPO / "configs" / "modeling"), version_base=None):
        cfg = compose(config_name="tessera_arm")
    assert bool(cfg.data.tessera_enabled) is True
    assert int(cfg.model.tessera_adapter_width) == 8
