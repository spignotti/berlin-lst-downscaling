"""TESSERA training path: loader replaces morphology/shadows; task uses the adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.data.embeddings.loader import (
    build_tessera_model_inputs,
    collate_tessera_batch,
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


def test_adapter_in_optimizer_and_nan_zero_filled() -> None:
    task = TesseraRealLSTTask(n_kept_channels=14, tessera_adapter_width=8, base_width=8, depth=2)
    opt = task.configure_optimizers()
    n_adapter = sum(p.numel() for p in task.adapter.parameters())
    n_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    n_model = sum(p.numel() for p in task.model.parameters())
    assert n_adapter > 0
    assert n_opt == n_model + n_adapter
    # NaN gaps are zero-filled at the model boundary (finite prediction).
    task.eval()
    kept = torch.randn(1, 14, 160, 160)
    embedding = torch.randn(1, 128, 160, 160)
    embedding[0, :, 0, 0] = float("nan")
    batch = RealBatch(
        features=kept,
        lst_prior=torch.zeros(1, 1, 160, 160),
        target_100m=torch.full((1, 1, 16, 16), 290.0),
        mask_100m=torch.ones(1, 1, 16, 16, dtype=torch.bool),
        metadata=[_patch_meta()],
        tessera_embedding=embedding,
    )
    with torch.inference_mode():
        out = task(batch)
    assert bool(torch.isfinite(out).all())


def test_load_patch_embedding_reads_window(tmp_path) -> None:
    from affine import Affine

    from berlin_lst_downscaling.data.embeddings.loader import load_patch_embedding
    from berlin_lst_downscaling.data.embeddings.product import write_tessera_year_product

    rng = np.random.default_rng(3)
    quantized = rng.integers(-20, 20, size=(320, 320, 128)).astype(np.int8)
    scale = np.full((320, 320), 0.5, dtype=np.float32)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 3200.0)
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
    embedding, valid = load_patch_embedding(str(tmp_path), 2024, row_100m=1, col_100m=2)
    assert embedding.shape == (160, 160, 128)
    assert valid.shape == (160, 160)
    assert bool(np.all(valid))


def test_collate_reads_windows_never_full_raster(tmp_path, monkeypatch) -> None:
    """Collate must serve every sample from a COG window, never a full read."""
    import types

    from affine import Affine

    import berlin_lst_downscaling.data.embeddings.loader as loader_mod
    from berlin_lst_downscaling.data.embeddings.product import write_tessera_year_product

    def _forbid_full_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("collate must not fall back to a full-raster read")

    monkeypatch.setattr(loader_mod, "load_scene_year_embedding", _forbid_full_read)

    rng = np.random.default_rng(11)
    quantized = rng.integers(-20, 20, size=(160, 320, 128)).astype(np.int8)
    scale = np.full((160, 320), 0.5, dtype=np.float32)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 1600.0)
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
    samples = []
    for col in (0, 16):
        samples.append(
            types.SimpleNamespace(
                features=rng.standard_normal((22, 160, 160)).astype(np.float32),
                meta=types.SimpleNamespace(year=2024, row=0, col=col),
            )
        )
    kept_batch, emb_batch = collate_tessera_batch(samples, str(tmp_path))
    assert tuple(kept_batch.shape) == (2, 14, 160, 160)
    assert tuple(emb_batch.shape) == (2, 128, 160, 160)
    assert bool(torch.isfinite(emb_batch).all())
    # Kept channels are exactly the loader-selected v6 subset.
    assert torch.equal(
        kept_batch[1],
        torch.from_numpy(select_kept_features(np.asarray(samples[1].features))),
    )


def test_tessera_task_checkpoint_roundtrip(tmp_path) -> None:
    task = TesseraRealLSTTask(n_kept_channels=14, tessera_adapter_width=8, base_width=8, depth=2)
    ckpt = str(tmp_path / "tessera.ckpt")
    from lightning.pytorch import Trainer

    trainer = Trainer(max_epochs=0, logger=False, enable_checkpointing=False)
    trainer.strategy.connect(task)
    trainer.save_checkpoint(ckpt)
    loaded = TesseraRealLSTTask.load_from_checkpoint(ckpt, map_location="cpu")
    assert loaded.n_kept_channels == 14
    assert loaded.tessera_adapter_width == 8
    assert loaded.adapter.proj.in_channels == 128
