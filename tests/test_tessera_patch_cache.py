"""Quantized TESSERA cache preserves window tensors without training-time I/O."""

import numpy as np
import pytest
import torch

from berlin_lst_downscaling.data.embeddings import loader
from berlin_lst_downscaling.modeling.contracts import RealSampleMeta
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, build_patch_cache
from berlin_lst_downscaling.modeling.patches import RealSample, collate_real_batch
from berlin_lst_downscaling.modeling.tessera_adapter import TesseraRealLSTTask


def _build(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    quantized = rng.integers(-127, 128, (160, 160, 128), dtype=np.int8)
    scale = rng.uniform(0.001, 0.1, (160, 160)).astype(np.float32)
    scale[0, 0] = np.nan

    class Raster:
        def __init__(self, kind):
            self.kind = kind

        def read(self, *args, window):
            assert (window.height, window.width) == (160, 160)
            assert (window.row_off, window.col_off) == (20, 30)
            return quantized.transpose(2, 0, 1) if self.kind == "int8" else scale

    monkeypatch.setattr(loader, "_year_dataset", lambda root, year, kind: Raster(kind))
    expected, _ = loader.load_patch_embedding("gs://offline/tessera", 2024, row_100m=2, col_100m=3)
    from berlin_lst_downscaling.data.embeddings.tessera import dequantize_tessera

    reference = dequantize_tessera(quantized, scale)
    np.testing.assert_array_equal(expected.view(np.uint32), reference.view(np.uint32))
    sample = RealSample(
        RealSampleMeta("p0", "scene", "validation", 2024, 2, 3, 0),
        np.zeros((22, 160, 160), dtype=np.float32),
        np.full((1, 160, 160), 290, dtype=np.float32),
        np.full((1, 16, 16), 290, dtype=np.float32),
        np.ones((1, 16, 16), dtype=bool),
    )
    provenance = {
        "channel_order": [str(i) for i in range(22)],
        "source_channels": 22,
        "tessera": {"root": "gs://offline/tessera", "encoding": "int8-scale-float32-v1"},
    }
    build_patch_cache(
        tmp_path / "cache",
        [sample],
        patch_ids=["p0"],
        active_channels=22,
        source_channels=22,
        provenance=provenance,
        max_bytes=100_000_000,
    )
    return PatchCache(tmp_path / "cache", provenance), expected, quantized, scale


def test_quantized_cache_bitwise_matches_window(tmp_path, monkeypatch):
    cache, expected, quantized, scale = _build(tmp_path, monkeypatch)
    np.testing.assert_array_equal(cache.arrays["tessera_int8"][0], quantized)
    np.testing.assert_array_equal(cache.arrays["tessera_scale"][0], scale)
    cached = cache[0]
    batch = collate_real_batch(
        [cached], n_active_channels=22, embeddings_root="gs://offline/tessera"
    )
    actual = batch.tessera_embedding.numpy()[0].transpose(1, 2, 0)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_cached_training_step_opens_no_storage(tmp_path, monkeypatch):
    cache, _, _, _ = _build(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not open cloud storage")

    monkeypatch.setattr(loader, "_year_dataset", forbidden)
    monkeypatch.setattr(loader, "load_patch_embedding", forbidden)
    monkeypatch.setattr("rasterio.open", forbidden)
    monkeypatch.setattr("berlin_lst_downscaling.data.io.storage.read_bytes", forbidden)
    task = TesseraRealLSTTask(base_width=4, depth=2)
    task.log = lambda *args, **kwargs: None
    optimizer = torch.optim.Adam(task.parameters())
    batch = collate_real_batch(
        [cache[0]], n_active_channels=22, embeddings_root="gs://offline/tessera"
    )
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_tessera_cache_disk_guard_before_source_read(tmp_path, monkeypatch):
    from berlin_lst_downscaling.modeling import patch_cache

    monkeypatch.setattr(
        patch_cache.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})()
    )
    provenance = {
        "channel_order": ["x"],
        "source_channels": 22,
        "tessera": {"root": "gs://offline/tessera", "encoding": "int8-scale-float32-v1"},
    }
    with pytest.raises(OSError, match="free"):
        build_patch_cache(
            tmp_path / "cache",
            [],
            patch_ids=["p0"],
            active_channels=1,
            source_channels=22,
            provenance=provenance,
            max_bytes=100_000_000,
        )


def test_tessera_prebuilt_roundtrip(tmp_path, monkeypatch):
    from berlin_lst_downscaling.modeling.prebuilt_cache import (
        fetch_prebuilt_cache,
        pack_prebuilt_cache,
        prebuilt_key_components,
        upload_prebuilt_cache,
    )

    cache, _, _, _ = _build(tmp_path, monkeypatch)
    # Two rows, one for each required split.
    provenance = dict(cache.manifest["provenance"])
    provenance.update(
        {
            "patch_index_fingerprints": {"index_policy_hash": "idx", "source_policy_hash": "src"},
            "scaler_digest": "scaler",
            "channel_indices": list(range(22)),
        }
    )
    from dataclasses import replace

    samples = [
        replace(cache[0], meta=replace(cache[0].meta, split="train", year=2018)),
        replace(cache[0], meta=replace(cache[0].meta, patch_id="p1", split="validation")),
    ]
    root = tmp_path / "two"
    build_patch_cache(
        root,
        samples,
        patch_ids=["p0", "p1"],
        active_channels=22,
        source_channels=22,
        provenance=provenance,
        max_bytes=100_000_000,
    )
    components = prebuilt_key_components(
        provenance, source_channel_order=provenance["channel_order"], splits=("train", "validation")
    )
    packed = tmp_path / "packed"
    manifest = pack_prebuilt_cache(
        root, packed, components=components, split_counts={"train": 1, "validation": 1}
    )
    assert {f["file"] for f in manifest["files"]} >= {"tessera_int8.npy", "tessera_scale.npy"}
    prefix = str(tmp_path / "objects")
    upload_prebuilt_cache(packed, prefix)
    dest = tmp_path / "nested" / "fetched"
    fetch_prebuilt_cache(prefix, dest, expected_components=components)
    fetched = PatchCache(dest, provenance)
    for key in cache.arrays:
        np.testing.assert_array_equal(fetched.arrays[key][0], cache.arrays[key][0])
    from berlin_lst_downscaling.modeling import prebuilt_cache

    monkeypatch.setattr(
        prebuilt_cache.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})()
    )
    with pytest.raises(OSError, match="packing cache needs"):
        pack_prebuilt_cache(
            root,
            tmp_path / "no-pack-space",
            components=components,
            split_counts={"train": 1, "validation": 1},
        )
    with pytest.raises(OSError, match="fetching cache needs"):
        fetch_prebuilt_cache(prefix, tmp_path / "no-fetch-space", expected_components=components)
    assert not (tmp_path / "no-pack-space").exists()
    assert not (tmp_path / "no-fetch-space").exists()


def test_tessera_estimate_includes_quantized_payload():
    from berlin_lst_downscaling.modeling.patch_cache import estimate_cache_bytes

    extra = estimate_cache_bytes(6250, 22, tessera=True) - estimate_cache_bytes(6250, 22)
    assert extra == 6250 * 160 * 160 * (128 + 4) + 2 * 8192


def test_cache_collate_refuses_missing_embedding(tmp_path, monkeypatch):
    from dataclasses import replace

    cache, _, _, _ = _build(tmp_path, monkeypatch)
    sample = replace(cache[0], tessera_int8=None, tessera_scale=None)
    with pytest.raises(RuntimeError, match="lacks quantized"):
        collate_real_batch(
            [sample],
            n_active_channels=22,
            embeddings_root="gs://offline/tessera",
            require_cached_tessera=True,
        )


@pytest.mark.parametrize("filename", ["tessera_int8.npy", "tessera_scale.npy"])
def test_cache_rejects_missing_tessera_array(tmp_path, monkeypatch, filename):
    cache, _, _, _ = _build(tmp_path, monkeypatch)
    (cache.root / filename).unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        PatchCache(cache.root, cache.manifest["provenance"])


def test_tessera_cache_budget_fails_before_source_read(tmp_path):
    from berlin_lst_downscaling.modeling.patch_cache import estimate_cache_bytes

    provenance = {
        "channel_order": ["x"],
        "source_channels": 22,
        "tessera": {"root": "gs://offline/tessera", "encoding": "int8-scale-float32-v1"},
    }
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        build_patch_cache(
            tmp_path / "cache",
            [],
            patch_ids=["p0"],
            active_channels=1,
            source_channels=22,
            provenance=provenance,
            max_bytes=estimate_cache_bytes(1, 1),
        )
    assert not (tmp_path / "cache").exists()
