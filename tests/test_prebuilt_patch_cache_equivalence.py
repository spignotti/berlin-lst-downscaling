"""Prebuilt cache versus job-local cache: identical samples and batches, fallback, timings.

Uses the real-raster fixture of ``test_parallel_patch_loading`` (22 and 28 channel
stacks). A temporary directory stands in for the GCS prefix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_parallel_patch_loading import (  # noqa: F401  (``world`` is a fixture)
    MAX_BYTES,
    _assert_same_epochs,
    _epochs,
    _patch_sources,
    _World,
    world,
)

from berlin_lst_downscaling.data.io import storage
from berlin_lst_downscaling.modeling import prebuilt_cache, real_task
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    PREBUILT_MANIFEST,
    PrebuiltCacheChecksumError,
    PrebuiltCacheMismatch,
    PrebuiltCacheMissing,
    pack_prebuilt_cache,
    upload_prebuilt_cache,
)
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule

_FINGERPRINTS = {
    "index_policy_hash": "idx",
    "source_policy_hash": "src",
    "patches_total": "5",
    "published_at": "t",
}


def _sources(monkeypatch: pytest.MonkeyPatch, scene: _World) -> None:
    _patch_sources(monkeypatch, scene)
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: dict(_FINGERPRINTS))


def _build(
    scene: _World, indices: tuple[int, ...], cache_root: Path, **kwargs: Any
) -> RealPatchDataModule:
    return RealPatchDataModule(
        scene.source,
        batch_size=2,
        n_active_channels=len(indices),
        channel_indices=indices,
        mode="stream",
        shuffle_train=True,
        seed=7,
        splits=kwargs.pop("splits", ("train", "validation")),
        cache_root=cache_root,
        cache_max_bytes=MAX_BYTES,
        **kwargs,
    )


def _publish(module: RealPatchDataModule, root: Path) -> str:
    components = module.prebuilt_components()
    assert module.cache_root is not None
    out_dir = root / "packed"
    pack_prebuilt_cache(
        module.cache_root,
        out_dir,
        components=components,
        split_counts=module.stats()["patches_per_split"],  # type: ignore[arg-type]
        shard_max_bytes=200_000,
    )
    prefix = prebuilt_cache.prebuilt_prefix(str(root / "gcs"), components)
    upload_prebuilt_cache(out_dir, prefix)
    return prefix


@pytest.fixture
def published(world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:  # noqa: F811
    _sources(monkeypatch, world)
    width = len(world.scaler.channels)
    indices = (0, 3, 7, width - 1)
    local = _build(world, indices, tmp_path / "a")
    local.setup("fit")
    return {
        "world": world,
        "indices": indices,
        "local": local,
        "prefix": _publish(local, tmp_path),
        "uri_root": str(tmp_path / "gcs"),
        "root": tmp_path,
    }


def _prebuilt(published: dict[str, Any], name: str = "b", **kwargs: Any) -> RealPatchDataModule:
    return _build(
        published["world"],
        published["indices"],
        published["root"] / name,
        cache_source="prebuilt",
        prebuilt_uri_root=published["uri_root"],
        **kwargs,
    )


def test_prebuilt_matches_job_local_samples_batches_and_order(published: dict[str, Any]) -> None:
    local: RealPatchDataModule = published["local"]
    module = _prebuilt(published)
    module.setup("fit")
    for split in ("train", "validation"):
        left, right = local._datasets[split], module._datasets[split]
        assert len(left) == len(right) > 0  # type: ignore[arg-type]
        for i in range(len(left)):  # type: ignore[arg-type]
            a, b = left[i], right[i]  # type: ignore[index]
            assert a.meta == b.meta
            for x, y in ((a.features, b.features), (a.lst_prior_k, b.lst_prior_k)):
                assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
            assert np.array_equal(a.target_100m, b.target_100m, equal_nan=True)
            assert np.array_equal(a.mask_100m, b.mask_100m)
    assert local.stats()["patch_ids"] == module.stats()["patch_ids"]
    assert local.stats()["train_shuffle_order"] == module.stats()["train_shuffle_order"]
    # Same seed: both loaders (train shuffled, validation fixed) give identical batches.
    _assert_same_epochs(_epochs(local), _epochs(module))
    for loader_a, loader_b in (
        (local.train_dataloader(), module.train_dataloader()),
        (local.val_dataloader(), module.val_dataloader()),
    ):
        for batch_a, batch_b in zip(loader_a, loader_b, strict=True):
            assert torch.equal(batch_a.features, batch_b.features)
            assert [m.patch_id for m in batch_a.metadata] == [m.patch_id for m in batch_b.metadata]


def test_stats_record_source_key_and_timings(
    published: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    local_stats = published["local"].stats()
    assert local_stats["patch_cache_source"] == "job_local"
    assert local_stats["prebuilt_download_seconds"] is None
    caplog.set_level(logging.INFO)
    module = _prebuilt(published)
    module.setup("fit")
    stats = module.stats()
    assert stats["patch_cache_source"] == "prebuilt"
    assert stats["patch_cache_prefix"] == published["prefix"]
    assert stats["patch_cache_key"] == published["prefix"].rsplit("/", 1)[1]
    for key in ("prebuilt_download_seconds", "prebuilt_verify_seconds", "prebuilt_load_seconds"):
        assert isinstance(stats[key], float) and stats[key] >= 0.0
    assert stats["prebuilt_bytes"] > 0  # type: ignore[operator]
    assert stats["prebuilt_shards"] > 0  # type: ignore[operator]
    build = stats["prebuilt_build_provenance"]
    assert isinstance(build, dict)
    assert set(build) == {"git_commit", "git_dirty", "builder_version", "created_at"}
    assert stats["prebuilt_build_commit"] == build["git_commit"]
    assert "prebuilt_commit_match" in stats and "prebuilt_running_commit" in stats
    ended = {
        r.fields["phase"]  # type: ignore[attr-defined]
        for r in caplog.records
        if r.message == "phase_end" and r.fields["phase"].startswith("prebuilt_")  # type: ignore[attr-defined]
    }
    assert ended == {"prebuilt_download", "prebuilt_verify", "prebuilt_load"}
    assert "cache_build" not in {
        r.fields["phase"]
        for r in caplog.records
        if r.message == "phase_end"  # type: ignore[attr-defined]
    }


def _empty_prefix_module(published: dict[str, Any], **kwargs: Any) -> RealPatchDataModule:
    return _build(
        published["world"],
        published["indices"],
        published["root"] / "c",
        cache_source="prebuilt",
        prebuilt_uri_root=str(published["root"] / "empty"),
        **kwargs,
    )


def test_missing_prebuilt_without_fallback_raises_and_never_builds(
    published: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    monkeypatch.setattr(RealPatchDataModule, "_build_cache", lambda self: built.append(1))
    module = _empty_prefix_module(published)
    with pytest.raises(PrebuiltCacheMissing):
        module.setup("fit")
    assert built == []
    assert not (published["root"] / "c").exists()


def test_missing_prebuilt_with_fallback_builds_job_local_and_warns(
    published: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    module = _empty_prefix_module(published, allow_job_local_fallback=True)
    module.setup("fit")
    assert module.stats()["patch_cache_source"] == "job_local_fallback"
    assert (published["root"] / "c" / "manifest.json").is_file()
    warnings = [r for r in caplog.records if r.message == "patch_cache_prebuilt_fallback"]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    _assert_same_epochs(_epochs(published["local"]), _epochs(module))


def test_mismatch_is_never_silently_rebuilt(
    published: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(published["prefix"]) / PREBUILT_MANIFEST
    manifest = json.loads(path.read_text())
    manifest["components"]["scaler_digest"] = "tampered"
    path.write_text(json.dumps(manifest))
    built: list[int] = []
    monkeypatch.setattr(RealPatchDataModule, "_build_cache", lambda self: built.append(1))
    with pytest.raises(PrebuiltCacheMismatch, match="scaler_digest"):
        _prebuilt(published).setup("fit")
    assert built == []


def test_checksum_failure_with_fallback_cleans_up_and_rebuilds(
    published: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    shard = next(Path(published["prefix"], "shards").glob("features-*.bin"))
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF
    shard.write_bytes(bytes(data))
    caplog.set_level(logging.WARNING)
    with pytest.raises(PrebuiltCacheChecksumError):
        _prebuilt(published, "strict").setup("fit")
    assert not (published["root"] / "strict").exists()
    module = _prebuilt(published, "lenient", allow_job_local_fallback=True)
    module.setup("fit")
    assert module.stats()["patch_cache_source"] == "job_local_fallback"
    fallback = [r for r in caplog.records if r.message == "patch_cache_prebuilt_fallback"]
    assert len(fallback) == 1


def test_existing_cache_root_is_refused(published: dict[str, Any]) -> None:
    (published["root"] / "b").mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        _prebuilt(published).setup("fit")


def test_prebuilt_never_touches_test_paths(
    published: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real_download, real_read, real_exists = (
        storage.download_to_file,
        storage.read_bytes,
        storage.exists,
    )
    root = str(published["uri_root"])

    def download(uri: object, path: object) -> None:
        seen.append(str(uri).removeprefix(root))
        real_download(uri, path)  # type: ignore[arg-type]

    def read(uri: object) -> bytes:
        seen.append(str(uri).removeprefix(root))
        return real_read(uri)  # type: ignore[arg-type]

    def exists(uri: object) -> bool:
        seen.append(str(uri).removeprefix(root))
        return real_exists(uri)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "download_to_file", download)
    monkeypatch.setattr(storage, "read_bytes", read)
    monkeypatch.setattr(storage, "exists", exists)
    _prebuilt(published).setup("fit")
    assert seen and all("test" not in uri for uri in seen)
    assert prebuilt_cache.ALLOWED_SPLITS == ("train", "validation")


def test_prebuilt_with_test_split_is_refused(published: dict[str, Any]) -> None:
    module = _prebuilt(published, splits=("train", "validation", "test"))
    with pytest.raises(ValueError, match="exactly train and validation"):
        module.setup("fit")


def test_prebuilt_requires_cache_root_and_uri_root(published: dict[str, Any]) -> None:
    world_ = published["world"]
    with pytest.raises(ValueError, match="prebuilt"):
        RealPatchDataModule(world_.source, cache_source="prebuilt", prebuilt_uri_root="/x")
    with pytest.raises(ValueError, match="prebuilt"):
        RealPatchDataModule(world_.source, cache_source="prebuilt", cache_root=Path("/x"))
    with pytest.raises(ValueError, match="cache_source"):
        RealPatchDataModule(world_.source, cache_source="bogus")
    with pytest.raises(ValueError, match=">= 1"):
        RealPatchDataModule(world_.source, prebuilt_download_workers=0)
