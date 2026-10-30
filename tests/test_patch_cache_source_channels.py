"""Offline tests: the job-local patch cache is sized by the source stack width."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6
from berlin_lst_downscaling.modeling import real_task
from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    validate_real_batch,
)
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    assert_samples_equal,
    build_patch_cache,
    estimate_cache_bytes,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealSample,
    RealSampleMeta,
    RealSourceConfig,
    ScalerChannel,
    ScalerSpec,
)
from berlin_lst_downscaling.modeling.random_forest import resolve_stage_selection
from berlin_lst_downscaling.modeling.real_task import RealLSTTask, RealPatchDataModule

N_V6 = 22
MAX_BYTES = 1 << 30
# v6 L3 arm: a non-contiguous named subset of the 22-channel stack.
V6_L3_INDICES = resolve_stage_selection(3, FEATURE_SCHEMA_VERSION_V6).indices


def _sample(patch_id: str, n_channels: int, seed: int, split: str = "train") -> RealSample:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_channels, REAL_PATCH_PX, REAL_PATCH_PX)).astype(np.float32)
    prior = np.full((1, REAL_PATCH_PX, REAL_PATCH_PX), 290.0, dtype=np.float32)
    target = (290.0 + rng.normal(size=(1, REAL_PATCH_CELLS, REAL_PATCH_CELLS))).astype(np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    mask[:, 0, :] = False
    meta = RealSampleMeta(
        patch_id=patch_id,
        scene_id="LC08_X",
        split=split,
        year=2018 if split == "train" else 2024,
        row=0,
        col=0,
        filled_feature_pixels=0,
    )
    return RealSample(meta, features, prior, target, mask)


def _provenance(source_channels: int, indices: tuple[int, ...]) -> dict[str, object]:
    return {
        "channel_order": [f"c{i}" for i in indices],
        "channel_indices": list(indices),
        "source_channels": source_channels,
    }


def _build(
    root: Path, samples: list[RealSample], source_channels: int, indices: tuple[int, ...]
) -> dict[str, object]:
    return build_patch_cache(
        root,
        iter(samples),
        patch_ids=[s.meta.patch_id for s in samples],
        active_channels=len(indices),
        source_channels=source_channels,
        provenance=_provenance(source_channels, indices),
        max_bytes=MAX_BYTES,
        channel_indices=indices,
    )


@pytest.mark.parametrize(
    ("source_channels", "indices"),
    [
        (N_V6, tuple(range(N_V6))),
        (N_V6, V6_L3_INDICES),
        (N_FEATURE_CHANNELS, tuple(range(N_FEATURE_CHANNELS))),
        (N_FEATURE_CHANNELS, (0, 3, 27)),
    ],
)
def test_cache_builds_and_reopens_for_source_width(
    tmp_path: Path, source_channels: int, indices: tuple[int, ...]
) -> None:
    samples = [_sample(f"p{i}", source_channels, i) for i in range(2)]
    manifest = _build(tmp_path / "cache", samples, source_channels, indices)
    assert manifest["provenance"]["source_channels"] == source_channels  # type: ignore[index]
    cache = PatchCache(tmp_path / "cache", _provenance(source_channels, indices))
    features = np.load(tmp_path / "cache" / "features.npy", mmap_mode="r")
    assert features.shape == (2, len(indices), REAL_PATCH_PX, REAL_PATCH_PX)
    for i, sample in enumerate(samples):
        assert_samples_equal(sample, cache[i], len(indices), channel_indices=indices)


def test_v6_l3_subset_is_non_contiguous() -> None:
    assert len(V6_L3_INDICES) == 18
    assert V6_L3_INDICES != tuple(range(18))
    assert max(V6_L3_INDICES) == N_V6 - 1


@pytest.mark.parametrize(("source_channels", "sample_channels"), [(N_V6, 28), (28, N_V6)])
def test_cache_rejects_source_width_mismatch(
    tmp_path: Path, source_channels: int, sample_channels: int
) -> None:
    samples = [_sample("p0", sample_channels, 0)]
    with pytest.raises(RuntimeError, match=r"source feature shape mismatch.*expected"):
        _build(tmp_path / "cache", samples, source_channels, tuple(range(10)))


@pytest.mark.parametrize("bad_index", [N_V6, 27, -1])
def test_cache_rejects_index_outside_v6_stack(tmp_path: Path, bad_index: int) -> None:
    samples = [_sample("p0", N_V6, 0)]
    with pytest.raises(ValueError, match="out of range for a 22-channel stack"):
        _build(tmp_path / "cache", samples, N_V6, (0, 1, bad_index))


@pytest.mark.parametrize("source_channels", [0, -1, True, 22.0])
def test_cache_rejects_invalid_source_channels(tmp_path: Path, source_channels: object) -> None:
    with pytest.raises(ValueError, match="source_channels"):
        build_patch_cache(
            tmp_path / "cache",
            iter([]),
            patch_ids=["p0"],
            active_channels=1,
            source_channels=source_channels,  # type: ignore[arg-type]
            provenance=_provenance(1, (0,)),
            max_bytes=MAX_BYTES,
        )


def test_cache_rejects_active_wider_than_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="active_channels"):
        _build(tmp_path / "cache", [_sample("p0", N_V6, 0)], N_V6, tuple(range(N_V6)) + (22,))


def test_cache_requires_source_channels_in_provenance(tmp_path: Path) -> None:
    provenance = _provenance(N_V6, (0,))
    del provenance["source_channels"]
    with pytest.raises(ValueError, match="provenance source_channels"):
        build_patch_cache(
            tmp_path / "cache",
            iter([_sample("p0", N_V6, 0)]),
            patch_ids=["p0"],
            active_channels=1,
            source_channels=N_V6,
            provenance=provenance,
            max_bytes=MAX_BYTES,
        )


def test_cache_reopen_rejects_other_source_width(tmp_path: Path) -> None:
    indices = tuple(range(4))
    _build(tmp_path / "cache", [_sample("p0", N_V6, 0)], N_V6, indices)
    with pytest.raises(RuntimeError, match="provenance"):
        PatchCache(tmp_path / "cache", _provenance(N_FEATURE_CHANNELS, indices))


def test_estimate_has_no_v3_upper_bound() -> None:
    assert estimate_cache_bytes(1, 64) > estimate_cache_bytes(1, N_FEATURE_CHANNELS)
    for bad in (0, True):
        with pytest.raises(ValueError):
            estimate_cache_bytes(1, bad)


def test_patch_cache_reopens_wider_than_v3_stack(tmp_path: Path) -> None:
    indices = tuple(range(30))
    _build(tmp_path / "cache", [_sample("p0", 32, 0)], 32, indices)
    assert (
        PatchCache(tmp_path / "cache", _provenance(32, indices)).arrays["features"].shape[1] == 30
    )


class _FakeReader:
    def __init__(self, samples: dict[str, RealSample], n_channels: int) -> None:
        self.samples = samples
        self.exclusions: Counter[str] = Counter()
        names = tuple(f"v6_c{i}" for i in range(n_channels))
        self.scaler = ScalerSpec(
            channel_order=names,
            channels=tuple(
                ScalerChannel(index=i + 1, name=n, transform="none", mean=0.0, std=1.0)
                for i, n in enumerate(names)
            ),
            policy_hash="test",
        )

    def preload(self, scene_ids: object) -> None:
        return None

    def admission_reason(self, ref: PatchRef) -> str | None:
        return None

    def read_patch(self, ref: PatchRef) -> RealSample:
        return self.samples[ref.patch_id]


def _refs() -> list[PatchRef]:
    return [
        PatchRef("t0", "LC08_X", "S2_X", "train", 2018, 0, 0, 240, "m"),
        PatchRef("t1", "LC08_X", "S2_X", "train", 2018, 16, 0, 240, "m"),
        PatchRef("v0", "LC08_X", "S2_X", "validation", 2024, 0, 0, 240, "m"),
    ]


@pytest.mark.parametrize("indices", [tuple(range(N_V6)), V6_L3_INDICES])
def test_cached_datamodule_v6_reader_yields_task_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indices: tuple[int, ...]
) -> None:
    samples = {
        ref.patch_id: _sample(ref.patch_id, N_V6, i, ref.split) for i, ref in enumerate(_refs())
    }
    monkeypatch.setattr(real_task, "RealPatchReader", lambda _source: _FakeReader(samples, N_V6))
    monkeypatch.setattr(
        real_task,
        "load_patch_refs",
        lambda _source, *, splits, scene_ids: [r for r in _refs() if r.split in splits],
    )
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: {"index": "x"})
    source = RealSourceConfig("local", "local", "local", "local")
    module = RealPatchDataModule(
        source,
        batch_size=2,
        n_active_channels=len(indices),
        channel_indices=indices,
        mode="stream",
        splits=("train", "validation"),
        cache_root=tmp_path / "cache",
        cache_max_bytes=MAX_BYTES,
    )
    module.setup("fit")

    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["provenance"]["source_channels"] == N_V6
    batch = next(iter(module.train_dataloader()))
    assert tuple(batch.features.shape) == (2, len(indices), REAL_PATCH_PX, REAL_PATCH_PX)
    validate_real_batch(batch, n_active_channels=len(indices))
    expected = np.stack([samples[p].features[np.asarray(indices)] for p in ("t0", "t1")])
    assert np.array_equal(batch.features.numpy(), expected)
    task = RealLSTTask(n_active_channels=len(indices), base_width=4, depth=1)
    assert tuple(task(batch).shape) == (2, 1, REAL_PATCH_PX, REAL_PATCH_PX)
    val = next(iter(module.val_dataloader()))
    assert tuple(val.features.shape) == (1, len(indices), REAL_PATCH_PX, REAL_PATCH_PX)


def _staging_dirs(parent: Path) -> list[Path]:
    return [p for p in parent.iterdir() if ".staging-" in p.name]


def test_failed_build_publishes_nothing_and_cleans_its_staging(tmp_path: Path) -> None:
    samples = [_sample(f"p{i}", N_V6, i) for i in range(2)]

    def failing():
        yield samples[0]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        build_patch_cache(
            tmp_path / "cache",
            failing(),
            patch_ids=["p0", "p1"],
            active_channels=N_V6,
            source_channels=N_V6,
            provenance=_provenance(N_V6, tuple(range(N_V6))),
            max_bytes=MAX_BYTES,
        )
    assert not (tmp_path / "cache").exists()
    assert _staging_dirs(tmp_path) == []


def test_interrupted_staging_does_not_block_rebuild(tmp_path: Path) -> None:
    leftover = tmp_path / ".cache.staging-dead"
    leftover.mkdir()
    (leftover / "features.npy").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="incomplete"):
        PatchCache(leftover, _provenance(N_V6, tuple(range(N_V6))))
    samples = [_sample(f"p{i}", N_V6, i) for i in range(2)]
    _build(tmp_path / "cache", samples, N_V6, tuple(range(N_V6)))
    assert (tmp_path / "cache" / "manifest.json").is_file()
    assert (leftover / "features.npy").read_bytes() == b"partial"
    assert not list((tmp_path / "cache").glob("*.partial"))


def test_existing_cache_is_never_overwritten(tmp_path: Path) -> None:
    samples = [_sample(f"p{i}", N_V6, i) for i in range(2)]
    indices = tuple(range(N_V6))
    _build(tmp_path / "cache", samples, N_V6, indices)
    before = (tmp_path / "cache" / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _build(tmp_path / "cache", samples, N_V6, indices)
    assert (tmp_path / "cache" / "manifest.json").read_bytes() == before
    assert _staging_dirs(tmp_path) == []


class _CountingReader(_FakeReader):
    def __init__(self, samples: dict[str, RealSample], n_channels: int) -> None:
        super().__init__(samples, n_channels)
        self.reads: list[str] = []

    def read_patch(self, ref: PatchRef) -> RealSample:
        self.reads.append(ref.patch_id)
        return super().read_patch(ref)


def _module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readers: list[_CountingReader],
    *,
    fingerprint: str = "x",
    mode: str = "stream",
    reuse_ready_cache: bool = False,
) -> RealPatchDataModule:
    samples = {
        ref.patch_id: _sample(ref.patch_id, N_V6, i, ref.split) for i, ref in enumerate(_refs())
    }

    def make_reader(_source: object) -> _CountingReader:
        reader = _CountingReader(samples, N_V6)
        readers.append(reader)
        return reader

    monkeypatch.setattr(real_task, "RealPatchReader", make_reader)
    monkeypatch.setattr(
        real_task,
        "load_patch_refs",
        lambda _source, *, splits, scene_ids: [r for r in _refs() if r.split in splits],
    )
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: {"index": fingerprint})
    return RealPatchDataModule(
        RealSourceConfig("local", "local", "local", "local"),
        batch_size=2,
        n_active_channels=N_V6,
        mode=mode,
        splits=("train", "validation"),
        cache_root=tmp_path / "cache",
        cache_max_bytes=MAX_BYTES,
        reuse_ready_cache=reuse_ready_cache,
    )


def test_ready_cache_is_refused_and_kept_without_explicit_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers: list[_CountingReader] = []
    _module(tmp_path, monkeypatch, readers).setup("fit")
    before = (tmp_path / "cache" / "manifest.json").read_bytes()
    readers.clear()
    second = _module(tmp_path, monkeypatch, readers)
    with pytest.raises(RuntimeError, match="refusing to reuse it across runs"):
        second.setup("fit")
    assert all(reader.reads == [] for reader in readers)
    assert (tmp_path / "cache" / "manifest.json").read_bytes() == before


def test_matching_ready_cache_reopens_without_reading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers: list[_CountingReader] = []
    first = _module(tmp_path, monkeypatch, readers)
    first.setup("fit")
    assert first.stats()["cache_reused"] is False
    first_batch = next(iter(first.train_dataloader()))
    readers.clear()

    second = _module(tmp_path, monkeypatch, readers, reuse_ready_cache=True)
    second.setup("fit")
    assert second.stats()["cache_reused"] is True
    assert all(reader.reads == [] for reader in readers)
    second_batch = next(iter(second.train_dataloader()))
    assert np.array_equal(first_batch.features.numpy(), second_batch.features.numpy())


def test_ready_cache_with_other_provenance_fails_and_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers: list[_CountingReader] = []
    _module(tmp_path, monkeypatch, readers).setup("fit")
    before = (tmp_path / "cache" / "manifest.json").read_bytes()
    other = _module(tmp_path, monkeypatch, readers, fingerprint="changed", reuse_ready_cache=True)
    with pytest.raises(RuntimeError, match="provenance"):
        other.setup("fit")
    assert (tmp_path / "cache" / "manifest.json").read_bytes() == before
    with pytest.raises(RuntimeError):
        other.train_dataloader()


def test_cache_root_without_manifest_is_refused_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "features.npy").write_bytes(b"x")
    module = _module(tmp_path, monkeypatch, [])
    with pytest.raises(RuntimeError, match="without a ready manifest"):
        module.setup("fit")
    assert (tmp_path / "cache" / "features.npy").read_bytes() == b"x"


def test_eager_cached_setup_reads_each_patch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers: list[_CountingReader] = []
    module = _module(tmp_path, monkeypatch, readers, mode="eager")
    module.setup("fit")
    assert sorted(readers[0].reads) == ["t0", "t1", "v0"]
    assert len(list(module.train_dataloader())) == 1
