"""Parallel job-local cache build: equivalence with serial, ordering, retries, setup state."""

from __future__ import annotations

import json
import random
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine

from berlin_lst_downscaling.modeling import patches, real_task
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
)
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    ordered_bounded_map,
    read_with_retry,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSourceConfig,
    ScalerChannel,
    ScalerSpec,
)
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule

MAX_BYTES = 1 << 30
CELLS = REAL_PATCH_CELLS
FEATURE_PX = 320  # two patches wide at 10 m
_TRANSFORMS = ("identity", "zscore", "log1p_zscore")

# (scene, split, row offset, col offset) in 100 m cells from the analysis origin.
_LAYOUT = [
    ("SCN_A", "train", 0, 0),
    ("SCN_A", "train", CELLS, 0),
    ("SCN_A", "train", 0, CELLS),
    ("SCN_B", "validation", 0, 0),
    ("SCN_B", "validation", CELLS, CELLS),
]


def _scaler(width: int) -> ScalerSpec:
    names = tuple(f"ch{i}" for i in range(width))
    return ScalerSpec(
        channel_order=names,
        channels=tuple(
            ScalerChannel(
                index=i + 1,
                name=n,
                transform=_TRANSFORMS[i % 3],
                mean=None if i % 3 == 0 else 0.5 + 0.1 * i,
                std=None if i % 3 == 0 else 1.5,
            )
            for i, n in enumerate(names)
        ),
        policy_hash="test",
    )


@dataclass
class _World:
    source: RealSourceConfig
    refs: list[PatchRef]
    uris: dict[str, tuple[str, str]]
    scaler: ScalerSpec
    tmp: Path


def _write(path: Path, array: np.ndarray, transform: Affine, dtype: str) -> None:
    array = array if array.ndim == 3 else array[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=dtype,
        crs="EPSG:25833",
        transform=transform,
    ) as dst:
        dst.write(array.astype(dtype))


def _build_world(tmp: Path, width: int) -> _World:
    rng = np.random.default_rng(width)
    reader = RealPatchReader(RealSourceConfig("l", "l", "l", "l"), scaler=_scaler(width))
    t100 = reader.analysis_100.transform
    t10 = reader.analysis_10.transform
    dcol100, drow100 = reader.analysis_offsets[2:]
    features_root = tmp / "features"
    uris: dict[str, tuple[str, str]] = {}
    masks: dict[str, Path] = {}
    size = 2 * CELLS
    for scene in ("SCN_A", "SCN_B"):
        mask = (rng.random((size, size)) < 0.7).astype(np.uint8)
        mask[:CELLS, :CELLS] = 1  # one fully eligible patch per scene
        lst = (295.0 + 5.0 * rng.random((size, size))).astype(np.float32)
        lst[(mask == 0) & (rng.random((size, size)) < 0.5)] = np.nan
        cog, flag, mask_path = (
            tmp / f"{scene}_lst.tif",
            tmp / f"{scene}_flag.tif",
            tmp / f"{scene}_m.tif",
        )
        _write(cog, lst, t100, "float32")
        _write(flag, np.zeros((size, size), dtype=np.uint8), t100, "uint8")
        _write(mask_path, mask, t100, "uint8")
        uris[scene] = (str(cog), str(flag))
        masks[scene] = mask_path
        feats = np.abs(rng.normal(size=(width, FEATURE_PX, FEATURE_PX))).astype(np.float32)
        feats[rng.random(feats.shape) < 0.01] = np.nan
        (features_root / scene).mkdir(parents=True)
        _write(features_root / scene / f"{scene}.tif", feats, t10, "float32")
    refs = []
    for i, (scene, split, dr, dc) in enumerate(_LAYOUT):
        with rasterio.open(masks[scene]) as src:
            window = rasterio.windows.Window(dc, dr, CELLS, CELLS)
            n_eligible = int((src.read(1, window=window) == 1).sum())
        refs.append(
            PatchRef(
                f"p{i}", scene, "S2", split, 2018 if split == "train" else 2024,
                drow100 + dr, dcol100 + dc, n_eligible, str(masks[scene]),
            )
        )  # fmt: skip
    source = RealSourceConfig("idx", "trn", str(features_root), "ard")
    return _World(source, refs, uris, _scaler(width), tmp)


@pytest.fixture(scope="module", params=[22, 28], ids=["w22", "w28"])
def world(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> _World:
    return _build_world(tmp_path_factory.mktemp("world"), request.param)


def _patch_sources(monkeypatch: pytest.MonkeyPatch, world: _World) -> None:
    monkeypatch.setattr(
        patches,
        "load_landsat_uris",
        lambda _root, ids: {s: world.uris[s] for s in ids if s in world.uris},
    )
    monkeypatch.setattr(
        real_task, "RealPatchReader", lambda source: RealPatchReader(source, scaler=world.scaler)
    )
    monkeypatch.setattr(
        real_task,
        "load_patch_refs",
        lambda _source, *, splits, scene_ids: [r for r in world.refs if r.split in splits],
    )
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: {"index": "x"})


def _module(
    world: _World,
    indices: tuple[int, ...],
    *,
    cache_root: Path | None = None,
    workers: int = 1,
) -> RealPatchDataModule:
    return RealPatchDataModule(
        world.source,
        batch_size=2,
        n_active_channels=len(indices),
        channel_indices=indices,
        mode="stream",
        shuffle_train=True,
        seed=7,
        splits=("train", "validation"),
        cache_root=cache_root,
        cache_max_bytes=MAX_BYTES,
        cache_build_workers=workers,
    )


def _epochs(module: RealPatchDataModule) -> list[list[tuple[object, ...]]]:
    out = []
    for loader in (module.train_dataloader(), module.val_dataloader()):
        for _ in range(2):
            out.append(
                [(b.features, b.lst_prior, b.target_100m, b.mask_100m, b.metadata) for b in loader]
            )
    return out


def _assert_same_epochs(left: list, right: list) -> None:
    assert len(left) == len(right)
    for l_epoch, r_epoch in zip(left, right, strict=True):
        assert len(l_epoch) == len(r_epoch)
        for l_batch, r_batch in zip(l_epoch, r_epoch, strict=True):
            for a, b in zip(l_batch[:4], r_batch[:4], strict=True):
                assert a.dtype == b.dtype
                assert np.array_equal(a.numpy(), b.numpy(), equal_nan=a.is_floating_point())
            assert l_batch[4] == r_batch[4]


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("selection", ["contiguous", "scattered"])
def test_parallel_cache_matches_serial_cache_and_uncached(
    world: _World,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workers: int,
    selection: str,
) -> None:
    _patch_sources(monkeypatch, world)
    width = len(world.scaler.channels)
    indices = tuple(range(width)) if selection == "contiguous" else (0, 3, 7, width - 1)
    uncached = _module(world, indices)
    uncached.setup("fit")
    serial = _module(world, indices, cache_root=tmp_path / "serial", workers=1)
    serial.setup("fit")
    parallel = _module(world, indices, cache_root=tmp_path / "parallel", workers=workers)
    parallel.setup("fit")

    reference = _epochs(uncached)
    _assert_same_epochs(reference, _epochs(serial))
    _assert_same_epochs(reference, _epochs(parallel))
    # 3 train rows with batch 2 -> a final incomplete batch must be present.
    assert [len(b[4]) for b in reference[0]] == [2, 1]

    a = PatchCache(tmp_path / "serial", _provenance(tmp_path / "serial"))
    b = PatchCache(tmp_path / "parallel", _provenance(tmp_path / "parallel"))
    assert a.patch_ids == b.patch_ids == tuple(r.patch_id for r in world.refs)
    assert a.manifest["metadata"] == b.manifest["metadata"]
    for key in a.arrays:
        assert a.arrays[key].tobytes() == b.arrays[key].tobytes()
    assert parallel.stats()["cache_build_workers"] == workers
    assert parallel.stats()["train_shuffle_order"] == uncached.stats()["train_shuffle_order"]


def _provenance(root: Path) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_text())["provenance"]


def test_partial_nodata_is_preserved(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, world)
    reader = RealPatchReader(world.source, scaler=world.scaler)
    reader.preload({r.scene_id for r in world.refs})
    samples = [reader.read_patch(r) for r in world.refs]
    assert all(s is not None for s in samples)
    assert any(s.meta.filled_feature_pixels > 0 for s in samples if s)
    partial_masks = [s for s in samples if s and 0 < s.mask_100m.sum() < CELLS * CELLS]
    assert partial_masks
    for s in samples:
        assert s is not None
        assert np.isfinite(s.features).all()
        assert np.isfinite(s.target_100m[s.mask_100m]).all()
    assert any(np.isnan(s.target_100m[~s.mask_100m]).any() for s in samples if s)
    ref = world.refs[0]
    with rasterio.open(world.source.features_root + f"/{ref.scene_id}/{ref.scene_id}.tif") as src:
        raw = src.read(2, window=rasterio.windows.Window(0, 0, REAL_PATCH_PX, REAL_PATCH_PX))
    ch = world.scaler.channels[1]
    finite = np.isfinite(raw)
    assert ch.mean is not None and ch.std is not None
    expected = (raw[finite].astype(np.float64) - ch.mean) / ch.std
    assert np.allclose(samples[0].features[1][finite], expected, atol=1e-5)  # type: ignore[union-attr]
    assert (samples[0].features[1][~finite] == 0).all()  # type: ignore[union-attr]


def test_reader_copy_reuses_scene_state_without_remote_reads(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sources(monkeypatch, world)
    reader = RealPatchReader(world.source, scaler=world.scaler)
    reader.preload({r.scene_id for r in world.refs})
    for ref in world.refs:
        assert reader.admission_reason(ref) is None
    reader.exclusions["x"] += 1
    copy = reader.copy_for_reads()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("remote ledger read")

    monkeypatch.setattr(patches, "load_landsat_uris", forbidden)
    assert copy.scaler is reader.scaler
    assert copy._priors is not reader._priors and copy._priors == reader._priors
    assert copy._landsat is not reader._landsat
    assert copy._landsat_queried is not reader._landsat_queried
    assert all(copy._priors[k] is v for k, v in reader._priors.items())
    assert copy.exclusions == Counter()
    sample = copy.read_patch(world.refs[0])
    expected = reader.read_patch(world.refs[0])
    assert sample is not None and expected is not None
    assert np.array_equal(sample.features, expected.features)
    copy.exclusions["y"] += 1
    copy._priors.clear()
    assert "y" not in reader.exclusions and reader._priors


def test_cache_prep_does_not_consume_rng(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _module(world, (0, 1, 2), cache_root=tmp_path / "c", workers=4).setup("fit")
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_skip_test_requests_only_train_and_validation(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    seen: list[tuple[str, ...]] = []

    def refs(_source: object, *, splits: tuple[str, ...], scene_ids: object) -> list[PatchRef]:
        seen.append(tuple(splits))
        if "test" in splits:
            raise AssertionError("test source accessed")
        return [r for r in world.refs if r.split in splits]

    monkeypatch.setattr(real_task, "load_patch_refs", refs)
    module = _module(world, (0, 1), cache_root=tmp_path / "c", workers=2)
    module.setup("fit")
    assert seen == [("train", "validation")]
    assert set(module.stats()["requested_per_split"]) == {"train", "validation"}  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        module.test_dataloader()


def test_setup_is_idempotent_and_failed_setup_can_retry(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    original = real_task.load_patch_refs
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> list[PatchRef]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient index read")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(real_task, "load_patch_refs", flaky)
    module = _module(world, (0, 1), cache_root=tmp_path / "c", workers=2)
    with pytest.raises(OSError, match="transient"):
        module.setup("fit")
    assert module.reader is None and module._datasets == {} and module._admitted == {}
    module.setup("fit")
    stats = module.stats()
    module.setup("fit")
    assert calls["n"] == 2
    assert module.stats() == stats
    assert stats["patches_per_split"] == {"train": 3, "validation": 2}


def test_failed_cache_build_resets_for_a_retry(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    sleeps: list[float] = []
    monkeypatch.setattr(real_task, "read_with_retry", partial(read_with_retry, sleep=sleeps.append))
    real_read = RealPatchReader.read_patch
    failing = {"on": True}

    def read(self: RealPatchReader, ref: PatchRef) -> RealSample | None:
        if failing["on"] and ref.patch_id == "p3":
            raise RasterioIOError("flaky")
        return real_read(self, ref)

    monkeypatch.setattr(RealPatchReader, "read_patch", read)
    module = _module(world, (0, 1), cache_root=tmp_path / "c", workers=3)
    with pytest.raises(RuntimeError, match=r"p3.*3 attempts") as info:
        module.setup("fit")
    assert isinstance(info.value.__cause__, RasterioIOError)
    assert sleeps == [0.5, 1.0]
    assert not (tmp_path / "c").exists()
    assert not [p for p in tmp_path.iterdir() if "staging" in p.name]
    with pytest.raises(RuntimeError):
        module.train_dataloader()
    failing["on"] = False
    module.setup("fit")
    assert len(list(module.train_dataloader())) == 2


# ── ordered_bounded_map ────────────────────────────────────────────────


def test_ordered_map_is_ordered_despite_out_of_order_completion() -> None:
    done: list[int] = []
    lock = threading.Lock()
    later_done = threading.Event()

    def fn(i: int) -> int:
        if i == 0:
            assert later_done.wait(5)
        with lock:
            done.append(i)
        if i == 1:
            later_done.set()
        return i * 10

    assert list(ordered_bounded_map(fn, range(6), max_workers=2)) == [0, 10, 20, 30, 40, 50]
    assert done.index(1) < done.index(0)


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_ordered_map_bounds_active_and_unconsumed_work(workers: int) -> None:
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "pulled": 0, "consumed": 0, "max_unconsumed": 0}

    def items():
        for i in range(40):
            with lock:
                state["pulled"] += 1
                state["max_unconsumed"] = max(
                    state["max_unconsumed"], state["pulled"] - state["consumed"]
                )
            yield i

    def fn(i: int) -> int:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        threading.Event().wait(0.001)
        with lock:
            state["active"] -= 1
        return i

    out = []
    for value in ordered_bounded_map(fn, items(), max_workers=workers):
        with lock:
            state["consumed"] += 1
        out.append(value)
    assert out == list(range(40))
    assert state["max_active"] <= workers
    assert state["max_unconsumed"] <= 2 * workers + 1  # +1: the item being yielded


def test_ordered_map_early_close_stops_submitting() -> None:
    pulled: list[int] = []

    def items():
        for i in range(1000):
            pulled.append(i)
            yield i

    gen = ordered_bounded_map(lambda i: i, items(), max_workers=2)
    assert next(gen) == 0
    gen.close()
    assert len(pulled) <= 6
    assert not [t for t in threading.enumerate() if t.name.startswith("patch-read")]


def test_ordered_map_propagates_failure_and_stops() -> None:
    def fn(i: int) -> int:
        if i == 2:
            raise ValueError("bad item")
        return i

    gen = ordered_bounded_map(fn, range(100), max_workers=2)
    assert [next(gen), next(gen)] == [0, 1]
    with pytest.raises(ValueError, match="bad item"):
        next(gen)
    assert not [t for t in threading.enumerate() if t.name.startswith("patch-read")]


@pytest.mark.parametrize("bad", [0, 5, True, 2.0, "2", None])
def test_ordered_map_rejects_bad_worker_counts(bad: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        next(ordered_bounded_map(lambda i: i, [1], max_workers=bad))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, 5, True, 2.0, "2", None])
def test_datamodule_rejects_bad_worker_counts(bad: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        RealPatchDataModule(
            RealSourceConfig("l", "l", "l", "l"),
            cache_build_workers=bad,  # type: ignore[arg-type]
        )


# ── retries ────────────────────────────────────────────────────────────


def _flaky(failures: int, error: Exception) -> tuple[Callable[[], str], list[int]]:
    calls: list[int] = []

    def read() -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return "ok"

    return read, calls


@pytest.mark.parametrize("error", [RasterioIOError("x"), TimeoutError("x"), ConnectionError("x")])
def test_transient_errors_retry_with_fixed_backoff(error: Exception) -> None:
    read, calls = _flaky(2, error)
    sleeps: list[float] = []
    assert read_with_retry(read, patch_id="p1", sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


def test_retry_exhaustion_names_patch_and_attempts() -> None:
    read, calls = _flaky(99, RasterioIOError("gone"))
    sleeps: list[float] = []
    with pytest.raises(RuntimeError, match=r"p9.*3 attempts") as info:
        read_with_retry(read, patch_id="p9", sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    assert isinstance(info.value.__cause__, RasterioIOError)


@pytest.mark.parametrize("error", [RuntimeError("contract"), ValueError("shape"), KeyError("k")])
def test_contract_errors_are_not_retried(error: Exception) -> None:
    read, calls = _flaky(99, error)
    sleeps: list[float] = []
    with pytest.raises(type(error)):
        read_with_retry(read, patch_id="p1", sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_unreadable_none_is_not_retried() -> None:
    calls: list[int] = []

    def read() -> None:
        calls.append(1)

    with pytest.raises(RuntimeError, match="p4"):
        read_with_retry(read, patch_id="p4", sleep=lambda _s: None)
    assert len(calls) == 1
