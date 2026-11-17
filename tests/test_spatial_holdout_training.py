"""Spatial holdout training option: patch-limited scaler fit and the split data module."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_parallel_patch_loading import MAX_BYTES, _build_world, _patch_sources, _World

from berlin_lst_downscaling.data.training.scaler import _Welford
from berlin_lst_downscaling.modeling.contracts import REAL_PATCH_CELLS
from berlin_lst_downscaling.modeling.patch_cache import PatchCache
from berlin_lst_downscaling.modeling.patches import (
    RealPatchReader,
    ScalerChannel,
    ScalerSpec,
    merge_scaler_moments,
    scaler_from_moments,
    scaler_moments,
)
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule

CELLS = REAL_PATCH_CELLS


def _spec() -> ScalerSpec:
    names = ("ident", "plain", "logged")
    transforms = ("identity", "zscore", "log1p_zscore")
    return ScalerSpec(
        channel_order=names,
        channels=tuple(
            ScalerChannel(index=i + 1, name=n, transform=t, mean=9.0, std=9.0)
            for i, (n, t) in enumerate(zip(names, transforms, strict=True))
        ),
        policy_hash="policy",
    )


# ------------------------------------------------------------------ moments


def test_merged_moments_equal_the_pooled_statistics_and_the_published_welford() -> None:
    rng = np.random.default_rng(0)
    spec = _spec()
    patches = []
    for _ in range(5):
        bands = np.abs(rng.normal(3.0, 2.0, size=(3, 20, 20))).astype(np.float32)
        bands[rng.random(bands.shape) < 0.05] = np.nan
        eligible = rng.random((20, 20)) < 0.7
        patches.append((bands, eligible))
    total = None
    welford = [_Welford(), _Welford()]
    pooled: list[list[np.ndarray]] = [[], []]
    for bands, eligible in patches:
        total = merge_scaler_moments(total, scaler_moments(bands, eligible, spec))
        for k, (channel, log) in enumerate(((1, False), (2, True))):
            band = bands[channel]
            vals = band[eligible & np.isfinite(band)].astype(np.float64)
            vals = np.log1p(vals) if log else vals
            welford[k].update(vals)
            pooled[k].append(vals)
    assert total is not None
    fitted = scaler_from_moments(spec, total)
    assert fitted.channel_order == spec.channel_order and fitted.policy_hash == "policy"
    assert fitted.channels[0] == spec.channels[0]  # identity channel is left alone
    assert total[0, 0] == sum(int(e.sum()) for _b, e in patches)
    for k, channel in enumerate((1, 2)):
        values = np.concatenate(pooled[k])
        assert fitted.channels[channel].mean == pytest.approx(values.mean(), abs=1e-8)
        assert fitted.channels[channel].std == pytest.approx(values.std(), abs=1e-8)
        assert fitted.channels[channel].mean == pytest.approx(welford[k].mean, abs=1e-8)
        assert fitted.channels[channel].std == pytest.approx(welford[k].std(), abs=1e-8)


def test_moments_reject_wrong_shapes_and_an_unfittable_channel() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="do not match"):
        scaler_moments(np.zeros((2, 4, 4), np.float32), np.ones((4, 4), bool), spec)
    empty = scaler_moments(np.ones((3, 4, 4), np.float32), np.zeros((4, 4), bool), spec)
    with pytest.raises(RuntimeError, match="no eligible finite pixel"):
        scaler_from_moments(spec, empty)


# -------------------------------------------------------------- data module


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> _World:
    return _build_world(tmp_path_factory.mktemp("holdout-world"), 22)


def _held_out(world: _World) -> frozenset[tuple[int, int]]:
    # Layout: train p0 (0,0), p1 (16,0), p2 (0,16); validation p3 (0,0), p4 (16,16).
    row0, col0 = world.refs[0].row, world.refs[0].col
    return frozenset({(row0 + CELLS, col0), (row0 + CELLS, col0 + CELLS)})


def _module(
    world: _World,
    *,
    holdout: frozenset[tuple[int, int]] | None,
    cache_root: Path | None = None,
    workers: int = 1,
    **kwargs: object,
) -> RealPatchDataModule:
    width = len(world.scaler.channels)
    options: dict = {
        "batch_size": 2,
        "n_active_channels": width,
        "channel_indices": tuple(range(width)),
        "mode": "stream",
        "seed": 7,
        "splits": ("train", "validation"),
        "cache_root": cache_root,
        "cache_max_bytes": MAX_BYTES,
        "cache_build_workers": workers,
        "holdout_anchors": holdout,
    }
    options.update(kwargs)
    return RealPatchDataModule(world.source, **options)


def _ids(loader) -> list[str]:  # noqa: ANN001
    return [meta.patch_id for batch in loader for meta in batch.metadata]


def _features(loader) -> np.ndarray:  # noqa: ANN001
    return np.concatenate([batch.features.numpy() for batch in loader])


def test_option_off_leaves_the_module_unchanged(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sources(monkeypatch, world)
    module = _module(world, holdout=None)
    module.setup("fit")
    stats = module.stats()
    assert "spatial_holdout" not in stats
    assert stats["patch_ids"] == {"train": ["p0", "p1", "p2"], "validation": ["p3", "p4"]}
    assert module.reader is not None and module.reader.scaler == world.scaler
    assert _ids(module.val_dataloader()) == ["p3", "p4"]
    with pytest.raises(RuntimeError, match="requires a spatial holdout"):
        module.holdout_val_dataloader()


@pytest.mark.parametrize("cached", [False, True], ids=["live", "cached"])
def test_holdout_drops_train_anchors_refits_the_scaler_and_splits_validation(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool
) -> None:
    _patch_sources(monkeypatch, world)
    module = _module(
        world, holdout=_held_out(world), cache_root=tmp_path / "cache" if cached else None
    )
    module.setup("fit")
    assert sorted(_ids(module.train_dataloader())) == ["p0", "p2"]
    assert _ids(module.val_dataloader()) == ["p3"]  # checkpoint selection: seen anchors
    assert _ids(module.holdout_val_dataloader()) == ["p4"]
    stats = module.stats()
    assert stats["patch_ids"] == {"train": ["p0", "p2"], "validation": ["p3", "p4"]}
    block = stats["spatial_holdout"]
    assert isinstance(block, dict)
    assert (block["train_removed"], block["validation_seen"], block["validation_held_out"]) == (
        1,
        1,
        1,
    )

    # The scaler is fitted on the two remaining train patches only.
    reference = RealPatchReader(world.source, scaler=world.scaler)
    total = None
    for ref in (world.refs[0], world.refs[2]):
        total = merge_scaler_moments(total, reference.read_scaler_moments(ref))
    assert total is not None
    expected = scaler_from_moments(world.scaler, total)
    assert module.holdout_scaler == expected != world.scaler
    assert block["scaler"] == asdict(expected)
    with_removed = merge_scaler_moments(total, reference.read_scaler_moments(world.refs[1]))
    assert scaler_from_moments(world.scaler, with_removed) != expected

    # Every loader scales with the refit, on held-out patches too.
    refit_reader = RealPatchReader(world.source, scaler=expected)
    sample = refit_reader.read_patch(world.refs[4])
    assert sample is not None
    assert np.array_equal(_features(module.holdout_val_dataloader())[0], sample.features)
    published = RealPatchReader(world.source, scaler=world.scaler).read_patch(world.refs[4])
    assert published is not None and not np.array_equal(published.features, sample.features)
    if cached:
        manifest = PatchCache(tmp_path / "cache", module.cache_provenance()).manifest
        plain = _module(world, holdout=None, cache_root=tmp_path / "plain")
        plain.setup("fit")
        assert manifest["provenance"]["scaler_digest"] != plain.cache_provenance()["scaler_digest"]


def test_holdout_scaler_and_tensors_do_not_depend_on_cache_or_worker_count(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    held = _held_out(world)
    live = _module(world, holdout=held)
    serial = _module(world, holdout=held, cache_root=tmp_path / "serial", workers=1)
    threaded = _module(world, holdout=held, cache_root=tmp_path / "threaded", workers=4)
    for module in (live, serial, threaded):
        module.setup("fit")
    assert live.holdout_scaler == serial.holdout_scaler == threaded.holdout_scaler
    for name in ("val_dataloader", "holdout_val_dataloader"):
        reference = _features(getattr(live, name)())
        assert np.array_equal(reference, _features(getattr(serial, name)()))
        assert np.array_equal(reference, _features(getattr(threaded, name)()))


def test_holdout_fails_closed_on_unsupported_setups(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, world)
    held = _held_out(world)
    with pytest.raises(ValueError, match="non-empty"):
        _module(world, holdout=frozenset())
    with pytest.raises(ValueError, match="job-local"):
        _module(
            world,
            holdout=held,
            cache_root=tmp_path / "c",
            cache_source="prebuilt",
            prebuilt_uri_root="gs://bucket/patch-cache",
        )
    with pytest.raises(ValueError, match="splits"):
        _module(world, holdout=held, splits=("train", "validation", "test"))
    with pytest.raises(ValueError, match="TESSERA arm needs the prebuilt"):
        _module(world, holdout=held, tessera_embeddings_root="gs://bucket/embeddings")
    row0, col0 = world.refs[0].row, world.refs[0].col
    nothing_unseen = _module(world, holdout=frozenset({(row0 + CELLS, col0)}))
    with pytest.raises(RuntimeError, match="validation patches on both sides"):
        nothing_unseen.setup("fit")
