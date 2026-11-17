"""TESSERA arm under the spatial holdout: the published cache is read-only and re-scaled.

Offline. The fixture is the real-raster world of ``test_parallel_patch_loading`` (22
channels, 1 % NaN features) with a scaler shaped like the real one: the 14 kept channels
are scaled (some ``log1p_zscore``), the two shadow channels are ``identity``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from test_parallel_patch_loading import _build_world, _patch_sources, _World
from test_prebuilt_patch_cache_equivalence import _FINGERPRINTS, _build, _publish
from test_spatial_holdout_run import lifecycle_cfg  # noqa: F401  (a fixture)
from test_spatial_holdout_training import _held_out, _module

from berlin_lst_downscaling.data.embeddings import loader
from berlin_lst_downscaling.data.embeddings.tessera import dequantize_tessera
from berlin_lst_downscaling.data.io import storage
from berlin_lst_downscaling.modeling import prebuilt_cache, real_task
from berlin_lst_downscaling.modeling.contracts import RealSampleMeta
from berlin_lst_downscaling.modeling.patch_cache import PatchCache
from berlin_lst_downscaling.modeling.patches import (
    RealPatchReader,
    RealSample,
    ScalerChannel,
    ScalerSpec,
)
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    pack_prebuilt_cache,
    upload_prebuilt_cache,
)
from berlin_lst_downscaling.modeling.published_cache_remap import (
    FeatureRemap,
    assert_zero_means_filled,
    scan_patch,
)
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule
from berlin_lst_downscaling.modeling.tessera_adapter import (
    TesseraRealLSTTask,
    tessera_kept_channel_indices,
)

ROOT = "gs://fake/tessera"
KEPT = tessera_kept_channel_indices()
# Documented bound: the stored float32 value carries a relative error of 2**-24, which
# the remap scales by std_published / std_refit; the fixture stays far below this bound.
FLOAT32_BOUND = 1e-5
SCALER_TOLERANCE = 1e-6


def _tessera_scaler(width: int = 22, kept_mean_override: dict[int, float] | None = None):
    channels = []
    for i in range(width):
        if i not in KEPT and i >= width - 2:
            channels.append(ScalerChannel(i + 1, f"ch{i}", "identity", None, None))
            continue
        log = i in KEPT and list(KEPT).index(i) % 5 == 4
        mean = (kept_mean_override or {}).get(i, 0.4137 + 0.0313 * i)
        channels.append(
            ScalerChannel(
                i + 1, f"ch{i}", "log1p_zscore" if log else "zscore", mean, 1.3 + 0.01 * i
            )
        )
    return ScalerSpec(tuple(c.name for c in channels), tuple(channels), "test")


@pytest.fixture(scope="module")
def base_world(tmp_path_factory: pytest.TempPathFactory) -> _World:
    return _build_world(tmp_path_factory.mktemp("tessera-holdout-world"), 22)


def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "tessera_window_fully_valid", lambda *a, **k: True)
    monkeypatch.setattr(
        loader,
        "load_patch_quantized",
        lambda *a, **k: (
            np.ones((160, 160, 128), np.int8),
            np.full((160, 160), 0.01, np.float32),
        ),
    )


def _sources(monkeypatch: pytest.MonkeyPatch, world: _World) -> None:
    _patch_sources(monkeypatch, world)
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: dict(_FINGERPRINTS))
    _stub_embeddings(monkeypatch)


def _published(world: _World, tmp: Path, *, tamper: str | None = None) -> RealPatchDataModule:
    """Build the TESSERA cache job-local with the published scaler and publish it."""
    local = _build(world, tuple(range(22)), tmp / "local", tessera_embeddings_root=ROOT)
    local.setup("fit")
    if tamper is None:
        _publish(local, tmp)
        return local
    copy = tmp / "tampered"
    shutil.copytree(tmp / "local", copy)
    manifest_path = copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"][0]["filled_feature_pixels"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    components = local.prebuilt_components()
    pack_prebuilt_cache(
        copy,
        tmp / "packed",
        components=components,
        split_counts=local.stats()["patches_per_split"],  # type: ignore[arg-type]
        shard_max_bytes=200_000,
    )
    upload_prebuilt_cache(
        tmp / "packed", prebuilt_cache.prebuilt_prefix(str(tmp / "gcs"), components)
    )
    return local


def _holdout_module(world: _World, tmp: Path, **kwargs: Any) -> RealPatchDataModule:
    return _build(
        world,
        tuple(range(22)),
        tmp / "holdout",
        tessera_embeddings_root=ROOT,
        cache_source="prebuilt",
        prebuilt_uri_root=str(tmp / "gcs"),
        holdout_anchors=_held_out(world),
        **kwargs,
    )


def test_remap_equals_a_job_local_holdout_build_and_flags_filled_pixels(
    base_world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = dataclasses.replace(base_world, scaler=_tessera_scaler())
    _sources(monkeypatch, world)
    local = _published(world, tmp_path)
    published_key = prebuilt_cache.prebuilt_key(local.prebuilt_components())
    published_bytes = (tmp_path / "local" / "features.npy").read_bytes()

    module = _holdout_module(world, tmp_path)
    module.setup("fit")

    # The published cache identity is untouched and the held-out anchor leaves train only.
    assert module.patch_cache_key == published_key
    assert module.cache_provenance() == local.cache_provenance()
    assert (tmp_path / "local" / "features.npy").read_bytes() == published_bytes
    stats = module.stats()
    assert stats["patch_ids"] == {"train": ["p0", "p2"], "validation": ["p3", "p4"]}
    assert stats["patch_cache_source"] == "prebuilt"

    def ids(loader_: Any) -> list[str]:
        return [m.patch_id for b in loader_ for m in b.metadata]

    assert sorted(ids(module.train_dataloader())) == ["p0", "p2"]
    assert ids(module.val_dataloader()) == ["p3"]
    assert ids(module.holdout_val_dataloader()) == ["p4"]
    block = stats["spatial_holdout"]
    assert (block["train_removed"], block["validation_seen"], block["validation_held_out"]) == (
        1,
        1,
        1,
    )

    # The refit equals the scaler a job-local holdout build computes from the same patches.
    reference = _module(world, holdout=_held_out(world))
    reference.setup("fit")
    assert reference.holdout_scaler is not None and module.holdout_scaler is not None
    for column in range(22):
        got, want = (
            module.holdout_scaler.channels[column],
            reference.holdout_scaler.channels[column],
        )
        if column in KEPT:
            assert got.mean == pytest.approx(want.mean, abs=SCALER_TOLERANCE)
            assert got.std == pytest.approx(want.std, abs=SCALER_TOLERANCE)
        else:
            assert got == world.scaler.channels[column]  # not kept: published entry
    assert block["scaler"] == dataclasses.asdict(module.holdout_scaler)

    # Model input: remapped cache versus a job-local read with the refit scaler.
    exact = RealPatchReader(world.source, scaler=module.holdout_scaler)
    raw = RealPatchReader(world.source, scaler=world.scaler)
    worst_filled = worst_other = 0.0
    n_filled = 0
    datasets = {
        "train": (module._datasets["train"], [world.refs[0], world.refs[2]]),
        "validation": (module._datasets["validation"], [world.refs[3]]),
        "validation_holdout": (module._datasets["validation_holdout"], [world.refs[4]]),
    }
    for dataset, refs in datasets.values():
        for index, ref in enumerate(refs):
            sample = dataset[index]
            want = exact.read_patch(ref)
            assert want is not None
            finite = np.isfinite(raw._read_feature_window(ref))
            diff = np.abs(sample.features[list(KEPT)] - want.features[list(KEPT)])
            keep_finite = finite[list(KEPT)]
            worst_filled = max(worst_filled, float(diff[~keep_finite].max()))
            worst_other = max(worst_other, float(diff[keep_finite].max()))
            n_filled += int((~keep_finite).sum())
    assert n_filled > 0
    assert worst_filled == 0.0
    assert worst_other < FLOAT32_BOUND

    # A loader batch carries the same model input and the cached embeddings.
    batch = next(iter(module.val_dataloader()))
    want = exact.read_patch(world.refs[3])
    assert want is not None
    assert batch.features.shape[1] == 14 and batch.tessera_embedding is not None
    assert np.abs(batch.features.numpy()[0] - want.features[list(KEPT)]).max() < FLOAT32_BOUND
    assert torch.all(batch.tessera_embedding == 0.01)
    window = dequantize_tessera(np.ones((2, 2, 128), np.int8), np.full((2, 2), 0.01, np.float32))
    assert float(window[0, 0, 0]) == pytest.approx(0.01)  # the embedding path is unchanged

    # The check is recorded: patches checked, filled and zero pixels.
    evidence = block["feature_remap"]
    assert evidence["patches_checked"] == 4 and evidence["patches_refit"] == 2
    cache = PatchCache(tmp_path / "holdout", module.cache_provenance())
    recorded = [m["filled_feature_pixels"] for m in cache.manifest["metadata"]]
    assert (
        evidence["filled_pixels_recorded"] == recorded[0] + recorded[2] + recorded[3] + recorded[4]
    )
    assert 0 < evidence["zero_pixels_kept_channels"] <= evidence["filled_pixels_recorded"]
    assert evidence["patch_cache_key"] == published_key


def test_a_training_step_opens_no_cloud_storage(
    base_world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = dataclasses.replace(base_world, scaler=_tessera_scaler())
    _sources(monkeypatch, world)
    _published(world, tmp_path)
    module = _holdout_module(world, tmp_path)
    module.setup("fit")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not open cloud storage")

    for target in ("_year_dataset", "load_patch_embedding", "load_patch_quantized"):
        monkeypatch.setattr(loader, target, forbidden)
    monkeypatch.setattr("rasterio.open", forbidden)
    monkeypatch.setattr(storage, "read_bytes", forbidden)
    task = TesseraRealLSTTask(base_width=4, depth=2)
    task.log = lambda *args, **kwargs: None
    loss = task.training_step(next(iter(module.train_dataloader())), 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_a_wrong_filled_count_aborts_the_load_naming_patch_and_counts(
    base_world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = dataclasses.replace(base_world, scaler=_tessera_scaler())
    _sources(monkeypatch, world)
    _published(world, tmp_path, tamper="filled")
    module = _holdout_module(world, tmp_path)
    with pytest.raises(RuntimeError, match=r"patch p0: \d+ stored zeros .* only 0 recorded"):
        module.setup("fit")


def test_a_kept_mean_the_input_can_take_aborts_the_load(
    base_world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = dataclasses.replace(
        base_world, scaler=_tessera_scaler(kept_mean_override={KEPT[0]: 0.5})
    )
    _sources(monkeypatch, world)
    _published(world, tmp_path)
    module = _holdout_module(world, tmp_path)
    with pytest.raises(RuntimeError, match="published mean 0.5"):
        module.setup("fit")


def test_zero_means_filled_proof_per_channel() -> None:
    spec = _tessera_scaler()
    assert_zero_means_filled(spec, tuple(range(22)), KEPT)

    def with_channel(index: int, **changes: Any) -> ScalerSpec:
        channels = list(spec.channels)
        channels[index] = dataclasses.replace(channels[index], **changes)
        return dataclasses.replace(spec, channels=tuple(channels))

    column = KEPT[0]
    log_column = KEPT[4]
    assert spec.channels[log_column].transform == "log1p_zscore"
    bad = {
        "float32-representable mean": with_channel(column, mean=0.25),
        "zero mean": with_channel(column, mean=0.0),
        "log1p of a float32 value": with_channel(
            log_column, mean=float(np.log1p(np.float64(np.float32(0.3))))
        ),
        "no std": with_channel(column, std=0.0),
        "no mean": with_channel(column, mean=None),
        "identity kept": with_channel(column, transform="identity", mean=None, std=None),
    }
    for name, broken in bad.items():
        with pytest.raises(RuntimeError, match="kept channel"):
            assert_zero_means_filled(broken, tuple(range(22)), KEPT)
            pytest.fail(name)


def test_scan_patch_flags_a_stored_zero_beyond_the_recorded_fills() -> None:
    features = np.ones((22, 160, 160), np.float32)
    features[KEPT[0], 0, :3] = 0.0  # three filled pixels in a kept channel
    meta = RealSampleMeta("p", "s", "train", 2018, 0, 0, 3)
    sample = RealSample(
        meta,
        features,
        np.zeros((1, 160, 160), np.float32),
        np.zeros((1, 16, 16), np.float32),
        np.ones((1, 16, 16), bool),
    )
    assert scan_patch(sample, KEPT) == (3, 3)
    features[KEPT[1], 5, 5] = 0.0  # one genuine zero: more zeros than fills
    with pytest.raises(RuntimeError, match=r"patch p: 4 stored zeros .* only 3 recorded"):
        scan_patch(sample, KEPT)


def test_remap_is_the_identity_when_the_scaler_does_not_change() -> None:
    spec = _tessera_scaler()
    chans = [spec.channels[c] for c in KEPT]
    remap = FeatureRemap(
        KEPT,
        tuple(c.mean for c in chans),
        tuple(c.std for c in chans),
        tuple(c.mean for c in chans),
        tuple(c.std for c in chans),
    )
    rng = np.random.default_rng(0)
    features = rng.normal(size=(22, 8, 8)).astype(np.float32)
    features[KEPT[2], 1, 1] = 0.0
    out = remap(features)
    assert out[KEPT[2], 1, 1] == 0.0
    assert np.abs(out - features).max() < 1e-6


def test_order_of_operations_differs_from_the_v6_holdout_and_both_refuse_the_wrong_cache(
    base_world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = dataclasses.replace(base_world, scaler=_tessera_scaler())
    _sources(monkeypatch, world)
    held = _held_out(world)
    # v6 holdout: the held-out train ref is dropped before admission, the scaler refit first.
    v6 = _module(world, holdout=held)
    v6.setup("fit")
    assert [r.patch_id for r in v6._requested["train"]] == ["p0", "p2"]
    assert v6.holdout_scaler is not None and v6.reader.scaler == v6.holdout_scaler
    # TESSERA holdout: the full train split is admitted with the published scaler (the key of
    # the published cache), the held-out rows are dropped after the cache is verified.
    _published(world, tmp_path)
    module = _holdout_module(world, tmp_path)
    module.setup("fit")
    assert [r.patch_id for r in module._requested["train"]] == ["p0", "p1", "p2"]
    assert module.reader is not None and module.reader.scaler == world.scaler
    # the TESSERA holdout never builds a cache job-local, and has no fallback to it
    with pytest.raises(ValueError, match="prebuilt patch cache without a job-local fallback"):
        _module(world, holdout=held, tessera_embeddings_root=ROOT)
    with pytest.raises(ValueError, match="prebuilt patch cache without a job-local fallback"):
        _holdout_module(world, tmp_path / "x", allow_job_local_fallback=True)


def test_fit_selects_on_seen_anchors_scores_unseen_after_and_records_the_remap(
    base_world: _World,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lifecycle_cfg: Any,  # noqa: F811
) -> None:
    from berlin_lst_downscaling.modeling import run as run_module

    world = dataclasses.replace(base_world, scaler=_tessera_scaler())
    _sources(monkeypatch, world)
    _published(world, tmp_path)
    module = _holdout_module(world, tmp_path)
    lifecycle_cfg.data.tessera_enabled = True
    lifecycle_cfg.data.tessera_embeddings_root = ROOT
    lifecycle_cfg.model.base_width = 4
    lifecycle_cfg.model.depth = 2
    info = {"assignment": "unit", "holdout_anchors": 2}
    result = run_module._fit_contract_lifecycle(
        lifecycle_cfg,
        "tessera-holdout",
        data_module=module,
        source_metadata={"spatial_holdout": info},
        reproducibility="test",
    )
    assert result.run_ok
    root = Path(str(lifecycle_cfg.output_root))
    summary = json.loads((root / "full_summary.json").read_text())
    rows = {
        row["patch_id"]: row
        for row in pq.read_table(root / "eval_validation_patches.parquet").to_pylist()
    }
    assert set(rows) == {"p3", "p4"}  # seen and held-out validation patches, scored once
    assert summary["validation_report"]["valid_cells"] == rows["p3"]["valid_cells"]
    seen_mae = rows["p3"]["abs_error_sum"] / rows["p3"]["valid_cells"]
    assert summary["best_metric"] == pytest.approx(seen_mae, rel=1e-3)
    assert summary["test"] == {"skipped": True}
    block = summary["spatial_holdout"]
    assert block["feature_remap"]["patches_checked"] == 4
    assert block["feature_remap"]["patch_cache_key"] == module.patch_cache_key
    assert block["scaler"] == json.loads(json.dumps(dataclasses.asdict(module.holdout_scaler)))
