"""Prebuilt patch cache: identity key, row-aligned shards, verified fetch, upload order.

A temporary directory stands in for the GCS prefix; every storage helper accepts
local paths, so no network or credentials are involved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_patch_cache_source_channels import _sample

from berlin_lst_downscaling.data.io import storage
from berlin_lst_downscaling.modeling import prebuilt_cache
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, build_patch_cache
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    CACHE_FILES,
    PREBUILT_MANIFEST,
    PrebuiltCacheChecksumError,
    PrebuiltCacheError,
    PrebuiltCacheMismatch,
    PrebuiltCacheMissing,
    fetch_prebuilt_cache,
    pack_prebuilt_cache,
    prebuilt_key,
    prebuilt_key_components,
    prebuilt_manifest_exists,
    prebuilt_prefix,
    upload_prebuilt_cache,
)

MAX_BYTES = 1 << 30
N_TRAIN = 3
N_VAL = 2


def _provenance(width: int, indices: tuple[int, ...]) -> dict[str, Any]:
    return {
        "source_roots": {"patch_index_root": "local"},
        "patch_index_fingerprints": {
            "index_policy_hash": "idx-hash",
            "source_policy_hash": "src-hash",
            "patches_total": "5",
            "published_at": "t",
        },
        "scaler_digest": "scaler-digest",
        "channel_order": [f"c{i}" for i in indices],
        "channel_indices": list(indices),
        "source_channels": width,
        "geometry": {"anchors": [[i, 0] for i in range(N_TRAIN + N_VAL)]},
    }


def _order(width: int) -> list[str]:
    return [f"c{i}" for i in range(width)]


def _make_cache(
    root: Path, width: int = 22, indices: tuple[int, ...] | None = None
) -> tuple[Path, dict[str, Any], dict[str, object], dict[str, int]]:
    indices = tuple(range(width)) if indices is None else indices
    provenance = _provenance(width, indices)
    splits = ["train"] * N_TRAIN + ["validation"] * N_VAL
    samples = [_sample(f"p{i}", width, i, split) for i, split in enumerate(splits)]
    cache_dir = root / "cache"
    build_patch_cache(
        cache_dir,
        iter(samples),
        patch_ids=[s.meta.patch_id for s in samples],
        active_channels=len(indices),
        source_channels=width,
        provenance=provenance,
        max_bytes=MAX_BYTES,
        channel_indices=indices,
    )
    components = prebuilt_key_components(
        provenance, source_channel_order=_order(width), splits=("train", "validation")
    )
    return cache_dir, provenance, components, {"train": N_TRAIN, "validation": N_VAL}


@pytest.fixture
def built(tmp_path: Path) -> dict[str, Any]:
    cache_dir, provenance, components, counts = _make_cache(tmp_path)
    out_dir = tmp_path / "prebuilt"
    manifest = pack_prebuilt_cache(
        cache_dir, out_dir, components=components, split_counts=counts, shard_max_bytes=1
    )
    prefix = prebuilt_prefix(str(tmp_path / "gcs"), components)
    return {
        "cache_dir": cache_dir,
        "provenance": provenance,
        "components": components,
        "counts": counts,
        "out_dir": out_dir,
        "manifest": manifest,
        "prefix": prefix,
        "root": tmp_path,
    }


@pytest.fixture
def uploaded(built: dict[str, Any]) -> dict[str, Any]:
    upload_prebuilt_cache(built["out_dir"], built["prefix"])
    return built


# --- key ---------------------------------------------------------------------


def _key(provenance: dict[str, Any], order: list[str], splits: tuple[str, ...]) -> str:
    return prebuilt_key(
        prebuilt_key_components(provenance, source_channel_order=order, splits=splits)
    )


def _mutate(path: tuple[str, ...], value: object) -> dict[str, Any]:
    provenance = _provenance(22, tuple(range(22)))
    node = provenance
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    return provenance


@pytest.mark.parametrize(
    "provenance",
    [
        _mutate(("patch_index_fingerprints", "index_policy_hash"), "other"),
        _mutate(("patch_index_fingerprints", "source_policy_hash"), "other"),
        _mutate(("scaler_digest",), "other"),
        _mutate(("source_channels",), 28),
        _mutate(("channel_indices",), [0, 1, 2]),
        _mutate(("geometry",), {"anchors": [[9, 9]]}),
        _mutate(("source_roots",), {"patch_index_root": "elsewhere"}),
    ],
    ids=["index", "source", "scaler", "width", "indices", "geometry", "roots"],
)
def test_key_changes_with_every_provenance_component(provenance: dict[str, Any]) -> None:
    base = _key(_provenance(22, tuple(range(22))), _order(22), ("train", "validation"))
    assert _key(provenance, _order(22), ("train", "validation")) != base


def test_key_changes_with_source_channel_order_and_is_stable() -> None:
    provenance = _provenance(22, tuple(range(22)))
    base = _key(provenance, _order(22), ("train", "validation"))
    assert _key(provenance, _order(22)[::-1], ("train", "validation")) != base
    assert _key(copy.deepcopy(provenance), _order(22), ("train", "validation")) == base


def test_components_are_exact_and_format_version_changes_key() -> None:
    components = prebuilt_key_components(
        _provenance(22, tuple(range(22))),
        source_channel_order=_order(22),
        splits=("train", "validation"),
    )
    assert set(components) == {
        "cache_format_version",
        "patch_cache_version",
        "index_policy_hash",
        "source_policy_hash",
        "scaler_digest",
        "source_channels",
        "source_channel_order",
        "channel_indices",
        "splits",
        "provenance_sha256",
    }
    bumped = {**components, "cache_format_version": 2}
    assert prebuilt_key(bumped) != prebuilt_key(components)
    assert len(prebuilt_key(components)) == 16
    assert prebuilt_prefix("gs://b/patch-cache/", components) == (
        f"gs://b/patch-cache/v1/{prebuilt_key(components)}"
    )


@pytest.mark.parametrize("splits", [("train", "validation", "test"), ("train",), ("test",)])
def test_key_components_reject_other_splits(splits: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="splits"):
        prebuilt_key_components(
            _provenance(22, tuple(range(22))), source_channel_order=_order(22), splits=splits
        )


# --- pack --------------------------------------------------------------------


@pytest.mark.parametrize("width", [22, 28])
@pytest.mark.parametrize("shard_bytes", ["tiny", "two_feature_rows"])
def test_pack_is_row_aligned_and_reassembles_byte_for_byte(
    tmp_path: Path, width: int, shard_bytes: str
) -> None:
    cache_dir, _, components, counts = _make_cache(tmp_path, width)
    features_row = int(np.prod(np.load(cache_dir / "features.npy", mmap_mode="r").shape[1:])) * 4
    limit = 1 if shard_bytes == "tiny" else 2 * features_row
    out_dir = tmp_path / "prebuilt"
    manifest = pack_prebuilt_cache(
        cache_dir, out_dir, components=components, split_counts=counts, shard_max_bytes=limit
    )
    assert not list(out_dir.rglob("*.partial"))
    assert manifest["splits"] == {
        "train": {"count": 3, "row_start": 0, "row_end": 3},
        "validation": {"count": 2, "row_start": 3, "row_end": 5},
    }
    assert manifest["total_count"] == 5
    assert manifest["key"] == prebuilt_key(components)
    assert manifest["prefix_suffix"] == f"v1/{prebuilt_key(components)}"
    assert json.loads((out_dir / PREBUILT_MANIFEST).read_text()) == manifest
    patch_ids = json.loads((cache_dir / "manifest.json").read_text())["patch_ids"]
    shards = manifest["shards"]
    by_name = {s["name"]: s for s in shards}
    for entry in manifest["files"]:
        original = (cache_dir / entry["file"]).read_bytes()
        assert entry["size"] == len(original)
        joined = b"".join((out_dir / "shards" / n).read_bytes() for n in entry["shards"])
        assert joined == original
        offset = 0
        for name in entry["shards"]:
            shard = by_name[name]
            data = (out_dir / "shards" / name).read_bytes()
            assert shard["size"] == len(data)
            assert shard["byte_offset"] == offset
            assert shard["sha256"] == hashlib.sha256(data).hexdigest()
            offset += len(data)
            if entry["file"] == "manifest.json":
                assert shard["row_start"] is None
            else:
                assert shard["patch_id_first"] == patch_ids[shard["row_start"]]
                assert shard["patch_id_last"] == patch_ids[shard["row_end"] - 1]
    feature_shards = [s for s in shards if s["file"] == "features.npy"]
    assert len(feature_shards) == (5 if shard_bytes == "tiny" else 3)
    assert {s["file"] for s in shards} == set(CACHE_FILES)


def test_pack_writes_the_manifest_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir, _, components, counts = _make_cache(tmp_path)
    out_dir = tmp_path / "prebuilt"
    seen: list[str] = []
    real_replace = os.replace

    def spy(src: object, dst: object) -> None:
        if Path(str(dst)).name == PREBUILT_MANIFEST:
            listed = json.loads(Path(str(src)).read_text())["shards"]
            assert all((out_dir / "shards" / s["name"]).is_file() for s in listed)
            seen.append("manifest")
        real_replace(str(src), str(dst))  # type: ignore[arg-type]

    monkeypatch.setattr(prebuilt_cache.os, "replace", spy)
    pack_prebuilt_cache(
        cache_dir, out_dir, components=components, split_counts=counts, shard_max_bytes=1
    )
    assert seen == ["manifest"]


@pytest.mark.parametrize(
    "counts",
    [
        {"train": 5, "validation": 0},
        {"train": 0, "validation": 5},
        {"train": 3, "validation": 2, "test": 1},
        {"train": 3},
    ],
)
def test_pack_rejects_empty_and_test_splits(tmp_path: Path, counts: dict[str, int]) -> None:
    cache_dir, _, components, _ = _make_cache(tmp_path)
    with pytest.raises(ValueError, match=r"empty split|test"):
        pack_prebuilt_cache(
            cache_dir, tmp_path / "prebuilt", components=components, split_counts=counts
        )
    assert not (tmp_path / "prebuilt").exists()


def test_pack_rejects_count_mismatch_and_existing_out_dir(tmp_path: Path) -> None:
    cache_dir, _, components, _ = _make_cache(tmp_path)
    with pytest.raises(ValueError, match="sum"):
        pack_prebuilt_cache(
            cache_dir,
            tmp_path / "prebuilt",
            components=components,
            split_counts={"train": 1, "validation": 1},
        )
    (tmp_path / "prebuilt").mkdir()
    with pytest.raises(FileExistsError):
        pack_prebuilt_cache(
            cache_dir,
            tmp_path / "prebuilt",
            components=components,
            split_counts={"train": 3, "validation": 2},
        )


# --- fetch -------------------------------------------------------------------


@pytest.mark.parametrize("width", [22, 28])
def test_fetch_roundtrip_is_byte_identical_and_loadable(tmp_path: Path, width: int) -> None:
    cache_dir, provenance, components, counts = _make_cache(tmp_path, width)
    out_dir = tmp_path / "prebuilt"
    pack_prebuilt_cache(
        cache_dir, out_dir, components=components, split_counts=counts, shard_max_bytes=100_000
    )
    prefix = prebuilt_prefix(str(tmp_path / "gcs"), components)
    upload_prebuilt_cache(out_dir, prefix)
    assert prebuilt_manifest_exists(prefix)
    dest = tmp_path / "fetched"
    result = fetch_prebuilt_cache(prefix, dest, expected_components=components)
    for name in CACHE_FILES:
        assert (dest / name).read_bytes() == (cache_dir / name).read_bytes()
    assert not (dest / ".prebuilt-parts").exists()
    assert not list(dest.glob("*.partial"))
    assert set(result) == {
        "prebuilt_download_seconds",
        "prebuilt_verify_seconds",
        "prebuilt_bytes",
        "prebuilt_shards",
        "prebuilt_build_provenance",
    }
    packed = json.loads((out_dir / PREBUILT_MANIFEST).read_text())
    assert result["prebuilt_shards"] == len(packed["shards"])
    assert result["prebuilt_build_provenance"] == {
        key: packed["provenance"][key] for key in ("git_commit", "builder_version", "created_at")
    }
    cache = PatchCache(dest, provenance)
    assert cache.patch_ids == tuple(f"p{i}" for i in range(5))


def _tamper(prefix: str, mutate: Any) -> None:
    path = Path(prefix) / PREBUILT_MANIFEST
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_policy_hash", "other"),
        ("source_policy_hash", "other"),
        ("scaler_digest", "other"),
        ("source_channels", 28),
        ("source_channel_order", ["x"]),
        ("channel_indices", [0]),
        ("provenance_sha256", "0" * 64),
        ("cache_format_version", 99),
        ("splits", ["train"]),
        ("patch_cache_version", 2),
    ],
)
def test_fetch_mismatch_names_the_field_and_downloads_nothing(
    uploaded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    _tamper(uploaded["prefix"], lambda m: m["components"].__setitem__(field, value))
    calls: list[str] = []
    monkeypatch.setattr(storage, "download_to_file", lambda uri, path: calls.append(str(uri)))
    dest = uploaded["root"] / "fetched"
    with pytest.raises(PrebuiltCacheMismatch, match=field) as excinfo:
        fetch_prebuilt_cache(uploaded["prefix"], dest, expected_components=uploaded["components"])
    assert excinfo.value.field == field
    assert calls == []
    assert not dest.exists()


def test_fetch_rejects_format_version_of_the_manifest(uploaded: dict[str, Any]) -> None:
    _tamper(uploaded["prefix"], lambda m: m.__setitem__("prebuilt_format_version", 2))
    with pytest.raises(PrebuiltCacheMismatch, match="cache_format_version"):
        fetch_prebuilt_cache(
            uploaded["prefix"],
            uploaded["root"] / "fetched",
            expected_components=uploaded["components"],
        )


def test_fetch_rejects_a_stored_test_split_and_bad_counts(uploaded: dict[str, Any]) -> None:
    kwargs = {"expected_components": uploaded["components"]}
    _tamper(
        uploaded["prefix"],
        lambda m: m["splits"].__setitem__("test", {"count": 1, "row_start": 5, "row_end": 6}),
    )
    with pytest.raises(PrebuiltCacheMismatch, match="splits"):
        fetch_prebuilt_cache(uploaded["prefix"], uploaded["root"] / "a", **kwargs)
    _tamper(uploaded["prefix"], lambda m: m["splits"].pop("test"))
    _tamper(uploaded["prefix"], lambda m: m.__setitem__("total_count", 99))
    with pytest.raises(PrebuiltCacheMismatch, match="total_count"):
        fetch_prebuilt_cache(uploaded["prefix"], uploaded["root"] / "b", **kwargs)
    _tamper(uploaded["prefix"], lambda m: m.__setitem__("total_count", 5))
    _tamper(uploaded["prefix"], lambda m: m["splits"]["train"].__setitem__("count", 0))
    with pytest.raises(PrebuiltCacheMismatch, match="splits"):
        fetch_prebuilt_cache(uploaded["prefix"], uploaded["root"] / "c", **kwargs)


def test_fetch_missing_artifact(built: dict[str, Any]) -> None:
    with pytest.raises(PrebuiltCacheMissing, match="no prebuilt patch cache at"):
        fetch_prebuilt_cache(
            built["prefix"], built["root"] / "fetched", expected_components=built["components"]
        )
    assert not (built["root"] / "fetched").exists()


def test_fetch_refuses_an_existing_cache(uploaded: dict[str, Any]) -> None:
    dest = uploaded["root"] / "fetched"
    dest.mkdir()
    (dest / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        fetch_prebuilt_cache(uploaded["prefix"], dest, expected_components=uploaded["components"])


def test_fetch_checksum_failure_after_retries_leaves_nothing(
    uploaded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    shard = next(
        s for s in uploaded["manifest"]["shards"] if s["file"] == "features.npy" and s["index"] == 1
    )
    path = Path(uploaded["prefix"]) / "shards" / shard["name"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    dest = uploaded["root"] / "fetched"
    caplog.set_level(logging.WARNING, logger=prebuilt_cache.logger.name)
    with pytest.raises(PrebuiltCacheChecksumError, match=shard["name"]):
        fetch_prebuilt_cache(
            uploaded["prefix"], dest, expected_components=uploaded["components"], retries=3
        )
    retries = [r for r in caplog.records if r.message == "prebuilt_cache_shard_retry"]
    assert len(retries) == 3
    assert not (dest / "manifest.json").exists()
    assert not list(dest.glob("*.npy"))
    assert not list(dest.rglob("*.partial"))


def test_fetch_retries_a_transient_truncated_download(
    uploaded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = storage.download_to_file
    state = {"truncated": False}

    def flaky(uri: object, local_path: object) -> None:
        real(uri, local_path)  # type: ignore[arg-type]
        if not state["truncated"] and "features-00001" in str(uri):
            state["truncated"] = True
            Path(str(local_path)).write_bytes(Path(str(local_path)).read_bytes()[:10])

    monkeypatch.setattr(storage, "download_to_file", flaky)
    caplog.set_level(logging.WARNING, logger=prebuilt_cache.logger.name)
    dest = uploaded["root"] / "fetched"
    fetch_prebuilt_cache(uploaded["prefix"], dest, expected_components=uploaded["components"])
    assert state["truncated"]
    retries = [r for r in caplog.records if r.message == "prebuilt_cache_shard_retry"]
    assert len(retries) == 1
    assert retries[0].levelno == logging.WARNING
    for name in CACHE_FILES:
        assert (dest / name).read_bytes() == (uploaded["cache_dir"] / name).read_bytes()


def test_fetch_missing_shard_is_a_checksum_error(uploaded: dict[str, Any]) -> None:
    shard = uploaded["manifest"]["shards"][0]
    (Path(uploaded["prefix"]) / "shards" / shard["name"]).unlink()
    with pytest.raises(PrebuiltCacheChecksumError, match=shard["name"]):
        fetch_prebuilt_cache(
            uploaded["prefix"],
            uploaded["root"] / "fetched",
            expected_components=uploaded["components"],
            retries=2,
        )


def test_fetch_concurrency_is_bounded(
    uploaded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = storage.download_to_file
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def tracked(uri: object, local_path: object) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.01)
            real(uri, local_path)  # type: ignore[arg-type]
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(storage, "download_to_file", tracked)
    fetch_prebuilt_cache(
        uploaded["prefix"],
        uploaded["root"] / "fetched",
        expected_components=uploaded["components"],
        max_workers=3,
    )
    assert 2 <= state["peak"] <= 3


def test_zero_workers_are_rejected(uploaded: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        fetch_prebuilt_cache(
            uploaded["prefix"],
            uploaded["root"] / "fetched",
            expected_components=uploaded["components"],
            max_workers=0,
        )
    with pytest.raises(ValueError, match="max_workers"):
        upload_prebuilt_cache(uploaded["out_dir"], uploaded["prefix"] + "-2", max_workers=0)
    assert issubclass(PrebuiltCacheMismatch, PrebuiltCacheError)


# --- upload ------------------------------------------------------------------


def test_upload_refuses_existing_without_overwrite(uploaded: dict[str, Any]) -> None:
    with pytest.raises(FileExistsError, match="--overwrite"):
        upload_prebuilt_cache(uploaded["out_dir"], uploaded["prefix"])
    upload_prebuilt_cache(uploaded["out_dir"], uploaded["prefix"], overwrite=True)
    assert prebuilt_manifest_exists(uploaded["prefix"])


def test_upload_manifest_goes_last_and_only_after_every_shard(
    built: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = storage.atomic_upload
    order: list[str] = []

    def recorder(local_path: object, dst: str, **kwargs: object) -> str:
        order.append(Path(dst).name)
        return real(local_path, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "atomic_upload", recorder)
    upload_prebuilt_cache(built["out_dir"], built["prefix"], max_workers=4)
    assert order[-1] == PREBUILT_MANIFEST
    assert order.count(PREBUILT_MANIFEST) == 1
    assert len(order) == len(built["manifest"]["shards"]) + 1


def test_interrupted_upload_has_no_manifest_and_rerun_succeeds(
    built: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = storage.atomic_upload
    calls = {"n": 0}

    def failing(local_path: object, dst: str, **kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("simulated interruption")
        return real(local_path, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "atomic_upload", failing)
    with pytest.raises(ConnectionError):
        upload_prebuilt_cache(built["out_dir"], built["prefix"], max_workers=1)
    assert not prebuilt_manifest_exists(built["prefix"])
    monkeypatch.setattr(storage, "atomic_upload", real)
    upload_prebuilt_cache(built["out_dir"], built["prefix"])
    dest = built["root"] / "fetched"
    fetch_prebuilt_cache(built["prefix"], dest, expected_components=built["components"])
    original = (built["cache_dir"] / "manifest.json").read_bytes()
    assert (dest / "manifest.json").read_bytes() == original
