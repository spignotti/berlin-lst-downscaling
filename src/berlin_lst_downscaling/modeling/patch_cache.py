"""Bounded, job-local cache for already-admitted Stage-1 patches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealSampleMeta,
)
from berlin_lst_downscaling.modeling.patches import RealSample

_ARRAYS = {
    "features": ("features.npy", np.dtype("float32")),
    "prior": ("prior.npy", np.dtype("float32")),
    "target": ("target.npy", np.dtype("float32")),
    "mask": ("mask.npy", np.dtype("bool")),
}


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_shape(count: int, channels: int) -> tuple[int, ...]:
    return count, channels, REAL_PATCH_PX, REAL_PATCH_PX


def estimate_cache_bytes(count: int, channels: int) -> int:
    """Return exact array payload bytes plus bounded NPY-header allowance."""
    if count <= 0 or not 1 <= channels <= 28:
        raise ValueError("cache requires positive count and 1–28 active channels")
    n = count
    per_sample = (
        channels * REAL_PATCH_PX * REAL_PATCH_PX * 4
        + REAL_PATCH_PX * REAL_PATCH_PX * 4
        + REAL_PATCH_CELLS * REAL_PATCH_CELLS * 4
        + REAL_PATCH_CELLS * REAL_PATCH_CELLS
    )
    return n * per_sample + len(_ARRAYS) * 8192 + n * 2048


class PatchCache:
    """Read-only memory-mapped patches after a validated ready manifest exists."""

    def __init__(self, root: Path, provenance: dict[str, object]) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"patch cache is incomplete: missing {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("version") != 1:
            raise RuntimeError("patch cache manifest version is unsupported")
        if self.manifest.get("provenance") != provenance:
            raise RuntimeError("patch cache provenance does not match the requested sources")
        if self.manifest.get("provenance_sha256") != _json_digest(provenance):
            raise RuntimeError("patch cache provenance digest is invalid")
        self.patch_ids = tuple(str(p) for p in self.manifest.get("patch_ids", []))
        self._metadata = self.manifest.get("metadata", [])
        if not self.patch_ids or len(self.patch_ids) != len(self._metadata):
            raise RuntimeError("patch cache manifest has inconsistent patch IDs and metadata")
        active_channels = self.manifest.get("active_channels")
        channel_order = provenance.get("channel_order")
        if (
            isinstance(active_channels, bool)
            or not isinstance(active_channels, int)
            or not 1 <= active_channels <= 28
            or not isinstance(channel_order, list | tuple)
            or len(channel_order) != active_channels
        ):
            raise RuntimeError("patch cache active channel count does not match provenance")
        if self.manifest.get("patch_ids_sha256") != _json_digest(list(self.patch_ids)):
            raise RuntimeError("patch cache patch-ID digest is invalid")
        if any(
            row.get("patch_id") != patch_id
            for row, patch_id in zip(self._metadata, self.patch_ids, strict=True)
        ):
            raise RuntimeError("patch cache metadata order differs from its patch-ID list")
        self.arrays: dict[str, np.ndarray] = {}
        shape_specs = {
            "features": _feature_shape(len(self.patch_ids), active_channels),
            "prior": (len(self.patch_ids), 1, REAL_PATCH_PX, REAL_PATCH_PX),
            "target": (len(self.patch_ids), 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS),
            "mask": (len(self.patch_ids), 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS),
        }
        if self.manifest.get("shapes") != {key: list(value) for key, value in shape_specs.items()}:
            raise RuntimeError("patch cache manifest shapes do not match the frozen patch geometry")
        expected_dtypes = {key: dtype.name for key, (_, dtype) in _ARRAYS.items()}
        if self.manifest.get("dtypes") != expected_dtypes:
            raise RuntimeError("patch cache manifest dtype declarations are invalid")
        for key, (filename, dtype) in _ARRAYS.items():
            path = root / filename
            if not path.is_file():
                raise RuntimeError(f"patch cache is incomplete: missing {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape_specs[key] or array.dtype != dtype:
                raise RuntimeError(
                    f"patch cache {key} shape/dtype mismatch: {array.shape}/{array.dtype}, "
                    f"expected {shape_specs[key]}/{dtype}"
                )
            self.arrays[key] = array
        actual_bytes = sum((root / filename).stat().st_size for filename, _ in _ARRAYS.values())
        if actual_bytes != self.manifest.get("array_bytes"):
            raise RuntimeError("patch cache array byte count differs from its ready manifest")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> RealSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        metadata = RealSampleMeta(**self._metadata[index])
        if metadata.patch_id != self.patch_ids[index]:
            raise RuntimeError(f"patch cache metadata ID drift at row {index}")
        return RealSample(
            meta=metadata,
            features=self.arrays["features"][index],
            lst_prior_k=self.arrays["prior"][index],
            target_100m=self.arrays["target"][index],
            mask_100m=self.arrays["mask"][index],
        )


def build_patch_cache(
    root: Path,
    samples: Iterable[RealSample],
    *,
    patch_ids: list[str],
    active_channels: int,
    provenance: dict[str, object],
    max_bytes: int,
    channel_indices: Iterable[int] | None = None,
) -> dict[str, object]:
    """Write a cache and publish its manifest last; partial roots never open."""
    if root.exists():
        raise FileExistsError(f"cache root already exists: {root}")
    if len(set(patch_ids)) != len(patch_ids) or not patch_ids:
        raise ValueError("patch_ids must be non-empty and unique")
    channel_order = provenance.get("channel_order")
    if not isinstance(channel_order, list | tuple) or len(channel_order) != active_channels:
        raise ValueError("provenance channel_order must match active_channels")
    if channel_indices is None:
        indices = tuple(range(active_channels))
    else:
        indices = tuple(int(i) for i in channel_indices)
        if len(indices) != active_channels:
            raise ValueError(
                f"channel_indices length {len(indices)} != active_channels {active_channels}"
            )
        if any(i < 0 or i >= 28 for i in indices):
            raise ValueError(f"channel_indices out of range for V3 stack: {indices}")
        if len(set(indices)) != len(indices):
            raise ValueError("channel_indices must be unique")
    index_array = np.asarray(indices, dtype=np.intp)
    estimate = estimate_cache_bytes(len(patch_ids), active_channels)
    if estimate > max_bytes:
        raise ValueError(f"estimated cache size {estimate} exceeds max_bytes {max_bytes}")
    if not provenance:
        raise ValueError("cache provenance must be non-empty")
    if not root.parent.is_dir():
        raise FileNotFoundError(f"cache parent directory does not exist: {root.parent}")
    available = shutil.disk_usage(root.parent).free
    if available < estimate:
        raise OSError(f"cache needs an estimated {estimate} bytes but only {available} are free")

    root.mkdir(parents=True, exist_ok=False)
    count = len(patch_ids)
    arrays = {
        "features": np.lib.format.open_memmap(
            root / _ARRAYS["features"][0], mode="w+", dtype="float32",
            shape=_feature_shape(count, active_channels),
        ),
        "prior": np.lib.format.open_memmap(
            root / _ARRAYS["prior"][0], mode="w+", dtype="float32",
            shape=(count, 1, REAL_PATCH_PX, REAL_PATCH_PX),
        ),
        "target": np.lib.format.open_memmap(
            root / _ARRAYS["target"][0], mode="w+", dtype="float32",
            shape=(count, 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS),
        ),
        "mask": np.lib.format.open_memmap(
            root / _ARRAYS["mask"][0], mode="w+", dtype="bool",
            shape=(count, 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS),
        ),
    }
    metadata: list[dict[str, object]] = []
    iterator = iter(samples)
    for index, expected_id in enumerate(patch_ids):
        try:
            sample = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(f"source ended after {index} of {count} admitted samples") from exc
        if sample.meta.patch_id != expected_id:
            raise RuntimeError(
                f"source order drift at row {index}: {sample.meta.patch_id!r} != {expected_id!r}"
            )
        if sample.features.shape != (28, REAL_PATCH_PX, REAL_PATCH_PX):
            raise RuntimeError(
                f"source feature shape mismatch for {expected_id}: {sample.features.shape}"
            )
        if sample.features.dtype != np.dtype("float32"):
            raise RuntimeError(f"source feature dtype mismatch for {expected_id}")
        if sample.lst_prior_k.shape != (1, REAL_PATCH_PX, REAL_PATCH_PX):
            raise RuntimeError(f"source prior shape mismatch for {expected_id}")
        if sample.lst_prior_k.dtype != np.dtype("float32"):
            raise RuntimeError(f"source prior dtype mismatch for {expected_id}")
        if sample.target_100m.shape != (1, REAL_PATCH_CELLS, REAL_PATCH_CELLS):
            raise RuntimeError(f"source target shape mismatch for {expected_id}")
        if sample.target_100m.dtype != np.dtype("float32"):
            raise RuntimeError(f"source target dtype mismatch for {expected_id}")
        if sample.mask_100m.shape != (1, REAL_PATCH_CELLS, REAL_PATCH_CELLS):
            raise RuntimeError(f"source mask shape mismatch for {expected_id}")
        if sample.mask_100m.dtype != np.dtype("bool"):
            raise RuntimeError(f"source mask dtype mismatch for {expected_id}")
        features = sample.features[index_array]
        if not np.isfinite(features).all() or not np.isfinite(sample.lst_prior_k).all():
            raise RuntimeError(f"non-finite model input in admitted source sample {expected_id}")
        valid = sample.mask_100m
        if not valid.any():
            raise RuntimeError(f"all-invalid admitted source sample {expected_id}")
        if not np.isfinite(sample.target_100m[valid]).all():
            raise RuntimeError(f"non-finite target under a valid mask for {expected_id}")
        arrays["features"][index] = features
        arrays["prior"][index] = sample.lst_prior_k
        arrays["target"][index] = sample.target_100m
        arrays["mask"][index] = sample.mask_100m
        metadata.append(asdict(sample.meta))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("source yielded more samples than the frozen patch ID list")

    for array in arrays.values():
        array.flush()
    del arrays

    shapes = {
        "features": list(_feature_shape(count, active_channels)),
        "prior": [count, 1, REAL_PATCH_PX, REAL_PATCH_PX],
        "target": [count, 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS],
        "mask": [count, 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS],
    }
    array_bytes = sum((root / filename).stat().st_size for filename, _ in _ARRAYS.values())
    if array_bytes > max_bytes:
        raise RuntimeError(f"actual cache size {array_bytes} exceeds max_bytes {max_bytes}")
    manifest: dict[str, object] = {
        "version": 1,
        "patch_ids": patch_ids,
        "patch_ids_sha256": _json_digest(patch_ids),
        "metadata": metadata,
        "provenance": provenance,
        "provenance_sha256": _json_digest(provenance),
        "active_channels": active_channels,
        "channel_indices": list(indices),
        "shapes": shapes,
        "dtypes": {key: dtype.name for key, (_, dtype) in _ARRAYS.items()},
        "array_bytes": array_bytes,
        "estimated_bytes": estimate,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    if array_bytes + len(payload.encode("utf-8")) > max_bytes:
        raise RuntimeError("cache arrays and manifest exceed max_bytes")
    manifest_path = root / "manifest.json.partial"
    manifest_path.write_text(payload, encoding="utf-8")
    os.replace(manifest_path, root / "manifest.json")
    PatchCache(root, provenance)
    return manifest


def assert_samples_equal(
    expected: RealSample,
    actual: RealSample,
    active_channels: int,
    *,
    channel_indices: Iterable[int] | None = None,
) -> None:
    """Fail unless a cached sample exactly preserves the source sample contract."""
    if asdict(expected.meta) != asdict(actual.meta):
        raise AssertionError(f"cached metadata differs for {expected.meta.patch_id}")
    if channel_indices is None:
        selected = expected.features[:active_channels]
    else:
        indices = tuple(int(i) for i in channel_indices)
        if len(indices) != active_channels:
            raise ValueError(
                f"channel_indices length {len(indices)} != active_channels {active_channels}"
            )
        selected = expected.features[np.asarray(indices, dtype=np.intp)]
    comparisons = (
        (selected, actual.features, True),
        (expected.lst_prior_k, actual.lst_prior_k, True),
        (expected.target_100m, actual.target_100m, True),
        (expected.mask_100m, actual.mask_100m, False),
    )
    for left, right, equal_nan in comparisons:
        if left.shape != right.shape or not np.array_equal(left, right, equal_nan=equal_nan):
            raise AssertionError(f"cached tensor differs for {expected.meta.patch_id}")
