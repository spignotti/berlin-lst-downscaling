"""Versioned, prebuilt patch cache: pack, upload and verified fetch.

The prebuilt artifact is exactly the job-local cache directory built by
``RealPatchDataModule`` (``features.npy``, ``prior.npy``, ``target.npy``,
``mask.npy``, ``manifest.json``; train rows then validation rows), cut into
byte shards on row boundaries. Concatenating the shards reproduces each file
byte for byte, after which ``PatchCache`` runs its existing exact provenance
check. One prefix holds one cache identity (train and validation together; the
test split is never cached). The prebuilt manifest is written and uploaded last,
so a prefix without it is incomplete and refused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from berlin_lst_downscaling.data.io import log_event, storage
from berlin_lst_downscaling.modeling.patch_cache import _json_digest
from berlin_lst_downscaling.modeling.timing import timed_phase

logger = logging.getLogger(__name__)

PREBUILT_CACHE_FORMAT_VERSION = 1
BUILDER_VERSION = 1
PATCH_CACHE_MANIFEST_VERSION = 1
PREBUILT_MANIFEST = "prebuilt_manifest.json"
DEFAULT_SHARD_MAX_BYTES = 512 * 1024**2
CACHE_FILES = ("features.npy", "prior.npy", "target.npy", "mask.npy", "manifest.json")
ALLOWED_SPLITS = ("train", "validation")
_CHUNK_BYTES = 64 * 1024**2
_PARTS_DIR = ".prebuilt-parts"
_SHARD_NAME = re.compile(r"^(features|prior|target|mask|manifest)-\d{5}\.bin$")


class PrebuiltCacheError(RuntimeError):
    """Base class for prebuilt patch cache failures."""


class PrebuiltCacheMissing(PrebuiltCacheError):
    """No complete prebuilt cache exists at the resolved prefix."""


class PrebuiltCacheMismatch(PrebuiltCacheError):
    """The stored cache identity differs from the one the run resolved."""

    def __init__(self, field: str, expected: object, actual: object) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"prebuilt patch cache mismatch in {field!r}: expected {expected!r}, found {actual!r}"
        )


class PrebuiltCacheChecksumError(PrebuiltCacheError):
    """A shard or assembled file failed its size or SHA-256 check."""


def prebuilt_key_components(
    provenance: Mapping[str, object],
    *,
    source_channel_order: Sequence[str],
    splits: Sequence[str],
) -> dict[str, object]:
    """Return the exact identity components a prebuilt cache is keyed on."""
    if tuple(splits) != ALLOWED_SPLITS:
        raise ValueError(
            f"prebuilt cache splits must be exactly {list(ALLOWED_SPLITS)}, got {list(splits)}; "
            "the test split is never cached"
        )
    fingerprints = cast(Mapping[str, str], provenance["patch_index_fingerprints"])
    return {
        "cache_format_version": PREBUILT_CACHE_FORMAT_VERSION,
        "patch_cache_version": PATCH_CACHE_MANIFEST_VERSION,
        "index_policy_hash": fingerprints["index_policy_hash"],
        "source_policy_hash": fingerprints["source_policy_hash"],
        "scaler_digest": provenance["scaler_digest"],
        "source_channels": int(cast(int, provenance["source_channels"])),
        "source_channel_order": list(source_channel_order),
        "channel_indices": list(cast(Sequence[int], provenance["channel_indices"])),
        "splits": list(splits),
        "provenance_sha256": _json_digest(provenance),
    }


def prebuilt_key(components: Mapping[str, object]) -> str:
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prebuilt_prefix(uri_root: str, components: Mapping[str, object]) -> str:
    return f"{uri_root.rstrip('/')}/v{PREBUILT_CACHE_FORMAT_VERSION}/{prebuilt_key(components)}"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _copy_range(src: Path, dst: Path, start: int, end: int) -> tuple[int, str]:
    """Copy bytes ``[start, end)`` of ``src`` to ``dst``; return ``(size, sha256)``."""
    digest = hashlib.sha256()
    remaining = end - start
    with src.open("rb") as reader, dst.open("wb") as writer:
        reader.seek(start)
        while remaining > 0:
            chunk = reader.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError(f"{src} ended before byte {end}")
            writer.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    return end - start, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def pack_prebuilt_cache(
    cache_dir: Path,
    out_dir: Path,
    *,
    components: Mapping[str, object],
    split_counts: Mapping[str, int],
    shard_max_bytes: int = DEFAULT_SHARD_MAX_BYTES,
    provenance_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Cut a ready job-local cache into checksummed row-aligned shards."""
    if shard_max_bytes < 1:
        raise ValueError("shard_max_bytes must be >= 1")
    if tuple(cast(Sequence[str], components.get("splits", ()))) != ALLOWED_SPLITS:
        raise ValueError(f"components splits must be exactly {list(ALLOWED_SPLITS)}")
    if set(split_counts) != set(ALLOWED_SPLITS):
        raise ValueError(
            f"split_counts keys must be exactly {list(ALLOWED_SPLITS)}, "
            f"got {sorted(split_counts)}; the test split is never cached"
        )
    for split in ALLOWED_SPLITS:
        if split_counts[split] <= 0:
            raise ValueError(f"empty split {split!r} in prebuilt cache")
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    patch_ids: list[str] = cache_manifest["patch_ids"]
    total = sum(split_counts[split] for split in ALLOWED_SPLITS)
    if total != len(patch_ids):
        raise ValueError(
            f"split counts sum to {total} but the cache holds {len(patch_ids)} patches"
        )
    if out_dir.exists():
        raise FileExistsError(str(out_dir))
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)

    shards: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for filename in CACHE_FILES:
        src = cache_dir / filename
        stem = src.stem
        file_size = src.stat().st_size
        names: list[str] = []
        if filename == "manifest.json":
            ranges: list[tuple[int, int, int, int]] = [(0, 0, 0, file_size)]
            header = 0
            row_bytes = 0
        else:
            array = np.load(src, mmap_mode="r")
            header = int(array.offset)  # type: ignore[attr-defined]
            if not array.flags.c_contiguous:
                raise RuntimeError(f"{src} is not C-contiguous; cannot shard on row boundaries")
            row_bytes = int(np.prod(array.shape[1:], dtype=np.int64)) * array.dtype.itemsize
            rows = int(array.shape[0])
            del array
            if rows != len(patch_ids):
                raise RuntimeError(f"{src} holds {rows} rows, manifest {len(patch_ids)}")
            rows_per_shard = max(1, shard_max_bytes // row_bytes)
            ranges = []
            for index, row_start in enumerate(range(0, rows, rows_per_shard)):
                row_end = min(rows, row_start + rows_per_shard)
                ranges.append((index, row_start, row_end, header + row_end * row_bytes))
            if header + rows * row_bytes != file_size:
                raise RuntimeError(f"{src} size {file_size} differs from header plus rows")
        offset = 0
        for index, row_start, row_end, end in ranges:
            name = f"{stem}-{index:05d}.bin"
            size, sha = _copy_range(src, shards_dir / name, offset, end)
            if filename == "manifest.json":
                first = last = None
                rs = re_ = None
            else:
                first, last = patch_ids[row_start], patch_ids[row_end - 1]
                rs, re_ = row_start, row_end
            shards.append(
                {
                    "file": filename,
                    "name": name,
                    "index": index,
                    "byte_offset": offset,
                    "size": size,
                    "sha256": sha,
                    "row_start": rs,
                    "row_end": re_,
                    "patch_id_first": first,
                    "patch_id_last": last,
                }
            )
            names.append(name)
            offset = end
        if offset != file_size:
            raise RuntimeError(f"shards of {src} end at {offset}, file size is {file_size}")
        files.append(
            {"file": filename, "size": file_size, "sha256": _file_sha256(src), "shards": names}
        )

    split_ranges: dict[str, dict[str, int]] = {}
    row = 0
    for split in ALLOWED_SPLITS:
        split_ranges[split] = {
            "count": split_counts[split],
            "row_start": row,
            "row_end": row + split_counts[split],
        }
        row += split_counts[split]
    key = prebuilt_key(components)
    manifest: dict[str, object] = {
        "prebuilt_format_version": PREBUILT_CACHE_FORMAT_VERSION,
        "key": key,
        "prefix_suffix": f"v{PREBUILT_CACHE_FORMAT_VERSION}/{key}",
        "components": dict(components),
        "splits": split_ranges,
        "total_count": total,
        "patch_ids_sha256": cache_manifest["patch_ids_sha256"],
        "channels": {
            "source_channel_order": components["source_channel_order"],
            "channel_indices": components["channel_indices"],
            "channel_order": cache_manifest["provenance"]["channel_order"],
        },
        "dtypes": cache_manifest["dtypes"],
        "shapes": cache_manifest["shapes"],
        "files": files,
        "shards": shards,
        "provenance": {
            "git_commit": _git_commit(),
            "builder_version": BUILDER_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            **dict(provenance_extra or {}),
        },
    }
    partial = out_dir / f"{PREBUILT_MANIFEST}.partial"
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, out_dir / PREBUILT_MANIFEST)
    return manifest


def _validate_workers(max_workers: int) -> None:
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")


def prebuilt_manifest_exists(prefix: str) -> bool:
    return storage.exists(f"{prefix.rstrip('/')}/{PREBUILT_MANIFEST}")


def upload_prebuilt_cache(
    local_dir: Path, prefix: str, *, overwrite: bool = False, max_workers: int = 8
) -> None:
    """Upload shards in parallel, then the manifest last (only after all shards)."""
    _validate_workers(max_workers)
    prefix = prefix.rstrip("/")
    if prebuilt_manifest_exists(prefix) and not overwrite:
        raise FileExistsError(
            f"prebuilt patch cache already exists at {prefix}; pass --overwrite to replace it"
        )
    manifest = json.loads((local_dir / PREBUILT_MANIFEST).read_text(encoding="utf-8"))
    names = [str(shard["name"]) for shard in manifest["shards"]]

    def upload(name: str) -> None:
        storage.atomic_upload(
            local_dir / "shards" / name, f"{prefix}/shards/{name}", overwrite=True
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in [pool.submit(upload, name) for name in names]:
            future.result()
    storage.atomic_upload(
        local_dir / PREBUILT_MANIFEST,
        f"{prefix}/{PREBUILT_MANIFEST}",
        overwrite=overwrite,
        if_generation_match=None if overwrite else 0,
    )
    log_event(
        logger,
        logging.INFO,
        "prebuilt_cache_uploaded",
        prefix=prefix,
        shards=len(names),
        overwrite=overwrite,
    )


def _check_manifest(
    manifest: Mapping[str, object], expected: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate identity and structure before any shard is downloaded."""
    version = manifest.get("prebuilt_format_version")
    if version != PREBUILT_CACHE_FORMAT_VERSION:
        raise PrebuiltCacheMismatch("cache_format_version", PREBUILT_CACHE_FORMAT_VERSION, version)
    stored = manifest.get("components")
    if not isinstance(stored, dict):
        raise PrebuiltCacheMismatch("components", dict(expected), stored)
    for name, value in expected.items():
        if name not in stored or stored[name] != value:
            raise PrebuiltCacheMismatch(name, value, stored.get(name))
    for name in stored:
        if name not in expected:
            raise PrebuiltCacheMismatch(name, None, stored[name])
    key = prebuilt_key(expected)
    if manifest.get("key") != key:
        raise PrebuiltCacheMismatch("key", key, manifest.get("key"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(ALLOWED_SPLITS):
        raise PrebuiltCacheMismatch(
            "splits", list(ALLOWED_SPLITS), sorted(splits) if isinstance(splits, dict) else splits
        )
    counts = [splits[split].get("count") for split in ALLOWED_SPLITS]
    if any(not isinstance(c, int) or c <= 0 for c in counts):
        raise PrebuiltCacheMismatch("splits", "positive counts", counts)
    if sum(counts) != manifest.get("total_count"):
        raise PrebuiltCacheMismatch("total_count", sum(counts), manifest.get("total_count"))
    shards = manifest.get("shards")
    files = manifest.get("files")
    if not isinstance(shards, list) or not isinstance(files, list):
        raise PrebuiltCacheError("prebuilt manifest lacks its shard or file list")
    for shard in shards:
        if not _SHARD_NAME.match(str(shard.get("name"))):
            raise PrebuiltCacheError(f"prebuilt manifest has an invalid shard name {shard!r}")
    for entry in files:
        if entry.get("file") not in CACHE_FILES:
            raise PrebuiltCacheError(f"prebuilt manifest has an unexpected file {entry!r}")
    if sorted(str(entry["file"]) for entry in files) != sorted(CACHE_FILES):
        raise PrebuiltCacheError("prebuilt manifest does not list exactly the cache files")
    return shards, files


def _download_shard(prefix: str, shard: Mapping[str, object], parts: Path, retries: int) -> None:
    name = str(shard["name"])
    final = parts / name
    partial = parts / f"{name}.partial"
    reason = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            storage.download_to_file(f"{prefix}/shards/{name}", partial)
            size = partial.stat().st_size
            if size != shard["size"]:
                raise PrebuiltCacheChecksumError(f"size {size} != {shard['size']}")
            if _file_sha256(partial) != shard["sha256"]:
                raise PrebuiltCacheChecksumError("sha256 differs")
            os.replace(partial, final)
            return
        except (PrebuiltCacheChecksumError, OSError) as exc:
            partial.unlink(missing_ok=True)
            reason = str(exc)
            log_event(
                logger,
                logging.WARNING,
                "prebuilt_cache_shard_retry",
                shard=name,
                attempt=attempt,
                retries=retries,
                reason=reason,
            )
    raise PrebuiltCacheChecksumError(
        f"prebuilt shard {name} at {prefix} failed after {retries} attempts: {reason}"
    )


def fetch_prebuilt_cache(
    prefix: str,
    dest: Path,
    *,
    expected_components: Mapping[str, object],
    max_workers: int = 8,
    retries: int = 3,
) -> dict[str, object]:
    """Download, verify and assemble a prebuilt cache into ``dest``.

    ``dest/manifest.json`` is placed last; on any failure it does not exist.
    """
    prefix = prefix.rstrip("/")
    manifest_uri = f"{prefix}/{PREBUILT_MANIFEST}"
    if not storage.exists(manifest_uri):
        raise PrebuiltCacheMissing(f"no prebuilt patch cache at {prefix}")
    try:
        manifest = json.loads(storage.read_bytes(manifest_uri))
    except (ValueError, FileNotFoundError) as exc:
        raise PrebuiltCacheError(f"unreadable prebuilt manifest at {manifest_uri}: {exc}") from exc
    shards, files = _check_manifest(manifest, expected_components)
    if (dest / "manifest.json").exists():
        raise FileExistsError(str(dest / "manifest.json"))
    _validate_workers(max_workers)
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    dest_existed = dest.exists()
    parts = dest / _PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    try:
        with timed_phase("prebuilt_download", logger=logger, prefix=prefix, shards=len(shards)):
            download_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for future in [
                    pool.submit(_download_shard, prefix, shard, parts, retries) for shard in shards
                ]:
                    future.result()
            download_seconds = time.perf_counter() - download_started
        with timed_phase("prebuilt_verify", logger=logger, prefix=prefix, files=len(files)):
            verify_started = time.perf_counter()
            for entry in sorted(files, key=lambda e: e["file"] == "manifest.json"):
                filename = str(entry["file"])
                partial = dest / f"{filename}.partial"
                with partial.open("wb") as writer:
                    for name in entry["shards"]:  # type: ignore[attr-defined]
                        part = parts / str(name)
                        with part.open("rb") as reader:
                            shutil.copyfileobj(reader, writer, _CHUNK_BYTES)
                        part.unlink()
                size = partial.stat().st_size
                if size != entry["size"] or _file_sha256(partial) != entry["sha256"]:
                    raise PrebuiltCacheChecksumError(
                        f"assembled {filename} from {prefix} fails its size or sha256 check"
                    )
                os.replace(partial, dest / filename)
            verify_seconds = time.perf_counter() - verify_started
    except BaseException:
        for filename in CACHE_FILES:
            (dest / filename).unlink(missing_ok=True)
            (dest / f"{filename}.partial").unlink(missing_ok=True)
        shutil.rmtree(parts, ignore_errors=True)
        if not dest_existed:
            shutil.rmtree(dest, ignore_errors=True)
        raise
    shutil.rmtree(parts, ignore_errors=True)
    raw_build = manifest.get("provenance")
    build = raw_build if isinstance(raw_build, dict) else {}
    return {
        "prebuilt_download_seconds": download_seconds,
        "prebuilt_verify_seconds": verify_seconds,
        "prebuilt_bytes": sum(int(cast(int, shard["size"])) for shard in shards),
        "prebuilt_shards": len(shards),
        "prebuilt_build_provenance": {
            key: build.get(key) for key in ("git_commit", "builder_version", "created_at")
        },
    }
