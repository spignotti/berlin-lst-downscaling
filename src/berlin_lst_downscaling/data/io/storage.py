"""Storage I/O helpers — local POSIX and GCS-backed atomic writes.

``atomic_write`` is the central function: it writes bytes to a temporary
location, then atomically moves the temp to the target URI.  For local
paths this uses ``os.replace`` (POSIX atomic).  For GCS it uses
``copy_blob`` + ``delete`` (object-store-renamable, eventually
consistent — see docstring for caveats).

All functions accept ``str | Path | OutputLocation`` as the URI argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Union
from uuid import uuid4

# ── URI type ─────────────────────────────────────────────────────────

UriLike = Union[str, Path, "OutputLocation"]


@dataclass(frozen=True)
class OutputLocation:
    """Normalised output location with prefix detection.

    Examples
    --------
    >>> OutputLocation("data/ard/ledger.parquet")
    OutputLocation(uri='data/ard/ledger.parquet', scheme='local')
    >>> OutputLocation("gs://berlin-lst-data/ard/ledger.parquet")
    OutputLocation(uri='gs://berlin-lst-data/ard/ledger.parquet', scheme='gcs')
    >>> OutputLocation("~/.mnt/berlin-lst/ard/ledger.parquet")
    OutputLocation(uri='~/.mnt/berlin-lst/ard/ledger.parquet', scheme='mounted')
    """

    uri: str
    scheme: Literal["local", "gcs", "mounted"] = "local"

    def __post_init__(self) -> None:
        obj = str(self.uri)
        if obj.startswith("gs://"):
            object.__setattr__(self, "scheme", "gcs")
        elif obj.startswith("~/.mnt/"):
            object.__setattr__(self, "scheme", "mounted")
        else:
            object.__setattr__(self, "scheme", "local")

    def __fspath__(self) -> str:
        return str(self.uri)

    def __str__(self) -> str:
        return str(self.uri)

    def __truediv__(self, other: str) -> OutputLocation:
        sep = "/" if self.scheme == "gcs" else os.sep
        base = str(self.uri).rstrip(sep)
        return OutputLocation(f"{base}{sep}{other}")


# ── GCS helpers (lazy import) ────────────────────────────────────────


def _gcs_client():
    from google.cloud import storage  # type: ignore[import-untyped]

    return storage.Client()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket_name, object_key)``."""
    path = uri.removeprefix("gs://")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return parts[0], parts[1]


# ── public API ───────────────────────────────────────────────────────


def exists(uri: UriLike) -> bool:
    """Return True if the URI exists (local file or GCS blob)."""
    loc = _as_loc(uri)
    if loc.scheme == "gcs":
        bucket_name, key = _parse_gs_uri(loc.uri)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        return blob.exists()
    return _resolve_local(loc.uri).exists()


def read_bytes(uri: UriLike) -> bytes:
    """Read the full content at the URI into bytes."""
    loc = _as_loc(uri)
    if loc.scheme == "gcs":
        bucket_name, key = _parse_gs_uri(loc.uri)
        client = _gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        return blob.download_as_bytes()
    return _resolve_local(loc.uri).read_bytes()


def read_text(uri: UriLike, encoding: str = "utf-8") -> str:
    """Read the full content at the URI into str."""
    return read_bytes(uri).decode(encoding)


def atomic_write(
    uri: UriLike,
    data: bytes | BinaryIO | str,
    overwrite: bool = True,
) -> None:
    """Write *data* to *uri* atomically.

    For **local paths** and **mounted paths** (``/`` or ``~/.mnt/...``):
      - Write to a ``.tmp/_{name}.{uuid}`` sibling, then ``os.replace``
        to the final path.
      - With ``overwrite=False``, raises ``FileExistsError`` if the
        target already exists.

    For **GCS paths** (``gs://``):
      - Upload to a ``.tmp/_{name}.{uuid}`` object, then
        ``copy_blob(tmp, bucket, final_key)`` + ``tmp.delete()``.

    *Note on GCS atomicity:* This is **not strictly atomic** — a reader
    between copy and delete sees the old blob.  Eventually consistent.
    For ledger + COG use this is tolerable: the caller's ``reconcile()``
    already checks file existence before acting on a scene.

    *Note on FUSE mounts:* ``os.replace`` over FUSE on macOS is
    **best-effort** atomic.
    """
    loc = _as_loc(uri)
    data_bytes = _to_bytes(data)

    if loc.scheme == "gcs":
        _atomic_write_gcs(loc.uri, data_bytes, overwrite)
    else:
        _atomic_write_local(loc.uri, data_bytes, overwrite)


# ── internals ────────────────────────────────────────────────────────


def _as_loc(uri: UriLike) -> OutputLocation:
    """Normalise URI to OutputLocation, handling Path and str."""
    if isinstance(uri, OutputLocation):
        return uri
    return OutputLocation(str(uri))


def _to_bytes(data: bytes | BinaryIO | str) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    # BinaryIO (file-like)
    chunk = data.read()
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    return bytes(chunk)


def _resolve_local(uri: str) -> Path:
    return Path(os.path.expanduser(uri))


def _atomic_write_local(uri: str, data: bytes, overwrite: bool) -> None:
    dst = _resolve_local(uri)
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))

    tmp_dir = dst.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"_{dst.name}.{uuid4().hex[:8]}"

    try:
        tmp_path.write_bytes(data)
        os.replace(str(tmp_path), str(dst))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    _prune_tmp(tmp_dir, max_age_s=3600)


def _atomic_write_gcs(uri: str, data: bytes, overwrite: bool) -> None:
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    final_blob = bucket.blob(key)
    if not overwrite and final_blob.exists():
        raise FileExistsError(uri)

    tmp_key = (
        Path(key).parent / ".tmp" / f"_{Path(key).name}.{uuid4().hex[:8]}"
    ).as_posix()
    tmp_blob = bucket.blob(tmp_key)

    try:
        tmp_blob.upload_from_string(data)
        bucket.copy_blob(tmp_blob, bucket, key)
        tmp_blob.delete()
    except BaseException:
        try:
            tmp_blob.delete()
        except Exception:  # noqa: S110 — best-effort cleanup, no logging needed
            pass
        raise


def _prune_tmp(tmp_dir: Path, max_age_s: int = 3600) -> None:
    """Remove orphaned temp files older than *max_age_s*."""
    import time

    now = time.time()
    for p in tmp_dir.iterdir():
        if p.is_file() and (now - p.stat().st_mtime) > max_age_s:
            p.unlink(missing_ok=True)


__all__ = [
    "OutputLocation",
    "atomic_write",
    "exists",
    "read_bytes",
    "read_text",
]
