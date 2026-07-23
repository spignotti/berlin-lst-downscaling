"""Storage I/O helpers — local POSIX and GCS-backed atomic writes.

``atomic_write`` is the central function: it writes bytes to a temporary
location, then atomically moves the temp to the target URI.  For local
paths this uses ``os.replace`` (POSIX atomic).  For GCS it uses
``copy_blob`` + ``delete`` (object-store-renamable, eventually
consistent — see docstring for caveats).

``atomic_upload`` is the file-based counterpart for large local files:
it streams the file directly without loading it into memory.

All functions accept ``str | Path | OutputLocation`` as the URI argument.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Union
from uuid import uuid4

_logger = logging.getLogger(__name__)

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

    tmp_key = (Path(key).parent / ".tmp" / f"_{Path(key).name}.{uuid4().hex[:8]}").as_posix()
    tmp_blob = bucket.blob(tmp_key)

    _gcs_upload_with_retry(tmp_blob, data, bucket, key)


def _gcs_upload_with_retry(tmp_blob, data, bucket, key):
    """Upload to GCS with retries for transient failures (429, 503, etc.)."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(
            (
                Exception,  # google.api_core.exceptions 429/503 inherit from Exception
            )
        ),
        reraise=True,
    )
    def _do_upload():
        try:
            tmp_blob.upload_from_string(data)
            bucket.copy_blob(tmp_blob, bucket, key)
        except Exception:
            # Clean up temp blob on failure before retry
            try:
                tmp_blob.delete()
            except Exception:  # noqa: S110 — best-effort cleanup
                pass
            raise

    try:
        _do_upload()
        tmp_blob.delete()
    except Exception:
        try:
            tmp_blob.delete()
        except Exception:  # noqa: S110 — best-effort cleanup
            pass
        raise


def _prune_tmp(tmp_dir: Path, max_age_s: int = 3600) -> None:
    """Remove orphaned temp files older than *max_age_s*."""
    import time as _time

    now = _time.time()
    for p in tmp_dir.iterdir():
        if p.is_file() and (now - p.stat().st_mtime) > max_age_s:
            p.unlink(missing_ok=True)


# ── atomic file upload ────────────────────────────────────────────────


def atomic_upload(
    local_path: Path | str,
    dst: str,
    overwrite: bool = True,
    if_generation_match: int | None = None,
) -> str:
    """Atomically upload a local file to GCS or copy it locally.

    For **GCS paths** (``gs://``):
      - Upload to a ``.tmp/_{name}.{uuid}`` object via
        ``upload_from_filename`` (streaming, no memory load),
      - ``copy_blob(tmp, bucket, final_key)``,
      - ``tmp.delete()``.

    For **local paths**:
      - ``shutil.copy2`` to a ``.tmp/_{name}.{uuid}`` sibling,
      - ``os.replace`` to the final path,
      - prune stale temp files.

    Parameters
    ----------
    local_path :
        Path to a local file that exists and is readable.
    dst :
        Final output URI (local path or ``gs://...``).
    overwrite :
        If ``False`` and *dst* exists, a :class:`FileExistsError` is raised.
    if_generation_match :
        GCS precondition: succeed only if the destination's current
        generation equals this value. Pass ``0`` to require the object
        to be absent (used for immutable bundle publication).

    Returns
    -------
    str
        The *dst* URI on success.
    """
    loc = _as_loc(dst)
    src_path = Path(local_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    if loc.scheme == "gcs":
        _atomic_upload_gcs(src_path, loc.uri, overwrite, if_generation_match)
    else:
        _atomic_upload_local(src_path, loc.uri, overwrite)

    return dst


def _atomic_upload_local(local_path: Path, dst_uri: str, overwrite: bool) -> None:
    """Copy a file to a local path atomically."""
    dst = _resolve_local(dst_uri)
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))

    tmp_dir = dst.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"_{dst.name}.{uuid4().hex[:8]}"

    try:
        shutil.copy2(str(local_path), str(tmp_path))
        os.replace(str(tmp_path), str(dst))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    _prune_tmp(tmp_dir, max_age_s=3600)


def _atomic_upload_gcs(
    local_path: Path,
    dst_uri: str,
    overwrite: bool,
    if_generation_match: int | None,
) -> None:
    """Upload a file to GCS atomically via temp object + rename."""
    bucket_name, key = _parse_gs_uri(dst_uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    final_blob = bucket.blob(key)
    if not overwrite and final_blob.exists():
        raise FileExistsError(dst_uri)

    tmp_key = (Path(key).parent / ".tmp" / f"_{Path(key).name}.{uuid4().hex[:8]}").as_posix()
    tmp_blob = bucket.blob(tmp_key)

    _gcs_upload_file_with_retry(tmp_blob, local_path, bucket, key, if_generation_match)


def _gcs_upload_file_with_retry(tmp_blob, local_path, bucket, key, if_generation_match):
    """Upload file to GCS with retries for transient failures."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _do_upload():
        try:
            tmp_blob.upload_from_filename(str(local_path))
            bucket.copy_blob(tmp_blob, bucket, key, if_generation_match=if_generation_match)
        except Exception:
            try:
                tmp_blob.delete()
            except Exception:  # noqa: S110 — best-effort cleanup
                pass
            raise

    try:
        _do_upload()
        tmp_blob.delete()
    except Exception:
        try:
            tmp_blob.delete()
        except Exception:  # noqa: S110 — best-effort cleanup
            pass
        raise


__all__ = [
    "OutputLocation",
    "atomic_write",
    "atomic_upload",
    "exists",
    "read_bytes",
]
