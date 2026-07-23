"""Streaming raw download with SHA-256 verification and atomic finalisation.

Uses stdlib ``hashlib`` and existing ``tenacity``; no new dependency.

Supports a ``local_cache_path`` argument so callers can hold a local copy
of the archive without loading it into RAM (required for the ~805 MB
vegetation-height ZIP, where the previous ``read_bytes()`` round-trip
would cost ~805 MB of heap).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from berlin_lst_downscaling.data.io.storage import atomic_upload, exists


@dataclass
class DownloadReceipt:
    """Result of a successful raw download."""

    uri: str  # final destination URI
    byte_count: int  # bytes written (0 when destination pre-existed)
    checksum: str  # SHA-256 hex digest
    local_cache_path: str | None = None  # local path to the archive, if retained


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def download_to_raw(
    url: str,
    destination: str,
    expected_checksum: str | None = None,
    local_cache_path: str | None = None,
) -> DownloadReceipt:
    """Stream-download *url* to raw storage at *destination*.

    Downloads to a temporary local file with chunked SHA-256 calculation,
    then atomically transfers to *destination* via :func:`atomic_upload`.

    If *expected_checksum* is given, the downloaded content must match.
    If *destination* already exists, the download is skipped (idempotent)
    but the SHA-256 is still computed so callers can record provenance.

    When *local_cache_path* is given, the archive is additionally
    materialised as a regular local file so callers can read it with
    memory-mapped I/O (no ``read_bytes`` round-trip through GCS).

    Raises
    ------
    requests.HTTPError
        On HTTP failure.
    ValueError
        On SHA-256 mismatch.
    """
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "download.tmp"
        need_download = not exists(destination)

        if need_download:
            # ── fresh download + streaming SHA-256 ──────────────────
            h = sha256()
            byte_count = 0

            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        h.update(chunk)
                        byte_count += len(chunk)

            checksum = h.hexdigest()

            if expected_checksum and checksum != expected_checksum:
                raise ValueError(
                    f"SHA-256 mismatch for {url}: expected {expected_checksum}, got {checksum}"
                )

            atomic_upload(tmp_path, destination)

            if local_cache_path:
                _ensure_local_cache(tmp_path, local_cache_path)

            return DownloadReceipt(
                uri=destination,
                byte_count=byte_count,
                checksum=checksum,
                local_cache_path=str(local_cache_path) if local_cache_path else None,
            )

        # ── destination pre-existed ─────────────────────────────────
        if destination.startswith("gs://"):
            # GCS: stream SHA-256 without loading the whole object.
            checksum = _stream_sha256_gcs(destination)
            if local_cache_path:
                _cache_from_gcs(destination, local_cache_path)
        else:
            # Local / mounted: the destination IS a local file.
            dst_path = Path(destination).expanduser()
            checksum = _stream_sha256_file(dst_path)
            if local_cache_path:
                _ensure_local_cache(dst_path, local_cache_path)

        return DownloadReceipt(
            uri=destination,
            byte_count=0,
            checksum=checksum,
            local_cache_path=str(local_cache_path) if local_cache_path else None,
        )


# ── helpers ───────────────────────────────────────────────────────────────


def _stream_sha256_file(path: Path) -> str:
    """Compute SHA-256 of a local file in 8 KiB chunks."""
    h = sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stream_sha256_gcs(uri: str) -> str:
    """Stream-download a GCS object and compute its SHA-256."""
    from google.cloud import storage  # type: ignore[import-untyped]

    bucket_name, key = uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)

    h = sha256()
    with blob.open("rb") as f:
        while True:
            # google-cloud-storage types f.read as bytes|str; in 'rb'
            # mode it always returns bytes.
            chunk: bytes = f.read(8192)  # type: ignore[assignment]
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cache_from_gcs(uri: str, cache_path: str) -> None:
    """Download a GCS object to a local cache path, skipping if cached."""
    from google.cloud import storage  # type: ignore[import-untyped]

    cp = Path(cache_path).expanduser()
    if cp.exists() and cp.stat().st_size > 0:
        return
    cp.parent.mkdir(parents=True, exist_ok=True)

    bucket_name, key = uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(key)
    blob.download_to_filename(str(cp))


def _ensure_local_cache(src: Path, cache_path: str) -> None:
    """Copy *src* to *cache_path* unless the cache already holds the file."""
    cp = Path(cache_path).expanduser()
    if cp.exists() and cp.stat().st_size > 0:
        return
    cp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(cp))


__all__ = [
    "DownloadReceipt",
    "download_to_raw",
]
