"""Thin GCS client wrapper for the berlin-lst-downscaling project.

Auth: relies on ADC via the ``GOOGLE_APPLICATION_CREDENTIALS`` env var
(bootstrap in ``berlin_lst_downscaling/__init__.py`` loads ``.env``).

All functions take a bucket *name* (not a ``Bucket`` handle) so callers
don't have to know about the storage client lifecycle.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from google.cloud import storage
from google.cloud.storage import Client

__all__ = [
    "get_client",
    "list_blobs",
    "download_blob",
    "upload_blob",
    "read_text",
]


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Return a cached GCS client (ADC auth)."""
    return storage.Client()


def list_blobs(
    bucket: str,
    prefix: str | None = None,
    max_results: int | None = None,
) -> list[str]:
    """List blob names in ``bucket``, optionally filtered by ``prefix``."""
    blobs = get_client().list_blobs(bucket, prefix=prefix, max_results=max_results)
    return [b.name for b in blobs]


def download_blob(bucket: str, name: str, dest: Path) -> None:
    """Download a single blob to ``dest`` (creates parent dirs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    get_client().bucket(bucket).blob(name).download_to_filename(str(dest))


def read_text(bucket: str, name: str) -> str | None:
    """Read a blob as text. Returns None if the blob does not exist."""
    blob = get_client().bucket(bucket).blob(name)
    if not blob.exists():
        return None
    return blob.download_as_text()


def upload_blob(bucket: str, src: Path, name: str) -> None:
    """Upload a local file to ``gs://<bucket>/<name>``."""
    get_client().bucket(bucket).blob(name).upload_from_filename(str(src))
