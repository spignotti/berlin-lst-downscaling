"""Small shared helpers used across pipeline modules."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """SHA-256 hex digest of a file (streamed in 64 KiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` localised to UTC.

    Naive datetimes are assumed UTC. Aware datetimes are converted.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


__all__ = ["sha256_bytes", "sha256_file", "ensure_utc"]
