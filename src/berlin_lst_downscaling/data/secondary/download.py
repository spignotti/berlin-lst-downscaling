"""Streaming raw download with SHA-256 verification and atomic finalisation.

Uses stdlib ``hashlib`` and existing ``tenacity``; no new dependency.
"""

from __future__ import annotations

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

    uri: str                 # final destination URI
    byte_count: int          # bytes written
    checksum: str            # SHA-256 hex digest


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def download_to_raw(
    url: str,
    destination: str,
    expected_checksum: str | None = None,
) -> DownloadReceipt:
    """Stream-download *url* to raw storage at *destination*.

    Downloads to a temporary local file with chunked SHA-256 calculation,
    then atomically transfers to *destination* via :func:`atomic_upload`.

    If *expected_checksum* is given, the downloaded content must match.
    If *destination* already exists, the download is skipped (idempotent).

    Raises
    ------
    requests.HTTPError
        On HTTP failure.
    ValueError
        On SHA-256 mismatch.
    """
    if exists(destination):
        return DownloadReceipt(uri=destination, byte_count=0, checksum="")

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "download.tmp"
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
                f"SHA-256 mismatch for {url}: "
                f"expected {expected_checksum}, got {checksum}"
            )

        atomic_upload(tmp_path, destination)

    return DownloadReceipt(
        uri=destination,
        byte_count=byte_count,
        checksum=checksum,
    )


__all__ = [
    "DownloadReceipt",
    "download_to_raw",
]
