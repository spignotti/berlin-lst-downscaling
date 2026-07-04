"""Ephemeral staging for raw input data.

Provides :class:`StageManager` — a scheme-aware manager that handles raw
input files (L2T COGs, AppEEARS exports, etc.) during ARD processing.

Design contract
--------------
Raw inputs are **ephemeral**: they exist only for the duration of a
processing run, then are deleted.  Final artefacts (COGs, STAC items)
are written to ``output_root`` and are never staged.

Supported URI schemes:

==============  =========================================================
Scheme          Mechanism
==============  =========================================================
``local``       ``pathlib.Path`` + ``shutil`` (POSIX)
``gcs``         ``google.cloud.storage`` (bucket → ``/vsigs/`` path)
``mounted``     ``pathlib.Path`` + ``shutil`` via rclone FUSE mount
==============  =========================================================

Usage
-----
.. code-block:: python

    with StageSession(stage_uri="data/tmp/ecostress_stage_abc123") as stage:
        # Download from NASA S3 → local tmp
        local_tif = download_to_tmp(granule_id)
        # Upload into stage
        stage.put(local_tif, f"{granule_id}/ECOv002_L2T_LSTE_..._LST.tif")
        # ... run pipeline, which reads from stage URI ...
        # cleanup runs automatically on exit

    # Or without context manager:
    stage = StageManager("gs://bucket/_staging/ecostress/run_001")
    stage.put(local_file, "granule_id/file.tif")
    # ... process ...
    stage.cleanup()   # always call explicitly when not using context manager

GCS credentials
--------------
``GOOGLE_APPLICATION_CREDENTIALS`` must be set (service-account JSON key)
or ADC must be configured for GCS operations to succeed.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from berlin_lst_downscaling.data.io.storage import OutputLocation

if TYPE_CHECKING:
    from google.cloud.storage import Blob, Bucket, Client

# ── URI type alias ────────────────────────────────────────────────────

UriLike = str | Path | OutputLocation


# ── GCS helpers ──────────────────────────────────────────────────────


def _gcs_client() -> Client:
    from google.cloud.storage import Client

    return Client()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket_name, object_key)`` from a ``gs://`` URI."""
    path = uri.removeprefix("gs://")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return parts[0], parts[1]


# ── stage manager ────────────────────────────────────────────────────


class StageManager:
    """Manage ephemeral raw-input staging for a single run.

    Parameters
    ----------
    uri :
        Stage root URI, e.g. ``data/tmp/ecostress_stage_abc`` or
        ``gs://bucket/_staging/ecostress/run_001``.
        The stage root is used as a prefix for all files put into the stage.
    run_id :
        Unique identifier for this run.  Included in the stage path when
        constructing URIs programmatically.  If ``None``, a short UUID is
        generated.
    persist :
        If ``True``, do not delete stage contents on :meth:`cleanup`.
        Default ``False`` (ephemeral).
    """

    def __init__(
        self,
        uri: UriLike,
        run_id: str | None = None,
        persist: bool = False,
    ) -> None:
        self._base_loc = OutputLocation(str(uri))
        self._run_id = run_id or uuid4().hex[:8]
        self._persist = persist

    @property
    def run_id(self) -> str:
        """Unique run identifier."""
        return self._run_id

    @property
    def uri(self) -> OutputLocation:
        """Full stage URI including run prefix."""
        return self._base_loc / self._run_id

    def put(self, local_path: Path, key: str) -> None:
        """Copy a local file into the stage.

        Parameters
        ----------
        local_path :
            Path to a local file that exists and is readable.
        key :
            Relative path within the stage, e.g.
            ``{granule_id}/ECOv002_L2T_LSTE_..._LST.tif``.
        """
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Stage source file not found: {local_path}")

        if self._base_loc.scheme == "gcs":
            self._put_gcs(local_path, key)
        else:
            self._put_local(local_path, key)

    def _put_local(self, local_path: Path, key: str) -> None:
        """Copy to a local or mounted stage path."""
        dst = Path(self.uri.uri) / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def _put_gcs(self, local_path: Path, key: str) -> None:
        """Upload a local file to a GCS stage prefix."""
        bucket_name, prefix = _parse_gs_uri(self.uri.uri)
        # key is relative to the stage root
        object_key = f"{prefix}/{key}"
        client = _gcs_client()
        bucket: Bucket = client.bucket(bucket_name)
        blob: Blob = bucket.blob(object_key)
        blob.upload_from_filename(str(local_path))

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys under ``prefix`` inside the stage.

        For GCS: lists blobs under ``stage_uri/run_id/prefix``.
        For local/mounted: lists files under ``stage_uri/run_id/prefix``.
        """
        stage_prefix = f"{self.uri.uri}/{prefix}" if prefix else self.uri.uri

        if self._base_loc.scheme == "gcs":
            return self._list_keys_gcs(stage_prefix)
        else:
            return self._list_keys_local(stage_prefix)

    def _list_keys_local(self, prefix: str) -> list[str]:
        """List files under a local/mounted prefix."""
        root = Path(prefix)
        if not root.is_dir():
            return []
        keys: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                # Return relative path from root
                rel = p.relative_to(root)
                keys.append(str(rel))
        return sorted(keys)

    def _list_keys_gcs(self, prefix: str) -> list[str]:
        """List blobs under a GCS prefix."""
        bucket_name, object_prefix = _parse_gs_uri(prefix)
        client = _gcs_client()
        bucket: Bucket = client.bucket(bucket_name)
        keys: list[str] = []
        for blob in bucket.list_blobs(prefix=object_prefix):
            keys.append(blob.name[len(object_prefix):].lstrip("/"))
        return sorted(keys)

    def stage_uri_for(self, key: str) -> str:
        """Return the full URI for a staged file.

        Examples
        --------
        >>> m = StageManager("data/tmp/stage", run_id="abc")
        >>> m.stage_uri_for("granule/file.tif")
        'data/tmp/stage/abc/granule/file.tif'
        >>> m = StageManager("gs://bucket/_staging/run001")
        >>> m.stage_uri_for("granule/file.tif")
        'gs://bucket/_staging/run001/granule/file.tif'
        """
        return f"{self.uri.uri}/{key}"

    def cleanup(self) -> None:
        """Delete all files in the stage.

        If ``persist=True`` was set at construction, this is a no-op.
        """
        if self._persist:
            return

        if self._base_loc.scheme == "gcs":
            self._cleanup_gcs()
        else:
            self._cleanup_local()

    def _cleanup_local(self) -> None:
        """Remove the local stage directory tree."""
        root = Path(self.uri.uri)
        if root.is_dir():
            shutil.rmtree(root)

    def _cleanup_gcs(self) -> None:
        """Delete all objects under the GCS stage prefix."""
        bucket_name, object_prefix = _parse_gs_uri(self.uri.uri)
        client = _gcs_client()
        bucket: Bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=object_prefix):
            blob.delete()


class StageSession(StageManager):
    """:class:`StageManager` with a guaranteed cleanup context manager.

    Use as a context manager — cleanup runs on ``__exit__`` even if
    processing raises an exception.

    Examples
    --------
    .. code-block:: python

        with StageSession("data/tmp/stage", run_id="abc") as stage:
            stage.put(local_file, "granule/file.tif")
            process(stage.uri.uri)
        # stage directory is now gone

    Parameters
    ----------
    uri, run_id, persist :
        Passed to :class:`StageManager`.
    """

    def __init__(
        self,
        uri: UriLike,
        run_id: str | None = None,
        persist: bool = False,
    ) -> None:
        # Normalise run_id to a short human-readable string if not given
        if run_id is None:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            run_id = f"{ts}-{uuid4().hex[:6]}"
        super().__init__(uri=uri, run_id=run_id, persist=persist)

    def __enter__(self) -> StageSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


# ── helpers ──────────────────────────────────────────────────────────


def stage_path(base_uri: UriLike, run_id: str) -> OutputLocation:
    """Construct a stage root URI for a given run.

    Parameters
    ----------
    base_uri :
        Stage root, e.g. ``data/tmp/ecostress_stage`` or
        ``gs://bucket/_staging/ecostress``.
    run_id :
        Unique run identifier.

    Returns
    -------
    OutputLocation
        Stage URI with run_id appended, e.g.
        ``data/tmp/ecostress_stage/abc123``.
    """
    loc = OutputLocation(str(base_uri))
    return loc / run_id


__all__ = [
    "StageManager",
    "StageSession",
    "stage_path",
    "UriLike",
]
