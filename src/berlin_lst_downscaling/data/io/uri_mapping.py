"""Read-time resolution of stored canonical asset URIs.

Published artifacts — the ``training/v1`` manifest, the WB3 patch index, and
the ARD/static/dynamic ledgers — record absolute ``gs://`` URIs that were
written while the canonical bucket was ``gs://berlin-lst-data``. The
2026-09-25 GCS cutover copied those objects byte-for-byte to
``gs://berlin-lst-training-data`` **without rewriting the stored provenance**,
so a reader must map the exact old canonical prefix onto the current bucket
when it opens a stored asset URI.

Only the canonical bucket prefix is mapped. A ``gs://`` URI on any other
bucket is a hard error (no silent fallback to the old account), and a
non-``gs://`` value — e.g. a local smoke path — is returned unchanged.
"""

from __future__ import annotations

LEGACY_CANONICAL_BUCKET = "gs://berlin-lst-data/"
CANONICAL_BUCKET = "gs://berlin-lst-training-data/"


def resolve_canonical_uri(uri: str) -> str:
    """Map a stored asset URI onto the current canonical bucket.

    Parameters
    ----------
    uri :
        A stored asset URI (absolute object URI, or a local path).

    Returns
    -------
    str
        The URI with the legacy canonical bucket prefix replaced by the
        current canonical bucket. Local paths and URIs already on the
        current bucket pass through unchanged.

    Raises
    ------
    ValueError
        If *uri* is a ``gs://`` URI on an unexpected bucket, so a stale root
        fails closed instead of reading the old account.
    """
    if uri.startswith(LEGACY_CANONICAL_BUCKET):
        return CANONICAL_BUCKET + uri[len(LEGACY_CANONICAL_BUCKET) :]
    if uri.startswith(CANONICAL_BUCKET):
        return uri
    if uri.startswith("gs://"):
        raise ValueError(
            f"asset URI {uri!r} does not belong to the canonical bucket "
            f"{CANONICAL_BUCKET!r}"
        )
    return uri


__all__ = ["CANONICAL_BUCKET", "LEGACY_CANONICAL_BUCKET", "resolve_canonical_uri"]
