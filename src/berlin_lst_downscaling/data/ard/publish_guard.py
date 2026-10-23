"""Refuse writes into published release prefixes.

Schema bumps make ``reconcile`` treat every finished row as changed.
Comments on the old configs do not stop that write. These checks run
before a ledger is opened, so a Sentinel-2 schema-9 run cannot land in
a published schema-8 root, a schema-10 mosaic cannot land in the published
schema-9 root, and a feature schema-4 run cannot land in
``features/v3``, ``features/v4``, or ``features/v5``.
"""

from __future__ import annotations

# Roots that already hold published Sentinel-2 ARD at schema 8.
_FROZEN_SENTINEL2_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/ard/full/2017-2026-cutoff-20260717T235959Z",
        "gs://berlin-lst-training-data/ard/full/2017-2026-qa-20261019",
        "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019",
    }
)

_PUBLISHED_SENTINEL2_V9_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-harmonized-v9"

# Roots that already hold a published 28-band feature release.
_FROZEN_FEATURE_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/features/v3",
        "gs://berlin-lst-training-data/features/v4",
        "gs://berlin-lst-training-data/features/v5",
    }
)


# Canonical TESSERA embedding product prefix (new in the TESSERA arm).
# Nothing is published there yet; the operator publishes each year once via
# scripts/operators/run_tessera_embeddings.py. After a year is published,
# add its directory URI to _FROZEN_TESSERA_ROOTS so republish paths fail
# closed instead of overwriting the canonical product.
TESSERA_PRODUCT_PREFIX = "gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate"

# Year directories already holding a published TESSERA product. Empty until
# the operator run publishes; add
# f"{TESSERA_PRODUCT_PREFIX}/<year>" here once that year's complete.json
# is published.
_FROZEN_TESSERA_ROOTS = frozenset[str]()


class FrozenPrefixError(RuntimeError):
    """A pipeline tried to publish into a prefix that already has a release."""


def _normalize(root: str) -> str:
    return root.strip().rstrip("/")


def assert_sentinel2_output_root(output_root: str, *, schema_version: int = 9) -> None:
    """Block schema-8 roots and prevent mosaic writes into published v9.

    Landsat and ECOSTRESS runs are not checked here. Their schema
    versions are unchanged, and the Landsat QA configs still publish
    into the QA roots.
    """
    if _normalize(output_root) in _FROZEN_SENTINEL2_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published Sentinel-2 prefix. "
            "Schema 9 is published only by configs/ard/sentinel2_harmonized_v9.yaml."
        )
    if schema_version >= 10 and _normalize(output_root) == _PUBLISHED_SENTINEL2_V9_ROOT:
        raise FrozenPrefixError(
            f"{output_root} is a published Sentinel-2 schema-9 prefix. "
            "Schema 10 requires the new root in configs/ard/sentinel2_mosaic_v10.yaml."
        )


def assert_feature_output_root(output_root: str) -> None:
    """Block feature-stack writes into features/v3, v4, or v5."""
    if _normalize(output_root) in _FROZEN_FEATURE_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published feature prefix. "
            "The next feature release is published only by configs/features/v6.yaml."
        )


def assert_tessera_output_root(output_root: str) -> None:
    """Block embedding writes into an already-published TESSERA year prefix.

    The canonical prefix itself stays writable until the operator publishes;
    once a year's ``complete.json`` is published, its directory URI belongs
    in :data:`_FROZEN_TESSERA_ROOTS` and any further write there raises.
    """
    if _normalize(output_root) in _FROZEN_TESSERA_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published TESSERA embedding prefix. "
            "Canonical TESSERA products are immutable; publish a new version "
            "under a new prefix instead of overwriting."
        )


__all__ = [
    "TESSERA_PRODUCT_PREFIX",
    "FrozenPrefixError",
    "assert_feature_output_root",
    "assert_sentinel2_output_root",
    "assert_tessera_output_root",
]
