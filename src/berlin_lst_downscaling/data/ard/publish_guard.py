"""Refuse writes into published release prefixes.

Schema bumps make ``reconcile`` treat every finished row as changed.
Comments on the old configs do not stop that write. These checks run
before a ledger is opened, so a Sentinel-2 schema-9 run cannot land in
a published schema-8 root and a feature schema-4 run cannot land in
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

# Roots that already hold a published 28-band feature release.
_FROZEN_FEATURE_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/features/v3",
        "gs://berlin-lst-training-data/features/v4",
        "gs://berlin-lst-training-data/features/v5",
    }
)


class FrozenPrefixError(RuntimeError):
    """A pipeline tried to publish into a prefix that already has a release."""


def _normalize(root: str) -> str:
    return root.strip().rstrip("/")


def assert_sentinel2_output_root(output_root: str) -> None:
    """Block Sentinel-2 writes into a published schema-8 prefix.

    Landsat and ECOSTRESS runs are not checked here. Their schema
    versions are unchanged, and the Landsat QA configs still publish
    into the QA roots.
    """
    if _normalize(output_root) in _FROZEN_SENTINEL2_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published Sentinel-2 prefix. "
            "Schema 9 is published only by configs/ard/sentinel2_harmonized_v9.yaml."
        )


def assert_feature_output_root(output_root: str) -> None:
    """Block feature-stack writes into features/v3, v4, or v5."""
    if _normalize(output_root) in _FROZEN_FEATURE_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published feature prefix. "
            "Schema 4 is published only by configs/features/v6.yaml."
        )


__all__ = [
    "FrozenPrefixError",
    "assert_feature_output_root",
    "assert_sentinel2_output_root",
]
