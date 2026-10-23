"""Refuse writes into published release prefixes.

Schema bumps make ``reconcile`` treat every finished row as changed.
Comments on the old configs do not stop that write. These checks run
before a ledger is opened, so a Sentinel-2 schema-9 run cannot land in
a published schema-8 root, a schema-10 mosaic cannot land in the published
schema-9 root, and a feature schema-4 run cannot land in
``features/v3``, ``features/v4``, or ``features/v5``.

The v6 chain roots (selection v4, S2 ARD v10, features/v6, training/v3,
patch-index v3, manifests/v4) are protected-after-publish in the same
style: old-schema writes into a new root are refused immediately, so a
legacy config can never pollute the new release. Once the v6 chain is
published, a follow-up adds the new roots to the unconditional frozen
sets below (one line each).
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

# New v6-chain roots (protected-after-publish; see module docstring).
SENTINEL2_V10_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
FEATURES_V6_ROOT = "gs://berlin-lst-training-data/features/v6"
TRAINING_V3_ROOT = "gs://berlin-lst-training-data/training/v3"
PATCH_INDEX_V3_ROOT = "gs://berlin-lst-training-data/training/patch-index/v3"
MANIFESTS_V4_ROOT = "gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z"
STATIC_V6_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2",
        "gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2",
        "gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2",
    }
)
EDA_V3_ROOT = "gs://berlin-lst-training-data/evaluation/eda/v3"

# Published training/patch-index roots (immutable; new writes go to v3).
_FROZEN_TRAINING_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/training/v1",
        "gs://berlin-lst-training-data/training/v2",
    }
)
_FROZEN_PATCH_INDEX_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/training/patch-index/v1",
        "gs://berlin-lst-training-data/training/patch-index/v2",
    }
)
_FROZEN_MANIFESTS_V3_ROOTS = frozenset(
    {
        "gs://berlin-lst-training-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2",
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


def assert_sentinel2_output_root(output_root: str, *, schema_version: int = 9) -> None:
    """Block schema-8 roots and prevent mosaic writes into published v9.

    Landsat and ECOSTRESS runs are not checked here. Their schema
    versions are unchanged, and the Landsat QA configs still publish
    into the QA roots.

    A legacy schema-9 (or older) Sentinel-2 run is additionally refused
    into the new v10 root so the old single-tile contract can never
    pollute the mosaic release; schema 10 is the only writer there.
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
    if schema_version < 10 and _normalize(output_root) == SENTINEL2_V10_ROOT:
        raise FrozenPrefixError(
            f"{output_root} is the schema-10 mosaic release. "
            "Schema 9 and older must not write there; "
            "use configs/ard/sentinel2_mosaic_v10.yaml."
        )


def assert_feature_output_root(output_root: str, *, schema_version: int | None = None) -> None:
    """Block feature-stack writes into features/v3, v4, or v5.

    A legacy feature schema (anything but 5) is additionally refused
    into the new features/v6 root so the published 28-band contract can
    never pollute the v6 release. The designated v6 run (schema 5)
    passes; callers that do not know their schema pass nothing and keep
    the previous behaviour.
    """
    if _normalize(output_root) in _FROZEN_FEATURE_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published feature prefix. "
            "The next feature release is published only by configs/features/v6.yaml."
        )
    if (
        schema_version is not None
        and schema_version != 5
        and _normalize(output_root) == FEATURES_V6_ROOT
    ):
        raise FrozenPrefixError(
            f"{output_root} is the v6 (schema 5) release. "
            "Older feature schemas must not write there."
        )


def assert_training_output_root(output_root: str) -> None:
    """Block training-data writes into the published v1/v2 roots."""
    if _normalize(output_root) in _FROZEN_TRAINING_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published training prefix. "
            "The next training release is published only by configs/training/v3.yaml."
        )


def assert_patch_index_output_root(output_root: str) -> None:
    """Block patch-index writes into the published v1/v2 roots."""
    if _normalize(output_root) in _FROZEN_PATCH_INDEX_ROOTS:
        raise FrozenPrefixError(
            f"{output_root} is a published patch-index prefix. "
            "The next patch index is published only by "
            "configs/training/patch_index_v3.yaml."
        )


def assert_manifests_output_root(output_root: str) -> None:
    """Block manifest writes into the published manifests/v3 release."""
    normalized = _normalize(output_root)
    if normalized in _FROZEN_MANIFESTS_V3_ROOTS or normalized.startswith(
        "gs://berlin-lst-training-data/manifests/v3/"
    ):
        raise FrozenPrefixError(
            f"{output_root} is a published manifest prefix. "
            "The v4 bundle is published only by "
            "configs/selection/full_2017_2026_v4.yaml."
        )


__all__ = [
    "EDA_V3_ROOT",
    "FEATURES_V6_ROOT",
    "MANIFESTS_V4_ROOT",
    "PATCH_INDEX_V3_ROOT",
    "SENTINEL2_V10_ROOT",
    "STATIC_V6_ROOTS",
    "TRAINING_V3_ROOT",
    "FrozenPrefixError",
    "assert_feature_output_root",
    "assert_manifests_output_root",
    "assert_patch_index_output_root",
    "assert_sentinel2_output_root",
    "assert_training_output_root",
]
