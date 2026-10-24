"""Derive expected scene counts from a selection manifest bundle.

The v6/v3 chain must not hard-code the 324-scene universe: the v4
AOI-overlap gate drops edge anchors, so the assessable count comes from
the upstream v4 bundle itself. These helpers read the bundle's
``manifest.parquet`` (anchor rows with years) and cross-check it against
``pairings.parquet`` (one row per coupled anchor). Pure table functions
are unit-testable; :func:`expected_counts_for_bundle` does the IO.
"""

from __future__ import annotations

import io

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import exists, read_bytes

# Training universe: 2017-2023 train, 2024 validation, 2025 test.
# 2026 is inference-only (metadata, never processed).
ASSESSABLE_YEAR_MIN = 2017
ASSESSABLE_YEAR_MAX = 2025


def anchor_years(manifest_table) -> dict[str, int]:
    """Map anchor scene_id -> year for manifest anchor rows."""
    cols = manifest_table.to_pydict()
    out: dict[str, int] = {}
    for i in range(manifest_table.num_rows):
        if str(cols["source"][i]) != "landsat-c2-l2":
            continue
        if str(cols["role"][i]) != "anchor":
            continue
        out[str(cols["scene_id"][i])] = int(cols["year"][i])
    return out


def expected_counts(manifest_table, pairings_table) -> tuple[int, int]:
    """Return ``(assessable_2017_2025, total_anchors)`` for the bundle.

    Every manifest anchor row must have exactly one pairing row (fail
    closed otherwise): the features universe is the 2017-2025 anchors,
    the training manifest universe is every anchor including 2026.
    """
    anchors = anchor_years(manifest_table)
    pairing_ids = {str(v) for v in pairings_table.column("landsat_scene_id").to_pylist()}
    if set(anchors) != pairing_ids:
        missing = sorted(set(anchors) - pairing_ids)
        extra = sorted(pairing_ids - set(anchors))
        raise ValueError(
            "manifest anchors and pairings disagree "
            f"(anchors without pairing: {missing[:5]}, "
            f"pairings without anchor: {extra[:5]})"
        )
    assessable = sum(
        1 for year in anchors.values() if ASSESSABLE_YEAR_MIN <= year <= ASSESSABLE_YEAR_MAX
    )
    return assessable, len(anchors)


def expected_counts_for_bundle(manifest_parquet_uri: str) -> tuple[int, int]:
    """Read a bundle from its manifest URI and return expected counts."""
    manifest_uri = manifest_parquet_uri.rstrip("/")
    bundle_root = manifest_uri.rsplit("/", 1)[0]
    pairings_uri = f"{bundle_root}/pairings.parquet"
    for uri in (manifest_uri, pairings_uri):
        if not exists(uri):
            raise FileNotFoundError(f"bundle artifact missing: {uri}")
    manifest_table = pq.read_table(io.BytesIO(read_bytes(manifest_uri)))
    pairings_table = pq.read_table(io.BytesIO(read_bytes(pairings_uri)))
    return expected_counts(manifest_table, pairings_table)


def expected_features_count_for_bundle(
    manifest_parquet_uri: str, ard_ledger_uri: str | None = None
) -> tuple[int, int, list[dict]]:
    """Return ``(expected_feature_scenes, assessable, exclusions)``.

    ``exclusions`` are pairings whose S2 partner failed the ARD v10 mosaic
    AOI coverage gate (conservative rule: excluded pairings drop out, no
    replacement partner). When ``ard_ledger_uri`` is None the expected
    count equals the assessable 2017-2025 universe. The dynamic universe is
    unaffected (per-Landsat-anchor products never read S2 ARD).
    """
    assessable, _total = expected_counts_for_bundle(manifest_parquet_uri)
    if ard_ledger_uri is None:
        return assessable, assessable, []
    from berlin_lst_downscaling.data.ard.coverage_gate import (
        excluded_pairings,
        excluded_s2_scene_ids,
    )

    manifest_uri = manifest_parquet_uri.rstrip("/")
    bundle_root = manifest_uri.rsplit("/", 1)[0]
    pairings_uri = f"{bundle_root}/pairings.parquet"
    pairings_table = pq.read_table(io.BytesIO(read_bytes(pairings_uri)))
    ard_table = pq.read_table(io.BytesIO(read_bytes(ard_ledger_uri)))
    manifest_table = pq.read_table(io.BytesIO(read_bytes(manifest_uri)))
    years = anchor_years(manifest_table)
    exclusions = excluded_pairings(pairings_table, set(excluded_s2_scene_ids(ard_table)))
    assessable_excluded = [
        row
        for row in exclusions
        if ASSESSABLE_YEAR_MIN <= years.get(row["landsat_scene_id"], -1) <= ASSESSABLE_YEAR_MAX
    ]
    return assessable - len(assessable_excluded), assessable, assessable_excluded


__all__ = [
    "ASSESSABLE_YEAR_MAX",
    "ASSESSABLE_YEAR_MIN",
    "anchor_years",
    "expected_counts",
    "expected_counts_for_bundle",
    "expected_features_count_for_bundle",
]
