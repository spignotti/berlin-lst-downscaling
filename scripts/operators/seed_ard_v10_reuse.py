#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Seed published Landsat/ECOSTRESS ARD rows into the v10 ledger.

The Sentinel-2 ARD v10 root holds fresh schema-10 mosaics, but the
features/training inventory resolves every anchor against a single ARD
ledger, so the new root must also carry the Landsat and ECOSTRESS rows.
Both contracts are unchanged (landsat-c2-l2 schema 7, ecostress schema 7
per ``data/ard/contract.py``), so the rows are copied from the published
reuse root — COGs stay referenced in place, no COG is copied.

Idempotent: already-seeded identical rows are skipped, a second run is a
no-op. Refuses to overwrite a differing existing row. Verifies every
copied row against the current contracts (schema version + policy hash)
and the referenced COGs before writing anything: on any failure nothing
is written (exit 1). ``run_ard`` with ``sources=["sentinel-2-l2a"]``
never touches these rows (reconcile only iterates manifest S2 rows), so
the S2 mosaic run keeps them and processes only S2.

Usage
-----
    uv run python scripts/operators/seed_ard_v10_reuse.py --dry-run

    uv run python scripts/operators/seed_ard_v10_reuse.py \\
        --source-ledger <reuse-root>/ledger.parquet \\
        --manifest-uri <v4-bundle>/manifest.parquet \\
        --dest-root gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.landsat_target import landsat_schema_hash
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow
from berlin_lst_downscaling.data.io import exists, read_bytes

_REPO = Path(__file__).resolve().parents[2]

REUSE_SOURCES = ("landsat-c2-l2", "ecostress")

DEFAULT_MANIFEST_URI = (
    "gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
)
DEFAULT_SOURCE_LEDGER = (
    "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019/ledger.parquet"
)
DEFAULT_DEST_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"

# Identity fields compared before accepting an already-seeded row.
# Volatile bookkeeping (attempts, run_id, timestamps, last error) is
# excluded: only content identity matters.
_IDENTITY_FIELDS = (
    "scene_id",
    "source",
    "year",
    "status",
    "schema_version",
    "schema_hash",
    "path_cog",
    "path_flag",
    "path_stac",
)


def _current_policy() -> tuple[str, str]:
    """Return the current (st_resampling, qa_aggregation) Landsat policy."""
    base = OmegaConf.load(_REPO / "configs" / "ard" / "_base.yaml")
    return str(base["landsat_st_resampling"]), str(base["landsat_qa_aggregation"])


def _expected_hash(source: str, st: str, qa: str) -> str:
    if source == "landsat-c2-l2":
        version = contract_for_source(source).schema_version
        return landsat_schema_hash(version, st, qa)
    return contract_for_source(source).schema_version_str()


def _identity(row: LedgerRow) -> tuple:
    return tuple(getattr(row, field) for field in _IDENTITY_FIELDS)


def _manifest_reuse_keys(manifest_uri: str) -> set[tuple[str, str]]:
    """Return the manifest's Landsat/ECOSTRESS ``(scene_id, source)`` keys."""
    table = pq.read_table(io.BytesIO(read_bytes(manifest_uri)))
    cols = table.to_pydict()
    keys: set[tuple[str, str]] = set()
    for i in range(table.num_rows):
        source = str(cols["source"][i])
        if source in REUSE_SOURCES:
            keys.add((str(cols["scene_id"][i]), source))
    return keys


def seed(
    source_ledger_uri: str,
    dest_root: str,
    manifest_uri: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Copy verified reuse rows into the destination ledger.

    Only rows whose ``(scene_id, source)`` appears in the selection
    bundle are copied: the v4 AOI-overlap gate drops edge anchors, and
    the ARD validator requires exact manifest/ledger key equality, so
    seeding dropped scenes would fail validation. Every manifest
    Landsat/ECOSTRESS key must resolve to a done, contract-valid reuse
    row (fail closed otherwise — a genuinely new scene needs
    processing, not seeding).

    Returns a ``{"copied", "staged", "skipped_identical",
    "skipped_not_done", "skipped_not_in_manifest"}`` summary. Raises
    ``RuntimeError`` before any write on any failure.
    """
    if not exists(source_ledger_uri):
        raise RuntimeError(f"source ledger missing: {source_ledger_uri}")
    if not exists(manifest_uri):
        raise RuntimeError(f"selection manifest missing: {manifest_uri}")
    st, qa = _current_policy()
    manifest_keys = _manifest_reuse_keys(manifest_uri)
    source = Ledger.open(source_ledger_uri)
    dest = Ledger.open(f"{dest_root.rstrip('/')}/ledger.parquet")

    staged: list[LedgerRow] = []
    skipped_identical = 0
    skipped_not_done = 0
    skipped_not_in_manifest = 0
    seen_manifest_keys: set[tuple[str, str]] = set()
    errors: list[str] = []
    for reuse_source in REUSE_SOURCES:
        for row in source.scenes_for_source(reuse_source):
            key = (row.scene_id, row.source)
            if key not in manifest_keys:
                skipped_not_in_manifest += 1
                continue
            seen_manifest_keys.add(key)
            if row.status != "done":
                skipped_not_done += 1
                continue
            expected_version = contract_for_source(reuse_source).schema_version
            if row.schema_version != expected_version:
                errors.append(
                    f"{row.scene_id} [{reuse_source}]: schema_version "
                    f"{row.schema_version} != current contract {expected_version}"
                )
                continue
            expected_hash = _expected_hash(reuse_source, st, qa)
            if row.schema_hash != expected_hash:
                errors.append(
                    f"{row.scene_id} [{reuse_source}]: schema_hash "
                    f"{row.schema_hash!r} != current contract {expected_hash!r}"
                )
                continue
            for label in ("path_cog", "path_flag"):
                uri = getattr(row, label)
                if not uri:
                    errors.append(f"{row.scene_id} [{reuse_source}]: missing {label}")
                elif not exists(uri):
                    errors.append(f"{row.scene_id} [{reuse_source}]: {label} missing: {uri}")
            if any(e.startswith(f"{row.scene_id} [{reuse_source}]") for e in errors):
                continue
            existing = dest.get(row.scene_id, row.source)
            if existing is not None:
                if _identity(existing) == _identity(row):
                    skipped_identical += 1
                else:
                    errors.append(
                        f"{row.scene_id} [{reuse_source}]: destination row differs — "
                        "refusing to overwrite (new root required)"
                    )
            else:
                staged.append(row)

    missing = sorted(manifest_keys - seen_manifest_keys)
    if missing:
        shown = ", ".join(f"{sid} [{src}]" for sid, src in missing[:10])
        errors.append(
            f"{len(missing)} manifest reuse key(s) without a done reuse row: "
            f"{shown} — these scenes need processing, not seeding"
        )

    if errors:
        raise RuntimeError(
            f"seeding refused with {len(errors)} finding(s), nothing written:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    if not dry_run:
        for row in staged:
            dest.upsert(row)
    return {
        "copied": len(staged) if not dry_run else 0,
        "staged": len(staged),
        "skipped_identical": skipped_identical,
        "skipped_not_done": skipped_not_done,
        "skipped_not_in_manifest": skipped_not_in_manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-ledger", default=DEFAULT_SOURCE_LEDGER)
    parser.add_argument("--dest-root", default=DEFAULT_DEST_ROOT)
    parser.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = seed(args.source_ledger, args.dest_root, args.manifest_uri, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    mode = "dry-run, nothing written" if args.dry_run else "seeded"
    print(
        f"seed {mode}: staged={summary['staged']} copied={summary['copied']} "
        f"identical={summary['skipped_identical']} "
        f"not_done={summary['skipped_not_done']} "
        f"not_in_manifest={summary['skipped_not_in_manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
