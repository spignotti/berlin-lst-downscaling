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

Only rows that fully verify are seeded. A manifest reuse key without a
valid done reuse row is a *processing gap*, not an error: the key is
reported in the machine-readable gap list (``--gap-out`` JSON plus the
``gaps`` return value, never truncated; the human summary line shows at
most 10) and left for the follow-up ``run_ard`` gap pass
(``configs/ard/landsat_ecostress_v10.yaml``), which processes exactly
those scenes into the v10 root with the current contracts. A gap is a
missing source row, a non-done source row, or a done row whose
referenced COG/flag is empty or missing **at its resolved canonical
URI**. Existence is checked on ``resolve_canonical_uri(path)``
(``data/io/uri_mapping.py``): reuse rows that reference the retired
``gs://berlin-lst-data`` bucket (pre-cutover provenance) verify against
their byte-identical copies under the canonical bucket and are seeded
with the stored URIs kept unchanged as provenance — downstream readers
resolve stored URIs at read time, so carrying the stale URIs forward is
safe and keeps the 2018-08-25 ECOSTRESS scenes out of the gap pass
(which would need a NASA Earthdata login that the VM does not have).
Paths are never rewritten.

Fail-closed: a contract/schema mismatch (schema_version or schema_hash),
a ``gs://`` URI on a foreign (non-canonical, non-legacy) bucket, or a
differing existing destination row aborts with ``RuntimeError`` before
anything is written — neither the ledger nor the gap file. A foreign
bucket is a hard error (not a gap) because ``resolve_canonical_uri``
fails closed on unexpected buckets: silently reprocessing an unknown
provenance root would be the wrong recovery.
``--dry-run`` previews without writing.

Idempotent: already-seeded identical rows are skipped, a second run is a
no-op. ``run_ard`` with ``sources=["sentinel-2-l2a"]`` never touches
these rows (reconcile only iterates manifest S2 rows), so the S2 mosaic
run keeps them and processes only S2. A non-done (failed/pending)
destination row for a key that now verifies from the reuse ledger is
replaced by the verified reuse row (logged); a differing *done*
destination row still aborts.

Usage
-----
    uv run python scripts/operators/seed_ard_v10_reuse.py --dry-run

    uv run python scripts/operators/seed_ard_v10_reuse.py \\
        --source-ledger <reuse-root>/ledger.parquet \\
        --manifest-uri <v4-bundle>/manifest.parquet \\
        --dest-root gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10 \\
        --gap-out gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10/seed_gaps.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.landsat_target import landsat_schema_hash
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow
from berlin_lst_downscaling.data.io import atomic_write, exists, read_bytes
from berlin_lst_downscaling.data.io.uri_mapping import resolve_canonical_uri

_REPO = Path(__file__).resolve().parents[2]

REUSE_SOURCES = ("landsat-c2-l2", "ecostress")

DEFAULT_MANIFEST_URI = (
    "gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/manifest.parquet"
)
DEFAULT_SOURCE_LEDGER = (
    "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019/ledger.parquet"
)
DEFAULT_DEST_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"

# Human summary lines show at most this many gap keys; the ``gaps``
# return value and the ``--gap-out`` JSON file always carry the full list.
_SUMMARY_GAP_LIMIT = 10

# Machine-readable gap reasons. ``missing_row``: no source-ledger row for
# the manifest key. ``not_done``: the source row is not done (detail holds
# the status). ``missing_cog``/``missing_flag``: the done row's referenced
# URI is empty, or the object is missing at its *resolved* canonical URI
# (detail holds both the stored and the resolved URI). Existence is
# checked on ``resolve_canonical_uri(path)`` so pre-cutover
# ``gs://berlin-lst-data`` provenance verifies against its byte-identical
# canonical copy; the stored URIs are seeded unchanged. A ``gs://`` URI on
# any other (foreign) bucket is a hard fail-closed error, never a gap.
_GAP_MISSING_ROW = "missing_row"
_GAP_NOT_DONE = "not_done"
_GAP_MISSING_COG = "missing_cog"
_GAP_MISSING_FLAG = "missing_flag"

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
    gap_out: str | None = None,
) -> dict:
    """Copy verified reuse rows into the destination ledger.

    Only rows whose ``(scene_id, source)`` appears in the selection
    bundle are copied: the v4 AOI-overlap gate drops edge anchors, and
    the ARD validator requires exact manifest/ledger key equality, so
    seeding dropped scenes would fail validation.

    A manifest key counts as covered only once a done, contract-valid
    reuse row with present COG/flag is staged or already seeded. Any
    other manifest key — no source row, a non-done source row, or a done
    row with an empty COG/flag or a COG/flag missing at its resolved
    canonical URI — is a *processing gap*: it is reported in the ``gaps``
    return value (full list of ``{"scene_id", "source", "reason",
    "detail"}`` dicts, sorted, never truncated; the detail for
    ``missing_cog``/``missing_flag`` holds both the stored and the
    resolved URI) and, unless ``dry_run``, in the ``--gap-out`` JSON
    file, while the verified rows are still written. The follow-up gap
    ``run_ard`` pass processes exactly those scenes into the v10 root.

    Still fail-closed (``RuntimeError`` before any write — neither the
    ledger nor the gap file): a schema_version/schema_hash mismatch
    against the current contracts, a ``gs://`` URI on a foreign bucket,
    or a differing *done* destination row (refusing to overwrite). A
    non-done destination row for a key that now verifies is replaced by
    the verified reuse row instead of aborting.

    Returns a ``{"copied", "staged", "skipped_identical",
    "replaced_nondone", "skipped_not_in_manifest", "gaps", "gap_count"}``
    summary. The human log line shows at most 10 gap keys; ``gaps`` and
    the gap file are complete.
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
    staged_keys: set[tuple[str, str]] = set()
    skipped_identical = 0
    replaced_nondone = 0
    skipped_not_in_manifest = 0
    gap_reasons: dict[tuple[str, str], dict] = {}
    errors: list[str] = []

    def _note_gap(key: tuple[str, str], reason: str, detail: str) -> None:
        if key not in staged_keys:
            gap_reasons.setdefault(
                key,
                {"scene_id": key[0], "source": key[1], "reason": reason, "detail": detail},
            )

    for reuse_source in REUSE_SOURCES:
        for row in source.scenes_for_source(reuse_source):
            key = (row.scene_id, row.source)
            if key not in manifest_keys:
                skipped_not_in_manifest += 1
                continue
            if key in staged_keys:
                continue
            if row.status != "done":
                _note_gap(key, _GAP_NOT_DONE, f"status={row.status}")
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
            # Existence is checked on the resolved canonical URI (legacy
            # ``gs://berlin-lst-data`` provenance verifies against its
            # canonical copy); the stored URIs are seeded unchanged.
            # A foreign-bucket gs:// URI fails closed as a hard error.
            cog, flag = row.path_cog, row.path_flag
            try:
                resolved_cog = resolve_canonical_uri(cog) if cog else ""
                resolved_flag = resolve_canonical_uri(flag) if flag else ""
            except ValueError as exc:
                errors.append(f"{row.scene_id} [{reuse_source}]: {exc} — refusing to seed")
                continue
            if not cog or not exists(resolved_cog):
                _note_gap(
                    key,
                    _GAP_MISSING_COG,
                    f"stored={cog or '<empty path_cog>'} "
                    f"resolved={resolved_cog or '<empty path_cog>'}",
                )
                continue
            if not flag or not exists(resolved_flag):
                _note_gap(
                    key,
                    _GAP_MISSING_FLAG,
                    f"stored={flag or '<empty path_flag>'} "
                    f"resolved={resolved_flag or '<empty path_flag>'}",
                )
                continue
            gap_reasons.pop(key, None)
            existing = dest.get(row.scene_id, row.source)
            if existing is not None:
                if _identity(existing) == _identity(row):
                    skipped_identical += 1
                    staged_keys.add(key)
                elif existing.status != "done":
                    print(
                        f"seed: replacing non-done dest row {row.scene_id} "
                        f"[{reuse_source}] status={existing.status} "
                        "with the verified reuse row"
                    )
                    replaced_nondone += 1
                    staged.append(row)
                    staged_keys.add(key)
                else:
                    errors.append(
                        f"{row.scene_id} [{reuse_source}]: destination row differs — "
                        "refusing to overwrite (new root required)"
                    )
            else:
                staged.append(row)
                staged_keys.add(key)

    for key in sorted(manifest_keys - staged_keys):
        _note_gap(key, _GAP_MISSING_ROW, "no reuse row for this manifest key")
    gaps = [gap_reasons[key] for key in sorted(gap_reasons)]

    if errors:
        raise RuntimeError(
            f"seeding refused with {len(errors)} finding(s), nothing written:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    if not dry_run:
        for row in staged:
            dest.upsert(row)
        if gap_out:
            atomic_write(gap_out, json.dumps({"gaps": gaps}, indent=2).encode())
    return {
        "copied": len(staged) if not dry_run else 0,
        "staged": len(staged),
        "skipped_identical": skipped_identical,
        "replaced_nondone": replaced_nondone,
        "skipped_not_in_manifest": skipped_not_in_manifest,
        "gaps": gaps,
        "gap_count": len(gaps),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-ledger", default=DEFAULT_SOURCE_LEDGER)
    parser.add_argument("--dest-root", default=DEFAULT_DEST_ROOT)
    parser.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI)
    parser.add_argument(
        "--gap-out",
        default=None,
        help=(
            "URI for the machine-readable gap list JSON "
            "(default: <dest-root>/seed_gaps.json; use '' to skip writing)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gap_out = args.gap_out
    if gap_out is None:
        gap_out = f"{args.dest_root.rstrip('/')}/seed_gaps.json"
    elif gap_out == "":
        gap_out = None
    try:
        summary = seed(
            args.source_ledger,
            args.dest_root,
            args.manifest_uri,
            dry_run=args.dry_run,
            gap_out=None if args.dry_run else gap_out,
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    mode = "dry-run, nothing written" if args.dry_run else "seeded"
    shown = ", ".join(
        f"{g['scene_id']} [{g['source']}]" for g in summary["gaps"][:_SUMMARY_GAP_LIMIT]
    )
    print(
        f"seed {mode}: staged={summary['staged']} copied={summary['copied']} "
        f"identical={summary['skipped_identical']} "
        f"replaced_nondone={summary['replaced_nondone']} "
        f"not_in_manifest={summary['skipped_not_in_manifest']} "
        f"gaps={summary['gap_count']}"
        + (f" gap_sample=[{shown}]" if summary["gap_count"] else "")
        + (f" gap_out={gap_out}" if gap_out and not args.dry_run else "")
    )
    if summary["gap_count"]:
        # Full machine-readable gap list (never truncated) for the log.
        print(f"GAPS_JSON: {json.dumps(summary['gaps'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
