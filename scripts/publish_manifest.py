#!/usr/bin/env python3
"""Publish a local manifest bundle to GCS.

Uploads manifest.parquet and pairings.parquet first, then
manifest_report.json last as the publication marker.

Usage:
    uv run python scripts/publish_manifest.py \
        --local-root data/manifest_build/v3/2017-2026-cutoff-20260717T235959Z \
        --publish-root gs://berlin-lst-data/manifests/v3/bundle-abc123-20260718T120000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish manifest bundle to GCS")
    parser.add_argument("--local-root", required=True, help="Local directory containing the bundle")
    parser.add_argument(
        "--publish-root",
        required=True,
        help="GCS URI to publish to (e.g. gs://bucket/manifests/v3/bundle-id)",
    )
    args = parser.parse_args()

    local = Path(args.local_root)
    publish = args.publish_root.rstrip("/")

    # Verify local bundle exists
    manifest_local = local / "manifest.parquet"
    pairings_local = local / "pairings.parquet"
    report_local = local / "manifest_report.json"

    for p in [manifest_local, pairings_local, report_local]:
        if not p.is_file():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1

    # Import storage helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from berlin_lst_downscaling.data.io.storage import atomic_upload, exists

    # Check if already published
    manifest_gcs = f"{publish}/manifest.parquet"
    if exists(manifest_gcs):
        print(f"ERROR: Bundle already published at {publish}", file=sys.stderr)
        print(
            "Published manifest bundles are immutable — use a new publish-root suffix (e.g. -r2).",
            file=sys.stderr,
        )
        return 1

    # Pre-flight: validate the local bundle before upload.
    import json as _json

    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.selection.validate import (
        validate_manifest_table,
        validate_pairings_table,
        validate_report_json,
    )

    print("Pre-flight: validating local bundle...")
    mf_table = pq.read_table(manifest_local)
    pf_table = pq.read_table(pairings_local)
    mf_hash = _file_hash(manifest_local)
    pf_hash = _file_hash(pairings_local)
    with open(report_local) as f:
        report = _json.load(f)
    errs = []
    for label, r in (
        ("manifest", validate_manifest_table(mf_table)),
        ("pairings", validate_pairings_table(pf_table, mf_table)),
        ("report", validate_report_json(report, mf_hash, pf_hash)),
    ):
        if not r.ok:
            errs.extend(f"{label}: {e}" for e in r.errors)
    if errs:
        print("ERROR: local bundle validation failed:", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("  Local bundle OK")

    # Upload in order: manifest, pairings, then report (publication marker).
    # All destinations must be absent — if_generation_match=0 enforces that
    # even if a concurrent publisher races us to the same prefix.
    print(f"Publishing to {publish}")
    for name, local_path in [
        ("manifest.parquet", manifest_local),
        ("pairings.parquet", pairings_local),
    ]:
        dst = f"{publish}/{name}"
        print(f"  Uploading {name}...")
        atomic_upload(local_path, dst, overwrite=False, if_generation_match=0)

    # Report last — this is the publication marker
    print("  Uploading manifest_report.json (publication marker)...")
    atomic_upload(
        report_local, f"{publish}/manifest_report.json", overwrite=False, if_generation_match=0
    )

    # Remote validation — re-read from GCS and verify
    print("\nVerifying published bundle...")
    from berlin_lst_downscaling.data.io.storage import read_bytes

    manifest_gcs = f"{publish}/manifest.parquet"
    pairings_gcs = f"{publish}/pairings.parquet"
    report_gcs = f"{publish}/manifest_report.json"

    try:
        manifest_table = pq.read_table(io.BytesIO(read_bytes(manifest_gcs)))
        pairings_table = pq.read_table(io.BytesIO(read_bytes(pairings_gcs)))
        report = _json.loads(read_bytes(report_gcs))

        # Verify hashes
        local_mf_hash = _file_hash(manifest_local)
        local_pf_hash = _file_hash(pairings_local)
        if report.get("manifest_hash") != local_mf_hash:
            print("ERROR: manifest hash mismatch after publish", file=sys.stderr)
            return 1
        if report.get("pairings_hash") != local_pf_hash:
            print("ERROR: pairings hash mismatch after publish", file=sys.stderr)
            return 1
        for label, r in (
            ("manifest", validate_manifest_table(manifest_table)),
            ("pairings", validate_pairings_table(pairings_table, manifest_table)),
            ("report", validate_report_json(report, local_mf_hash, local_pf_hash)),
        ):
            if not r.ok:
                print(f"ERROR: remote {label} validation failed", file=sys.stderr)
                for e in r.errors:
                    print(f"  {e}", file=sys.stderr)
                return 1
        print(
            f"  Remote validation passed: {manifest_table.num_rows} manifest rows, "
            f"{pairings_table.num_rows} pairings"
        )
    except Exception as exc:
        print(f"WARNING: Remote validation failed: {exc}", file=sys.stderr)
        print("Bundle was published but could not be verified.", file=sys.stderr)
        return 1

    print(f"\nPublished bundle: {publish}")
    print(f"Use: manifest_uri={manifest_gcs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
