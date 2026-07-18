#!/usr/bin/env python3
"""Publish a local manifest bundle to GCS.

Uploads manifest.parquet and pairings.parquet first, then
manifest_report.json last as the publication marker.

Usage:
    uv run python scripts/publish_manifest.py \
        --local-root data/ard/manifests/v3/2017-2026-cutoff-20260717T235959Z \
        --publish-root gs://berlin-lst-data/manifests/v3/bundle-abc123-20260718T120000
"""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--local-root", required=True,
                        help="Local directory containing the bundle")
    parser.add_argument("--publish-root", required=True,
                        help="GCS URI to publish to (e.g. gs://bucket/manifests/v3/bundle-id)")
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
        print("Use a new publish-root or delete the existing bundle.", file=sys.stderr)
        return 1

    # Upload in order: manifest, pairings, then report (publication marker)
    print(f"Publishing to {publish}")
    for name, local_path in [
        ("manifest.parquet", manifest_local),
        ("pairings.parquet", pairings_local),
    ]:
        dst = f"{publish}/{name}"
        print(f"  Uploading {name}...")
        atomic_upload(local_path, dst, overwrite=False)

    # Report last — this is the publication marker
    print("  Uploading manifest_report.json (publication marker)...")
    atomic_upload(report_local, f"{publish}/manifest_report.json", overwrite=False)

    print(f"\nPublished bundle: {publish}")
    print(f"Use: manifest_uri={manifest_gcs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
