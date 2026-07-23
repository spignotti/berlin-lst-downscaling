# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
# ]
# ///
"""Standalone manifest bundle validator — offline + upstream identity checks.

Usage
-----
    # Offline validation (no network)
    uv run python scripts/validate_manifest.py \
        --manifest data/ard/manifest.parquet \
        --pairings data/ard/pairings.parquet \
        --report data/ard/manifest_report.json

    # With upstream PC/CMR identity resolution
    uv run python scripts/validate_manifest.py \
        --manifest data/ard/manifest.parquet \
        --pairings data/ard/pairings.parquet \
        --report data/ard/manifest_report.json \
        --resolve-upstream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import pyarrow.parquet as pq


def _file_hash(path: str) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manifest bundle")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairings", required=True)
    parser.add_argument("--report", required=False)
    parser.add_argument(
        "--resolve-upstream",
        action="store_true",
        help="Resolve PC/CMR item identity (requires network)",
    )
    args = parser.parse_args()

    from berlin_lst_downscaling.data.selection.validate import (
        validate_manifest_table,
        validate_pairings_table,
        validate_report_json,
    )

    all_ok = True

    # ── Load and validate manifest ──────────────────────────────────────
    print(f"Loading manifest: {args.manifest}")
    manifest_table = pq.read_table(args.manifest)
    print(f"  {manifest_table.num_rows} rows, {manifest_table.num_columns} columns")

    r = validate_manifest_table(manifest_table)
    if not r.ok:
        all_ok = False
        for e in r.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
    else:
        print("  Manifest: OK")

    # ── Load and validate pairings ──────────────────────────────────────
    print(f"Loading pairings: {args.pairings}")
    pairings_table = pq.read_table(args.pairings)
    print(f"  {pairings_table.num_rows} rows, {pairings_table.num_columns} columns")

    r = validate_pairings_table(pairings_table, manifest_table)
    if not r.ok:
        all_ok = False
        for e in r.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
    for w in r.warnings:
        print(f"  WARN: {w}", file=sys.stderr)
    if r.ok:
        print("  Pairings: OK")

    # ── Validate report ─────────────────────────────────────────────────
    if args.report:
        print(f"Loading report: {args.report}")
        with open(args.report) as f:
            report = json.load(f)

        mf_hash = _file_hash(args.manifest)
        pf_hash = _file_hash(args.pairings)

        r = validate_report_json(report, mf_hash, pf_hash)
        if not r.ok:
            all_ok = False
            for e in r.errors:
                print(f"  ERROR: {e}", file=sys.stderr)
        else:
            print("  Report: OK")

    # ── Upstream resolution ─────────────────────────────────────────────
    if args.resolve_upstream:
        print("Resolving upstream identities...")
        upstream_ok = _resolve_pc_items(manifest_table)
        if not upstream_ok:
            all_ok = False

    if all_ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nFAILED: errors found in bundle.", file=sys.stderr)
        return 1


def _resolve_pc_items(table) -> bool:
    """Resolve Planetary Computer STAC items by exact ID (requires network).

    Returns True if all items resolved successfully, False on any error.
    Uses bounded retries for rate-limited operations.
    """
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    from berlin_lst_downscaling.data.acquisition.pc_client import (
        get_catalog,
        resolve_item_from_href,
    )

    sources = table.column("source").to_pylist()
    ids = table.column("scene_id").to_pylist()
    hrefs = table.column("item_href").to_pylist()

    pc_rows = [
        (sid, href)
        for sid, src, href in zip(ids, sources, hrefs, strict=True)
        if src in ("landsat-c2-l2", "sentinel-2-l2a") and href is not None
    ]

    if not pc_rows:
        print("  No PC STAC rows to resolve.")
        return True

    cat = get_catalog()
    errors = []
    resolved = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        reraise=True,
    )
    def _resolve_one(sid, expected_href):
        # Try direct HREF resolution first (faster, avoids catalog search)
        if expected_href:
            try:
                item = resolve_item_from_href(expected_href, expected_id=sid)
                return item
            except Exception:  # noqa: S110 — fallback to catalog search
                pass
        # Fallback to catalog search
        search = cat.search(
            collections=["landsat-c2-l2"] if "LC" in sid else ["sentinel-2-l2a"],
            ids=[sid],
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"Not found: {sid}")
        return items[0]

    for i, (sid, expected_href) in enumerate(pc_rows):
        try:
            _resolve_one(sid, expected_href)
            resolved += 1
        except Exception as exc:
            errors.append(f"  {sid}: {exc}")

        if (i + 1) % 50 == 0:
            print(f"  ... resolved {i + 1}/{len(pc_rows)}")

    if errors:
        print(f"\n  {len(errors)} upstream errors:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return False

    print(f"  All {resolved} items resolved successfully.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
