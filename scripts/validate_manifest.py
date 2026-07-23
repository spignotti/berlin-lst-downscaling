# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
# ]
# ///
"""Standalone manifest bundle validator — offline + upstream identity checks.

Loads the canonical v3 bundle (manifest + pairings + report) through the
shared loader that ARD/Dynamic consume, so validation works for both
local and gs:// URIs.

Usage
-----
    uv run python scripts/validate_manifest.py \
        --manifest gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet

    # With upstream PC/CMR identity resolution (network):
    uv run python scripts/validate_manifest.py \
        --manifest gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet \
        --resolve-upstream
"""

from __future__ import annotations

import argparse
import sys

from berlin_lst_downscaling.data.selection.validate import (
    load_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manifest bundle")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.parquet inside a canonical v3 bundle root",
    )
    parser.add_argument(
        "--resolve-upstream",
        action="store_true",
        help="Resolve PC STAC item identity (requires network)",
    )
    args = parser.parse_args()

    bundle, result = load_bundle(args.manifest)

    print(f"Bundle root: {bundle.bundle_dir.rstrip('/')}")
    print(f"  manifest rows:    {bundle.manifest_table.num_rows}")
    print(f"  pairings rows:    {bundle.pairings_table.num_rows}")
    print(f"  manifest_report:  {'present' if bundle.report else 'absent'}")

    all_ok = result.ok
    if all_ok:
        print("  Bundle: OK")
    else:
        for e in result.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
    for w in result.warnings:
        print(f"  WARN: {w}", file=sys.stderr)

    if args.resolve_upstream:
        if bundle.manifest_table.num_rows == 0:
            print("  No manifest rows to resolve upstream.")
        else:
            upstream_ok = _resolve_pc_items(bundle.manifest_table)
            if not upstream_ok:
                all_ok = False

    if all_ok:
        print("\nAll checks passed.")
        return 0
    print("\nFAILED: errors found in bundle.", file=sys.stderr)
    return 1


def _resolve_pc_items(table) -> bool:
    """Resolve Planetary Computer STAC items by exact ID (requires network)."""
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
        # Direct HREF resolution is faster; fall back to catalog search on miss.
        if expected_href:
            try:
                return resolve_item_from_href(expected_href, expected_id=sid)
            except Exception:  # noqa: S110 — fallback path below
                pass
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
