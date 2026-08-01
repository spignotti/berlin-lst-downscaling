#!/usr/bin/env python3
"""Canonical ERA5-Land monthly cache prefetcher.

Ensures the monthly ERA5-Land NetCDF cache exists under
``_raw/dynamic/era5_land/YYYY-MM/`` for every primary month
extracted from a manifest. The script is idempotent: existing cache
files are not re-downloaded, but missing months trigger CDS retrieval.

Usage::

    uv run python scripts/cache_era5_months.py \\
        --manifest-uri gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet \\
        --output-root gs://berlin-lst-data/dynamic/full

    uv run python scripts/cache_era5_months.py \\
        --manifest-uri gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet \\
        --output-root gs://berlin-lst-data/dynamic/full \\
        --years 2025
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _primary_months(manifest_uri: str, years: list[int] | None = None) -> set[tuple[int, int]]:
    from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors

    report = load_landsat_anchors(manifest_uri, years=years)
    if not report.ok:
        raise RuntimeError(f"Manifest load failed: {report.errors}")
    months: set[tuple[int, int]] = set()
    for scene in report.scenes:
        acq = scene.acquisition_datetime
        months.add((acq.year, acq.month))
        prev_month = 12 if acq.month == 1 else acq.month - 1
        prev_year = acq.year - 1 if acq.month == 1 else acq.year
        months.add((prev_year, prev_month))
    return months


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure canonical ERA5-Land monthly cache in GCS",
    )
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args = parser.parse_args()

    months = sorted(_primary_months(args.manifest_uri, years=args.years))
    print(f"[cache-era5] {len(months)} months to ensure", flush=True)

    from berlin_lst_downscaling.data.dynamic.era5 import _ensure_month_cached  # noqa: PLC2701

    t0 = time.perf_counter()
    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cache_era5_") as tmp:
        for year, month in months:
            print(f"  {year:04d}-{month:02d} …", end="", flush=True)
            t1 = time.perf_counter()
            path = _ensure_month_cached(
                args.output_root,
                year,
                month,
                run_id="prefetch",
                local_dir=Path(tmp),
            )
            elapsed = time.perf_counter() - t1
            if path is None:
                missing.append(f"{year:04d}-{month:02d}")
                print(f" FAILED ({elapsed:.1f}s)")
            else:
                print(f" OK ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\n[cache-era5] finished in {elapsed:.1f}s")
    if missing:
        print("Missing months:", ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
