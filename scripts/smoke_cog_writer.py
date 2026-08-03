#!/usr/bin/env python3
"""Writer smoke test for COG layout recovery.

Creates large numeric and flag COGs using the writer, validates them
with the strict validator, and cleans up. Used to prove the writer
produces strict-clean COGs before recovery begins.

Usage::

    # Local smoke (no GCS)
    uv run python scripts/smoke_cog_writer.py

    # GCS smoke (requires ADC)
    uv run python scripts/smoke_cog_writer.py --gcs --output-root gs://berlin-lst-data-recovery/smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


def _create_large_numeric_cog(output_path: Path, *, size: int = 2048) -> None:
    """Create a large numeric COG for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    # Create a float32 raster
    data = np.random.default_rng(42).standard_normal((1, size, size)).astype(np.float32)

    # Berlin extent in EPSG:25833
    transform = from_bounds(
        369190, 5838410, 369190 + size * 10, 5838410 + size * 10,
        size, size,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:25833",
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "lst")
        dst.update_tags(1, units="K")


def _create_large_flag_cog(output_path: Path, *, size: int = 2048) -> None:
    """Create a large flag COG for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    # Create a uint8 flag raster
    data = np.random.default_rng(42).integers(0, 4, size=(1, size, size), dtype=np.uint8)

    # Berlin extent in EPSG:25833
    transform = from_bounds(
        369190, 5838410, 369190 + size * 10, 5838410 + size * 10,
        size, size,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs="EPSG:25833",
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "flag")


def _copy_to_cog(
    src_path: Path,
    dst_path: Path,
    *,
    is_flag: bool = False,
) -> None:
    """Copy a GTiff to COG using the GDAL COG driver."""
    from rasterio.shutil import copy as rio_copy

    options = {
        "driver": "COG",
        "blocksize": 512,
        "compress": "ZSTD" if is_flag else "LZW",
        "predictor": 1 if is_flag else 3,
        "bigtiff": "IF_SAFER",
        "overview_resampling": "nearest" if is_flag else "cubic",
    }

    rio_copy(str(src_path), str(dst_path), **options)


def _validate_strict(cog_path: Path) -> tuple[bool, list[str], list[str]]:
    """Validate a COG using the strict validator."""
    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog

    result = validate_strict_cog(str(cog_path))
    return result.valid, list(result.errors), list(result.warnings)


def _run_local_smoke(tmp_dir: Path) -> int:
    """Run local smoke test."""
    print("=== Local Writer Smoke Test ===\n")

    # Test 1: Large numeric COG
    print("1. Creating large numeric COG (2048x2048 float32)...")
    numeric_src = tmp_dir / "numeric_staging.tif"
    numeric_cog = tmp_dir / "numeric_cog.tif"
    _create_large_numeric_cog(numeric_src)
    _copy_to_cog(numeric_src, numeric_cog, is_flag=False)

    valid, errors, warnings = _validate_strict(numeric_cog)
    if not valid:
        print("   FAIL: numeric COG not strict-clean")
        print(f"   Errors: {errors}")
        print(f"   Warnings: {warnings}")
        return 1
    print("   PASS: numeric COG is strict-clean")

    # Test 2: Large flag COG
    print("\n2. Creating large flag COG (2048x2048 uint8)...")
    flag_src = tmp_dir / "flag_staging.tif"
    flag_cog = tmp_dir / "flag_cog.tif"
    _create_large_flag_cog(flag_src)
    _copy_to_cog(flag_src, flag_cog, is_flag=True)

    valid, errors, warnings = _validate_strict(flag_cog)
    if not valid:
        print("   FAIL: flag COG not strict-clean")
        print(f"   Errors: {errors}")
        print(f"   Warnings: {warnings}")
        return 1
    print("   PASS: flag COG is strict-clean")

    # Test 3: Verify file sizes
    numeric_size = numeric_cog.stat().st_size
    flag_size = flag_cog.stat().st_size
    print("\n3. File sizes:")
    print(f"   Numeric: {numeric_size / 1024 / 1024:.2f} MB")
    print(f"   Flag: {flag_size / 1024 / 1024:.2f} MB")

    if numeric_size < 1024 * 1024:  # < 1 MB
        print("   WARNING: Numeric COG seems too small")
    if flag_size < 1024 * 1024:  # < 1 MB
        print("   WARNING: Flag COG seems too small")

    print("\n=== All Local Smoke Tests PASSED ===")
    return 0


def _run_gcs_smoke(output_root: str) -> int:
    """Run GCS smoke test."""
    print("=== GCS Writer Smoke Test ===\n")

    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    from berlin_lst_downscaling.data.ard.cog_recovery_gcs import (
        create_scratch_object,
        delete_scratch_object,
        snapshot_gcs_metadata,
    )

    bucket_name = output_root.replace("gs://", "").split("/")[0]
    prefix = "/".join(output_root.replace("gs://", "").split("/")[1:])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Upload numeric COG
        print("1. Creating and uploading numeric COG...")
        numeric_src = tmp_path / "numeric_staging.tif"
        numeric_cog = tmp_path / "numeric_cog.tif"
        _create_large_numeric_cog(numeric_src)
        _copy_to_cog(numeric_src, numeric_cog, is_flag=False)

        numeric_key = f"{prefix}/smoke_numeric.tif"
        with open(numeric_cog, "rb") as f:
            numeric_data = f.read()

        try:
            result = create_scratch_object(
                bucket_name, numeric_key, numeric_data, content_type="image/tiff"
            )
            print(f"   Uploaded: {result['uri']}")
            print(f"   Generation: {result['generation']}")
            print(f"   CRC32C: {result['crc32c']}")
        except Exception as exc:
            print(f"   FAIL: {exc}")
            return 1

        # Validate remote
        numeric_uri = f"gs://{bucket_name}/{numeric_key}"
        cog_result = validate_strict_cog(numeric_uri)
        if not cog_result.valid:
            print("   FAIL: remote numeric COG not strict-clean")
            print(f"   Errors: {cog_result.errors}")
            print(f"   Warnings: {cog_result.warnings}")
            delete_scratch_object(bucket_name, numeric_key)
            return 1
        print("   PASS: remote numeric COG is strict-clean")

        # Test 2: Upload flag COG
        print("\n2. Creating and uploading flag COG...")
        flag_src = tmp_path / "flag_staging.tif"
        flag_cog = tmp_path / "flag_cog.tif"
        _create_large_flag_cog(flag_src)
        _copy_to_cog(flag_src, flag_cog, is_flag=True)

        flag_key = f"{prefix}/smoke_flag.tif"
        with open(flag_cog, "rb") as f:
            flag_data = f.read()

        try:
            result = create_scratch_object(
                bucket_name, flag_key, flag_data, content_type="image/tiff"
            )
            print(f"   Uploaded: {result['uri']}")
            print(f"   Generation: {result['generation']}")
            print(f"   CRC32C: {result['crc32c']}")
        except Exception as exc:
            print(f"   FAIL: {exc}")
            delete_scratch_object(bucket_name, numeric_key)
            return 1

        # Validate remote
        flag_uri = f"gs://{bucket_name}/{flag_key}"
        cog_result = validate_strict_cog(flag_uri)
        if not cog_result.valid:
            print("   FAIL: remote flag COG not strict-clean")
            print(f"   Errors: {cog_result.errors}")
            print(f"   Warnings: {cog_result.warnings}")
            delete_scratch_object(bucket_name, numeric_key)
            delete_scratch_object(bucket_name, flag_key)
            return 1
        print("   PASS: remote flag COG is strict-clean")

        # Test 3: Verify metadata
        print("\n3. Verifying metadata...")
        numeric_meta = snapshot_gcs_metadata(numeric_uri)
        flag_meta = snapshot_gcs_metadata(flag_uri)
        print(f"   Numeric: {numeric_meta['size']} bytes, {numeric_meta['content_type']}")
        print(f"   Flag: {flag_meta['size']} bytes, {flag_meta['content_type']}")

        # Cleanup
        print("\n4. Cleaning up...")
        delete_scratch_object(bucket_name, numeric_key)
        delete_scratch_object(bucket_name, flag_key)
        print("   Cleaned up scratch objects")

    print("\n=== All GCS Smoke Tests PASSED ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Writer smoke test")
    parser.add_argument("--gcs", action="store_true", help="Run GCS smoke test")
    parser.add_argument("--output-root", default=None, help="GCS output root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.gcs:
        if not args.output_root:
            print("ERROR: --output-root required for GCS smoke", file=sys.stderr)
            return 1
        return _run_gcs_smoke(args.output_root)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            return _run_local_smoke(Path(tmp_dir))


if __name__ == "__main__":
    raise SystemExit(main())
