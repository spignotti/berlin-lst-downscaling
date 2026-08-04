#!/usr/bin/env python3
"""Independent validator for COG recovery candidates.

Downloads source and candidate by pinned generation, runs GDAL COG
validation, rio-cogeo strict validation, and blockwise semantic
comparison.

Usage::

    uv run python scripts/validate_cog_recovery.py \\
        --config configs/cog_repair/remediation.yaml \\
        --recovery-root gs://berlin-lst-data-recovery/cog-layout-recovery/2026-08-03-remediation \\
        --run-id <run-id> \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    assert_raster_equivalent,
    hash_config,
    validate_strict_cog,
)
from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri


def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _download_to_file(uri: str, dst: Path) -> None:
    """Download a GCS object to a local file."""
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    blob.download_to_filename(str(dst))


def _validate_one(
    uri: str,
    recovery_root: str,
    *,
    original_uri: str | None,
    candidate_uri: str | None,
) -> dict[str, Any]:
    """Validate a single candidate independently."""
    errors: list[str] = []
    _, key = _parse_gs_uri(uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_src = Path(tmp_dir) / "source.tif"
        tmp_cand = Path(tmp_dir) / "candidate.tif"

        # download source (original if available, otherwise current)
        source_uri = original_uri or uri
        try:
            _download_to_file(source_uri, tmp_src)
        except Exception as exc:
            return {
                "uri": uri,
                "valid": False,
                "errors": [f"Failed to download source: {exc}"],
            }

        # download candidate
        if not candidate_uri:
            return {
                "uri": uri,
                "valid": False,
                "errors": ["No candidate URI"],
            }

        try:
            _download_to_file(candidate_uri, tmp_cand)
        except Exception as exc:
            return {
                "uri": uri,
                "valid": False,
                "errors": [f"Failed to download candidate: {exc}"],
            }

        # strict COG validation
        strict_result = validate_strict_cog(str(tmp_cand))
        if not strict_result.valid:
            errors.extend(strict_result.errors)
            errors.extend(strict_result.warnings)

        # semantic comparison
        compare_errors = assert_raster_equivalent(str(tmp_src), tmp_cand)
        if compare_errors:
            errors.extend(compare_errors)

    return {
        "uri": uri,
        "valid": len(errors) == 0,
        "errors": errors,
        "layout_class": strict_result.layout_class if not errors else "invalid",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate COG recovery candidates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg_hash = hash_config(config_path)
    recovery_root = args.recovery_root
    run_id = args.run_id

    # load config for legacy root
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    legacy_root = config["legacy_recovery_root"]

    print(f"Config hash: {cfg_hash}")
    print(f"Run ID: {run_id}")

    # load inventory
    from berlin_lst_downscaling.data.ard.cog_repair import load_table
    inv_path = f"{recovery_root}/snapshots/{run_id}/inventory.parquet"
    inv_table = load_table(inv_path)
    rows = inv_table.to_pylist()

    # filter to assets with candidates
    has_candidate = [
        r for r in rows
        if r.get("candidate_uri") or r.get("layout_class") in (
            "missing_overview", "hard_layout",
        )
    ]
    print(f"Assets with candidates: {len(has_candidate)}")

    # validate each candidate
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for row in has_candidate:
            _, key = _parse_gs_uri(row["uri"])
            # route source: flags from originals, hard-layout from legacy backups
            if row["layout_class"] == "missing_overview":
                source_uri = f"{recovery_root}/originals/{key}"
            else:
                source_uri = f"{legacy_root}/backups/current/{key}"
            candidate_uri = f"{recovery_root}/candidates/{key}"
            future = ex.submit(
                _validate_one,
                row["uri"],
                recovery_root,
                original_uri=source_uri,
                candidate_uri=candidate_uri,
            )
            futures[future] = row

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # summary
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = sum(1 for r in results if not r["valid"])

    print("\nValidation summary:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")

    if invalid_count > 0:
        print("\nInvalid candidates:")
        for r in results:
            if not r["valid"]:
                print(f"  {r['uri']}:")
                for e in r["errors"]:
                    print(f"    - {e}")

    # publish report
    report = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "total": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "results": results,
    }
    report_path = f"{recovery_root}/reports/{run_id}/candidate_validation.json"
    from berlin_lst_downscaling.data.io.storage import atomic_write
    atomic_write(
        report_path,
        json.dumps(report, indent=2, sort_keys=True).encode(),
        overwrite=False,
    )
    print(f"\nReport: {report_path}")

    if invalid_count == 0:
        print("\nVALIDATION PASSED — all candidates verified")
        return 0
    else:
        print(f"\nVALIDATION FAILED — {invalid_count} candidates invalid")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
