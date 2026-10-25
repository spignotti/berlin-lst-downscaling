#!/usr/bin/env python3
"""Validate dynamic pipeline products against expected inventory.

Read-only validator: checks ledger status, artifact existence, role
consistency, COG band contract, and geometry mapping for a given
dynamic run root.

Usage::

    uv run python scripts/validators/validate_dynamic.py \\
        --output-root gs://berlin-lst-training-data/dynamic/full \\
        --expected-role anchor \\
        --expected-scenes 324

    uv run python scripts/validators/validate_dynamic.py \\
        --output-root gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2 \\
        --expected-role anchor --expected-scenes-from-manifest <v4 manifest> \\
        --check-bands --era5-channel-set v6

    uv run python scripts/validators/validate_dynamic.py \\
        --output-root gs://berlin-lst-training-data/dynamic/inference/2026 \\
        --expected-role inference \\
        --expected-scenes 21
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import resolve_canonical_uri

# ── ERA5 contracts ───────────────────────────────────────────────────
# v5 is the published 8-band dynamic/full contract. v6 is the 4-band
# scene_value roster; its names, units, and ranges come from the v6
# feature schema in src (single source of truth, never duplicated here).

_ERA5_BAND_NAMES = [
    "t2m_scene",
    "ssrd_scene",
    "ssrd_antecedent_72h_mean",
    "vpd_scene",
    "wind_speed_10m_scene",
    "tp_0_24h",
    "tp_24_48h",
    "tp_48_72h",
]

_ERA5_BAND_UNITS = {
    "t2m_scene": "K",
    "ssrd_scene": "W/m²",
    "ssrd_antecedent_72h_mean": "W/m²",
    "vpd_scene": "kPa",
    "wind_speed_10m_scene": "m/s",
    "tp_0_24h": "mm",
    "tp_24_48h": "mm",
    "tp_48_72h": "mm",
}

_ERA5_INPUT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]

_ERA5_CHANNEL_SETS = ("v5", "v6")


def _v6_era5_contract() -> tuple[list[str], dict[str, str], dict[str, tuple[float, float]]]:
    """Return the v6 ERA5 roster (names, units, valid ranges).

    Single source of truth: the v6 feature schema in src. The names are
    never duplicated in this validator.
    """
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )

    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    era5_channels = [c for c in schema.channels if c.family == "era5"]
    names = [c.name for c in era5_channels]
    units = {c.name: c.unit for c in era5_channels}
    ranges = {c.name: c.valid_range for c in era5_channels if c.valid_range is not None}
    return names, units, ranges


def _read_ledger(output_root: str) -> dict:
    ledger_path = f"{output_root.rstrip('/')}/_state/dynamic/ledger.parquet"
    if ledger_path.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes

        table = pq.read_table(io.BytesIO(read_bytes(ledger_path)))
    else:
        table = pq.read_table(ledger_path)
    return _summarise(table)


def _summarise(table) -> dict:
    sources = table.column("source").to_pylist()
    statuses = table.column("status").to_pylist()
    roles = table.column("role").to_pylist()

    counts = Counter(zip(sources, statuses, strict=False))
    non_done = []
    missing_role = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pydict()
        if row["status"][0] != "done":
            non_done.append(
                {
                    "item_id": row["item_id"][0],
                    "source": row["source"][0],
                    "status": row["status"][0],
                    "attempts": int(row["attempts"][0]),
                    "last_error": row["last_error"][0],
                }
            )
        if row["role"][0] is None:
            missing_role += 1

    return {
        "total_rows": table.num_rows,
        "counts": dict(counts),
        "non_done": non_done,
        "role_counts": dict(Counter(roles)),
        "missing_role": missing_role,
    }


def _validate_cog_bands(cog_uri: str, channel_set: str = "v5") -> list[str]:
    """Check a COG against the selected ERA5 band contract.

    v5 is the published eight-band bilinear contract. v6 is the four-band
    scene_value roster: same dtype/count/name checks plus a plausible-range
    check per band and a spatial-constancy check over valid (finite) pixels.
    An unknown channel set is an error, never a silent pass.
    """
    if channel_set not in _ERA5_CHANNEL_SETS:
        return [
            f"unknown era5 channel set {channel_set!r}: expected one of {list(_ERA5_CHANNEL_SETS)}"
        ]
    errors = []
    try:
        try:
            from berlin_lst_downscaling.data.io.gdal_gcs import ensure_gdal_gcs_auth

            ensure_gdal_gcs_auth()
        except Exception:  # noqa: S110 — auth is best-effort; open fails loudly
            pass

        import numpy as np
        import rasterio

        if channel_set == "v6":
            expected_names, _units, valid_ranges = _v6_era5_contract()
            with rasterio.open(cog_uri) as src:
                if src.count != len(expected_names):
                    errors.append(f"band count: expected {len(expected_names)}, got {src.count}")
                if src.dtypes[0] != "float32":
                    errors.append(f"dtype: expected float32, got {src.dtypes[0]}")
                descriptions = list(src.descriptions)
                if descriptions != expected_names:
                    errors.append(f"band names: expected {expected_names}, got {descriptions}")
                # Range + constancy per band, on a downsampled overview window.
                for i in range(src.count):
                    name = descriptions[i] if i < len(descriptions) else f"band_{i + 1}"
                    lo_hi = valid_ranges.get(name)
                    out_h = max(1, src.height // 32)
                    out_w = max(1, src.width // 32)
                    arr = src.read(i + 1, out_shape=(out_h, out_w)).astype(np.float64)
                    valid = arr[np.isfinite(arr)]
                    if valid.size == 0:
                        errors.append(f"band {name} is entirely NaN (sampled overview)")
                        continue
                    if lo_hi is not None:
                        lo, hi = lo_hi
                        if valid.min() < lo or valid.max() > hi:
                            errors.append(
                                f"band {name} sampled range "
                                f"{valid.min():.4g}..{valid.max():.4g} outside [{lo}, {hi}]"
                            )
                    ptp = float(valid.max() - valid.min())
                    mean = float(np.mean(valid))
                    tol = 1e-4 * max(1.0, abs(mean))
                    if ptp > tol:
                        errors.append(
                            f"band {name} is not spatially constant "
                            f"(scene_value ptp={ptp:.4g} > tol={tol:.4g})"
                        )
            return errors

        with rasterio.open(cog_uri) as src:
            if src.count != 8:
                errors.append(f"band count: expected 8, got {src.count}")
            if src.dtypes[0] != "float32":
                errors.append(f"dtype: expected float32, got {src.dtypes[0]}")
            descriptions = list(src.descriptions)
            if descriptions != _ERA5_BAND_NAMES:
                errors.append(f"band names: expected {_ERA5_BAND_NAMES}, got {descriptions}")
    except Exception as exc:
        errors.append(f"cannot open COG: {exc}")
    return errors


def _validate_era5_provenance(prov_uri: str, channel_set: str = "v5") -> list[str]:
    """Check ERA5 provenance carries its published input-variable list.

    The six CDS input variables are identical for v5 and v6. The v6 writer
    additionally records ``source_metadata.channel_set == "v6"``; a release
    validated with ``--era5-channel-set v6`` must carry it, and a release
    validated as v5 must not claim v6. Any mismatch is an error, never a
    silent pass.
    """
    if channel_set not in _ERA5_CHANNEL_SETS:
        return [
            f"unknown era5 channel set {channel_set!r}: expected one of {list(_ERA5_CHANNEL_SETS)}"
        ]
    errors = []
    try:
        from berlin_lst_downscaling.data.io.storage import read_bytes

        prov = json.loads(read_bytes(prov_uri))
        variables = prov.get("source_metadata", {}).get("era5_variables", [])
        if variables != _ERA5_INPUT_VARIABLES:
            errors.append(
                f"provenance era5_variables: expected {_ERA5_INPUT_VARIABLES}, got {variables}"
            )
        recorded = prov.get("source_metadata", {}).get("channel_set")
        if channel_set == "v6":
            if recorded != "v6":
                errors.append(
                    f"provenance channel_set: expected 'v6' for the v6 contract, got {recorded!r}"
                )
        else:
            if recorded is not None and recorded != "v5":
                errors.append(
                    f"provenance channel_set: expected v5 (absent or 'v5'), got {recorded!r}"
                )
    except Exception as exc:
        errors.append(f"cannot read provenance: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dynamic pipeline products")
    parser.add_argument("--output-root", required=True, help="Dynamic run output root")
    parser.add_argument("--expected-role", required=True, help="Expected dataset role")
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=None,
        help="Expected total scene count per source (e.g. 324 or 21); "
        "mutually exclusive with --expected-scenes-from-manifest",
    )
    parser.add_argument(
        "--expected-scenes-from-manifest",
        default=None,
        metavar="MANIFEST_PARQUET",
        help="Derive the expected scene count per source from a selection "
        "manifest bundle (assessable 2017-2025 anchors, matching the anchor "
        "role this step validates); mutually exclusive with --expected-scenes",
    )
    parser.add_argument("--progress-only", action="store_true", help="Quick status only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--check-bands",
        action="store_true",
        help="Also validate COG band count and provenance (slower)",
    )
    parser.add_argument(
        "--era5-channel-set",
        choices=list(_ERA5_CHANNEL_SETS),
        default="v5",
        help="ERA5 band contract to validate: 'v5' is the published 8-band "
        "bilinear dynamic/full contract (default); 'v6' is the 4-band "
        "scene_value roster for dynamic/lod2-area-exact-v2. A COG or "
        "provenance that does not match the selected roster is an error.",
    )
    args = parser.parse_args()

    try:
        from berlin_lst_downscaling.data.io.gdal_gcs import ensure_gdal_gcs_auth

        ensure_gdal_gcs_auth()
    except Exception:  # noqa: S110 — auth is best-effort; COG opens fail loudly
        pass

    if args.expected_scenes is not None and args.expected_scenes_from_manifest is not None:
        parser.error("--expected-scenes and --expected-scenes-from-manifest are exclusive")
    if args.expected_scenes_from_manifest is not None:
        from berlin_lst_downscaling.data.selection.bundle_counts import (
            expected_counts_for_bundle,
        )

        assessable, _ = expected_counts_for_bundle(args.expected_scenes_from_manifest)
        print(f"  Expected scenes from manifest bundle: {assessable}")
        args.expected_scenes = assessable

    try:
        ledger = _read_ledger(args.output_root)
    except Exception as e:
        print(f"ERROR: Failed to load ledger: {e}", file=sys.stderr)
        return 1

    if args.progress_only:
        if args.json:
            # Convert tuple keys to strings for JSON serialization
            serializable = {
                **ledger,
                "counts": {f"{k[0]}/{k[1]}": v for k, v in ledger["counts"].items()},
            }
            print(json.dumps(serializable, indent=2, default=str))
        else:
            print(f"Ledger: {ledger['total_rows']} rows")
            print(f"Counts: {ledger['counts']}")
            print(f"Non-done: {len(ledger['non_done'])} items")
            print(f"Roles: {ledger['role_counts']}")
            print(f"Missing role: {ledger['missing_role']}")
        return 0

    errors = []
    warnings = []

    expected_per_source = args.expected_scenes
    if expected_per_source is not None:
        for src in ("era5_land", "shadow_building", "shadow_vegetation"):
            done = ledger["counts"].get((src, "done"), 0)
            if done != expected_per_source:
                errors.append(f"{src}: {done}/{expected_per_source} done")
            else:
                print(f"  {src}: {done}/{expected_per_source} done ✓")

    if ledger["non_done"]:
        for item in ledger["non_done"]:
            errors.append(
                f"  non-done: {item['item_id']} status={item['status']} "
                f"attempts={item['attempts']} error={item['last_error']}"
            )

    if ledger["missing_role"] > 0:
        errors.append(f"{ledger['missing_role']} rows have a null role (required)")
    for role, count in ledger["role_counts"].items():
        if role is None:
            continue
        if role != args.expected_role:
            errors.append(f"Unexpected role '{role}': {count} items")

    # ── band/provenance checks (optional, slower) ────────────────────
    if args.check_bands:
        print(f"\n  Checking ERA5 band contracts (sample, channel_set={args.era5_channel_set}) ...")

        # Read raw table for artifact inspection
        ledger_path = f"{args.output_root.rstrip('/')}/_state/dynamic/ledger.parquet"
        if ledger_path.startswith("gs://"):
            from berlin_lst_downscaling.data.io.storage import read_bytes

            table = pq.read_table(io.BytesIO(read_bytes(ledger_path)))
        else:
            table = pq.read_table(ledger_path)

        era5_done = [
            table.slice(i, 1).to_pydict()
            for i in range(table.num_rows)
            if table.slice(i, 1).to_pydict()["source"][0] == "era5_land"
            and table.slice(i, 1).to_pydict()["status"][0] == "done"
        ]

        # Check first 5 ERA5 products for band contract
        sample_size = min(5, len(era5_done))
        band_errors = 0
        for row_dict in era5_done[:sample_size]:
            cog_uri = row_dict.get("output_uri", [None])[0]
            prov_uri = row_dict.get("provenance_uri", [None])[0]
            cog_uri = resolve_canonical_uri(cog_uri) if cog_uri else cog_uri
            prov_uri = resolve_canonical_uri(prov_uri) if prov_uri else prov_uri
            if cog_uri:
                errs = _validate_cog_bands(cog_uri, channel_set=args.era5_channel_set)
                if errs:
                    band_errors += 1
                    for e in errs:
                        errors.append(f"  {row_dict['item_id'][0]}: {e}")
            if prov_uri:
                errs = _validate_era5_provenance(prov_uri, channel_set=args.era5_channel_set)
                if errs:
                    band_errors += 1
                    for e in errs:
                        errors.append(f"  {row_dict['item_id'][0]}: {e}")

        if band_errors == 0 and sample_size > 0:
            print(f"  ERA5 band contract ({args.era5_channel_set}): {sample_size} samples OK ✓")
        elif sample_size == 0:
            warnings.append("No ERA5 done rows to check")

    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors:
        print("\nAll source counts and roles verified.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
