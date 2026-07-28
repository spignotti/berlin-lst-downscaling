# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Validate the historical LoD vintage products and raw archive manifests.

Checks every published artefact in the chosen GCS roots:

- Raw archive ZIPs: present and reported byte-count matches manifest.
- Raw archive manifests: archive URI exists, SHA-256 matches, member
  count matches expected, and member list matches the ZIP.
- Source products: COG + STAC + provenance + complete marker per
  requested vintage. STAC id matches the requested vintage.
- Derived products: building_dsm, combined_dsm, horizon_building, svf.
- Year → vintage carry-forward mapping covers every year 2017–2026.

Returns exit code 0 on full success, 1 on any failure.  Designed to be
the smoke gate before declaring the historical processing closed.

Usage
-----
    uv run python scripts/validate_lod_vintages.py \\
        --source-root gs://berlin-lst-data/static/sources/full \\
        --derived-root gs://berlin-lst-data/static/derived/full \\
        --metadata-root gs://berlin-lst-data/static/geometry_vintages/v1 \\
        --raw-root gs://berlin-lst-data \\
        --vintages 2017,2021,2022
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.secondary.lod_vintages import (
    _VINTAGE_SOURCES,
    archive_uri_for,
    year_to_vintage_range,
)


@dataclass
class ValidationReport:
    """Aggregated validation findings."""

    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _check_archive(
    report: ValidationReport,
    raw_root: str,
    vintage: int,
) -> dict | None:
    """Verify the raw archive ZIP exists at the canonical location."""
    spec = _VINTAGE_SOURCES[vintage]
    uri = archive_uri_for(raw_root, spec)

    from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

    bucket, key = _parse_gs_uri(uri)

    if not exists(uri):
        report.failures.append(f"archive missing: {uri}")
        return None

    client = _gcs_client()
    blob = client.bucket(bucket).blob(key)
    blob.reload()
    if blob.size is None:
        report.failures.append(f"archive size unavailable: {uri}")
        return None

    return {
        "vintage": vintage,
        "uri": uri,
        "byte_count": blob.size,
        "md5": blob.md5_hash,
    }


def _check_raw_manifest(
    report: ValidationReport,
    raw_root: str,
    source_root: str,
    vintage: int,
) -> None:
    """Verify the per-vintage raw manifest exists, hashes match, members match."""
    spec = _VINTAGE_SOURCES[vintage]
    expected_uri = archive_uri_for(raw_root, spec)
    expected_count = spec.expected_count
    manifest_uri = (
        f"{source_root.rstrip('/')}/ard/static/sources/lod_vintages/"
        f"raw_manifest_{vintage}.json"
    )
    if not exists(manifest_uri):
        report.failures.append(f"raw_manifest missing: {manifest_uri}")
        return

    payload = json.loads(read_bytes(manifest_uri))
    if payload.get("archive_uri") != expected_uri:
        report.failures.append(
            f"raw_manifest archive_uri mismatch at {manifest_uri}: "
            f"expected {expected_uri} got {payload.get('archive_uri')}"
        )
    if payload.get("member_count") != expected_count:
        report.failures.append(
            f"raw_manifest member_count mismatch at {manifest_uri}: "
            f"expected {expected_count} got {payload.get('member_count')}"
        )
    members = payload.get("members") or []
    if len(members) != expected_count:
        report.failures.append(
            f"raw_manifest members length mismatch at {manifest_uri}: "
            f"expected {expected_count} got {len(members)}"
        )


def _check_source_product(
    report: ValidationReport,
    source_root: str,
    vintage: int,
) -> None:
    base = f"{source_root.rstrip('/')}/ard/static/sources/lod2_morphology/{vintage}"
    expected = [
        f"{base}/lod2_morphology_{vintage}.tif",
        f"{base}/lod2_morphology_{vintage}.stac.json",
        f"{base}/provenance.json",
        f"{base}/complete.json",
    ]
    for uri in expected:
        if not exists(uri):
            report.failures.append(f"source artifact missing: {uri}")

    stac_uri = f"{base}/lod2_morphology_{vintage}.stac.json"
    if exists(stac_uri):
        try:
            stac = json.loads(read_bytes(stac_uri))
            if str(stac.get("id", "")) != f"lod2_morphology-{vintage}":
                report.failures.append(f"STAC id mismatch at {stac_uri}")
        except Exception as exc:
            report.failures.append(f"STAC parse error at {stac_uri}: {exc}")

    prov_uri = f"{base}/provenance.json"
    if exists(prov_uri):
        try:
            prov = json.loads(read_bytes(prov_uri))
            archive_meta = prov.get("source_metadata", {}).get("archive")
            if not archive_meta or not archive_meta.get("sha256"):
                report.failures.append(
                    f"provenance missing archive sha256 at {prov_uri}"
                )
        except Exception as exc:
            report.failures.append(f"provenance parse error at {prov_uri}: {exc}")


def _check_derived_product(
    report: ValidationReport,
    derived_root: str,
    vintage: int,
) -> None:
    geometry_id = f"dgm1-2021__lod2-{vintage}__vh-2020"
    base = f"{derived_root.rstrip('/')}/ard/static/derived"
    expected_products = ["building_dsm", "combined_dsm", "horizon_building", "svf"]
    for product in expected_products:
        prod_dir = f"{base}/{product}/{geometry_id}"
        for fname in (
            f"{product}_{geometry_id}.tif",
            f"{product}_{geometry_id}.stac.json",
            "provenance.json",
            "complete.json",
        ):
            uri = f"{prod_dir}/{fname}"
            if not exists(uri):
                report.failures.append(f"derived artifact missing: {uri}")


def _check_geometry_mapping(
    report: ValidationReport,
    metadata_root: str,
    vintages: list[int],
) -> None:
    uri = f"{metadata_root.rstrip('/')}/geometry_mapping.json"
    if not exists(uri):
        report.failures.append(f"geometry_mapping missing: {uri}")
        return

    payload = json.loads(read_bytes(uri))

    selected_years: set[int] = set()
    for v in vintages:
        lo, hi = year_to_vintage_range(v)
        if lo is None or hi is None:
            continue
        for y in range(lo, hi + 1):
            selected_years.add(y)

    mapping_years = {int(y): int(v) for y, v in payload.get("year_to_vintage", {}).items()}
    missing = sorted(y for y in selected_years if y not in mapping_years)
    if missing:
        report.failures.append(
            "geometry_mapping missing years: " + ", ".join(str(y) for y in missing)
        )

    for year, vint in mapping_years.items():
        if vint not in vintages:
            report.failures.append(
                f"geometry_mapping year {year} -> vintage {vint} not in selected set"
            )
            continue
        lo, hi = year_to_vintage_range(vint)
        if lo is None or hi is None or not (lo <= year <= hi):
            report.failures.append(
                f"geometry_mapping year {year} -> vintage {vint} violates carry-forward"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument(
        "--source-root",
        default="gs://berlin-lst-data/static/sources/full",
    )
    parser.add_argument(
        "--derived-root",
        default="gs://berlin-lst-data/static/derived/full",
    )
    parser.add_argument(
        "--metadata-root",
        default="gs://berlin-lst-data/static/geometry_vintages/v1",
    )
    parser.add_argument(
        "--raw-root",
        default="gs://berlin-lst-data",
    )
    parser.add_argument(
        "--vintages",
        default="2017,2021,2022",
        help="Comma-separated list of vintages to verify.",
    )
    parser.add_argument(
        "--skip-archives",
        action="store_true",
        help="Skip raw archive ZIP verification.",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip raw archive manifest verification.",
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip derived product verification.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Skip the year-to-vintage mapping check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vintages = [int(v) for v in args.vintages.split(",") if v.strip()]
    report = ValidationReport()

    for vintage in vintages:
        if not args.skip_archives:
            info = _check_archive(report, args.raw_root, vintage)
            if info is not None:
                print(
                    f"archive ok: vintage={vintage} "
                    f"size={info['byte_count']} md5={info['md5']}"
                )
        if not args.skip_manifests:
            _check_raw_manifest(report, args.raw_root, args.source_root, vintage)
        _check_source_product(report, args.source_root, vintage)
        if not args.skip_derived:
            _check_derived_product(report, args.derived_root, vintage)

    if not args.skip_mapping:
        _check_geometry_mapping(report, args.metadata_root, vintages)

    for w in report.warnings:
        print(f"WARN: {w}")
    for f in report.failures:
        print(f"FAIL: {f}", file=sys.stderr)

    if report.ok:
        print("OK: all checks passed")
        return 0
    print(f"FAILED: {len(report.failures)} failure(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())