#!/usr/bin/env python3
"""Review class support on the published Stadtstruktur release.

Source, purpose, grain
----------------------
Source: a local copy of ``gs://berlin-lst-training-data/evaluation/stadtstruktur/v1/``
and the published v3 scene manifest. The release is not downloaded again here
and its files are not modified.
Purpose: write area, mixture, transition, and scene-year tables for issue #74.
Grain: see ``berlin_lst_downscaling.data.stadtstruktur.class_review``.

Usage
-----
uv run python scripts/operators/review_stadtstruktur_classes.py \
    --release-dir /tmp/stadtstruktur-v1 \
    --manifest /tmp/stadtstruktur-manifest/manifest.parquet \
    --output-dir /tmp/stadtstruktur-class-review-v1

Upload an already validated directory with ``--upload-only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from berlin_lst_downscaling.data.io.storage import atomic_upload
from berlin_lst_downscaling.data.stadtstruktur.class_review import (
    APPROVAL,
    RELEASE_ID,
    annotate_inside_shares,
    build_class_review,
    load_aoi,
    slim_polygons,
)
from berlin_lst_downscaling.data.stadtstruktur.sources import RELEASE_ID as SOURCE_RELEASE_ID

SOURCE_URI = "gs://berlin-lst-training-data/evaluation/stadtstruktur/v1"
MANIFEST_URI = (
    "gs://berlin-lst-training-data/manifests/v3/"
    "2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
)
YEARS = (2020, 2021, 2022, 2023, 2024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inside_cell_ids(cells: pd.DataFrame) -> set[str]:
    flags = cells["inside_aoi"].tolist()
    return {
        str(cell_id)
        for cell_id, flag in zip(cells["cell_id"].tolist(), flags, strict=True)
        if bool(flag)
    }


def _anchor_counts(manifest_path: Path) -> dict[int, int]:
    manifest = pd.read_parquet(manifest_path)
    anchors = manifest.loc[manifest["role"].astype(str) == "anchor"]
    counts: dict[int, int] = {}
    for year in anchors["year"].tolist():
        parsed = int(year)
        counts[parsed] = counts.get(parsed, 0) + 1
    return counts


def _upload(output_dir: Path, output_root: str) -> None:
    root = output_root.rstrip("/")
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "complete.json"
    )
    for path in files:
        relative = path.relative_to(output_dir).as_posix()
        print(f"upload {relative}", flush=True)
        atomic_upload(path, f"{root}/{relative}")
    complete = output_dir / "complete.json"
    if not complete.exists():
        raise RuntimeError("complete.json is missing; refuse to publish a partial review")
    atomic_upload(complete, f"{root}/complete.json")


def build(release_dir: Path, manifest_path: Path, output_dir: Path, aoi_path: Path) -> None:
    """Read the local release and write the review tables."""
    report_path = release_dir / "qa" / "report.json"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("release_id") != SOURCE_RELEASE_ID:
        raise RuntimeError(f"expected {SOURCE_RELEASE_ID}")
    aoi = load_aoi(str(aoi_path))
    file_sha256: dict[str, str] = {"qa/report.json": _sha256(report_path)}
    inside_by_year = {}
    outside_area_by_year: dict[int, float] = {}
    share_area_by_year: dict[int, float] = {}
    slim_by_year = {}
    block_shares_by_year = {}
    for year in YEARS:
        shares_path = release_dir / "grid" / "100m" / str(year) / "shares.parquet"
        cells_path = release_dir / "grid" / "100m" / str(year) / "cells.parquet"
        blocks_path = release_dir / "source" / str(year) / "blocks.parquet"
        blocks_1km_path = release_dir / "grid" / "1000m" / str(year) / "shares.parquet"
        for path in (shares_path, cells_path, blocks_path, blocks_1km_path):
            file_sha256[path.relative_to(release_dir).as_posix()] = _sha256(path)
        print(f"class review {year}: reading", flush=True)
        shares = pd.read_parquet(shares_path)
        cells = pd.read_parquet(cells_path)
        inside, inside_area, outside_area = annotate_inside_shares(shares, _inside_cell_ids(cells))
        share_area_by_year[year] = float(shares["area_m2"].sum())
        outside_area_by_year[year] = outside_area
        inside_by_year[year] = inside
        slim_by_year[year] = slim_polygons(gpd.read_parquet(blocks_path), aoi)
        block_shares_by_year[year] = pd.read_parquet(blocks_1km_path)
        print(
            f"class review {year}: inside {inside_area / 1_000_000:.2f} km², "
            f"{len(slim_by_year[year])} polygons",
            flush=True,
        )
    review = build_class_review(
        source_report=source_report,
        inside_by_year=inside_by_year,
        outside_area_by_year=outside_area_by_year,
        share_area_by_year=share_area_by_year,
        slim_by_year=slim_by_year,
        block_shares_by_year=block_shares_by_year,
        anchor_counts=_anchor_counts(manifest_path),
        file_sha256=file_sha256,
        source_release_uri=SOURCE_URI,
        manifest_uri=MANIFEST_URI,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    review.support.to_parquet(output_dir / "support.parquet", index=False)
    review.crosswalk.to_parquet(output_dir / "crosswalk.parquet", index=False)
    review.transitions.to_parquet(output_dir / "transitions.parquet", index=False)
    review.transition_flows.to_parquet(output_dir / "transition_flows.parquet", index=False)
    review.mixture.to_parquet(output_dir / "mixture.parquet", index=False)
    review.block_sensitivity.to_parquet(output_dir / "block_sensitivity.parquet", index=False)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in review.figures.items():
        (figure_dir / name).write_text(svg, encoding="utf-8")
    _write_json(output_dir / "report.json", review.report)
    _write_json(
        output_dir / "complete.json",
        {
            "release_id": RELEASE_ID,
            "completed_at": datetime.now(UTC).isoformat(),
            "qa_pass": True,
            "approval": APPROVAL,
        },
    )
    print(f"wrote {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Stadtstruktur class support")
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aoi",
        type=Path,
        default=Path("data/boundaries/berlin_landesgrenze.geojson"),
    )
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--upload-only", action="store_true")
    args = parser.parse_args()
    if args.upload_only:
        if not args.output_root:
            raise SystemExit("--output-root is required with --upload-only")
        _upload(args.output_dir, args.output_root)
        return
    if args.release_dir is None or args.manifest is None:
        raise SystemExit("--release-dir and --manifest are required")
    build(args.release_dir, args.manifest, args.output_dir, args.aoi)
    if args.output_root:
        _upload(args.output_dir, args.output_root)


if __name__ == "__main__":
    main()
