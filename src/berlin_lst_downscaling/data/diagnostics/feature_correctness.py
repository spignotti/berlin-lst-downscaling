"""Read-only checks for Sentinel-2 baselines, eligibility gaps, and LoD2 heights.

The operator prints a cost estimate and then, only with ``--yes``, writes
``summary.json`` and ``summary.md`` under a local directory. An optional
upload is accepted only at
``gs://berlin-lst-training-data/evaluation/diagnostics/<run-id>/``.

This module does not start a virtual machine or a cloud training job.
Metadata-only mode reads STAC properties and one local LoD2 tile. It
does not open feature rasters.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from berlin_lst_downscaling.data.ard.paths import cog_path, flag_path
from berlin_lst_downscaling.data.ard.s2_reflectance import resolve_s2_reflectance_scale
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNELS
from berlin_lst_downscaling.data.features.paths import feature_cog
from berlin_lst_downscaling.data.io.storage import atomic_upload
from berlin_lst_downscaling.data.io.storage import exists as storage_exists
from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.secondary.citygml import (
    describe_lod2_height_locations,
    parse_lod2_buildings_legacy,
    parse_lod2_buildings_with_stats,
)
from berlin_lst_downscaling.data.training.contracts import INFERENCE_DEFERRED_REASON

_LOGGER = logging.getLogger("berlin_lst_downscaling.data.diagnostics")

DIAGNOSTICS_PREFIX = "gs://berlin-lst-training-data/evaluation/diagnostics/"
EXPECTED_SCENE_COUNT = 324
SCOPES = ("s2-baseline", "eligibility-gaps", "lod2-parser")
_ASSET_KEYS = ("feature_stack", "landsat_cog", "landsat_flag")
_RESULTS_NAME = "results.jsonl"
_WINDOW = 640  # 10 m pixels; divisible by the 10x10 aggregation
_LANDSAT_SOURCE = "landsat-c2-l2"
_S2_SOURCE = "sentinel-2-l2a"

# One STAC document per scene, one feature COG of about 1 GiB, one CityGML tile.
_STAC_SECONDS = 0.5
_FEATURE_GIB_PER_SCENE = 1.0
_FEATURE_MINUTES_PER_SCENE = 1.0
_LOD2_SECONDS = 5.0


class DiagnosticsRefused(RuntimeError):
    """The requested diagnostics plan is outside the read-only contract."""


def cost_estimate(
    *,
    scopes: list[str],
    scene_count: int,
    lod2_tile_count: int,
    metadata_only: bool,
) -> dict[str, Any]:
    """Return a runtime and transfer estimate. Does not open any dataset.

    Parameters
    ----------
    scopes :
        Requested check names.
    scene_count :
        Scenes the S2 and eligibility checks would visit.
    lod2_tile_count :
        CityGML tiles the parser check would read.
    metadata_only :
        When true, raster scans are not part of the estimate.
    """
    stac_seconds = _STAC_SECONDS * scene_count if "s2-baseline" in scopes else 0.0
    if metadata_only or "eligibility-gaps" not in scopes:
        feature_gib = 0.0
        feature_minutes = 0.0
    else:
        feature_gib = _FEATURE_GIB_PER_SCENE * scene_count
        feature_minutes = _FEATURE_MINUTES_PER_SCENE * scene_count
    lod2_seconds = _LOD2_SECONDS * lod2_tile_count if "lod2-parser" in scopes else 0.0
    return {
        "scopes": list(scopes),
        "scene_count": int(scene_count),
        "lod2_tile_count": int(lod2_tile_count),
        "metadata_only": bool(metadata_only),
        "stac_seconds": stac_seconds,
        "feature_transfer_gib": feature_gib,
        "feature_minutes": feature_minutes,
        "lod2_seconds": lod2_seconds,
        "starts_vm": False,
        "writes_releases": False,
        "upload_prefix": DIAGNOSTICS_PREFIX,
    }


def format_cost_estimate(estimate: dict[str, Any]) -> str:
    """Render the estimate for the operator console."""
    lines = [
        "Feature-correctness diagnostics — cost estimate",
        f"Scopes: {', '.join(estimate['scopes']) or '(none)'}",
        f"Scenes: {estimate['scene_count']}",
        f"LoD2 tiles: {estimate['lod2_tile_count']}",
        f"Metadata only: {estimate['metadata_only']}",
        (
            "(a) S2 processing baseline: one STAC metadata read per scene "
            f"(about {estimate['stac_seconds']:.0f} s). No reflectance raster."
        ),
        (
            "(b) Eligibility gaps: "
            f"about {estimate['feature_transfer_gib']:.0f} GiB of feature COGs, "
            f"about {estimate['feature_minutes']:.0f} min. "
            "Reads are windowed; the full 28-band raster is not held in memory."
        ),
        f"(c) LoD2 parser: about {estimate['lod2_seconds']:.0f} s of local CPU.",
        "This process does not start a virtual machine or a cloud training job.",
        "It does not write feature, ARD, or training releases.",
        f"An upload is accepted only under {DIAGNOSTICS_PREFIX}<run-id>/.",
    ]
    return "\n".join(lines)


def validate_plan(
    scopes: list[str],
    *,
    metadata_only: bool,
    lod2_tiles: list[str],
) -> None:
    """Reject plans that would scan rasters in metadata-only mode.

    Metadata-only LoD2 analysis accepts exactly one tile. Eligibility
    gap counts need the feature stack, so that scope is refused.
    """
    unknown = [scope for scope in scopes if scope not in SCOPES]
    if unknown:
        raise DiagnosticsRefused(f"unknown scope: {unknown}")
    if not scopes:
        raise DiagnosticsRefused("at least one --scope is required")
    if metadata_only and "eligibility-gaps" in scopes:
        raise DiagnosticsRefused(
            "metadata-only cannot run eligibility-gaps; that scope reads feature rasters"
        )
    if metadata_only and "lod2-parser" in scopes and len(lod2_tiles) != 1:
        raise DiagnosticsRefused("metadata-only LoD2 analysis requires exactly one --lod2-tile")


def assert_output_dir(path: Path) -> Path:
    """Refuse an output directory inside ``docs/results`` or a release tree."""
    resolved = path.expanduser().resolve()
    text = resolved.as_posix()
    if "/docs/results" in text:
        raise DiagnosticsRefused(f"refusing to write under docs/results: {resolved}")
    return resolved


def assert_upload_uri(uri: str) -> str:
    """Accept only ``evaluation/diagnostics/<run-id>/`` in the canonical bucket."""
    cleaned = uri.strip()
    if not cleaned.endswith("/"):
        cleaned = cleaned + "/"
    prefix = DIAGNOSTICS_PREFIX
    if not cleaned.startswith(prefix):
        raise DiagnosticsRefused(f"upload must stay under {prefix}<run-id>/, got {uri}")
    run_id = cleaned[len(prefix) :].strip("/")
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise DiagnosticsRefused(
            "upload run-id must be a single path segment under evaluation/diagnostics/"
        )
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(char not in allowed for char in run_id):
        raise DiagnosticsRefused(f"upload run-id has unsupported characters: {run_id}")
    return cleaned


def scene_baseline_row(record: dict[str, Any]) -> dict[str, Any]:
    """Resolve one scene's BOA offset from inline STAC properties."""
    properties = record.get("properties")
    if not isinstance(properties, dict):
        raise DiagnosticsRefused(
            f"{record.get('scene_id', record.get('s2_scene_id'))}: "
            "S2 baseline needs STAC properties (metadata-only does not guess by date)"
        )
    scale = resolve_s2_reflectance_scale(properties)
    return {
        "scene_id": record.get("scene_id"),
        "year": record.get("year"),
        "s2_scene_id": record.get("s2_scene_id") or record.get("scene_id"),
        "processing_baseline": scale.processing_baseline,
        "boa_add_offset": scale.boa_add_offset,
        "decision": scale.decision,
        "quantification_value": scale.quantification_value,
    }


def _channel_invalid(values: np.ndarray, index: int) -> np.ndarray:
    """True where one feature channel is non-finite or outside its range."""
    channel = FEATURE_CHANNELS[index]
    bad = ~np.isfinite(values)
    if channel.valid_range is not None:
        low, high = channel.valid_range
        finite = np.isfinite(values)
        bad = bad | (finite & ((values < low) | (values > high)))
    return bad


def exclusion_reason_counts(
    target_valid: np.ndarray,
    bands: np.ndarray,
) -> dict[str, Any]:
    """Count why target-valid 100 m cells fail the strict 100/100 feature support.

    ``bands`` is ``(channel, y10, x10)`` in ``FEATURE_CHANNELS`` order.
    Leftover rows and columns that do not fill a 10 m block are dropped,
    matching ``GeoBox.zoom_out(10)``. A channel counts once per excluded
    cell when any of that cell's subpixels is invalid. Eligible cells are
    target-valid cells whose 100 subpixels are valid on every channel.

    Parameters
    ----------
    target_valid :
        Boolean Landsat target mask on the floored 100 m grid.
    bands :
        Feature stack window or scene, already in memory. Production
        scans call :func:`scan_feature_stack` so a full scene is not
        loaded at once.
    """
    stack = np.asarray(bands)
    if stack.ndim != 3 or stack.shape[0] != len(FEATURE_CHANNELS):
        raise ValueError(f"bands must be ({len(FEATURE_CHANNELS)}, H, W), got {stack.shape}")
    height10, width10 = int(stack.shape[1]), int(stack.shape[2])
    height100, width100 = height10 // 10, width10 // 10
    target = np.asarray(target_valid, dtype=bool)
    if target.shape != (height100, width100):
        raise ValueError(
            f"target_valid shape {target.shape} != floored 100 m grid {(height100, width100)}"
        )
    used = stack[:, : height100 * 10, : width100 * 10]
    any_invalid = np.zeros((len(FEATURE_CHANNELS), height100, width100), dtype=bool)
    sub_bad = np.zeros((height100 * 10, width100 * 10), dtype=bool)
    for index in range(len(FEATURE_CHANNELS)):
        bad = _channel_invalid(used[index], index)
        sub_bad |= bad
        any_invalid[index] = bad.reshape(height100, 10, width100, 10).any(axis=(1, 3))
    support = (~sub_bad).reshape(height100, 10, width100, 10).sum(axis=(1, 3))
    eligible = target & (support == 100)
    excluded = target & ~eligible
    per_channel = {
        channel.name: int((excluded & any_invalid[index]).sum())
        for index, channel in enumerate(FEATURE_CHANNELS)
    }
    return {
        "target_valid_cells": int(target.sum()),
        "eligible_cells": int(eligible.sum()),
        "excluded_cells": int(excluded.sum()),
        "incomplete_support_cells": int(excluded.sum()),
        "edge_truncated_cells": 0,
        "dropped_edge_rows": int(height10 - height100 * 10),
        "dropped_edge_cols": int(width10 - width100 * 10),
        "per_channel": per_channel,
    }


def _assert_aligned(feature: rasterio.DatasetReader, landsat: rasterio.DatasetReader) -> None:
    """Require the Landsat grid to be the 100 m parent of the feature grid."""
    left = feature.transform
    right = landsat.transform
    if abs(left.c - right.c) > 1e-3 or abs(left.f - right.f) > 1e-3:
        raise DiagnosticsRefused("feature stack and Landsat target origins differ")
    if abs(abs(left.a) * 10 - abs(right.a)) > 1e-2:
        raise DiagnosticsRefused("Landsat resolution is not 10x the feature grid")
    if abs(abs(left.e) * 10 - abs(right.e)) > 1e-2:
        raise DiagnosticsRefused("Landsat resolution is not 10x the feature grid")


def _target_valid_from_landsat(
    st_path: str,
    flag_path_uri: str,
    *,
    height100: int,
    width100: int,
    feature: rasterio.DatasetReader,
) -> np.ndarray:
    """Read the Landsat target. The 100 m grid is small relative to the 10 m stack."""
    with rasterio.open(st_path) as st_src, rasterio.open(flag_path_uri) as flag_src:
        _assert_aligned(feature, st_src)
        _assert_aligned(feature, flag_src)
        if st_src.height < height100 or st_src.width < width100:
            raise DiagnosticsRefused(
                f"Landsat grid {st_src.height}x{st_src.width} is smaller than "
                f"the floored feature grid {height100}x{width100}"
            )
        window = Window(0, 0, width100, height100)  # type: ignore[call-arg]
        st = st_src.read(1, window=window).astype(np.float32)
        flag = flag_src.read(1, window=window)
    return (flag == 0) & (st >= LST_RANGE_K[0]) & (st <= LST_RANGE_K[1])


def scan_feature_stack(
    feature_stack: str,
    landsat_cog: str,
    landsat_flag: str,
) -> dict[str, Any]:
    """Window the feature COG band by band and count exclusion reasons.

    Each window is ``640`` 10 m pixels on a side. One band is read at a
    time. The 28-band scene array is not materialized.
    """
    if _WINDOW % 10 != 0:
        raise RuntimeError("diagnostic window must be a multiple of 10")
    try:
        return _scan_feature_stack_open(feature_stack, landsat_cog, landsat_flag)
    except RasterioIOError as exc:
        raise DiagnosticsRefused(
            f"cannot open raster {feature_stack} "
            f"(landsat_cog={landsat_cog}, landsat_flag={landsat_flag}): {exc}"
        ) from exc


def _scan_feature_stack_open(
    feature_stack: str,
    landsat_cog: str,
    landsat_flag: str,
) -> dict[str, Any]:
    """Read one feature stack. Raster open failures propagate to the caller."""
    with rasterio.open(feature_stack) as src:
        if src.count != len(FEATURE_CHANNELS):
            raise DiagnosticsRefused(
                f"{feature_stack} has {src.count} bands, expected {len(FEATURE_CHANNELS)}"
            )
        height100, width100 = src.height // 10, src.width // 10
        dropped_rows = int(src.height % 10)
        dropped_cols = int(src.width % 10)
        target = _target_valid_from_landsat(
            landsat_cog,
            landsat_flag,
            height100=height100,
            width100=width100,
            feature=src,
        )
        any_invalid = np.zeros((len(FEATURE_CHANNELS), height100, width100), dtype=bool)
        support = np.zeros((height100, width100), dtype=np.int32)
        for row0 in range(0, height100 * 10, _WINDOW):
            for col0 in range(0, width100 * 10, _WINDOW):
                row1 = min(row0 + _WINDOW, height100 * 10)
                col1 = min(col0 + _WINDOW, width100 * 10)
                window = Window(col0, row0, col1 - col0, row1 - row0)  # type: ignore[call-arg]
                sub_bad = np.zeros((row1 - row0, col1 - col0), dtype=bool)
                cell_r0, cell_c0 = row0 // 10, col0 // 10
                cell_r1, cell_c1 = row1 // 10, col1 // 10
                for index in range(src.count):
                    values = src.read(index + 1, window=window)
                    bad = _channel_invalid(values, index)
                    sub_bad |= bad
                    cell_bad = bad.reshape(cell_r1 - cell_r0, 10, cell_c1 - cell_c0, 10).any(
                        axis=(1, 3)
                    )
                    any_invalid[index, cell_r0:cell_r1, cell_c0:cell_c1] |= cell_bad
                good = (~sub_bad).reshape(cell_r1 - cell_r0, 10, cell_c1 - cell_c0, 10)
                support[cell_r0:cell_r1, cell_c0:cell_c1] += good.sum(axis=(1, 3))
        eligible = target & (support == 100)
        excluded = target & ~eligible
        per_channel = {
            channel.name: int((excluded & any_invalid[index]).sum())
            for index, channel in enumerate(FEATURE_CHANNELS)
        }
        return {
            "target_valid_cells": int(target.sum()),
            "eligible_cells": int(eligible.sum()),
            "excluded_cells": int(excluded.sum()),
            "incomplete_support_cells": int(excluded.sum()),
            "edge_truncated_cells": 0,
            "dropped_edge_rows": dropped_rows,
            "dropped_edge_cols": dropped_cols,
            "per_channel": per_channel,
        }


def rollup_exclusions(per_scene: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum exclusion counts by calendar year and across every scene."""
    by_year: dict[str, dict[str, Any]] = {}
    total_channel: dict[str, int] = {}
    excluded = 0
    for row in per_scene:
        year = str(row.get("year"))
        bucket = by_year.setdefault(year, {"excluded_cells": 0, "per_channel": {}})
        bucket["excluded_cells"] += int(row["excluded_cells"])
        excluded += int(row["excluded_cells"])
        for name, count in row["per_channel"].items():
            bucket["per_channel"][name] = bucket["per_channel"].get(name, 0) + int(count)
            total_channel[name] = total_channel.get(name, 0) + int(count)
    return {"excluded_cells": excluded, "per_channel": total_channel, "by_year": by_year}


def _height_distribution(heights: list[float]) -> dict[str, Any]:
    """Summarise heights. An empty sample has null statistics, not zeros."""
    if not heights:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "max": None,
            "mean": None,
            "histogram_1m": {},
        }
    values = np.asarray(heights, dtype=np.float64)
    bins = np.floor(values).astype(int)
    histogram = {
        str(int(edge)): int(count)
        for edge, count in zip(*np.unique(bins, return_counts=True), strict=True)
    }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "histogram_1m": histogram,
    }


def _xml_members(path: Path) -> list[bytes]:
    """Read one CityGML document, or every XML member of a tile ZIP."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith((".gml", ".xml"))]
            if not names:
                raise DiagnosticsRefused(f"{path} has no CityGML member")
            return [archive.read(name) for name in names]
    return [path.read_bytes()]


def analyze_lod2_tile(path: Path) -> dict[str, Any]:
    """Compare height locations and the legacy versus current parser.

    ``buildingpart_only_measured_height`` counts buildings whose positive
    ``measuredHeight`` lives only on a ``BuildingPart``. The legacy parser
    drops those buildings. The current parser emits the parts. Neither
    distribution contains a silent 0 m height.
    """
    locations = {
        "source_buildings": 0,
        "buildingpart_only_measured_height": 0,
        "missing_measured_height": 0,
        "parts_missing_measured_height": 0,
        "parts": 0,
    }
    current_heights: list[float] = []
    legacy_heights: list[float] = []
    parse_totals: dict[str, int] = {}
    for raw in _xml_members(path):
        found = describe_lod2_height_locations(raw)
        for key, value in found.items():
            locations[key] = locations.get(key, 0) + int(value)
        buildings, stats = parse_lod2_buildings_with_stats(raw)
        for building in buildings:
            if building.measured_height is None or building.measured_height <= 0:
                raise DiagnosticsRefused("current parser emitted a missing or non-positive height")
            current_heights.append(float(building.measured_height))
        for building in parse_lod2_buildings_legacy(raw):
            if building.measured_height is None or building.measured_height <= 0:
                continue
            legacy_heights.append(float(building.measured_height))
        for key, value in stats.as_dict().items():
            parse_totals[key] = parse_totals.get(key, 0) + int(value)
    return {
        "tile": str(path),
        "height_locations": locations,
        "parse": parse_totals,
        "current": _height_distribution(current_heights),
        "legacy": _height_distribution(legacy_heights),
    }


def _is_inference_deferred(reason: object) -> bool:
    """True for 2026 rows whose feature stack was never materialised."""
    return str(reason or "") == INFERENCE_DEFERRED_REASON


def catalog_from_frames(
    training: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    features_root: str,
    ard_root: str,
) -> list[dict[str, Any]]:
    """Join a training manifest to Sentinel-2 ``item_href`` values.

    Feature and Landsat paths are derived from the release roots. STAC
    properties are not fetched here.

    Rows whose ``exclusion_reason`` is ``inference_deferred`` are omitted.
    ``no_eligible_cells`` rows stay: their stacks exist and the eligibility
    scope needs them.
    """
    needed = {"scene_id", "year", "s2_scene_id"}
    missing = needed.difference(training.columns)
    if missing:
        raise DiagnosticsRefused(f"training manifest is missing {sorted(missing)}")
    if "scene_id" not in selection.columns or "item_href" not in selection.columns:
        raise DiagnosticsRefused("selection manifest needs scene_id and item_href")
    hrefs = selection
    if "source" in hrefs.columns:
        hrefs = hrefs.loc[hrefs["source"].astype(str) == _S2_SOURCE]
    href_by_scene: dict[str, str] = {}
    for row in hrefs.to_dict(orient="records"):
        href = row.get("item_href")
        if isinstance(href, str) and href:
            href_by_scene[str(row["scene_id"])] = href
    records: list[dict[str, Any]] = []
    for row in training.to_dict(orient="records"):
        if _is_inference_deferred(row.get("exclusion_reason")):
            continue
        scene_id = str(row["scene_id"])
        year = int(row["year"])
        s2_scene_id = str(row["s2_scene_id"])
        feature_stack = row.get("feature_stack")
        if not isinstance(feature_stack, str) or not feature_stack:
            feature_stack = feature_cog(features_root, scene_id)
        records.append(
            {
                "scene_id": scene_id,
                "year": year,
                "s2_scene_id": s2_scene_id,
                "split": row.get("split"),
                "status": row.get("status"),
                "feature_stack": feature_stack,
                "landsat_cog": cog_path(ard_root, _LANDSAT_SOURCE, year, scene_id),
                "landsat_flag": flag_path(ard_root, _LANDSAT_SOURCE, year, scene_id),
                "item_href": href_by_scene.get(s2_scene_id),
            }
        )
    return records


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load scene records from JSON, or from local/remote manifests when asked."""
    if args.scenes_json is not None:
        payload = json.loads(Path(args.scenes_json).read_text(encoding="utf-8"))
        records = payload["scenes"] if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise DiagnosticsRefused("--scenes-json must be a list or an object with scenes")
        return [
            record
            for record in records
            if isinstance(record, dict)
            and not _is_inference_deferred(record.get("exclusion_reason"))
        ]
    if args.training_manifest is None:
        raise DiagnosticsRefused(
            "pass --scenes-json or --training-manifest; the default manifest is not opened"
        )
    training = pd.read_parquet(args.training_manifest)
    if args.selection_manifest is None:
        raise DiagnosticsRefused("--selection-manifest is required with --training-manifest")
    selection = pd.read_parquet(args.selection_manifest)
    return catalog_from_frames(
        training,
        selection,
        features_root=args.features_root,
        ard_root=args.ard_root,
    )


def _properties_from_href(href: str) -> dict[str, Any]:
    """Read STAC properties from a local or remote item. Tests pass inline properties."""
    import pystac

    item = pystac.Item.from_file(href)
    return dict(item.properties)


def _with_properties(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("properties"), dict):
        return record
    href = record.get("item_href")
    if not isinstance(href, str) or not href:
        raise DiagnosticsRefused(f"{record.get('scene_id')}: no STAC properties or item_href")
    copied = dict(record)
    copied["properties"] = _properties_from_href(href)
    return copied


def missing_record_assets(
    records: list[dict[str, Any]],
    *,
    exists: Callable[[str], bool] | None = None,
) -> list[str]:
    """Return ``feature_stack``, ``landsat_cog``, and ``landsat_flag`` paths that are absent.

    Parameters
    ----------
    records :
        Selected scene records.
    exists :
        Existence probe. Tests pass a fake filesystem. The default reads
        :func:`storage_exists` at call time so a monkeypatch applies.
    """
    probe = storage_exists if exists is None else exists
    missing: list[str] = []
    for record in records:
        scene_id = record.get("scene_id", "")
        for key in _ASSET_KEYS:
            path = record.get(key)
            label = path if isinstance(path, str) and path else f"{scene_id}:{key}"
            if not isinstance(path, str) or not path or not probe(path):
                missing.append(str(label))
    return missing


def assert_record_assets_exist(
    records: list[dict[str, Any]],
    *,
    exists: Callable[[str], bool] | None = None,
) -> None:
    """Refuse when any selected stack, COG, or flag path is missing."""
    missing = missing_record_assets(records, exists=exists)
    if missing:
        listed = "\n".join(missing)
        raise DiagnosticsRefused(f"missing assets ({len(missing)}):\n{listed}")


def _append_jsonl(output_dir: Path, row: dict[str, Any]) -> None:
    """Append one scene result so a later crash still leaves partial output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / _RESULTS_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _progress(scope: str, index: int, total: int, scene_id: object) -> None:
    """Print one flushed progress line. Block buffering hid these on the VM."""
    print(f"{scope} {index}/{total} {scene_id}", flush=True)


def _write_reports(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Feature-correctness diagnostics", ""]
    lines.append(f"Run id: `{summary.get('run_id', '')}`")
    lines.append("")
    if summary.get("error"):
        lines.append("## Error")
        lines.append("")
        lines.append(f"`{summary.get('error')}`")
        lines.append("")
        lines.append("```")
        lines.append(str(summary.get("traceback", "")))
        lines.append("```")
        lines.append("")
    lines.append("```")
    lines.append(summary.get("estimate_text", ""))
    lines.append("```")
    lines.append("")
    baselines = summary.get("s2_baseline")
    if isinstance(baselines, list):
        lines.append("## Sentinel-2 processing baseline")
        lines.append("")
        lines.append("| scene | year | baseline | BOA_ADD_OFFSET | decision |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in baselines:
            lines.append(
                f"| {row.get('s2_scene_id')} | {row.get('year')} | "
                f"{row.get('processing_baseline')} | {row.get('boa_add_offset')} | "
                f"{row.get('decision')} |"
            )
        lines.append("")
    gaps = summary.get("eligibility_gaps")
    if isinstance(gaps, dict):
        lines.append("## Eligibility gaps")
        lines.append("")
        lines.append(f"Excluded target-valid cells: {gaps.get('excluded_cells')}")
        lines.append("")
        lines.append("| channel | excluded cells where the channel is invalid |")
        lines.append("| --- | ---: |")
        for name, count in gaps.get("per_channel", {}).items():
            lines.append(f"| {name} | {count} |")
        lines.append("")
        lines.append("### By year")
        lines.append("")
        for year, bucket in sorted(gaps.get("by_year", {}).items()):
            lines.append(f"- {year}: {bucket.get('excluded_cells')} excluded cells")
        lines.append("")
    tiles = summary.get("lod2_parser")
    if isinstance(tiles, list):
        lines.append("## LoD2 parser")
        lines.append("")
        for tile in tiles:
            locations = tile["height_locations"]
            lines.append(f"### {tile['tile']}")
            lines.append("")
            lines.append(
                f"- buildings with BuildingPart-only measuredHeight: "
                f"{locations['buildingpart_only_measured_height']}"
            )
            lines.append(
                f"- buildings missing direct measuredHeight: {locations['missing_measured_height']}"
            )
            lines.append(
                f"- current heights: {tile['current']['count']} "
                f"(min {tile['current']['min']}, p50 {tile['current']['p50']}, "
                f"max {tile['current']['max']})"
            )
            lines.append(
                f"- legacy heights: {tile['legacy']['count']} "
                f"(min {tile['legacy']['min']}, p50 {tile['legacy']['p50']}, "
                f"max {tile['legacy']['max']})"
            )
            lines.append("")
    return "\n".join(lines)


def _upload_tree(source: Path, destination: str) -> None:
    root = assert_upload_uri(destination)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            atomic_upload(path, f"{root}{relative}")


def build_parser() -> argparse.ArgumentParser:
    """Command line for the diagnostics operator."""
    parser = argparse.ArgumentParser(
        description=(
            "Read-only feature-correctness diagnostics. Prints a cost estimate "
            "before any read beyond the estimate itself. Does not start cloud resources."
        )
    )
    parser.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        choices=SCOPES,
        required=True,
        help="Check to run. Repeat for more than one.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--upload-uri", default=None)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Required to execute after the estimate."
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Run S2 baseline and one local LoD2 tile without the VM and without rasters.",
    )
    parser.add_argument("--scenes-json", type=Path, default=None)
    parser.add_argument("--lod2-tile", action="append", default=[], dest="lod2_tiles")
    parser.add_argument(
        "--expected-scene-count",
        type=int,
        default=None,
        help=(
            "When set, refuse if the selected record count differs. "
            f"The feature release has {EXPECTED_SCENE_COUNT} stacks."
        ),
    )
    parser.add_argument("--training-manifest", default=None)
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument(
        "--features-root",
        default="gs://berlin-lst-training-data/features/v5",
    )
    parser.add_argument(
        "--ard-root",
        default="gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019",
    )
    parser.add_argument("--run-id", default="local")
    return parser


def _selected_scene_count(args: argparse.Namespace) -> tuple[int, list[dict[str, Any]] | None]:
    """Load records when a scene source is present and return the selected count.

    ``None`` for the record list means no scene source was passed (LoD2-only
    or a bare estimate). The count is then zero rather than a hard-coded
    release size.
    """
    if args.scenes_json is None and args.training_manifest is None:
        return 0, None
    records = _load_records(args)
    return len(records), records


def _check_expected_count(actual: int, expected: int | None) -> None:
    """Refuse when the caller pinned a scene count that does not match."""
    if expected is None:
        return
    if actual != expected:
        raise DiagnosticsRefused(
            f"selected scene count {actual} != --expected-scene-count {expected}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the operator. Returns a process status code."""
    args = build_parser().parse_args(argv)
    scopes = list(args.scopes or [])
    try:
        scene_count, preloaded = _selected_scene_count(args)
    except (OSError, json.JSONDecodeError, DiagnosticsRefused) as exc:
        print(f"ERROR: {exc}")
        return 2
    estimate = cost_estimate(
        scopes=scopes,
        scene_count=scene_count,
        lod2_tile_count=len(args.lod2_tiles),
        metadata_only=args.metadata_only,
    )
    text = format_cost_estimate(estimate)
    print(text)
    try:
        validate_plan(scopes, metadata_only=args.metadata_only, lod2_tiles=args.lod2_tiles)
        _check_expected_count(scene_count, args.expected_scene_count)
    except DiagnosticsRefused as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.estimate_only:
        return 0
    if not args.yes:
        print("ERROR: refusing to run without --yes.")
        return 2
    if args.output_dir is None:
        print("ERROR: --output-dir is required to execute.")
        return 2
    output_dir: Path | None = None
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "scopes": scopes,
        "metadata_only": args.metadata_only,
        "estimate": estimate,
        "estimate_text": text,
    }
    try:
        output_dir = assert_output_dir(args.output_dir)
        if args.upload_uri is not None:
            assert_upload_uri(args.upload_uri)
        needs_scenes = "s2-baseline" in scopes or "eligibility-gaps" in scopes
        records: list[dict[str, Any]] = (
            preloaded if preloaded is not None else (_load_records(args) if needs_scenes else [])
        )
        if needs_scenes and not args.metadata_only:
            assert_record_assets_exist(records)
        if "s2-baseline" in scopes:
            baselines = []
            total = len(records)
            for index, record in enumerate(records, start=1):
                _progress("s2-baseline", index, total, record.get("scene_id"))
                row = scene_baseline_row(_with_properties(record))
                baselines.append(row)
                _append_jsonl(output_dir, {"scope": "s2-baseline", **row})
            summary["s2_baseline"] = baselines
        if "eligibility-gaps" in scopes:
            per_scene = []
            total = len(records)
            for index, record in enumerate(records, start=1):
                _progress("eligibility-gaps", index, total, record.get("scene_id"))
                counted = scan_feature_stack(
                    str(record["feature_stack"]),
                    str(record["landsat_cog"]),
                    str(record["landsat_flag"]),
                )
                counted["scene_id"] = record.get("scene_id")
                counted["year"] = record.get("year")
                per_scene.append(counted)
                _append_jsonl(output_dir, {"scope": "eligibility-gaps", **counted})
            rolled = rollup_exclusions(per_scene)
            rolled["scenes"] = per_scene
            summary["eligibility_gaps"] = rolled
        if "lod2-parser" in scopes:
            if not args.lod2_tiles:
                raise DiagnosticsRefused("lod2-parser requires at least one --lod2-tile")
            summary["lod2_parser"] = [analyze_lod2_tile(Path(tile)) for tile in args.lod2_tiles]
        _write_reports(output_dir, summary)
        if args.upload_uri is not None:
            _upload_tree(output_dir, args.upload_uri)
    except DiagnosticsRefused as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        print(f"ERROR: {exc}", flush=True)
        if output_dir is not None:
            _write_reports(output_dir, summary)
            if args.upload_uri is not None:
                _upload_tree(output_dir, args.upload_uri)
        return 1
    _LOGGER.info("diagnostics wrote %s", output_dir)
    return 0


__all__ = [
    "DIAGNOSTICS_PREFIX",
    "EXPECTED_SCENE_COUNT",
    "SCOPES",
    "DiagnosticsRefused",
    "analyze_lod2_tile",
    "assert_output_dir",
    "assert_upload_uri",
    "catalog_from_frames",
    "cost_estimate",
    "exclusion_reason_counts",
    "format_cost_estimate",
    "main",
    "missing_record_assets",
    "rollup_exclusions",
    "scan_feature_stack",
    "scene_baseline_row",
    "validate_plan",
]
