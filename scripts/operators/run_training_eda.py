#!/usr/bin/env python3
"""Build the read-only EDA tables for the published Berlin training release.

The default run scans every published scene and every ``no_eligible_cells``
scene, then writes Parquet tables, ``summary.json``, and per-scene
observations. A completed scene is skipped when its gap record is already
in ``--output-dir``. 2026 is inventory only.

No figure, map, dashboard, or PDF is written.

Usage
-----
uv run python scripts/operators/run_training_eda.py --output-dir /tmp/eda-v1

uv run python scripts/operators/run_training_eda.py \\
    --output-dir /tmp/eda-v1 --tables-only

uv run python scripts/operators/run_training_eda.py \\
    --output-dir /tmp/eda-v1 --upload-only \\
    --upload-uri gs://berlin-lst-training-data/evaluation/eda/v2
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from berlin_lst_downscaling.data.eda.frames import (
    between_within_table,
    block_table,
    correlation_table,
    coverage_table,
    exclusion_table,
    feature_distribution_table,
    feature_scale_table,
    hot_cold_table,
    inventory_table,
    lst_distribution_table,
    missing_scene_dates,
    neighbor_table,
    patch_acceptance_table,
    prepare_manifest,
    stadtstruktur_feature_table,
    stadtstruktur_lst_table,
    support_table,
)
from berlin_lst_downscaling.data.eda.load import (
    acquisition_calendar,
    inside_cells,
    patch_index,
    read_json,
    scaler_payload,
    stadtstruktur_shares,
    training_cells,
    training_manifest,
)
from berlin_lst_downscaling.data.eda.rules import (
    EVAL_SPLITS,
    FEATURES_ROOT,
    GAP_CLASSES,
    PATCH_INDEX_ROOT,
    RELEASE_ID,
    STADTSTRUKTUR_ROOT,
    TRAINING_ROOT,
    V3_MANIFEST_URI,
    proposal_b_fingerprint,
    season_of_month,
)
from berlin_lst_downscaling.data.eda.scan import scan_scene
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.io import (
    RunLogSession,
    atomic_upload,
    log_event,
    resolve_canonical_uri,
)
from berlin_lst_downscaling.data.stadtstruktur.sources import RELEASE_ID as STADTSTRUKTUR_RELEASE_ID
from berlin_lst_downscaling.data.training.contracts import (
    EXPECTED_V3_CONFIG_HASH,
    NO_ELIGIBLE_CELLS_REASON,
    training_policy_hash,
)
from berlin_lst_downscaling.data.training.paths import eligibility_provenance

_LOGGER = logging.getLogger("berlin_lst_downscaling.data.eda")

_NARROW = [
    "scene_id",
    "year",
    "month",
    "season",
    "split",
    "cell_id",
    "row",
    "col",
    "block_row",
    "block_col",
    "lst_k",
    "lst_c",
    "lst_anomaly_k",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upload_tree(source: Path, destination: str) -> None:
    root = destination.rstrip("/")
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            atomic_upload(path, f"{root}/{relative}")


def _observation_paths(output_dir: Path) -> list[Path]:
    directory = output_dir / "observations"
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def _read_observations(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _scenes_to_scan(manifest: pd.DataFrame) -> pd.DataFrame:
    """Published scenes and scenes excluded only because no cell was eligible."""
    status = manifest["status"].astype(str)
    reason = manifest["exclusion_reason"].fillna("").astype(str)
    split = manifest["split"].astype(str)
    selected = ((status == "published") | (reason == NO_ELIGIBLE_CELLS_REASON)) & (
        split.isin(EVAL_SPLITS)
    )
    return manifest.loc[selected].sort_values("scene_id")


def _gap_path(output_dir: Path, scene_id: str) -> Path:
    return output_dir / "gaps" / f"{scene_id}.json"


def _scan_one(
    output_dir: Path,
    row: pd.Series,
    calendar: pd.DataFrame,
    training_root: str,
) -> None:
    scene_id = str(row["scene_id"])
    marker = _gap_path(output_dir, scene_id)
    observation = output_dir / "observations" / f"{scene_id}.parquet"
    arrays = output_dir / "gaps" / f"{scene_id}.npz"
    if marker.exists() and arrays.exists() and (observation.exists() or _marker_is_empty(marker)):
        log_event(_LOGGER, logging.INFO, f"scene_skip {scene_id}")
        return
    provenance = read_json(eligibility_provenance(training_root, scene_id))
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{scene_id}: provenance has no inputs")
    dated = calendar.loc[calendar["scene_id"] == scene_id]
    month = None if dated.empty else dated.iloc[0]["month"]
    missing = month is None or (isinstance(month, float) and np.isnan(month))
    month_value = None if missing else int(month)
    season = season_of_month(month_value)
    scanned = scan_scene(
        scene_id=scene_id,
        year=int(row["year"]),
        split=str(row["split"]),
        month=month_value,
        season=season,
        s2_scene_id=str(row["s2_scene_id"]),
        landsat_cog=resolve_canonical_uri(str(inputs["landsat_cog"])),
        landsat_flag=resolve_canonical_uri(str(inputs["landsat_flag"])),
        feature_stack=resolve_canonical_uri(str(row["feature_stack"])),
        feature_valid=resolve_canonical_uri(str(inputs["feature_valid"])),
        provenance=provenance,
    )
    eligible = scanned.gap_counts["target_valid_support_100"]
    if eligible != scanned.provenance_eligible:
        raise RuntimeError(
            f"{scene_id}: scanned eligible {eligible} != provenance {scanned.provenance_eligible}"
        )
    if eligible != int(row["eligible_cells"]):
        raise RuntimeError(
            f"{scene_id}: scanned eligible {eligible} != manifest {int(row['eligible_cells'])}"
        )
    if eligible:
        _write_frame(scanned.observations, observation)
    elif observation.exists():
        observation.unlink()
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays,
        **{name: scanned.gap_cells[name] for name in GAP_CLASSES},
    )
    _write_json(
        marker,
        {
            "scene_id": scene_id,
            "year": int(row["year"]),
            "season": season,
            "split": str(row["split"]),
            "empty": eligible == 0,
            "provenance_target_valid": scanned.provenance_target_valid,
            "provenance_eligible": scanned.provenance_eligible,
            **scanned.gap_counts,
        },
    )
    log_event(_LOGGER, logging.INFO, f"scene_done {scene_id} eligible={eligible}")


def _marker_is_empty(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("empty"))


def _load_gap_markers(output_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "gaps").glob("*.json"))
    ]


def _unique_gap_table(output_dir: Path, markers: list[dict[str, object]]) -> pd.DataFrame:
    counts: dict[tuple[str, int, str, str], int] = defaultdict(int)
    unique: dict[tuple[str, int, str, str], set[tuple[int, int]]] = defaultdict(set)
    marker_by_id = {str(marker["scene_id"]): marker for marker in markers}
    for path in sorted((output_dir / "gaps").glob("*.npz")):
        marker = marker_by_id[path.stem]
        with np.load(path) as payload:
            for name in GAP_CLASSES:
                key = (str(marker["split"]), int(marker["year"]), str(marker["season"]), name)
                pairs = payload[name]
                counts[key] += int(pairs.shape[0])
                unique[key].update((int(row), int(col)) for row, col in pairs)
    rows = []
    for key, cells in unique.items():
        split, year, season, gap_class = key
        rows.append(
            {
                "split": split,
                "year": year,
                "season": season,
                "gap_class": gap_class,
                "n_scene_cells": counts[key],
                "n_unique_cells": len(cells),
            }
        )
    return pd.DataFrame(rows)


def _by_vintage(frame: pd.DataFrame, builder, shares: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return builder(frame, shares)
    pieces = []
    years = frame["year"].astype(int)
    for vintage_years in (
        [2017, 2018, 2019, 2020],
        [2021],
        [2022],
        [2023],
        [2024, 2025],
    ):
        part = frame.loc[years.isin(vintage_years)]
        if part.empty:
            continue
        pieces.append(builder(part, shares))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _save(frame: pd.DataFrame, directory: Path, name: str) -> None:
    _write_frame(frame, directory / name)


def _assemble(output_dir: Path, shares: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, int]:
    paths = _observation_paths(output_dir)
    narrow = _read_observations(paths, _NARROW)
    if narrow.empty:
        raise RuntimeError("scan produced no eligible observations")
    tables = output_dir / "tables"
    _save(lst_distribution_table(narrow, "lst_k"), tables, "lst_distribution.parquet")
    _save(lst_distribution_table(narrow, "lst_anomaly_k"), tables, "lst_scene_anomaly.parquet")
    _save(between_within_table(narrow, "lst_k"), tables, "lst_between_within.parquet")
    _save(block_table(narrow), tables, "lst_block_1000m.parquet")
    _save(neighbor_table(narrow), tables, "neighbor_gradients.parquet")
    _save(hot_cold_table(narrow), tables, "hot_cold_counts.parquet")
    _save(support_table(shares, narrow), tables, "stadtstruktur_support.parquet")
    lst_groups = _by_vintage(narrow, stadtstruktur_lst_table, shares)
    _save(lst_groups, tables, "stadtstruktur_lst.parquet")

    feature_frames: list[pd.DataFrame] = []
    scale_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    key_columns = ["scene_id", "year", "month", "season", "split", "row", "col"]
    for name in FEATURE_CHANNEL_NAMES:
        column = _read_observations(paths, [*key_columns, name])
        feature_frames.append(feature_distribution_table(column))
        scale_frames.append(feature_scale_table(column))
        profile = _by_vintage(column, stadtstruktur_feature_table, shares)
        profile_frames.append(profile)
        del column
    _save(pd.concat(feature_frames, ignore_index=True), tables, "feature_distribution.parquet")
    _save(pd.concat(scale_frames, ignore_index=True), tables, "feature_within_between.parquet")
    profiles = pd.concat(profile_frames, ignore_index=True)
    _save(profiles, tables, "stadtstruktur_feature_profile.parquet")

    wide_columns = ["scene_id", "lst_k", *FEATURE_CHANNEL_NAMES]
    wide = _read_observations(paths, wide_columns)
    _write_frame(correlation_table(wide), tables / "correlation.parquet")
    nan_lst = int(narrow["lst_k"].isna().sum())
    population = {
        "n_observations": int(len(narrow)),
        "n_unique_cells": int(narrow["cell_id"].nunique()),
        "n_scenes_with_observations": int(narrow["scene_id"].nunique()),
        "n_nonfinite_lst": nan_lst,
        "n_training_scenes": int((manifest["split"].astype(str) == "train").sum()),
    }
    del narrow, wide
    return population


def _summary(
    *,
    output_dir: Path,
    population: dict[str, int],
    scaler: dict[str, object],
    data_gaps: list[str],
    n_scenes_scanned: int,
) -> None:
    policy = training_policy_hash()
    payload = {
        "release_id": RELEASE_ID,
        "git_commit": _git_commit(),
        "training_root": TRAINING_ROOT,
        "features_root": FEATURES_ROOT,
        "stadtstruktur_root": STADTSTRUKTUR_ROOT,
        "stadtstruktur_release_id": STADTSTRUKTUR_RELEASE_ID,
        "patch_index_root": PATCH_INDEX_ROOT,
        "v3_manifest": V3_MANIFEST_URI,
        "training_policy_hash": policy,
        "scaler_policy_hash": scaler.get("policy_hash"),
        "v3_config_hash": scaler.get("v3_config_hash"),
        "expected_v3_config_hash": EXPECTED_V3_CONFIG_HASH,
        "proposal_b_fingerprint": proposal_b_fingerprint(),
        "scaler_transform_policy": scaler.get("transform_policy"),
        "rules": {
            "hot_quantile": 0.90,
            "cold_quantile": 0.10,
            "hot_cold": (
                "A cell is hot at or above the scene q90, and cold at or below the scene q10."
            ),
            "neighbor": (
                "4-neighborhood in the same scene. Both cells are eligible. "
                "A missing neighbor is omitted."
            ),
            "coarse_within_to_between": 0.01,
            "lst_unit": "K",
            "celsius": "lst_c = lst_k - 273.15",
            "feature_values": (
                "Mean of the 100 valid 10 m pixels in an eligible cell, in the "
                "published physical unit. Precipitation stays in mm. Shadows are "
                "the fraction of shadowed subpixels. These are not scaler outputs."
            ),
            "vintage": (
                "2017-2020 use 2020, 2021-2023 use the same year, "
                "2024-2025 use 2024. 2026 is unassigned."
            ),
            "weights": (
                "observation_area weights each scene-cell by class area. "
                "cell_area keeps one mean scene-anomaly per cell."
            ),
            "blocks": (
                "1 km blocks are groups of 100 canonical cells. "
                "Statistics use eligible 100 m cells only."
            ),
            "evaluation_limit": (
                "100 m LST describes the 1000-to-100 m learning problem. "
                "It does not validate a 10 m prediction."
            ),
        },
        "population": population,
        "n_scenes_scanned": n_scenes_scanned,
        "data_gaps": data_gaps,
        "tables": sorted(path.name for path in (output_dir / "tables").glob("*.parquet")),
    }
    _write_json(output_dir / "summary.json", payload)


def _table_pass(
    output_dir: Path,
    manifest: pd.DataFrame,
    cells: pd.DataFrame,
    calendar: pd.DataFrame,
    inside: pd.DataFrame,
    patches: pd.DataFrame,
) -> list[str]:
    prepared = prepare_manifest(manifest, calendar)
    tables = output_dir / "tables"
    _save(inventory_table(prepared, cells), tables, "inventory_by_split_year_season.parquet")
    _save(exclusion_table(prepared), tables, "exclusions.parquet")
    _save(coverage_table(cells, inside), tables, "coverage_cells.parquet")
    _save(patch_acceptance_table(prepared, patches), tables, "patch_acceptance.parquet")
    audit_columns = [
        "scene_id",
        "year",
        "split",
        "status",
        "eligible_cells",
        "exclusion_reason",
        "s2_scene_id",
    ]
    _write_frame(manifest.loc[:, audit_columns], output_dir / "audit" / "manifest.parquet")
    gaps = missing_scene_dates(manifest, calendar)
    data_gaps = [
        "The cause of a failed feature_valid subpixel is not stored and is not inferred.",
        "2026 is an inference reserve. It has no thermal or feature row.",
        "Scaler z-scores and log1p precipitation are not the physical EDA units.",
        "2025 is described here, so this pass is not untouched holdout evidence.",
    ]
    if gaps:
        data_gaps.append(f"Scenes without an acquisition month: {', '.join(gaps)}")
    return data_gaps


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if args.upload_only:
        if not args.upload_uri:
            raise SystemExit("--upload-only requires --upload-uri")
        _upload_tree(output_dir, args.upload_uri)
        print(f"uploaded {output_dir} to {args.upload_uri}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLogSession(str(output_dir), pipeline="eda", run_id=RELEASE_ID):
        log_event(_LOGGER, logging.INFO, "eda_start")
        manifest = training_manifest(args.training_root)
        cells = training_cells(args.training_root)
        calendar = acquisition_calendar(args.v3_manifest)
        inside = inside_cells(args.stadtstruktur_root)
        shares = stadtstruktur_shares(args.stadtstruktur_root)
        patches = patch_index(args.patch_index_root)
        scaler = scaler_payload(args.training_root)
        policy = training_policy_hash()
        if scaler.get("policy_hash") != policy:
            raise RuntimeError("published scaler policy hash does not match training_policy_hash()")
        if scaler.get("v3_config_hash") != EXPECTED_V3_CONFIG_HASH:
            raise RuntimeError("published scaler V3 config hash does not match the pinned hash")
        data_gaps = _table_pass(output_dir, manifest, cells, calendar, inside, patches)
        if args.tables_only:
            _summary(
                output_dir=output_dir,
                population={"n_observations": 0},
                scaler=scaler,
                data_gaps=[*data_gaps, "Raster scan was not run."],
                n_scenes_scanned=0,
            )
            print(f"tables only: {output_dir}")
            return
        selected = _scenes_to_scan(manifest)
        for _, scene in selected.iterrows():
            _scan_one(output_dir, scene, calendar, args.training_root)
        markers = _load_gap_markers(output_dir)
        tables = output_dir / "tables"
        _save(pd.DataFrame(markers), tables, "eligibility_gap.parquet")
        _save(_unique_gap_table(output_dir, markers), tables, "eligibility_gap_cells.parquet")
        outside = coverage_table(cells, inside)
        outside_all = outside.loc[outside["grain"] == "all"].iloc[0]
        n_outside = int(outside_all["n_unique_eligible_outside"])
        if n_outside:
            data_gaps.append(f"{n_outside} eligible cells fall outside the inside-AOI lattice.")
        population = _assemble(output_dir, shares, manifest)
        _summary(
            output_dir=output_dir,
            population=population,
            scaler=scaler,
            data_gaps=data_gaps,
            n_scenes_scanned=len(markers),
        )
        log_event(_LOGGER, logging.INFO, f"eda_done observations={population['n_observations']}")
    if args.upload_uri:
        _upload_tree(output_dir, args.upload_uri)
        print(f"uploaded {output_dir} to {args.upload_uri}")
    print(f"eda tables: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only EDA tables for the training release.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-root", default=TRAINING_ROOT)
    parser.add_argument("--v3-manifest", default=V3_MANIFEST_URI)
    parser.add_argument("--stadtstruktur-root", default=STADTSTRUKTUR_ROOT)
    parser.add_argument("--patch-index-root", default=PATCH_INDEX_ROOT)
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--upload-uri", default="")
    parser.add_argument("--upload-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
