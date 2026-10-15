#!/usr/bin/env python3
"""Re-read a Stadtstruktur class review and check its contract.

Usage
-----
uv run python scripts/validators/validate_stadtstruktur_class_review.py \
    --review-dir /tmp/stadtstruktur-class-review-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from berlin_lst_downscaling.data.stadtstruktur.class_review import (
    APPROVAL,
    EVAL_SCENE_VINTAGE,
    RELEASE_ID,
)
from berlin_lst_downscaling.data.stadtstruktur.sources import CANDIDATE_B

_LEVELS = ("typ", "ststrnr", "published_b", "proposal_b", "proposal_c")
_ASSIGNMENT = {
    2017: 2020,
    2018: 2020,
    2019: 2020,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2024,
    2025: 2024,
}


def _fail(messages: list[str]) -> int:
    for message in messages:
        print(f"FAIL {message}")
    print(f"{len(messages)} failure(s)")
    return 1


def _walk_keys(value: object, keys: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            _walk_keys(child, keys)
    elif isinstance(value, list):
        for child in value:
            _walk_keys(child, keys)


def validate(review_dir: Path) -> int:
    """Return 0 when the review matches the pre-registered contract."""
    report = json.loads((review_dir / "report.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("release_id") != RELEASE_ID:
        failures.append(f"release_id is {report.get('release_id')}")
    if report.get("approval") != APPROVAL:
        failures.append(f"approval is {report.get('approval')}")
    recommendation = report.get("recommendation", {})
    if recommendation.get("approval") != APPROVAL:
        failures.append("recommendation approval is not awaiting review")
    if recommendation.get("recommended_grouping") == "approved":
        failures.append("recommendation marks a grouping approved")
    keys: list[str] = []
    _walk_keys(report, keys)
    banned = [key for key in keys if "mae" in key.lower()]
    if banned:
        failures.append(f"report keys include a model-error field: {banned[:5]}")
    support = pd.read_parquet(review_dir / "support.parquet")
    if any("mae" in column.lower() for column in support.columns):
        failures.append("support columns include a model-error field")
    crosswalk = pd.read_parquet(review_dir / "crosswalk.parquet")
    mapped = {int(value) for value in crosswalk["typ"]}
    if mapped != set(CANDIDATE_B):
        failures.append("crosswalk typ set differs from the published candidate map")
    if crosswalk["typ"].duplicated().any():
        failures.append("duplicate typ in the crosswalk")
    for column in ("published_b", "proposal_b", "proposal_c"):
        if crosswalk[column].isna().any():
            failures.append(f"crosswalk {column} has a null group")
    if (crosswalk["proposal_b"] == "vegetated_open").any():
        failures.append("proposal B still pools open land as vegetated_open")
    scenes = {int(row["scene_year"]): row for row in report["scene_year_assignment"]}
    if set(EVAL_SCENE_VINTAGE) != set(_ASSIGNMENT):
        failures.append("module scene map drifted from the validator")
    for year, vintage in _ASSIGNMENT.items():
        row = scenes.get(year)
        if row is None or int(row["vintage"]) != vintage:
            failures.append(f"scene year {year} is not assigned to {vintage}")
            continue
        if "anchor_times" in row:
            failures.append("scene row multiplies anchors into area")
    for year_rows in report["reconciliation"]:
        year = int(year_rows["vintage"])
        if float(year_rows["share_abs_diff_m2"]) > 1.0:
            failures.append(f"{year}: share area is not conserved")
        if float(year_rows["centre_split_abs_diff_m2"]) > 1.0:
            failures.append(f"{year}: inside and outside shares do not sum")
        if float(year_rows["gap_plus_source_inside_abs_diff_m2"]) > 1.0:
            failures.append(f"{year}: street gap identity failed")
        if float(year_rows["street_gap_m2"]) <= 0:
            failures.append(f"{year}: street gap is missing")
        classified = float(year_rows["inside_centre_area_m2"])
        for level in _LEVELS:
            level_area = float(
                support.loc[
                    (support["vintage"] == year) & (support["level"] == level),
                    "area_m2",
                ].sum()
            )
            if abs(level_area - classified) > 1.0:
                failures.append(f"{year} {level}: class areas do not sum to the classified area")
        class_ids = set(
            support.loc[
                (support["vintage"] == year) & (support["level"] == "typ"),
                "class_id",
            ].astype(str)
        )
        if "street" in class_ids or "unclassified_street" in class_ids:
            failures.append(f"{year}: a street class entered the stratified table")
    if report["recommendation"]["recommended_grouping"] not in {"proposal_b"}:
        failures.append("recommendation left the pre-registered working grouping")
    required = [
        "support.parquet",
        "crosswalk.parquet",
        "transitions.parquet",
        "transition_flows.parquet",
        "mixture.parquet",
        "block_sensitivity.parquet",
        "figures/area_proposal_b_2024.svg",
        "figures/mixture_by_level_2024.svg",
        "report.json",
    ]
    for name in required:
        if not (review_dir / name).exists():
            failures.append(f"missing {name}")
    if failures:
        return _fail(failures)
    print("stadtstruktur class review: ok")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Stadtstruktur class review")
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(validate(args.review_dir))


if __name__ == "__main__":
    main()
