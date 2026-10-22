#!/usr/bin/env python3
"""Check an EDA directory against the pre-registered contract.

Usage
-----
uv run python scripts/validators/validate_training_eda.py --self-check

uv run python scripts/validators/validate_training_eda.py \\
    --eda-dir /tmp/eda-v1 \\
    --manifest gs://berlin-lst-training-data/training/v2/manifest.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from berlin_lst_downscaling.data.eda.frames import frames_self_check
from berlin_lst_downscaling.data.eda.load import read_frame
from berlin_lst_downscaling.data.eda.rules import (
    BANNED_SUFFIXES,
    EVAL_SPLITS,
    PROPOSAL_B,
    RELEASE_ID,
    proposal_b_fingerprint,
)
from berlin_lst_downscaling.data.eda.stats import quantiles_monotone, self_check
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.stadtstruktur.class_review import PROPOSAL_B_GROUPS
from berlin_lst_downscaling.data.training.contracts import (
    EXPECTED_V3_CONFIG_HASH,
    training_policy_hash,
)

_TABLES = (
    "inventory_by_split_year_season.parquet",
    "exclusions.parquet",
    "eligibility_gap.parquet",
    "eligibility_gap_cells.parquet",
    "coverage_cells.parquet",
    "patch_acceptance.parquet",
    "lst_distribution.parquet",
    "lst_scene_anomaly.parquet",
    "lst_between_within.parquet",
    "lst_block_1000m.parquet",
    "neighbor_gradients.parquet",
    "hot_cold_counts.parquet",
    "feature_distribution.parquet",
    "feature_within_between.parquet",
    "correlation.parquet",
    "stadtstruktur_support.parquet",
    "stadtstruktur_lst.parquet",
    "stadtstruktur_feature_profile.parquet",
)


def _fail(messages: list[str]) -> int:
    for message in messages:
        print(f"FAIL {message}")
    print(f"{len(messages)} failure(s)")
    return 1


def _banned(root: Path) -> list[str]:
    failures = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in BANNED_SUFFIXES:
            failures.append(f"visual output is present: {path.name}")
    return failures


def _quantile_failures(frame: pd.DataFrame) -> list[str]:
    failures = []
    observed = frame[frame["weighting"].astype(str) == "observation"]
    for record in observed.itertuples(index=False):
        names = ("n", "min", "q05", "q25", "median", "q75", "q95", "max")
        stats = {name: getattr(record, name) for name in names}
        if not quantiles_monotone(stats):
            failures.append(f"quantiles are not monotone for {getattr(record, 'grain', '')}")
            break
    return failures


def validate(eda_dir: Path, manifest_uri: str) -> list[str]:
    """Return contract failures for one EDA directory."""
    failures = [*self_check(), *frames_self_check(), *_banned(eda_dir)]
    summary_path = eda_dir / "summary.json"
    if not summary_path.exists():
        return [*failures, "summary.json is missing"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("release_id") != RELEASE_ID:
        failures.append(f"release_id is {summary.get('release_id')}")
    if summary.get("training_policy_hash") != training_policy_hash():
        failures.append("training policy hash does not match the code")
    if summary.get("proposal_b_fingerprint") != proposal_b_fingerprint():
        failures.append("proposal B fingerprint does not match the code")
    if summary.get("v3_config_hash") != EXPECTED_V3_CONFIG_HASH:
        failures.append("V3 config hash does not match the pinned hash")
    if summary.get("scaler_policy_hash") != summary.get("training_policy_hash"):
        failures.append("scaler policy hash differs from the training policy hash")
    if int(summary.get("population", {}).get("n_observations", 0)) <= 0:
        failures.append("population has no observations")
    for name in _TABLES:
        if not (eda_dir / "tables" / name).exists():
            failures.append(f"missing table {name}")
    if failures and any(item.startswith("missing table") for item in failures):
        return failures

    thermal = pd.read_parquet(eda_dir / "tables" / "lst_distribution.parquet")
    if (thermal["year"].dropna().astype(int) == 2026).any():
        failures.append("2026 appears in the LST distribution")
    if thermal["split"].dropna().astype(str).isin(["inference"]).any():
        failures.append("inference split appears in the LST distribution")
    failures.extend(_quantile_failures(thermal))

    blocks = pd.read_parquet(eda_dir / "tables" / "lst_block_1000m.parquet")
    if (blocks["n_eligible"] > 100).any():
        failures.append("a 1 km block has more than 100 eligible cells")
    if (blocks["coverage"] > 1.0 + 1e-9).any():
        failures.append("a 1 km block coverage exceeds 1")

    support = pd.read_parquet(eda_dir / "tables" / "stadtstruktur_support.parquet")
    proposal = support[support["level"].astype(str) == "proposal_b"]
    known = set(PROPOSAL_B.values())
    unknown = sorted(set(proposal["class_id"].astype(str)) - known)
    if unknown:
        failures.append(f"proposal B table has unknown groups: {unknown[:5]}")
    present = set(proposal["class_id"].astype(str))
    for required in ("water", "construction"):
        if required not in present:
            failures.append(f"{required} is missing from the proposal B support")
    roles = {spec.group_id: spec.role for spec in PROPOSAL_B_GROUPS}
    for record in proposal.itertuples(index=False):
        expected = roles.get(str(record.class_id))
        if expected is not None and str(record.role) != expected:
            failures.append(f"{record.class_id} role is {record.role}")
            break

    weighted = pd.read_parquet(eda_dir / "tables" / "stadtstruktur_lst.parquet")
    broken = weighted[(weighted["n"] > 0) & ~(weighted["weight_sum_m2"] > 0)]
    if len(broken):
        failures.append("a class row has observations and a zero area weight")

    features = pd.read_parquet(eda_dir / "tables" / "feature_distribution.parquet")
    found = set(features["channel"].astype(str))
    rosters = [
        set(feature_schema(version).names)
        for version in (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V6)
    ]
    if found not in rosters:
        failures.append("feature distribution does not cover one feature roster")

    gap = pd.read_parquet(eda_dir / "tables" / "eligibility_gap.parquet")
    if (gap["target_valid_support_100"] != gap["provenance_eligible"]).any():
        failures.append("scanned eligibility does not match provenance")
    audit = pd.read_parquet(eda_dir / "audit" / "manifest.parquet")
    source = read_frame(manifest_uri)
    audit_ids = set(audit["scene_id"].astype(str))
    source_ids = set(source["scene_id"].astype(str))
    if audit_ids != source_ids:
        failures.append("audit manifest scene ids differ from the training manifest")
    merged = audit.merge(source, on="scene_id", suffixes=("_audit", "_source"))
    if (merged["eligible_cells_audit"] != merged["eligible_cells_source"]).any():
        failures.append("audit eligible counts differ from the training manifest")
    scanned = gap.merge(audit, on="scene_id", suffixes=("_gap", "_audit"))
    if (scanned["target_valid_support_100"] != scanned["eligible_cells"]).any():
        failures.append("gap eligibility does not match the manifest")
    eval_gap = gap[gap["split"].astype(str).isin(EVAL_SPLITS)]
    if len(eval_gap) != len(gap):
        failures.append("gap table contains a non-evaluation split")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the training-data EDA tables.")
    parser.add_argument("--eda-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check or not args.eda_dir:
        failures = [*self_check(), *frames_self_check()]
        if failures:
            return _fail(failures)
        print("GO eda self-check")
        if not args.eda_dir:
            return 0
    if not args.manifest:
        return _fail(["--manifest is required when --eda-dir is set"])
    failures = validate(Path(args.eda_dir), args.manifest)
    if failures:
        return _fail(failures)
    print(f"GO {RELEASE_ID}: {args.eda_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
