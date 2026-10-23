#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fail-closed preflight for the v6 republish chain (docs/republish-v6.md).

Verifies, before any write, that the step's config points at the exact
v6/v3/v10/v4 upstream roots and that every upstream completion marker /
contract exists. Each step refuses to launch otherwise (exit 1). The
script writes nothing.

Usage
-----
    uv run python scripts/operators/preflight_republish_v6.py \\
        --step ard_v10 --config-dir configs/ard --config-name sentinel2_mosaic_v10

    uv run python scripts/operators/preflight_republish_v6.py \\
        --step features_v6 --config-dir configs/features --config-name v6

Steps: selection_v4, ard_v10, static_sources_v6, static_derived_v6,
dynamic_v6, features_v6, training_v3, patch_index_v3, eda_v3.
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[2]

# ── canonical chain roots ─────────────────────────────────────────────

MANIFEST_V4_ROOT = "gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z"
MANIFEST_V4_URI = f"{MANIFEST_V4_ROOT}/manifest.parquet"
PAIRINGS_V4_URI = f"{MANIFEST_V4_ROOT}/pairings.parquet"
REPORT_V4_URI = f"{MANIFEST_V4_ROOT}/manifest_report.json"

ARD_V10_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10"
ARD_LANDSAT_REUSE_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019"

STATIC_SOURCES_V6 = "gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
STATIC_DERIVED_V6 = "gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
DYNAMIC_V6 = "gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2"
FEATURES_V6 = "gs://berlin-lst-training-data/features/v6"
TRAINING_V3 = "gs://berlin-lst-training-data/training/v3"
PATCH_INDEX_V3 = "gs://berlin-lst-training-data/training/patch-index/v3"
EDA_V3 = "gs://berlin-lst-training-data/evaluation/eda/v3"
GEOMETRY_MAPPING = "gs://berlin-lst-training-data/static/geometry_vintages/v1/geometry_mapping.json"

STEPS = (
    "selection_v4",
    "ard_v10",
    "static_sources_v6",
    "static_derived_v6",
    "dynamic_v6",
    "features_v6",
    "training_v3",
    "patch_index_v3",
    "eda_v3",
)


def _load_config(config_dir: str, config_name: str):
    base = _REPO / config_dir / "_base.yaml"
    over = _REPO / config_dir / f"{config_name}.yaml"
    if base.exists():
        cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.load(over))
    else:
        cfg = OmegaConf.load(over)
    # Resolve ${output_root} style interpolations (e.g. checkpoint_dir).
    with suppress(Exception):
        OmegaConf.resolve(cfg)
    return cfg


def _failures_for(step: str, cfg) -> list[str]:
    """Collect fail-closed findings for one chain step."""
    from berlin_lst_downscaling.data.io import exists, read_bytes

    failures: list[str] = []

    def need(uri: str, label: str) -> None:
        if not exists(uri):
            failures.append(f"missing upstream {label}: {uri}")

    def want_root(value: object, expected: str, label: str) -> None:
        if str(value or "").rstrip("/") != expected:
            failures.append(f"{label} is {value!r}, expected {expected!r}")

    if step == "selection_v4":
        want_root(cfg.get("output_root"), MANIFEST_V4_ROOT, "output_root")
        if cfg.get("sentinel2", {}).get("mosaic", {}).get("enabled") is not True:
            failures.append("sentinel2.mosaic.enabled must be true for selection v4")
        # The v4 bundle is immutable once published.
        if exists(MANIFEST_V4_URI):
            failures.append(
                f"v4 bundle already published ({MANIFEST_V4_URI}) — "
                "a new bundle requires a new manifests/v4 root"
            )
    elif step == "ard_v10":
        want_root(cfg.get("output_root"), ARD_V10_ROOT, "output_root")
        if str(cfg.get("manifest_uri") or "").rstrip("/") != MANIFEST_V4_URI:
            failures.append(
                f"manifest_uri is {cfg.get('manifest_uri')!r}, expected {MANIFEST_V4_URI!r}"
            )
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        need(PAIRINGS_V4_URI, "v4 pairings.parquet")
        need(REPORT_V4_URI, "v4 manifest_report.json")
        need(f"{ARD_LANDSAT_REUSE_ROOT}/ledger.parquet", "reuse Landsat/ECOSTRESS ledger")
    elif step == "static_sources_v6":
        want_root(cfg.get("source_root"), STATIC_SOURCES_V6, "source_root")
    elif step == "static_derived_v6":
        want_root(cfg.get("source_root"), STATIC_SOURCES_V6, "source_root")
        want_root(cfg.get("derived_root"), STATIC_DERIVED_V6, "derived_root")
        need(f"{STATIC_SOURCES_V6}/ledger.parquet", "static-sources v6 ledger")
    elif step == "dynamic_v6":
        want_root(cfg.get("output_root"), DYNAMIC_V6, "output_root")
        want_root(cfg.get("source_root"), STATIC_SOURCES_V6, "source_root")
        want_root(cfg.get("derived_root"), STATIC_DERIVED_V6, "derived_root")
        if str(cfg.get("manifest_uri") or "").rstrip("/") != MANIFEST_V4_URI:
            failures.append(
                f"manifest_uri is {cfg.get('manifest_uri')!r}, expected {MANIFEST_V4_URI!r}"
            )
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        need(GEOMETRY_MAPPING, "geometry mapping")
        need(f"{STATIC_SOURCES_V6}/ledger.parquet", "static-sources v6 ledger")
        need(
            f"{STATIC_DERIVED_V6}/_state/static/derived/ledger.parquet",
            "static-derived v6 ledger",
        )
    elif step == "features_v6":
        want_root(cfg.get("output_root"), FEATURES_V6, "output_root")
        want_root(cfg.get("ard_root"), ARD_V10_ROOT, "ard_root")
        if str(cfg.get("manifest_uri") or "").rstrip("/") != MANIFEST_V4_URI:
            failures.append(
                f"manifest_uri is {cfg.get('manifest_uri')!r}, expected {MANIFEST_V4_URI!r}"
            )
        want_root(cfg.get("static_sources_root"), STATIC_SOURCES_V6, "static_sources_root")
        want_root(cfg.get("static_derived_root"), STATIC_DERIVED_V6, "static_derived_root")
        want_root(cfg.get("dynamic_root"), DYNAMIC_V6, "dynamic_root")
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        need(GEOMETRY_MAPPING, "geometry mapping")
        need(f"{ARD_V10_ROOT}/ledger.parquet", "ARD v10 ledger")
        need(f"{STATIC_SOURCES_V6}/ledger.parquet", "static-sources v6 ledger")
        need(
            f"{STATIC_DERIVED_V6}/_state/static/derived/ledger.parquet",
            "static-derived v6 ledger",
        )
        need(
            f"{DYNAMIC_V6}/_state/dynamic/ledger.parquet",
            "dynamic v6 ledger",
        )
        from berlin_lst_downscaling.data.ard.publish_guard import (
            FrozenPrefixError,
            assert_feature_output_root,
        )

        try:
            assert_feature_output_root(
                str(cfg.output_root),
                schema_version=int(cfg.get("feature_schema_version", 5)),
            )
        except FrozenPrefixError as exc:
            failures.append(f"publish_guard: {exc}")
    elif step == "training_v3":
        want_root(cfg.get("output_root"), TRAINING_V3, "output_root")
        want_root(cfg.get("features_root"), FEATURES_V6, "features_root")
        want_root(cfg.get("ard_root"), ARD_V10_ROOT, "ard_root")
        if str(cfg.get("manifest_uri") or "").rstrip("/") != MANIFEST_V4_URI:
            failures.append(
                f"manifest_uri is {cfg.get('manifest_uri')!r}, expected {MANIFEST_V4_URI!r}"
            )
        want_root(cfg.get("static_sources_root"), STATIC_SOURCES_V6, "static_sources_root")
        want_root(cfg.get("static_derived_root"), STATIC_DERIVED_V6, "static_derived_root")
        want_root(cfg.get("dynamic_root"), DYNAMIC_V6, "dynamic_root")
        need(f"{FEATURES_V6}/ledger.parquet", "features/v6 ledger")
        need(f"{ARD_V10_ROOT}/ledger.parquet", "ARD v10 ledger")
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        if exists(f"{TRAINING_V3}/complete.json"):
            failures.append(
                f"training/v3 already published ({TRAINING_V3}/complete.json) — "
                "a new release requires a new training root"
            )
    elif step == "patch_index_v3":
        want_root(cfg.get("source_root"), TRAINING_V3, "source_root")
        want_root(cfg.get("output_root"), PATCH_INDEX_V3, "output_root")
        need(f"{TRAINING_V3}/complete.json", "training/v3 complete.json")
        need(f"{TRAINING_V3}/manifest.parquet", "training/v3 manifest.parquet")
        marker_uri = f"{TRAINING_V3}/complete.json"
        if exists(marker_uri):
            try:
                marker = json.loads(read_bytes(marker_uri))
            except Exception as exc:
                failures.append(f"training/v3 marker unreadable: {exc}")
            else:
                if not marker.get("policy_hash"):
                    failures.append("training/v3 marker carries no policy_hash")
        if exists(f"{PATCH_INDEX_V3}/complete.json"):
            failures.append(
                f"patch-index v3 already published ({PATCH_INDEX_V3}/complete.json) — "
                "a new index requires a new patch-index root"
            )
    elif step == "eda_v3":
        cfg_map = OmegaConf.to_container(cfg, resolve=True) or {}
        for key, expected in (
            ("training_root", TRAINING_V3),
            ("features_root", FEATURES_V6),
            ("v3_manifest", MANIFEST_V4_URI),
            ("patch_index_root", PATCH_INDEX_V3),
            ("upload_uri", EDA_V3),
        ):
            if str(cfg_map.get(key) or "").rstrip("/") != expected:
                failures.append(f"{key} is {cfg_map.get(key)!r}, expected {expected!r}")
        need(f"{TRAINING_V3}/complete.json", "training/v3 complete.json")
        need(f"{TRAINING_V3}/manifest.parquet", "training/v3 manifest.parquet")
        need(f"{PATCH_INDEX_V3}/complete.json", "patch-index v3 complete.json")
    else:
        failures.append(f"unknown step: {step}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", required=True, choices=STEPS)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config_dir, args.config_name)
    failures = _failures_for(args.step, cfg)
    if failures:
        print(f"preflight {args.step} FAILED ({args.config_name}):")
        for finding in failures:
            print(f"  - {finding}")
        return 1
    print(f"preflight {args.step} OK ({args.config_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
