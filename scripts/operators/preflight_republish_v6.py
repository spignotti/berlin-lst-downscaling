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
lod_vintages_v6, dynamic_v6, features_v6, training_v3, patch_index_v3, eda_v3.
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[2]

# Canonical chain roots, single-sourced from publish_guard so the guards
# and the preflight can never disagree about the release roots.
from berlin_lst_downscaling.data.ard.publish_guard import (  # noqa: E402
    EDA_V3_ROOT,
    FEATURES_V6_ROOT,
    MANIFESTS_V4_ROOT,
    PATCH_INDEX_V3_ROOT,
    SENTINEL2_V10_ROOT,
    STATIC_V6_ROOTS,
    TRAINING_V3_ROOT,
)
from berlin_lst_downscaling.data.features.paths import (  # noqa: E402
    ledger_path as features_ledger_path,
)
from berlin_lst_downscaling.data.secondary.paths import (  # noqa: E402
    derived_product_cog,
    derived_product_dir,
    source_product_cog,
    source_product_dir,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    cells_parquet as training_cells_parquet,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    manifest_parquet as training_manifest_parquet,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    patch_index_completion as patch_index_completion_uri,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    patch_index_parquet as patch_index_parquet_uri,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    release_completion as training_release_completion,
)
from berlin_lst_downscaling.data.training.paths import (  # noqa: E402
    scaler_json as training_scaler_json,
)

# ── canonical chain roots ─────────────────────────────────────────────

MANIFEST_V4_ROOT = MANIFESTS_V4_ROOT
MANIFEST_V4_URI = f"{MANIFEST_V4_ROOT}/manifest.parquet"
PAIRINGS_V4_URI = f"{MANIFEST_V4_ROOT}/pairings.parquet"
REPORT_V4_URI = f"{MANIFEST_V4_ROOT}/manifest_report.json"

ARD_V10_ROOT = SENTINEL2_V10_ROOT
ARD_LANDSAT_REUSE_ROOT = "gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019"


def _static_root(kind: str) -> str:
    matches = [root for root in STATIC_V6_ROOTS if f"/{kind}/" in root]
    if len(matches) != 1:
        raise RuntimeError(f"expected one static {kind} root, found {matches}")
    return matches[0]


STATIC_SOURCES_V6 = _static_root("static/sources")
STATIC_DERIVED_V6 = _static_root("static/derived")
DYNAMIC_V6 = _static_root("dynamic")
FEATURES_V6 = FEATURES_V6_ROOT
TRAINING_V3 = TRAINING_V3_ROOT
PATCH_INDEX_V3 = PATCH_INDEX_V3_ROOT
EDA_V3 = EDA_V3_ROOT
GEOMETRY_MAPPING = "gs://berlin-lst-training-data/static/geometry_vintages/v1/geometry_mapping.json"

STEPS = (
    "selection_v4",
    "ard_v10",
    "static_sources_v6",
    "static_derived_v6",
    "lod_vintages_v6",
    "dynamic_v6",
    "features_v6",
    "training_v3",
    "patch_index_v3",
    "eda_v3",
)


# ── v6 LoD vintage evidence ───────────────────────────────────────────
# Steps 5–7 (dynamic, features, training) resolve every scene year
# through geometry_mapping v1, so they need the full per-vintage product
# set in the v6 roots — not just the 2024 products steps 3–4 publish.
# These helpers enumerate exactly those URIs so the preflights (and the
# tests) can never disagree about the required evidence.

V6_HISTORICAL_VINTAGES = (2017, 2021, 2022)
V6_ALL_VINTAGES = (2017, 2021, 2022, 2024)

# Morphology-derived products rebuilt per vintage by step 4b
# (vegetation_dsm/horizon_vegetation are vintage-agnostic and reused
# from the 2024 geometry bundle).
_V6_PER_VINTAGE_DERIVED = ("building_dsm", "combined_dsm", "horizon_building", "svf")


def _vintage_geometry_id(vintage: int) -> str:
    return f"dgm1-2021__lod2-{vintage}__vh-2020"


def _derived_cog_and_marker(product: str, geometry_id: str, vintage: int) -> list[tuple[str, str]]:
    """(COG, marker) for one derived product via the writer's path helpers.

    Uses :mod:`berlin_lst_downscaling.data.secondary.paths` (the same
    helpers the derived pipeline publishes through), so the preflight can
    never disagree with the published layout about these URIs.
    """
    return [
        (
            derived_product_cog(STATIC_DERIVED_V6, product, geometry_id),
            f"v6 {product}/{vintage} COG",
        ),
        (
            f"{derived_product_dir(STATIC_DERIVED_V6, product, geometry_id)}/complete.json",
            f"v6 {product}/{vintage} marker",
        ),
    ]


def v6_vintage_horizon_uris() -> list[tuple[str, str]]:
    """(uri, label) for every horizon COG shadows resolve under v6."""
    out: list[tuple[str, str]] = []
    for vintage in V6_ALL_VINTAGES:
        out.extend(
            _derived_cog_and_marker("horizon_building", _vintage_geometry_id(vintage), vintage)
        )
    out.extend(_derived_cog_and_marker("horizon_vegetation", _vintage_geometry_id(2024), 2024))
    return out


def v6_vintage_product_uris() -> list[tuple[str, str]]:
    """(uri, label) for the full per-vintage evidence steps 6–7 consume."""
    out: list[tuple[str, str]] = []
    for vintage in V6_ALL_VINTAGES:
        revision = str(vintage)
        out.append(
            (
                source_product_cog(STATIC_SOURCES_V6, "lod2_morphology", revision),
                f"v6 lod2_morphology/{vintage} COG",
            )
        )
        out.append(
            (
                f"{source_product_dir(STATIC_SOURCES_V6, 'lod2_morphology', revision)}"
                "/complete.json",
                f"v6 lod2_morphology/{vintage} marker",
            )
        )
    for vintage in V6_HISTORICAL_VINTAGES:
        out.append(
            (
                f"{STATIC_SOURCES_V6}/ard/static/sources/lod_vintages/raw_manifest_{vintage}.json",
                f"v6 raw_manifest_{vintage}",
            )
        )
    out.append(
        (
            f"{source_product_dir(STATIC_SOURCES_V6, 'lod2_morphology', '2024')}/provenance.json",
            "v6 lod2_morphology/2024 provenance",
        )
    )
    for vintage in V6_ALL_VINTAGES:
        geometry_id = _vintage_geometry_id(vintage)
        for product in _V6_PER_VINTAGE_DERIVED:
            out.extend(_derived_cog_and_marker(product, geometry_id, vintage))
    vegetation_geometry = _vintage_geometry_id(2024)
    for product in ("vegetation_dsm", "horizon_vegetation"):
        out.extend(_derived_cog_and_marker(product, vegetation_geometry, 2024))
    return out


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
    elif step == "lod_vintages_v6":
        want_root(cfg.get("source_root"), STATIC_SOURCES_V6, "source_root")
        want_root(cfg.get("derived_root"), STATIC_DERIVED_V6, "derived_root")
        want_root(
            cfg.get("raw_root"),
            "gs://berlin-lst-training-data",
            "raw_root",
        )
        vintages = sorted(int(v) for v in list(cfg.get("vintages", [])))
        if vintages != [2017, 2021, 2022]:
            failures.append(f"vintages is {vintages!r}, expected [2017, 2021, 2022]")
        if int(cfg.get("svf_max_radius", -1)) != 10:
            failures.append(f"svf_max_radius is {cfg.get('svf_max_radius')!r}, expected 10 (100 m)")
        if int(cfg.get("svf_n_directions", -1)) != 16:
            failures.append(f"svf_n_directions is {cfg.get('svf_n_directions')!r}, expected 16")
        need(f"{STATIC_SOURCES_V6}/ledger.parquet", "static-sources v6 ledger")
        need(
            f"{STATIC_DERIVED_V6}/_state/static/derived/ledger.parquet",
            "static-derived v6 ledger",
        )
        need(GEOMETRY_MAPPING, "geometry mapping")
        for vintage, archive in (
            (2017, "LoD1_2017.zip"),
            (2021, "LoD2_BE_1_33_2021.zip"),
            (2022, "LoD2_2022.zip"),
        ):
            need(
                f"gs://berlin-lst-training-data/lod_vintages/{vintage}/{archive}",
                f"raw archive {vintage}",
            )
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
        for uri, label in v6_vintage_horizon_uris():
            need(uri, label)
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
        for uri, label in v6_vintage_product_uris():
            need(uri, label)
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
        # Features ledger via the same helper the features pipeline and
        # the training release gate publish/read through
        # (data.features.paths.ledger_path): <root>/_state/features/ledger.parquet.
        need(features_ledger_path(FEATURES_V6), "features/v6 ledger")
        # ARD v10 ledger: the ARD pipeline opens Ledger at
        # f"{output_root}/ledger.parquet" (data/ard/pipeline.py); there is
        # no ledger helper in data/ard/paths.py, so this stays an f-string.
        need(f"{ARD_V10_ROOT}/ledger.parquet", "ARD v10 ledger")
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        need(GEOMETRY_MAPPING, "geometry mapping")
        for uri, label in v6_vintage_product_uris():
            need(uri, label)
        if exists(training_release_completion(TRAINING_V3)):
            failures.append(
                f"training/v3 already published ({TRAINING_V3}/complete.json) — "
                "a new release requires a new training root"
            )
    elif step == "patch_index_v3":
        want_root(cfg.get("source_root"), TRAINING_V3, "source_root")
        want_root(cfg.get("output_root"), PATCH_INDEX_V3, "output_root")
        # Source-release artifacts via data.training.paths, matching what
        # patch_index._verify_source_release + build require: the release
        # marker, the scene manifest, and — for a training/v3 source — the
        # scaler carrying the v6 v3_config_hash.
        need(training_release_completion(TRAINING_V3), "training/v3 complete.json")
        need(training_manifest_parquet(TRAINING_V3), "training/v3 manifest.parquet")
        need(training_scaler_json(TRAINING_V3), "training/v3 scaler.json")
        marker_uri = training_release_completion(TRAINING_V3)
        if exists(marker_uri):
            try:
                marker = json.loads(read_bytes(marker_uri))
            except Exception as exc:
                failures.append(f"training/v3 marker unreadable: {exc}")
            else:
                if not marker.get("policy_hash"):
                    failures.append("training/v3 marker carries no policy_hash")
        if exists(patch_index_completion_uri(PATCH_INDEX_V3)):
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
        # Release artifacts via data.training.paths, matching what the EDA
        # runner reads (data.eda.load + run_training_eda.run): the training
        # manifest/cells/scaler, the v4 selection bundle for the
        # acquisition calendar, and the patch-index parquet behind its
        # completion marker.
        need(training_release_completion(TRAINING_V3), "training/v3 complete.json")
        need(training_manifest_parquet(TRAINING_V3), "training/v3 manifest.parquet")
        need(training_cells_parquet(TRAINING_V3), "training/v3 cells.parquet")
        need(training_scaler_json(TRAINING_V3), "training/v3 scaler.json")
        need(MANIFEST_V4_URI, "v4 manifest.parquet")
        need(patch_index_parquet_uri(PATCH_INDEX_V3), "patch-index v3 patch_index.parquet")
        need(patch_index_completion_uri(PATCH_INDEX_V3), "patch-index v3 complete.json")
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
