# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Read-only V3 feature-release preflight (hard go/no-go gate).

Verifies every data-side precondition that must hold before the single
guarded full V3 publication:

- exactly the expected number of assessable scenes (324 for the full
  training universe),
- the canonical ``features/v3`` prefix is completely empty (a partially
  published release must be investigated, not resumed over),
- the V3 config hash computed exactly as the pipeline will (manifests,
  geometry mapping, ARD / static-source / static-derived / dynamic
  ledgers, AOI, vegetation carry-forward, LoD coverage evidence + LoD
  COG content fingerprints),
- the four-vintage smoke fixture scenes resolve as assessable with the
  expected LoD vintages (proves the smoke gate is meaningful against
  the current data).

Git-state checks (clean pushed HEAD) run in the VM wrapper, not here, so
this script never spawns subprocesses. The script writes nothing. Exit
code is non-zero on any failed precondition.

Usage
-----
    uv run python scripts/operators/preflight_feature_release.py
    uv run python scripts/operators/preflight_feature_release.py --config-name full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.dynamic.geometry import load_geometry_mapping
from berlin_lst_downscaling.data.features.lod_coverage import (
    resolve_lod_coverage_artifacts,
)
from berlin_lst_downscaling.data.features.schema import config_hash_for_features
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.qa.inventory import build_inventory

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs" / "features"

# Smoke fixture: one scene per LoD vintage (mirrors configs/features/smoke.yaml).
SMOKE_SCENES = {
    "LC08_L2SP_193023_20170720_02_T1": 2017,
    "LC08_L2SP_192024_20210910_02_T1": 2021,
    "LC09_L2SP_193023_20220624_02_T1": 2022,
    "LC09_L2SP_193023_20240629_02_T1": 2024,
}


def load_config(config_name: str) -> OmegaConf:
    base = OmegaConf.load(_CONFIG_DIR / "_base.yaml")
    over = OmegaConf.load(_CONFIG_DIR / f"{config_name}.yaml")
    return OmegaConf.merge(base, over)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", default="full")
    args = parser.parse_args()

    cfg = load_config(args.config_name)
    failures: list[str] = []

    # ── 1. inventory + assessable count ───────────────────────────────
    inventory = build_inventory(
        manifest_uri=str(cfg.manifest_uri),
        ard_root=str(cfg.ard_root),
        static_sources_root=str(cfg.static_sources_root),
        static_derived_root=str(cfg.static_derived_root),
        dynamic_root=str(cfg.dynamic_root),
        geometry_mapping_uri=str(cfg.geometry_mapping_uri),
    )
    if not inventory.ok:
        failures.append(f"inventory failed: {inventory.errors}")
    assessed = inventory.assessed
    expected = int(cfg.get("expected_scene_count", 0) or 0)
    if expected and assessed != expected:
        failures.append(f"assessed {assessed} != expected {expected}")

    # ── 2. V3 prefix must be empty ────────────────────────────────────
    from google.cloud import storage

    output_root = str(cfg.output_root)
    prefix = f"{output_root.removeprefix('gs://berlin-lst-data/')}/"
    bucket = storage.Client().get_bucket("berlin-lst-data")
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        failures.append(
            f"canonical prefix not empty: {len(blobs)} blobs under {output_root} "
            f"(first: {blobs[0].name})"
        )

    # ── 3. config hash (identical inputs to the pipeline) ─────────────
    mapping_report = load_geometry_mapping(str(cfg.geometry_mapping_uri))
    if not mapping_report.ok or mapping_report.mapping is None:
        failures.append(f"geometry mapping failed: {mapping_report.errors}")
        mapping = None
    else:
        mapping = mapping_report.mapping
    aoi_uri = str(cfg.aoi_mask_uri)
    if not exists(aoi_uri):
        failures.append(f"AOI mask missing: {aoi_uri}")
    veg_vintage = int(cfg.get("vegetation_carry_forward_vintage", 2024))
    veg_geometry_id = ""
    if mapping is not None:
        veg_geometry_id = str(mapping.vintages.get(veg_vintage, {}).get("geometry_id", ""))
        if not veg_geometry_id:
            failures.append(f"vegetation carry-forward geometry missing for {veg_vintage}")
    config_hash = ""
    if not failures:
        lod_artifacts = resolve_lod_coverage_artifacts(str(cfg.static_sources_root))
        config_hash = config_hash_for_features(
            manifest_hash=inventory.fingerprints["manifest"],
            geometry_mapping_hash=mapping.content_hash,
            ard_ledger_hash=inventory.fingerprints["ard_ledger"],
            static_sources_ledger_hash=inventory.fingerprints.get("static_sources_ledger", ""),
            static_derived_ledger_hash=inventory.fingerprints["static_derived_ledger"],
            dynamic_ledger_hash=inventory.fingerprints["dynamic_ledger"],
            aoi_fingerprint=sha256_bytes(read_bytes(aoi_uri))[:16],
            vegetation_carry_forward_geometry_id=veg_geometry_id,
            lod_coverage_fingerprints={str(v): a.fingerprint for v, a in lod_artifacts.items()},
            lod_cog_fingerprints={str(v): a.cog_fingerprint for v, a in lod_artifacts.items()},
        )

    # ── 4. smoke fixture scenes resolve with expected vintages ────────
    by_id = {s.scene_id: s for s in inventory.scenes}
    for scene_id, vintage in SMOKE_SCENES.items():
        scene = by_id.get(scene_id)
        if scene is None or not scene.assessable:
            failures.append(f"smoke scene {scene_id} not assessable")
        elif scene.lod_vintage != vintage:
            failures.append(f"smoke scene {scene_id}: vintage {scene.lod_vintage} != {vintage}")

    report = {
        "assessed": assessed,
        "expected_scene_count": expected,
        "prefix_empty": not blobs,
        "config_hash": config_hash,
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())