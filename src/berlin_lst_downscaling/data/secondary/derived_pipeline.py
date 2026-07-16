"""Pipeline B — derived geometry product computation.

Consumes finalized Pipeline A source products from GCS and produces:
- building_dsm, vegetation_dsm, combined_dsm
- horizon_building, horizon_vegetation (36-band cubes)
- svf

Pipeline B refuses any input that is not a finalized GCS product with
valid COG, STAC, provenance, and completion marker.
"""

from __future__ import annotations

import time
from uuid import uuid4

from omegaconf import DictConfig

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.paths import ledger_path
from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
from berlin_lst_downscaling.data.secondary.reports import (
    format_secondary_report,
    persist_secondary_report,
    secondary_qa_report,
)
from berlin_lst_downscaling.data.secondary.source_products import (
    resolve_source_products,
)


def run_derived(cfg: DictConfig) -> int:
    """Execute the derived geometry pipeline (Pipeline B).

    Returns 0 on success, 1 if any items failed.
    """
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    derived_root = str(cfg.get("derived_root", "data/static/derived"))
    t0 = time.perf_counter()

    _banner(cfg, run_id, source_root, derived_root)

    # ── 0. preflight: resolve source products ────────────────────────
    report = resolve_source_products(source_root)
    if not report.ok:
        print("  SOURCE RESOLUTION FAILED:")
        for err in report.errors:
            print(f"    {err}")
        return 1

    print(f"  Resolved {len(report.resolved)} source products")
    for r in report.resolved:
        print(f"    {r.source}/{r.revision} — {r.cog_uri}")

    # ── 1. build upstream map ────────────────────────────────────────
    src_map = {f"{r.source}/{r.revision}": r for r in report.resolved}

    led = SecondaryLedger.open(ledger_path(derived_root))
    failed = 0

    # ── 2. derive DSMs ───────────────────────────────────────────────
    failed += _run_dsm_products(led, cfg, run_id, derived_root, src_map)

    # ── 3. derive horizons + SVF ─────────────────────────────────────
    failed += _run_horizon_svf(led, cfg, run_id, derived_root, src_map)

    # ── 4. final report ──────────────────────────────────────────────
    report = secondary_qa_report(led, run_id)
    print(format_secondary_report(report))
    persist_secondary_report(report, derived_root)

    elapsed = time.perf_counter() - t0
    print(f"  Duration: {elapsed:.1f}s")
    return 0 if failed == 0 else 1


# ── DSM stage ────────────────────────────────────────────────────────


def _run_dsm_products(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    derived_root: str,
    src_map: dict,
) -> int:
    """Compute and publish building/vegetation/combined DSMs."""
    from berlin_lst_downscaling.data.secondary.dsm import (
        config_hash_for_dsm,
        prepare_building_dsm,
        prepare_combined_dsm,
        prepare_vegetation_dsm,
    )

    grid = canon_grid_10m()
    failed = 0

    # Resolve required upstream sources
    terrain = src_map.get("terrain_height/2021")
    vh = src_map.get("vegetation_height/2020")
    lod2 = src_map.get("lod2_morphology/2024")

    if terrain is None or vh is None or lod2 is None:
        missing = [s for s, v in [
            ("terrain_height/2021", terrain),
            ("vegetation_height/2020", vh),
            ("lod2_morphology/2024", lod2),
        ] if v is None]
        print(f"  DSM skipped — missing upstream: {missing}")
        return 1

    upstream_hashes = {
        "terrain": "2021",
        "lod2": "2024",
        "vh": "2020",
    }

    # Building DSM
    bldg_item = "building_dsm"
    bldg_hash = config_hash_for_dsm("building", **upstream_hashes)
    todo = reconcile([(bldg_item, bldg_item, "2024")], led, bldg_hash)

    if todo:
        print("  Processing building_dsm...")
        led.upsert(SecondaryLedgerRow(
            item_id=bldg_item, source="building_dsm",
            period_or_vintage="2024", status="exporting",
            run_id=run_id,
        ))
        try:
            prepared = prepare_building_dsm(
                terrain.cog_uri, lod2.cog_uri, derived_root, run_id,
                item_key="2024", upstream_hashes=upstream_hashes,
            )
            artifacts = finalize_secondary_product(
                prepared, grid, derived_root, run_id,
            )
            led.upsert(SecondaryLedgerRow(
                item_id=bldg_item, source="building_dsm",
                period_or_vintage="2024", status="done",
                run_id=run_id, config_hash=bldg_hash,
                output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            ))
            print(f"  building_dsm OK — {artifacts.cog_uri}")
        except Exception as exc:
            print(f"  building_dsm FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=bldg_item, source="building_dsm",
                period_or_vintage="2024", status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1

    # Vegetation DSM
    veg_item = "vegetation_dsm"
    veg_hash = config_hash_for_dsm("vegetation", **upstream_hashes)
    todo = reconcile([(veg_item, veg_item, "2020")], led, veg_hash)

    if todo:
        print("  Processing vegetation_dsm...")
        led.upsert(SecondaryLedgerRow(
            item_id=veg_item, source="vegetation_dsm",
            period_or_vintage="2020", status="exporting",
            run_id=run_id,
        ))
        try:
            prepared = prepare_vegetation_dsm(
                terrain.cog_uri, vh.cog_uri, derived_root, run_id,
                item_key="2020", upstream_hashes=upstream_hashes,
            )
            artifacts = finalize_secondary_product(
                prepared, grid, derived_root, run_id,
            )
            led.upsert(SecondaryLedgerRow(
                item_id=veg_item, source="vegetation_dsm",
                period_or_vintage="2020", status="done",
                run_id=run_id, config_hash=veg_hash,
                output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            ))
            print(f"  vegetation_dsm OK — {artifacts.cog_uri}")
        except Exception as exc:
            print(f"  vegetation_dsm FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=veg_item, source="vegetation_dsm",
                period_or_vintage="2020", status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1

    # Combined DSM — depends on building and vegetation DSMs
    # Read their COGs from the derived root after publication
    combined_uri = _product_cog(derived_root, "building_dsm", "building_dsm", "2024")
    veg_dsm_uri = _product_cog(derived_root, "vegetation_dsm", "vegetation_dsm", "2020")

    from berlin_lst_downscaling.data.io.storage import exists
    if exists(combined_uri) and exists(veg_dsm_uri):
        combined_item = "combined_dsm"
        combined_hash = config_hash_for_dsm("combined", **upstream_hashes)
        todo = reconcile([(combined_item, combined_item, "2024")], led, combined_hash)

        if todo:
            print("  Processing combined_dsm...")
            led.upsert(SecondaryLedgerRow(
                item_id=combined_item, source="combined_dsm",
                period_or_vintage="2024", status="exporting",
                run_id=run_id,
            ))
            try:
                prepared = prepare_combined_dsm(
                    combined_uri, veg_dsm_uri, derived_root, run_id,
                    item_key="2024", upstream_hashes=upstream_hashes,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id=combined_item, source="combined_dsm",
                    period_or_vintage="2024", status="done",
                    run_id=run_id, config_hash=combined_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                print(f"  combined_dsm OK — {artifacts.cog_uri}")
            except Exception as exc:
                print(f"  combined_dsm FAILED: {exc}")
                led.upsert(SecondaryLedgerRow(
                    item_id=combined_item, source="combined_dsm",
                    period_or_vintage="2024", status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1

    return failed


# ── Horizons + SVF stage ────────────────────────────────────────────


def _run_horizon_svf(
    led: SecondaryLedger,
    cfg: DictConfig,
    run_id: str,
    derived_root: str,
    src_map: dict,
) -> int:
    """Compute horizons and SVF from combined DSM."""
    from berlin_lst_downscaling.data.io.storage import exists
    from berlin_lst_downscaling.data.secondary.horizon import (
        config_hash_for_horizon,
        prepare_horizon,
    )
    from berlin_lst_downscaling.data.secondary.svf import (
        config_hash_for_svf,
        prepare_svf,
    )

    grid = canon_grid_10m()
    failed = 0

    combined_uri = _product_cog(derived_root, "combined_dsm", "combined_dsm", "2024")

    if not exists(combined_uri):
        print("  Horizons/SVF skipped — combined_dsm not available")
        return 1

    max_radius_m = cfg.get("horizon_max_radius_m", 200)
    svf_max_radius = cfg.get("svf_max_radius", 3)
    svf_n_dir = cfg.get("svf_n_directions", 16)

    for component in ["building", "vegetation"]:
        item = f"horizon_{component}"
        c_hash = config_hash_for_horizon(component, max_radius_m, "2024")
        todo = reconcile([(item, item, "2024")], led, c_hash)

        if todo:
            print(f"  Processing horizon_{component}...")
            led.upsert(SecondaryLedgerRow(
                item_id=item, source=f"horizon_{component}",
                period_or_vintage="2024", status="exporting",
                run_id=run_id,
            ))
            try:
                prepared = prepare_horizon(
                    combined_uri, derived_root, run_id,
                    item_key="2024", component=component,
                    upstream_hash="2024", max_radius_m=max_radius_m,
                )
                artifacts = finalize_secondary_product(
                    prepared, grid, derived_root, run_id,
                )
                led.upsert(SecondaryLedgerRow(
                    item_id=item, source=f"horizon_{component}",
                    period_or_vintage="2024", status="done",
                    run_id=run_id, config_hash=c_hash,
                    output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                    provenance_uri=artifacts.provenance_uri,
                    completion_uri=artifacts.completion_uri,
                ))
                print(f"  horizon_{component} OK — {artifacts.cog_uri}")
            except Exception as exc:
                print(f"  horizon_{component} FAILED: {exc}")
                led.upsert(SecondaryLedgerRow(
                    item_id=item, source=f"horizon_{component}",
                    period_or_vintage="2024", status="failed",
                    run_id=run_id, last_error=str(exc),
                ))
                failed += 1

    # SVF
    svf_item = "svf"
    svf_hash = config_hash_for_svf(svf_max_radius, svf_n_dir, "2024")
    todo = reconcile([(svf_item, svf_item, "2024")], led, svf_hash)

    if todo:
        print("  Processing svf...")
        led.upsert(SecondaryLedgerRow(
            item_id=svf_item, source="svf",
            period_or_vintage="2024", status="exporting",
            run_id=run_id,
        ))
        try:
            prepared = prepare_svf(
                combined_uri, derived_root, run_id,
                item_key="2024", upstream_hash="2024",
                max_radius=svf_max_radius, n_directions=svf_n_dir,
            )
            artifacts = finalize_secondary_product(
                prepared, grid, derived_root, run_id,
            )
            led.upsert(SecondaryLedgerRow(
                item_id=svf_item, source="svf",
                period_or_vintage="2024", status="done",
                run_id=run_id, config_hash=svf_hash,
                output_uri=artifacts.cog_uri, stac_uri=artifacts.stac_uri,
                provenance_uri=artifacts.provenance_uri,
                completion_uri=artifacts.completion_uri,
            ))
            print(f"  svf OK — {artifacts.cog_uri}")
        except Exception as exc:
            print(f"  svf FAILED: {exc}")
            led.upsert(SecondaryLedgerRow(
                item_id=svf_item, source="svf",
                period_or_vintage="2024", status="failed",
                run_id=run_id, last_error=str(exc),
            ))
            failed += 1

    return failed


# ── helpers ──────────────────────────────────────────────────────────


def _product_cog(root: str, source: str, category: str, vintage: str) -> str:
    """Build the COG URI for a derived product."""
    return (
        f"{root.rstrip('/')}/ard/static/derived"
        f"/{category}/{vintage}/{source}_{vintage}.tif"
    )


def _banner(
    cfg: DictConfig, run_id: str, source_root: str, derived_root: str,
) -> None:
    width = 60
    print("=" * width, flush=True)
    print(f"Derived Geometry Pipeline (B) — mode={cfg.mode}", flush=True)
    print(f"  run_id      : {run_id}", flush=True)
    print(f"  source_root : {source_root}", flush=True)
    print(f"  derived_root: {derived_root}", flush=True)
    print("=" * width, flush=True)
