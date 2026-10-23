"""ARD pipeline orchestration — run, per-scene processing, STAC construction.

The outer :func:`run` dispatches by ``cfg.mode``:

* **smoke** — discover scene ID via lightweight STAC search, then process
  one scene per source (accepts double-load for simplicity).
* **full** — read a manifest of scenes, reconcile against ledger, process
  only scenes that need work (new, failed, interrupted, schema-changed).

Per-scene logic is in per-source runner functions registered in
:data:`_RUNNERS`: load → mask → write COG → write STAC → update ledger.

Supported sources: ``landsat-c2-l2``, ``sentinel-2-l2a``, ``ecostress``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import xarray as xr
from omegaconf import DictConfig

from berlin_lst_downscaling.common.config import BERLIN_BBOX
from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.acquisition.ecostress import load_ecostress_scene
from berlin_lst_downscaling.data.acquisition.landsat import load_landsat_scene
from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import (
    s2_mosaic_enabled as _s2_mosaic_enabled,
)
from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import s2_mosaic_group
from berlin_lst_downscaling.data.acquisition.sentinel2 import (
    S2_SWIR_BANDS,
    load_s2_scene,
)
from berlin_lst_downscaling.data.ard.aoi import compute_aoi_metrics
from berlin_lst_downscaling.data.ard.contract import Contract, contract_for_source
from berlin_lst_downscaling.data.ard.idempotency import reconcile
from berlin_lst_downscaling.data.ard.landsat_target import (
    LWIR11_NODATA_DN,
    landsat_schema_hash,
    policy_from_config,
    usgs_30m_geobox_covering,
)
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow
from berlin_lst_downscaling.data.ard.masking import (
    landsat_uses_area_mean,
    mask_ecostress,
    mask_landsat,
    mask_landsat_area_mean,
    mask_s2,
)
from berlin_lst_downscaling.data.ard.paths import cog_path, flag_path, stac_path
from berlin_lst_downscaling.data.ard.product import finalize_ard_product
from berlin_lst_downscaling.data.ard.publish_guard import assert_sentinel2_output_root
from berlin_lst_downscaling.data.ard.reports import qa_report
from berlin_lst_downscaling.data.ard.s2_mosaic import (
    aoi_mask_uri,
    coverage_metrics,
    load_aoi_on_grid,
    mosaic_s2_items,
    recorded_s2_members,
    s2_aoi_identity,
    s2_input_contract,
    s2_input_contract_hash,
    validate_missing_fraction,
)
from berlin_lst_downscaling.data.ard.s2_reflectance import (
    reflectance_scale_from_item,
    reflectance_scale_from_items,
)
from berlin_lst_downscaling.data.ard.solar_position import solar_position
from berlin_lst_downscaling.data.ard.validate import validate_cog, validate_flag_cog
from berlin_lst_downscaling.data.ard.writer import (
    write_cog_atomic,
    write_flag_cog_atomic,
)
from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)


def _load_s2_mosaic_aoi(target: xr.DataArray, cfg: DictConfig):
    return load_aoi_on_grid(aoi_mask_uri(cfg), target)


def _assert_s2_resume_contract(ledger: Ledger, scene_id: str, expected_hash: str) -> None:
    """Fail before reconciliation or ledger mutation when v10 inputs have drifted."""
    row = ledger.get(scene_id, "sentinel-2-l2a")
    if row is not None and (row.schema_version == 10 or row.status == "done"):
        if row.schema_version != 10 or row.schema_hash != expected_hash:
            log_event(
                _logger,
                logging.ERROR,
                "s2_input_contract_mismatch",
                scene_id=scene_id,
                status=row.status,
                stored_hash=row.schema_hash,
                expected_hash=expected_hash,
            )
            raise RuntimeError(
                f"scene {scene_id}: Sentinel-2 input contract is missing or changed "
                f"(stored={row.schema_hash!r}, expected={expected_hash!r}). "
                "Use a new output_root; existing products and ledger are preserved."
            )


def _schema_hash(source: str, contract: Contract, cfg: DictConfig) -> str:
    """Ledger schema hash. Landsat rows also record the target policy."""
    if source == "landsat-c2-l2":
        st_resampling, qa_aggregation = policy_from_config(cfg)
        return landsat_schema_hash(contract.schema_version, st_resampling, qa_aggregation)
    return contract.schema_version_str()


# ── per-source runner registry ───────────────────────────────────────


def _run_landsat_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
    ecostress_raw_dir: str | None = None,  # unused by Landsat
) -> xr.Dataset:
    """Load + mask a Landsat C2 L2 scene."""
    effective_date = scene_date or cfg.scene_date
    bbox = tuple(cfg.bbox)
    resolution = int(cfg.target_resolution_low)
    if landsat_uses_area_mean(cfg):
        target = canon_grid_for_resolution(resolution)
        st_resampling, _qa = policy_from_config(cfg)
        ds, loaded_ids = load_landsat_scene(
            date=effective_date,
            bbox=bbox,
            bands=["lwir11"],
            resolution=resolution,
            items=items,
            geobox=target,
            resampling=st_resampling,
            nodata=LWIR11_NODATA_DN,
        )
        qa_ds, _qa_ids = load_landsat_scene(
            date=effective_date,
            bbox=bbox,
            bands=["qa_pixel"],
            items=items,
            geobox=usgs_30m_geobox_covering(target),
            resampling="nearest",
        )
        if scene_id not in loaded_ids:
            raise RuntimeError(
                f"Scene {scene_id!r} not in loaded items {loaded_ids}. "
                f"Date {effective_date!r} or bbox may not cover the requested scene."
            )
        return mask_landsat_area_mean(ds, qa_ds, cfg)
    ds, loaded_ids = load_landsat_scene(
        date=effective_date,
        bbox=bbox,
        resolution=resolution,
        items=items,
        resampling="nearest",
    )
    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Scene {scene_id!r} not in loaded items {loaded_ids}. "
            f"Date {effective_date!r} or bbox may not cover the requested scene."
        )
    return mask_landsat(ds, cfg)


def _run_sentinel2_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
    ecostress_raw_dir: str | None = None,  # unused by S2
) -> xr.Dataset:
    """Load + mask a Sentinel-2 L2A scene."""
    if not items:
        raise ValueError(
            f"scene {scene_id}: Sentinel-2 reflectance needs the STAC item "
            "so BOA_ADD_OFFSET can be chosen from metadata"
        )
    if _s2_mosaic_enabled(cfg):
        return _run_sentinel2_mosaic(scene_id, cfg, items, scene_date)
    reflectance = reflectance_scale_from_items(items)
    effective_date = scene_date or cfg.scene_date
    bbox = tuple(cfg.bbox)
    ds_10m, loaded_ids = load_s2_scene(
        date=effective_date,
        bbox=bbox,
        resolution=int(cfg.target_resolution_high),
        bands=["B02", "B03", "B04", "B08", "SCL"],
        items=items,
    )
    # SWIR bands + SCL are loaded natively on the 20m grid so B11/B12
    # can be masked at native resolution before the bilinear 20→10m
    # resampling inside mask_s2.
    ds_20m, _ = load_s2_scene(
        date=effective_date,
        bbox=bbox,
        resolution=20,
        bands=list(S2_SWIR_BANDS),
        items=items,
    )
    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Scene {scene_id!r} not in loaded items {loaded_ids}. "
            f"Date {effective_date!r} or bbox may not cover the requested scene."
        )
    # S2 uses NOAA computation (PC items lack view:sun_*)
    az, el = _solar_for_scene(ds_10m, cfg)
    masked = mask_s2(ds_10m, ds_20m, cfg, az, el, reflectance=reflectance)
    return masked


def _run_sentinel2_mosaic(
    scene_id: str, cfg: DictConfig, items: list[Any], scene_date: str | None
) -> xr.Dataset:
    """Mask and harmonize each exact member before any mosaic or COG write."""
    import numpy as np

    threshold = validate_missing_fraction(
        cfg.sentinel2.mosaic.get("max_missing_aoi_fraction", 0.005)
    )
    if int(cfg.target_resolution_high) != 10:
        raise ValueError("Sentinel-2 schema-10 mosaics require the canonical 10m grid")
    item_ids = [item.id for item in items]
    if len(set(item_ids)) != len(item_ids) or scene_id not in item_ids:
        raise ValueError(f"Invalid Sentinel-2 mosaic members for scene {scene_id}")
    group = s2_mosaic_group(items[0], bbox=cfg.bbox)
    if any(s2_mosaic_group(item, bbox=cfg.bbox) != group for item in items):
        raise ValueError(f"Sentinel-2 mosaic members have inconsistent groups for scene {scene_id}")
    members = []
    for item in sorted(items, key=lambda item: item.id):
        reflectance = reflectance_scale_from_item(item)
        # Nearest resampling is explicit: both loads carry the SCL
        # classification mask, which must never be interpolated, and the
        # footprint/donor choice must match selection's nearest SCL mosaic.
        ds_10m, ids_10m = load_s2_scene(
            date=scene_date or cfg.scene_date,
            bbox=tuple(cfg.bbox),
            resolution=10,
            bands=["B02", "B03", "B04", "B08", "SCL"],
            items=[item],
            resampling="nearest",
        )
        ds_20m, ids_20m = load_s2_scene(
            date=scene_date or cfg.scene_date,
            bbox=tuple(cfg.bbox),
            resolution=20,
            bands=list(S2_SWIR_BANDS),
            items=[item],
            resampling="nearest",
        )
        if ids_10m != [item.id] or ids_20m != [item.id]:
            raise RuntimeError(f"Sentinel-2 mosaic member {item.id} did not load exactly")
        scl = ds_10m["SCL"]
        if "time" in scl.dims:
            if scl.sizes["time"] != 1:
                raise ValueError(f"Sentinel-2 mosaic member {item.id} has multiple times")
            scl = scl.isel(time=0)
        scl_values = scl.transpose("y", "x").values
        footprint = np.isfinite(scl_values) & (scl_values != 0)
        az, el = _solar_for_scene(ds_10m, cfg)
        masked = mask_s2(ds_10m, ds_20m, cfg, az, el, reflectance=reflectance)
        members.append((item.id, masked, footprint))
    mosaic, footprint = mosaic_s2_items(members)
    provenance = mosaic.attrs["s2_reflectance"]
    provenance["s2_mosaic_group"] = group
    try:
        aoi = _load_s2_mosaic_aoi(mosaic["flag"], cfg)
        coverage = coverage_metrics(footprint, aoi, threshold)
    except Exception as exc:
        log_event(
            _logger,
            logging.ERROR,
            "s2_mosaic_prewrite",
            scene_id=scene_id,
            provenance=provenance,
            coverage_passed=False,
            error=str(exc),
        )
        raise
    provenance["coverage"] = coverage
    log_event(
        _logger,
        logging.INFO if coverage["coverage_passed"] else logging.ERROR,
        "s2_mosaic_prewrite",
        scene_id=scene_id,
        provenance=provenance,
        **coverage,
    )
    if not coverage["coverage_passed"]:
        raise ValueError(
            f"Sentinel-2 mosaic AOI coverage failed: missing {coverage['aoi_missing_px']}/"
            f"{coverage['aoi_total_px']} pixels ({coverage['missing_aoi_fraction']:.8f}) "
            f"> max_missing_aoi_fraction={threshold}; item_ids={provenance['item_ids']}"
        )
    return mosaic


def _run_ecostress_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
    ecostress_raw_dir: str | None = None,
) -> xr.Dataset:
    """Load + mask an ECOSTRESS L2T granule from local COGs.

    No STAC search — scene_id comes from the manifest.  The raw_dir
    (``cfg.ecostress.raw_dir`` or ``ecostress_raw_dir`` override) is
    expected to contain one sub-directory per granule, each with
    ECO_L2T_LSTE layer COGs.
    """
    bbox = tuple(cfg.bbox) if cfg.get("bbox") else None
    raw_dir = ecostress_raw_dir or str(cfg.ecostress.raw_dir)

    ds, loaded_ids = load_ecostress_scene(
        granule_id=scene_id,
        raw_dir=raw_dir,
        bbox=bbox,
    )

    if scene_id not in loaded_ids:
        raise RuntimeError(
            f"Granule {scene_id!r} not in loaded IDs {loaded_ids}. "
            "Check that the granule exists in the raw_dir."
        )

    masked = mask_ecostress(ds, cfg)
    return masked


# Registry maps source key → per-source runner function.
# Each runner returns a masked xr.Dataset ready for COG writing.
_RUNNERS: dict[str, Callable[..., Any]] = {
    "landsat-c2-l2": _run_landsat_scene,
    "sentinel-2-l2a": _run_sentinel2_scene,
    "ecostress": _run_ecostress_scene,
}

# ── main entry ───────────────────────────────────────────────────────


def run(cfg: DictConfig, run_id: str | None = None) -> int:
    """Execute the ARD pipeline — manifest-driven (mode=full).

    ``smoke_primary`` builds the 8-row smoke bundle and runs ``mode=full``
    end-to-end.

    Returns 0 on success, 1 if any scene failed.
    """
    if run_id is None:
        run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    sources = list(cfg.sources)
    contracts = {}
    for source in sources:
        version = (
            cfg.get("sentinel2", {}).get("schema_version") if source == "sentinel-2-l2a" else None
        )
        contract = contract_for_source(source, schema_version=version)
        if source == "sentinel-2-l2a":
            if _s2_mosaic_enabled(cfg) != (contract.schema_version == 10):
                raise ValueError(
                    "Sentinel-2 mosaic.enabled and schema_version=10 must be set together"
                )
            if _s2_mosaic_enabled(cfg):
                validate_missing_fraction(
                    cfg.sentinel2.mosaic.get("max_missing_aoi_fraction", 0.005)
                )
        contracts[source] = contract
    if "sentinel-2-l2a" in sources:
        assert_sentinel2_output_root(
            output_root, schema_version=contracts["sentinel-2-l2a"].schema_version
        )

    led = Ledger.open(f"{output_root}/ledger.parquet")
    log_event(_logger, logging.INFO, "start", mode=cfg.mode, sources=list(cfg.sources))

    for source in sources:
        contract = contracts[source]
        _process_manifest(source, contract, cfg, led, run_id)

    report = qa_report(led, cfg, run_id)
    failed_count = sum(report.get("per_source", {}).get(s, {}).get("failed", 0) for s in sources)
    log_event(_logger, logging.INFO, "qa_report", **report)

    return 0 if failed_count == 0 else 1


def _process_manifest(
    source: str,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
) -> None:
    """Process scenes listed in a manifest Parquet (mode=full).

    Loads the canonical v3/v4 bundle, validates it once, and resolves
    exact STAC items from item_href before passing them to loaders.
    """
    import pyarrow.compute as pc  # type: ignore[attr-defined]

    from berlin_lst_downscaling.data.io.storage import exists as _exists
    from berlin_lst_downscaling.data.selection.validate import load_bundle

    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri or not _exists(manifest_uri):
        raise FileNotFoundError(
            f"mode=full requires the published manifest bundle at {manifest_uri!r}"
        )

    bundle, validation = load_bundle(manifest_uri, require_item_href=True)
    if not validation.ok:
        raise RuntimeError("manifest bundle validation failed: " + "; ".join(validation.errors))
    tbl = bundle.manifest_table
    if tbl.num_rows == 0:
        return

    mask = pc.equal(tbl.column("source"), source)  # type: ignore[attr-defined]
    rows = tbl.filter(mask)
    if rows.num_rows == 0:
        return

    if source == "sentinel-2-l2a":
        manifest_version = (tbl.schema.metadata or {}).get(
            b"schema_version", b"4" if "s2_items" in tbl.column_names else b"3"
        )
        has_v4 = manifest_version == b"4" or any(
            column in tbl.column_names for column in ("s2_items", "s2_mosaic_group")
        )
        if has_v4 and (contract.schema_version != 10 or not _s2_mosaic_enabled(cfg)):
            raise ValueError("Sentinel-2 v4 membership requires schema-10 with mosaic.enabled=true")
        if contract.schema_version == 10 and (not has_v4 or not _s2_mosaic_enabled(cfg)):
            raise ValueError(
                "Sentinel-2 schema-10 requires recorded v4 membership and mosaic.enabled=true"
            )

    # Collect manifest rows — carry scene_id, year, item_href, acq time
    scenes: list[tuple[str, str, int]] = []
    scene_meta: dict[str, dict] = {}
    for i in range(rows.num_rows):
        r = rows.slice(i, 1).to_pydict()
        sid = str(r["scene_id"][0])
        yr = int(r["year"][0])
        scenes.append((sid, source, yr))

        # Extract v3 fields
        href = r.get("item_href", [None])[0]
        acq_dt = r.get("acquisition_datetime", [None])[0]
        scene_meta[sid] = {
            "item_href": href,
            "acquisition_datetime": acq_dt,
            "year": yr,
            "s2_items": r.get("s2_items", [None])[0],
            "s2_mosaic_group": r.get("s2_mosaic_group", [None])[0],
        }

    s2_contracts: dict[str, dict] = {}
    if source == "sentinel-2-l2a" and contract.schema_version == 10:
        # Validate every row before AOI reads, reconcile, resolution, or writes.
        for sid, meta in scene_meta.items():
            recorded_s2_members(sid, meta)
        aoi_identity = s2_aoi_identity(cfg)
        for sid, meta in scene_meta.items():
            s2_contracts[sid] = s2_input_contract(sid, meta["year"], meta, cfg, aoi_identity)
            _assert_s2_resume_contract(ledger, sid, s2_input_contract_hash(s2_contracts[sid]))

    max_attempts = int(cfg.max_scene_attempts)
    todo = reconcile(scenes, ledger, contract, max_attempts=max_attempts)
    log_event(
        _logger,
        logging.INFO,
        "manifest_todo",
        source=source,
        total=len(scenes),
        n_todo=len(todo),
    )

    if source == "ecostress":
        _process_ecostress_todo(
            todo,
            source,
            contract,
            cfg,
            ledger,
            run_id,
            scene_meta,
        )
    else:
        for scene_id, _source, year, _reason in todo:
            meta = scene_meta.get(scene_id, {})
            _run_scene(
                scene_id,
                source,
                year,
                contract,
                cfg,
                ledger,
                run_id,
                manifest_meta=meta,
                scene_date=meta.get("acquisition_datetime"),
                s2_scene_contract=s2_contracts.get(scene_id),
            )


def _resolve_manifest_items(
    scene_id: str,
    source: str,
    meta: dict,
    *,
    require_s2_members: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[Any] | None:
    """Resolve exact STAC items from manifest metadata.

    Recorded S2 members are resolved exactly, sorted by ID, with no fallback.
    The opt-in mosaic path requires this set. For legacy rows, tries directly
    from item_href first (faster, avoids catalog search). Falls back to
    ID-based search if item_href is missing.
    Returns None when resolution fails; ECOSTRESS callers skip this path.
    """
    from berlin_lst_downscaling.data.acquisition.pc_client import (
        resolve_exact_item,
        resolve_item_from_href,
    )

    recorded = meta.get("s2_items") if source == "sentinel-2-l2a" else None
    if require_s2_members or recorded is not None:
        recorded = recorded_s2_members(scene_id, meta)
        group = meta.get("s2_mosaic_group")
        resolved = []
        for member in sorted(recorded, key=lambda member: member["item_id"]):
            item = resolve_item_from_href(member["item_href"], expected_id=member["item_id"])
            if item.id != member["item_id"]:
                raise ValueError(
                    f"scene {scene_id}: recorded member ID mismatch for {member['item_id']}"
                )
            if s2_mosaic_group(item, bbox=bbox or BERLIN_BBOX) != group:
                raise ValueError(
                    f"scene {scene_id}: recorded member {item.id} s2_mosaic_group mismatch"
                )
            resolved.append(item)
        return resolved

    item_href = meta.get("item_href")

    # Try direct HREF resolution first (preferred path)
    if item_href:
        try:
            item = resolve_item_from_href(item_href, expected_id=scene_id)
            return [item]
        except Exception as exc:
            log_event(
                _logger,
                logging.WARNING,
                "href_resolve_failed",
                scene_id=scene_id,
                item_href=item_href,
                error=str(exc),
            )

    # Fallback: resolve by ID from catalog
    try:
        item = resolve_exact_item(collection=source, scene_id=scene_id)
        return [item]
    except Exception as exc:
        log_event(
            _logger,
            logging.WARNING,
            "exact_item_resolve_failed",
            scene_id=scene_id,
            error=str(exc),
        )
        return None


def _process_ecostress_todo(
    todo: list[tuple[str, str, int, str]],
    source: str,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    scene_meta: dict[str, dict],
) -> None:
    """Process ECOSTRESS scenes with pipeline-internal staging.

    For each scene: download granule from NASA Earthdata → stage →
    process → cleanup.  Stage lifecycle is managed by ``StageSession``.
    """
    from berlin_lst_downscaling.data.acquisition.ecostress import (
        download_and_stage_granule,
    )
    from berlin_lst_downscaling.data.io.staging import StageSession

    stage_base = cfg.get("ecostress.stage_base", "data/smoke/ecostress_stage")
    with StageSession(stage_base, run_id=run_id) as stage:  # StageSession cleans up on exit
        for scene_id, _src, year, _reason in todo:
            try:
                download_and_stage_granule(scene_id, stage)
                granule_raw_dir = str(stage.uri.uri)
            except Exception as exc:
                log_event(
                    _logger,
                    logging.WARNING,
                    "scene_failed",
                    scene_id=scene_id,
                    source=source,
                    error=f"Stage download failed: {exc}",
                )
                continue

            _run_scene(
                scene_id,
                source,
                year,
                contract,
                cfg,
                ledger,
                run_id,
                scene_date=scene_meta.get(scene_id, {}).get("acquisition_datetime"),
                ecostress_raw_dir=granule_raw_dir,
            )
        # Stage cleaned up automatically on StageSession.__exit__


# ── per-scene processing ─────────────────────────────────────────────


def _run_scene(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    cfg: DictConfig,
    ledger: Ledger,
    run_id: str,
    items: list[Any] | None = None,
    scene_date: str | None = None,
    ecostress_raw_dir: str | None = None,
    manifest_meta: dict | None = None,
    s2_scene_contract: dict | None = None,
) -> None:
    """Process one scene: load → mask → write COG+STAC → update ledger.

    Parameters
    ----------
    items :
        Pre-fetched STAC items for the scene (manifest-driven).
    scene_date :
        Overrides ``cfg.scene_date`` for this scene. Used by
        ``mode=full`` where each manifest row carries its own date.
    ecostress_raw_dir :
        Per-scene override for ECOSTRESS raw_dir. Used by the
        manifest-driven ECOSTRESS pipeline where each granule is
        downloaded and staged on demand.
    manifest_meta :
        Exact item references and optional v4 mosaic group/membership.
        Resolution occurs inside the attempt so failure updates the ledger.
    s2_scene_contract :
        Optional v10 contract prepared by the manifest's read-only preflight.
        Its hash is persisted for exporting, failed, and completed attempts.
    """
    effective_date = scene_date or cfg.scene_date
    log_event(
        _logger,
        logging.INFO,
        "scene_start",
        scene_id=scene_id,
        source=source,
        scene_date=effective_date,
    )
    t0 = time.perf_counter()

    is_mosaic = source == "sentinel-2-l2a" and contract.schema_version == 10
    scene_hash: str | None = None
    attempt_started = False
    # Also protect private/direct callers before begin_attempt replaces a row.
    existing = ledger.get(scene_id, source) if is_mosaic else None
    if existing is not None and (existing.schema_version == 10 or existing.status == "done"):
        if s2_scene_contract is None:
            s2_scene_contract = s2_input_contract(
                scene_id, year, manifest_meta or {}, cfg, s2_aoi_identity(cfg)
            )
        _assert_s2_resume_contract(ledger, scene_id, s2_input_contract_hash(s2_scene_contract))

    try:
        if is_mosaic:
            if not _s2_mosaic_enabled(cfg):
                raise ValueError("Sentinel-2 schema-10 requires mosaic.enabled=true")
            recorded_s2_members(scene_id, manifest_meta or {})
            if s2_scene_contract is None:
                s2_scene_contract = s2_input_contract(
                    scene_id, year, manifest_meta or {}, cfg, s2_aoi_identity(cfg)
                )
            scene_hash = s2_input_contract_hash(s2_scene_contract)
            log_event(
                _logger,
                logging.INFO,
                "s2_mosaic_input_contract",
                scene_id=scene_id,
                input_contract_hash=scene_hash,
                input_contract=s2_scene_contract,
            )
        _attempts = ledger.begin_attempt(
            LedgerRow(
                scene_id=scene_id,
                source=source,
                year=year,
                run_id=run_id,
                **({"schema_hash": scene_hash, "schema_version": 10} if is_mosaic else {}),
            )
        )
        attempt_started = True
        if manifest_meta is not None:
            items = _resolve_manifest_items(
                scene_id,
                source,
                manifest_meta,
                require_s2_members=source == "sentinel-2-l2a" and _s2_mosaic_enabled(cfg),
                bbox=tuple(cfg.bbox),
            )
        runner = _RUNNERS.get(source)
        if runner is None:
            raise ValueError(f"Unknown source: {source}")

        masked = runner(
            scene_id=scene_id,
            source=source,
            year=year,
            contract=contract,
            cfg=cfg,
            ledger=ledger,
            run_id=run_id,
            items=items,
            scene_date=scene_date,
            ecostress_raw_dir=ecostress_raw_dir,
        )

        if is_mosaic:
            masked.attrs["s2_reflectance"]["input_contract"] = s2_scene_contract
            masked.attrs["s2_reflectance"]["input_contract_hash"] = scene_hash

        log_event(_logger, logging.INFO, "scene_writing", scene_id=scene_id, source=source)
        root = str(cfg.output_root)
        cog_dst = cog_path(root, source, year, scene_id)

        # Split data bands from flag band
        flag_da: xr.DataArray | None = masked.get("flag")
        data_bands = [v for v in masked.data_vars if v != "flag"]
        if data_bands:
            ds_data = masked[data_bands]
        else:
            ds_data = masked

        write_cog_atomic(ds_data, cog_dst, contract, overwrite=True)

        flag_dst = flag_path(root, source, year, scene_id)
        if flag_da is not None and contract.flag_mode == "separate":
            write_flag_cog_atomic(flag_da, flag_dst, contract, overwrite=True)

        # Canonical grid per source
        if source == "landsat-c2-l2":
            expected_grid = canon_grid_for_resolution(100)
        elif source == "sentinel-2-l2a":
            expected_grid = canon_grid_for_resolution(10)
        elif source == "ecostress":
            expected_grid = canon_grid_for_resolution(70)
        else:
            expected_grid = canon_grid_for_resolution(10)

        vig = validate_cog(cog_dst, contract, expected_grid)
        if vig.ok:
            log_event(_logger, logging.INFO, "cog_validated", scene_id=scene_id, source=source)
        else:
            raise RuntimeError(f"COG validation failed: {'; '.join(vig.errors)}")

        if flag_da is not None and contract.flag_mode == "separate":
            vif = validate_flag_cog(flag_dst, expected_grid)
            if not vif.ok:
                raise RuntimeError(f"Flag COG validation failed: {'; '.join(vif.errors)}")

        aoi_clear_px: int | None = None
        aoi_cloudy_px: int | None = None
        aoi_shadow_px: int | None = None
        aoi_cirrus_px: int | None = None
        aoi_saturated_px: int | None = None
        aoi_fill_px: int | None = None
        aoi_total_px: int | None = None
        aoi_overlap_px: int | None = None
        aoi_clear_frac: float | None = None
        if flag_da is not None and contract.flag_mode == "separate":
            aoi_res = (
                int(cfg.target_resolution_low)
                if source == "landsat-c2-l2"
                else int(cfg.target_resolution_high)
            )
            aoi_base = str(cfg.aoi.mask_base)
            aoi_uri = f"{aoi_base.rstrip('/')}/aoi_{aoi_res}m.tif"
            try:
                _raw = compute_aoi_metrics(flag_dst, aoi_uri, contract)
                aoi_clear_px = _int_or_none(_raw.get("aoi_clear_px"))
                aoi_cloudy_px = _int_or_none(_raw.get("aoi_cloudy_px"))
                aoi_shadow_px = _int_or_none(_raw.get("aoi_shadow_px"))
                aoi_cirrus_px = _int_or_none(_raw.get("aoi_cirrus_px"))
                aoi_saturated_px = _int_or_none(_raw.get("aoi_saturated_px"))
                aoi_fill_px = _int_or_none(_raw.get("aoi_fill_px"))
                aoi_total_px = _int_or_none(_raw.get("aoi_total_px"))
                aoi_overlap_px = _int_or_none(_raw.get("aoi_overlap_px"))
                aoi_clear_frac = _float_or_none(_raw.get("aoi_clear_frac"))
            except Exception as _exc:
                # AOI metrics are best-effort; log and continue without them
                log_event(
                    _logger,
                    logging.WARNING,
                    "aoi_metrics_error",
                    scene_id=scene_id,
                    aoi_uri=aoi_uri,
                    error=str(_exc),
                )

        min_overlap = int(cfg.aoi.min_overlap_px)
        if aoi_overlap_px is not None and aoi_overlap_px < min_overlap:
            log_event(
                _logger,
                logging.WARNING,
                "low_aoi_overlap",
                scene_id=scene_id,
                aoi_overlap_px=aoi_overlap_px,
                min_overlap_px=min_overlap,
            )

        if source == "ecostress":
            # ECOSTRESS L2T native grid is fixed at 70 m (acquisition
            # hardcodes canon_grid_70m) — never the configurable low/high
            # resolutions of the other sensors.
            resolution = 70
        elif source == "landsat-c2-l2":
            resolution = int(cfg.target_resolution_low)
        else:
            resolution = int(cfg.target_resolution_high)
        source_metadata = masked.attrs.get("s2_reflectance")
        if source_metadata is not None and not isinstance(source_metadata, dict):
            source_metadata = None
        finalize_ard_product(
            scene_id=scene_id,
            source=source,
            year=year,
            root=root,
            contract=contract,
            masked=masked,
            run_id=run_id,
            flag_uri=flag_dst if contract.flag_mode == "separate" else None,
            target_resolution=resolution,
            source_metadata=source_metadata,
        )
        stac_dst = stac_path(root, source, year, scene_id)
        elapsed = time.perf_counter() - t0
        ledger.upsert(
            LedgerRow(
                scene_id=scene_id,
                source=source,
                year=year,
                path_cog=cog_dst,
                path_flag=flag_dst if contract.flag_mode == "separate" else None,
                path_stac=stac_dst,
                status="done",
                schema_hash=scene_hash if is_mosaic else _schema_hash(source, contract, cfg),
                schema_version=contract.schema_version,
                run_id=run_id,
                updated_at=datetime.now(UTC),
                aoi_clear_px=aoi_clear_px,
                aoi_cloudy_px=aoi_cloudy_px,
                aoi_shadow_px=aoi_shadow_px,
                aoi_cirrus_px=aoi_cirrus_px,
                aoi_saturated_px=aoi_saturated_px,
                aoi_fill_px=aoi_fill_px,
                aoi_total_px=aoi_total_px,
                aoi_overlap_px=aoi_overlap_px,
                aoi_clear_frac=aoi_clear_frac,
            )
        )
        _attempts = row.attempts if (row := ledger.get(scene_id, source)) else 0
        log_event(
            _logger,
            logging.INFO,
            "scene_done",
            scene_id=scene_id,
            source=source,
            attempts=_attempts,
            elapsed_s=round(elapsed, 2),
        )

    except Exception as exc:
        if not attempt_started:
            ledger.begin_attempt(
                LedgerRow(
                    scene_id=scene_id,
                    source=source,
                    year=year,
                    run_id=run_id,
                    **({"schema_hash": scene_hash, "schema_version": 10} if is_mosaic else {}),
                )
            )
        elapsed = time.perf_counter() - t0
        _attempts = row.attempts if (row := ledger.get(scene_id, source)) else 0
        log_event(
            _logger,
            logging.ERROR,
            "scene_failed",
            scene_id=scene_id,
            source=source,
            attempts=_attempts,
            error=str(exc),
            elapsed_s=round(elapsed, 2),
        )
        # Note: attempts is auto-managed by upsert
        ledger.upsert(
            LedgerRow(
                scene_id=scene_id,
                source=source,
                year=year,
                status="failed",
                last_error=str(exc),
                run_id=run_id,
                updated_at=datetime.now(UTC),
                **({"schema_hash": scene_hash, "schema_version": 10} if is_mosaic else {}),
            )
        )
        # do not re-raise — outer loop continues


# ── solar position ───────────────────────────────────────────────────


def _solar_for_scene(
    ds: xr.Dataset,
    cfg: DictConfig,
) -> tuple[float, float]:
    """Return ``(azimuth_deg, elevation_deg)`` for the scene.

    Computed from the dataset's acquisition time via NOAA.  Falls back to
    ``cfg.scene_date`` at 10:00 UTC when the dataset has no time coordinate.
    """
    try:
        dt64 = ds.time.values[0]
        ts = dt64.astype("datetime64[us]").tolist()
        dt = datetime.fromtimestamp(ts.timestamp(), tz=UTC)
    except (IndexError, AttributeError, ValueError):
        date_str = cfg.scene_date
        dt = datetime.fromisoformat(date_str).replace(
            hour=10,
            minute=0,
            second=0,
            tzinfo=UTC,
        )
    return solar_position(dt)


def _int_or_none(val) -> int | None:
    """Convert value to int, returning None for None or NaN."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN
        return None
    return int(val)


def _float_or_none(val) -> float | None:
    """Convert value to float, returning None for None or NaN."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN
        return None
    return float(val)


__all__ = [
    "run",
]
