"""Stage-1 raw-input inventory — resolve published inputs for every paired anchor.

Composes the canonical v3 manifest bundle, the ARD ledger, the static
source/derived ledgers, the dynamic ledger, and ``geometry_mapping.json``
into one resolved record per paired Landsat anchor scene. Nothing is
synthesised from scene IDs where a ledger path exists — the ledgers are
the single source of truth for artifact URIs.

Exclusion reasons are explicit: a pair that is not assessable (missing
ledger row, missing dynamic products, or a ``role=inference`` 2026
scene) is recorded with a reason and never silently dropped.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pyarrow.parquet as pq

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.dynamic.geometry import load_geometry_mapping
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.qa.contracts import (
    STATIC_DERIVED_MORPHOLOGY_PRODUCTS,
    STATIC_DERIVED_OPTIONAL_PRODUCTS,
)
from berlin_lst_downscaling.data.secondary.imperviousness import vintage_for_scene_year
from berlin_lst_downscaling.data.selection.validate import load_bundle

# 2026 anchors are inference scenes, outside the training universe. They
# are reported as a single, expected exclusion across every QA gate.
INFERENCE_EXCLUSION_REASON = "dynamic role=inference (2026)"

# Metadata-only derived products (upstream of the shadow computation).
_METADATA_DERIVED_PRODUCTS = ("horizon_building", "horizon_vegetation")

# Static source products — vintage-fixed, metadata-checked once per run.
_STATIC_SOURCE_FAMILIES = (
    "imperviousness",
    "vegetation_height",
    "terrain_height",
    "lod2_morphology",
)


@dataclass
class ResolvedScene:
    """One resolved paired Landsat anchor scene."""

    scene_id: str
    year: int
    s2_scene_id: str
    geometry_id: str
    landsat_cog: str
    landsat_flag: str
    s2_cog: str
    s2_flag: str
    dynamic: dict[str, str]  # source -> COG URI (era5_land, shadow_*)
    static_derived: dict[str, str]  # product -> COG URI (morphology, in-support)
    static_derived_meta: dict[str, str]  # product -> COG URI (horizons, metadata-only)
    static_sources: dict[str, str]  # source -> COG URI (lod2, vh, imperv — feature inputs)
    exclusion_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def assessable(self) -> bool:
        return self.exclusion_reason is None


@dataclass
class InventoryReport:
    """Result of resolving the full training-input inventory."""

    scenes: list[ResolvedScene]
    total_pairings: int
    assessed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    static_sources: dict[str, str]  # (source, vintage) -> COG URI
    fingerprints: dict[str, str]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _read_table(uri: str):
    """Read a Parquet table from a local path or GCS URI."""
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _fingerprint(uri: str) -> str:
    """Return a short SHA-256 fingerprint of a ledger/manifest file."""
    return sha256_bytes(read_bytes(uri))[:16]


def _rows_by_key(table, key_col: str) -> dict:
    """Return ``{key: row_dict}`` from a pyarrow table column."""
    cols = table.to_pydict()
    out: dict = {}
    for i in range(table.num_rows):
        out[str(cols[key_col][i])] = {name: cols[name][i] for name in cols}
    return out


def build_inventory(
    *,
    manifest_uri: str,
    ard_root: str,
    static_sources_root: str,
    static_derived_root: str,
    dynamic_root: str,
    geometry_mapping_uri: str,
    scene_ids: list[str] | None = None,
) -> InventoryReport:
    """Resolve every paired Landsat anchor against the published ledgers.

    ``scene_ids`` restricts the resolution to the given Landsat scene IDs
    (used by the smoke run). Returns an :class:`InventoryReport` with one
    :class:`ResolvedScene` per pairing, explicit exclusion reasons, and
    fingerprints of every source ledger.
    """
    errors: list[str] = []
    fingerprints: dict[str, str] = {}

    # ── manifest bundle + pairings ─────────────────────────────────────
    bundle, bundle_validation = load_bundle(manifest_uri)
    if not bundle_validation.ok:
        errors.extend(bundle_validation.errors)
    fingerprints["manifest"] = _fingerprint(manifest_uri)

    pairings: dict[str, dict] = {}
    pair_table = bundle.pairings_table
    if pair_table.num_rows:
        cols = pair_table.to_pydict()
        for i in range(pair_table.num_rows):
            pairings[str(cols["landsat_scene_id"][i])] = {
                "sentinel2_scene_id": str(cols["sentinel2_scene_id"][i]),
            }

    manifest_rows = _rows_by_key(bundle.manifest_table, "scene_id")

    # ── ARD ledger ─────────────────────────────────────────────────────
    ard_ledger_uri = f"{ard_root.rstrip('/')}/ledger.parquet"
    fingerprints["ard_ledger"] = _fingerprint(ard_ledger_uri)
    ard_rows: dict[tuple[str, str], dict] = {}
    ard_table = _read_table(ard_ledger_uri)
    for row in _rows_by_key(ard_table, "scene_id").values():
        ard_rows[(str(row["source"]), str(row["scene_id"]))] = row

    # ── static source ledger (vintage-fixed roster) ────────────────────
    static_sources: dict[str, str] = {}
    source_rows: dict[str, dict] = {}
    src_ledger_uri = f"{static_sources_root.rstrip('/')}/ledger.parquet"
    if exists(src_ledger_uri):
        fingerprints["static_sources_ledger"] = _fingerprint(src_ledger_uri)
        for row in _rows_by_key(_read_table(src_ledger_uri), "item_id").values():
            if str(row["status"]) != "done":
                continue
            source = str(row["source"])
            if source not in _STATIC_SOURCE_FAMILIES:
                continue
            output_uri = row.get("output_uri")
            if output_uri:
                item_id = str(row["item_id"])
                static_sources[f"{source}/{row['period_or_vintage']}"] = str(output_uri)
                source_rows[item_id] = row
    else:
        errors.append(f"static sources ledger missing: {src_ledger_uri}")

    # ── static derived ledger ──────────────────────────────────────────
    derived_ledger_uri = f"{static_derived_root.rstrip('/')}/_state/static/derived/ledger.parquet"
    fingerprints["static_derived_ledger"] = _fingerprint(derived_ledger_uri)
    derived_rows: dict[tuple[str, str], dict] = {}
    derived_cols = _read_table(derived_ledger_uri).to_pydict()
    for i in range(len(derived_cols.get("item_id", []))):
        key = (str(derived_cols["source"][i]), str(derived_cols["period_or_vintage"][i]))
        derived_rows[key] = {name: derived_cols[name][i] for name in derived_cols}

    # ── dynamic ledger ─────────────────────────────────────────────────
    dynamic_ledger_uri = f"{dynamic_root.rstrip('/')}/_state/dynamic/ledger.parquet"
    fingerprints["dynamic_ledger"] = _fingerprint(dynamic_ledger_uri)
    dynamic_rows: dict[tuple[str, str], dict] = {}
    for row in _rows_by_key(_read_table(dynamic_ledger_uri), "item_id").values():
        dynamic_rows[(str(row["source"]), str(row["period_or_vintage"]))] = row

    # ── geometry mapping ───────────────────────────────────────────────
    mapping_report = load_geometry_mapping(geometry_mapping_uri)
    if not mapping_report.ok:
        errors.extend(mapping_report.errors)
    mapping = mapping_report.mapping
    if mapping is not None:
        fingerprints["geometry_mapping"] = mapping.content_hash

    # ── resolve per pairing ────────────────────────────────────────────
    scenes: list[ResolvedScene] = []
    exclusion_reasons: dict[str, int] = {}

    pair_ids = sorted(pairings.keys())
    if scene_ids:
        pair_ids = [sid for sid in pair_ids if sid in set(scene_ids)]

    for ls_id in pair_ids:
        year = None
        if ls_id in manifest_rows:
            row = manifest_rows[ls_id]
            year = int(row["year"]) if row["year"] is not None else None
        resolved = _resolve_scene(
            ls_id=ls_id,
            year=year,
            pairings=pairings,
            manifest_rows=manifest_rows,
            ard_rows=ard_rows,
            derived_rows=derived_rows,
            source_rows=source_rows,
            dynamic_rows=dynamic_rows,
            mapping=mapping,
            ard_root=ard_root,
            dynamic_root=dynamic_root,
        )
        if resolved.exclusion_reason:
            exclusion_reasons[resolved.exclusion_reason] = (
                exclusion_reasons.get(resolved.exclusion_reason, 0) + 1
            )
        scenes.append(resolved)

    total = len(pairings) if not scene_ids else len(pair_ids)
    assessed = sum(1 for s in scenes if s.assessable)
    excluded = len(scenes) - assessed

    return InventoryReport(
        scenes=scenes,
        total_pairings=total,
        assessed=assessed,
        excluded=excluded,
        exclusion_reasons=exclusion_reasons,
        static_sources=static_sources,
        fingerprints=fingerprints,
        errors=errors,
    )


def _resolve_scene(
    *,
    ls_id: str,
    year: int | None,
    pairings: dict[str, dict],
    manifest_rows: dict[str, dict],
    ard_rows: dict[tuple[str, str], dict],
    derived_rows: dict[tuple[str, str], dict],
    source_rows: dict[str, dict],
    dynamic_rows: dict[tuple[str, str], dict],
    mapping,
    ard_root: str,
    dynamic_root: str,
) -> ResolvedScene:
    """Resolve a single pairing; returns the scene (assessable or excluded)."""
    errors: list[str] = []
    exclusion: str | None = None

    s2_id = pairings[ls_id]["sentinel2_scene_id"]

    # 2026 anchors are inference scenes — by design outside the training
    # universe. They are excluded up front regardless of artifact state,
    # so an incomplete 2026 ledger never produces a hard finding.
    if year is not None and year >= 2026:
        return ResolvedScene(
            scene_id=ls_id, year=year, s2_scene_id=s2_id, geometry_id="",
            landsat_cog="", landsat_flag="", s2_cog="", s2_flag="",
            dynamic={}, static_derived={}, static_derived_meta={},
            static_sources={}, exclusion_reason=INFERENCE_EXCLUSION_REASON, errors=errors,
        )

    # ── Landsat ARD row ────────────────────────────────────────────────
    ls_row = ard_rows.get(("landsat-c2-l2", ls_id))
    if ls_row is None or ls_row["status"] != "done":
        return ResolvedScene(
            scene_id=ls_id,
            year=year or 0,
            s2_scene_id=s2_id,
            geometry_id="",
            landsat_cog="",
            landsat_flag="",
            s2_cog="",
            s2_flag="",
            dynamic={},
            static_derived={},
            static_derived_meta={},
            static_sources={},
            exclusion_reason="ard landsat not done",
            errors=errors,
        )
    ls_cog = str(ls_row.get("path_cog") or "")
    ls_flag = str(ls_row.get("path_flag") or "")
    if not ls_cog or not ls_flag:
        errors.append(f"landsat {ls_id}: missing COG/flag path in ARD ledger")

    # ── Sentinel-2 ARD row ─────────────────────────────────────────────
    s2_row = ard_rows.get(("sentinel-2-l2a", s2_id))
    s2_cog = s2_flag = ""
    if s2_row is None or s2_row["status"] != "done":
        exclusion = "ard sentinel2 not done"
    else:
        s2_cog = str(s2_row.get("path_cog") or "")
        s2_flag = str(s2_row.get("path_flag") or "")
        if not s2_cog or not s2_flag:
            errors.append(f"sentinel2 {s2_id}: missing COG/flag path in ARD ledger")

    # ── geometry profile ───────────────────────────────────────────────
    geometry_id = ""
    if year is None or mapping is None:
        if exclusion is None:
            exclusion = "missing geometry mapping"
    else:
        vintage = mapping.year_to_vintage.get(year)
        if vintage is None:
            if exclusion is None:
                exclusion = f"year {year} not in geometry mapping"
        else:
            vdata = mapping.vintages.get(vintage, {})
            geometry_id = str(vdata.get("geometry_id", ""))

    # ── static derived morphology (in-support) ─────────────────────────
    static_derived: dict[str, str] = {}
    static_derived_meta: dict[str, str] = {}
    if geometry_id:
        for product in (*STATIC_DERIVED_MORPHOLOGY_PRODUCTS, *STATIC_DERIVED_OPTIONAL_PRODUCTS):
            row = derived_rows.get((product, geometry_id))
            if row is not None and row["status"] == "done" and row.get("output_uri"):
                static_derived[product] = str(row["output_uri"])
        for product in _METADATA_DERIVED_PRODUCTS:
            row = derived_rows.get((product, geometry_id))
            if row is not None and row["status"] == "done" and row.get("output_uri"):
                static_derived_meta[product] = str(row["output_uri"])

    # ── static source products (feature-stack morphology inputs) ───────
    # Resolved per scene year → vintage for the three semantic predictor
    # source products.  The feature pipeline reads these COGs directly
    # instead of the derived DSM products.
    static_src: dict[str, str] = {}
    if year is not None:
        # LoD2 morphology — vintage from geometry mapping
        if geometry_id and mapping is not None:
            lod_vintage = mapping.year_to_vintage.get(year)
            if lod_vintage is not None:
                item_id = f"lod2_morphology_{lod_vintage}"
                row = source_rows.get(item_id)
                if row is not None and row.get("output_uri"):
                    static_src["lod2_morphology"] = str(row["output_uri"])
        # Vegetation height — fixed 2020 carry-forward
        vh_row = source_rows.get("vegetation_height_2020")
        if vh_row is not None and vh_row.get("output_uri"):
            static_src["vegetation_height"] = str(vh_row["output_uri"])
        # Imperviousness — year-dependent vintage
        imp_vintage = vintage_for_scene_year(year)
        item_id = f"imperviousness_{imp_vintage}"
        imp_row = source_rows.get(item_id)
        if imp_row is not None and imp_row.get("output_uri"):
            static_src["imperviousness"] = str(imp_row["output_uri"])

    # ── dynamic products ───────────────────────────────────────────────
    dynamic: dict[str, str] = {}
    for source in ("era5_land", "shadow_building", "shadow_vegetation"):
        row = dynamic_rows.get((source, ls_id))
        if row is None or row["status"] != "done" or not row.get("output_uri"):
            if exclusion is None:
                exclusion = f"dynamic {source} not done"
            continue
        if row.get("role") == "inference":
            if exclusion is None:
                exclusion = INFERENCE_EXCLUSION_REASON
            continue
        dynamic[source] = str(row["output_uri"])

    return ResolvedScene(
        scene_id=ls_id,
        year=year or 0,
        s2_scene_id=s2_id,
        geometry_id=geometry_id,
        landsat_cog=ls_cog,
        landsat_flag=ls_flag,
        s2_cog=s2_cog,
        s2_flag=s2_flag,
        dynamic=dynamic,
        static_derived=static_derived,
        static_derived_meta=static_derived_meta,
        static_sources=static_src,
        exclusion_reason=exclusion,
        errors=errors,
    )


__all__ = [
    "INFERENCE_EXCLUSION_REASON",
    "InventoryReport",
    "ResolvedScene",
    "build_inventory",
]
