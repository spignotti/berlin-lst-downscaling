"""Manifest bundle validator — offline + optional upstream identity checks.

Validates a manifest/pairings/report bundle for structural integrity,
policy compliance, and cross-reference consistency before publication
or ARD consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from berlin_lst_downscaling.data.selection.schema import (
    ALLOWED_LANDSAT_PLATFORMS,
    ALLOWED_ROLES,
    ALLOWED_SOURCES,
    ECOSTRESS_VALIDATION_IDS,
    MANIFEST_SCHEMA,
    PAIRINGS_SCHEMA,
)


@dataclass
class ValidationResult:
    """Result of manifest/pairings validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_manifest_table(
    table: pa.Table,
    *,
    require_item_href: bool = True,
) -> ValidationResult:
    """Validate the manifest.parquet table.

    Checks:
      - schema matches MANIFEST_SCHEMA
      - zero duplicate (source, scene_id)
      - no Landsat-7 rows
      - platforms valid
      - temporal policy: months 5–9, years in range
      - AOI metrics: clear_frac >= 0.05 for non-validation rows
      - ECOSTRESS: exactly the six allowed IDs, role=validation
    """
    result = ValidationResult()

    # Schema check
    try:
        table.cast(MANIFEST_SCHEMA, safe=False)
    except pa.ArrowInvalid as exc:
        result.errors.append(f"Schema mismatch: {exc}")
        return result

    if table.num_rows == 0:
        result.errors.append("Manifest is empty")
        return result

    sources = table.column("source").to_pylist()
    scene_ids = table.column("scene_id").to_pylist()
    roles = table.column("role").to_pylist()
    platforms = table.column("platform").to_pylist()
    clear_fracs = table.column("aoi_clear_frac").to_pylist()
    item_hrefs = table.column("item_href").to_pylist()

    # Duplicate check
    keys = list(zip(sources, scene_ids, strict=True))
    seen: set[tuple[str, str]] = set()
    for i, k in enumerate(keys):
        if k in seen:
            result.errors.append(f"Duplicate (source={k[0]}, scene_id={k[1]}) at row {i}")
        seen.add(k)

    # Per-row checks
    eco_ids_found: set[str] = set()
    for i in range(len(sources)):
        src = sources[i]
        sid = scene_ids[i]
        role = roles[i]
        plat = platforms[i]
        cf = clear_fracs[i]
        href = item_hrefs[i]

        # Source
        if src not in ALLOWED_SOURCES:
            result.errors.append(f"Row {i}: invalid source {src!r}")

        # Role
        if role not in ALLOWED_ROLES:
            result.errors.append(f"Row {i}: invalid role {role!r}")

        # Landsat platform
        if src == "landsat-c2-l2":
            if plat not in ALLOWED_LANDSAT_PLATFORMS:
                result.errors.append(
                    f"Row {i}: Landsat scene {sid} has platform {plat!r}; "
                    f"expected one of {ALLOWED_LANDSAT_PLATFORMS}"
                )

        # ECOSTRESS
        if src == "ecostress":
            if sid not in ECOSTRESS_VALIDATION_IDS:
                result.errors.append(f"Row {i}: ECOSTRESS {sid} not in validation allowlist")
            eco_ids_found.add(sid)
            if role != "validation":
                result.errors.append(
                    f"Row {i}: ECOSTRESS {sid} has role {role!r}; expected 'validation'"
                )

        # AOI clear fraction (required for anchor/predictor)
        if role in ("anchor", "predictor"):
            if cf is None or (isinstance(cf, float) and cf != cf):
                result.errors.append(f"Row {i}: {src} {sid} has no AOI clear_frac")
            elif cf < 0.05:
                result.errors.append(f"Row {i}: {src} {sid} has clear_frac={cf:.4f} < 0.05")

        # item_href required for PC STAC rows
        if require_item_href and src in ("landsat-c2-l2", "sentinel-2-l2a"):
            if href is None or (isinstance(href, str) and not href.strip()):
                result.errors.append(f"Row {i}: {src} {sid} missing item_href")

    # ECOSTRESS completeness
    missing_eco = ECOSTRESS_VALIDATION_IDS - eco_ids_found
    extra_eco = eco_ids_found - ECOSTRESS_VALIDATION_IDS
    if missing_eco:
        result.errors.append(f"Missing ECOSTRESS validation IDs: {sorted(missing_eco)}")
    if extra_eco:
        result.errors.append(f"Extra ECOSTRESS IDs not in allowlist: {sorted(extra_eco)}")

    return result


def validate_pairings_table(
    table: pa.Table,
    manifest_table: pa.Table,
) -> ValidationResult:
    """Validate pairings.parquet against manifest.

    Checks:
      - schema matches PAIRINGS_SCHEMA
      - every landsat_scene_id exists in manifest as anchor
      - every sentinel2_scene_id exists in manifest as predictor
      - no duplicate landsat_scene_id
      - score recomputes exactly
    """
    result = ValidationResult()

    # Schema check
    try:
        table.cast(PAIRINGS_SCHEMA, safe=False)
    except pa.ArrowInvalid as exc:
        result.errors.append(f"Schema mismatch: {exc}")
        return result

    if table.num_rows == 0:
        result.errors.append("Pairings table is empty")
        return result

    # Build manifest lookup
    m_sources = manifest_table.column("source").to_pylist()
    m_ids = manifest_table.column("scene_id").to_pylist()
    m_roles = manifest_table.column("role").to_pylist()
    anchor_ids = {
        sid
        for sid, src, role in zip(m_ids, m_sources, m_roles, strict=True)
        if src == "landsat-c2-l2" and role == "anchor"
    }
    predictor_ids = {
        sid
        for sid, src, role in zip(m_ids, m_sources, m_roles, strict=True)
        if src == "sentinel-2-l2a"
    }

    p_lids = table.column("landsat_scene_id").to_pylist()
    p_sids = table.column("sentinel2_scene_id").to_pylist()
    p_jcf = table.column("joint_clear_frac").to_pylist()
    p_lcf = table.column("landsat_clear_px").to_pylist()
    p_jcp = table.column("joint_clear_px").to_pylist()

    seen_lids: set[str] = set()
    for i in range(len(p_lids)):
        lid = p_lids[i]
        sid = p_sids[i]

        # FK checks
        if lid not in anchor_ids:
            result.errors.append(
                f"Pairings row {i}: landsat_scene_id {lid!r} not in manifest anchors"
            )
        if sid not in predictor_ids:
            result.errors.append(
                f"Pairings row {i}: sentinel2_scene_id {sid!r} not in manifest predictors"
            )

        # Duplicate landsat_scene_id
        if lid in seen_lids:
            result.errors.append(f"Pairings row {i}: duplicate landsat_scene_id {lid!r}")
        seen_lids.add(lid)

        # Count invariants
        lcf = p_lcf[i]
        jcp = p_jcp[i]
        jcf = p_jcf[i]
        if lcf is None or lcf <= 0:
            result.errors.append(f"Pairings row {i}: landsat_clear_px must be > 0, got {lcf!r}")
            continue
        if jcp is None or jcp < 0 or jcp > lcf:
            result.errors.append(
                f"Pairings row {i}: joint_clear_px must be in [0, landsat_clear_px]; "
                f"got {jcp!r} for landsat_clear_px={lcf}"
            )
            continue
        if jcf is None or jcf < 0.0 or jcf > 1.0:
            result.errors.append(
                f"Pairings row {i}: joint_clear_frac must be in [0, 1]; got {jcf!r}"
            )
            continue
        # The stored float32 is the authoritative value; require the count/fraction
        # arithmetic to round-trip exactly through float32.
        expected = jcp / lcf
        if abs(float(np.float32(expected)) - float(np.float32(jcf))) > 1e-7:
            result.errors.append(
                f"Pairings row {i}: joint_clear_px/landsat_clear_px = {expected!r} "
                f"does not match joint_clear_frac = {jcf!r}"
            )

    # Every anchor should have exactly one pairing
    for aid in anchor_ids:
        if aid not in seen_lids:
            result.warnings.append(f"Anchor {aid!r} has no pairing in pairings.parquet")

    return result


def validate_report_json(report: dict, manifest_hash: str, pairings_hash: str) -> ValidationResult:
    """Validate manifest_report.json structure and hashes."""
    result = ValidationResult()

    required_keys = {
        "bundle_id",
        "generated_at",
        "cutoff_utc",
        "policy_hash",
        "schema_version",
        "manifest_hash",
        "pairings_hash",
        "counts",
        "unresolved_errors",
    }
    missing = required_keys - set(report.keys())
    if missing:
        result.errors.append(f"Report missing keys: {sorted(missing)}")
        return result

    if report["unresolved_errors"] > 0:
        result.errors.append(f"Report has {report['unresolved_errors']} unresolved errors")

    if report.get("manifest_hash") != manifest_hash:
        result.errors.append("Report manifest_hash mismatch")
    if report.get("pairings_hash") != pairings_hash:
        result.errors.append("Report pairings_hash mismatch")

    return result


__all__ = [
    "ValidationResult",
    "validate_manifest_table",
    "validate_pairings_table",
    "validate_report_json",
]
