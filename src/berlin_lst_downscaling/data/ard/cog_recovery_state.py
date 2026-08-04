"""Fail-closed recovery state for COG layout repair.

Provides immutable per-object events, inventory/classification schemas,
metadata snapshots, and a deterministic reducer for recovery orchestration.

Every state transition is recorded as an immutable JSON event stored as a
create-only GCS object.  Parquet snapshots are derived views and never
authoritative resume state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# ── versioned schemas ─────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"

INVENTORY_SCHEMA_FIELDS: list[str] = [
    "uri",
    "asset_kind",
    "source",
    "partition",
    "year",
    "current_generation",
    "current_metageneration",
    "current_crc32c",
    "current_size",
    "current_content_type",
    "current_metadata_json",
    "current_descriptor_json",
    "original_generation",
    "original_metageneration",
    "original_crc32c",
    "original_size",
    "original_descriptor_json",
    "hard_delete_time",
    "restore_token",
    "layout_signature",
    "layout_class",
    "evidence_class",
    "recovery_bucket_uri",
    "original_backup_uri",
    "current_backup_uri",
    "candidate_uri",
    "status",
]

# ── event types ───────────────────────────────────────────────────────


class EventType(StrEnum):
    """Immutable event types for recovery state machine."""
    SNAPSHOTTED = "SNAPSHOTTED"
    BACKUP_VERIFIED = "BACKUP_VERIFIED"
    ORIGINAL_CAPTURE_INTENT = "ORIGINAL_CAPTURE_INTENT"
    ORIGINAL_RESTORED = "ORIGINAL_RESTORED"
    ORIGINAL_COPIED = "ORIGINAL_COPIED"
    BASELINE_RECONSTRUCTED = "BASELINE_RECONSTRUCTED"
    ORIGINAL_VERIFIED = "ORIGINAL_VERIFIED"
    CANDIDATE_STAGED = "CANDIDATE_STAGED"
    CANDIDATE_VERIFIED = "CANDIDATE_VERIFIED"
    PROMOTION_INTENT = "PROMOTION_INTENT"
    PROMOTED = "PROMOTED"
    CANONICAL_VERIFIED = "CANONICAL_VERIFIED"
    ROLLBACK_INTENT = "ROLLBACK_INTENT"
    ROLLED_BACK = "ROLLED_BACK"
    BLOCKED = "BLOCKED"


class OperationType(StrEnum):
    """High-level recovery operations."""
    SNAPSHOT = "snapshot"
    BACKUP = "backup"
    ORIGINAL_CAPTURE = "original_capture"
    CANDIDATE_STAGING = "candidate_staging"
    CANDIDATE_VERIFICATION = "candidate_verification"
    PROMOTION = "promotion"
    ROLLBACK = "rollback"
    FINAL_VERIFICATION = "final_verification"


# ── valid state transitions ───────────────────────────────────────────

VALID_TRANSITIONS: dict[str, list[str]] = {
    "PENDING": [EventType.SNAPSHOTTED, EventType.BLOCKED],
    EventType.SNAPSHOTTED: [EventType.BACKUP_VERIFIED, EventType.BLOCKED],
    EventType.BACKUP_VERIFIED: [
        EventType.ORIGINAL_CAPTURE_INTENT,
        EventType.CANDIDATE_STAGED,
        EventType.BLOCKED,
    ],
    EventType.ORIGINAL_CAPTURE_INTENT: [EventType.ORIGINAL_RESTORED, EventType.BLOCKED],
    EventType.ORIGINAL_RESTORED: [EventType.ORIGINAL_COPIED, EventType.BLOCKED],
    EventType.ORIGINAL_COPIED: [EventType.BASELINE_RECONSTRUCTED, EventType.BLOCKED],
    EventType.BASELINE_RECONSTRUCTED: [EventType.ORIGINAL_VERIFIED, EventType.BLOCKED],
    EventType.ORIGINAL_VERIFIED: [EventType.CANDIDATE_STAGED, EventType.BLOCKED],
    EventType.CANDIDATE_STAGED: [EventType.CANDIDATE_VERIFIED, EventType.BLOCKED],
    EventType.CANDIDATE_VERIFIED: [EventType.PROMOTION_INTENT, EventType.BLOCKED],
    EventType.PROMOTION_INTENT: [
        EventType.PROMOTED,
        EventType.ROLLBACK_INTENT,
        EventType.BLOCKED,
    ],
    EventType.PROMOTED: [
        EventType.CANONICAL_VERIFIED,
        EventType.ROLLBACK_INTENT,
        EventType.BLOCKED,
    ],
    EventType.CANONICAL_VERIFIED: [],
    EventType.ROLLBACK_INTENT: [EventType.ROLLED_BACK, EventType.BLOCKED],
    EventType.ROLLED_BACK: [EventType.CANDIDATE_STAGED, EventType.BLOCKED],
    EventType.BLOCKED: [],
}


# ── utility functions ─────────────────────────────────────────────────


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()


def make_operation_id(prefix: str) -> str:
    """Generate a unique operation identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImmutableEvent:
    """Immutable event record for a recovery state transition."""
    uri: str
    run_id: str
    sequence: int
    event_type: str
    timestamp: str
    operation_id: str = ""
    prev_event_hash: str = ""
    generation_before: int | None = None
    generation_after: int | None = None
    metageneration_before: int | None = None
    metageneration_after: int | None = None
    crc32c: str | None = None
    descriptor_json: str | None = None
    details_json: str | None = None
    config_hash: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "operation_id": self.operation_id,
            "prev_event_hash": self.prev_event_hash,
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
            "metageneration_before": self.metageneration_before,
            "metageneration_after": self.metageneration_after,
            "crc32c": self.crc32c,
            "descriptor_json": self.descriptor_json,
            "details_json": self.details_json,
            "config_hash": self.config_hash,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImmutableEvent:
        details = d.get("details_json")
        if isinstance(details, dict):
            details = json.dumps(details, sort_keys=True)
        return cls(
            uri=d["uri"],
            run_id=d["run_id"],
            sequence=d["sequence"],
            event_type=d["event_type"],
            timestamp=d["timestamp"],
            operation_id=d.get("operation_id", ""),
            prev_event_hash=d.get("prev_event_hash", ""),
            generation_before=d.get("generation_before"),
            generation_after=d.get("generation_after"),
            metageneration_before=d.get("metageneration_before"),
            metageneration_after=d.get("metageneration_after"),
            crc32c=d.get("crc32c"),
            descriptor_json=d.get("descriptor_json"),
            details_json=details,
            config_hash=d.get("config_hash", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class ObjectDescriptor:
    """Captures the full mutable GCS object metadata contract."""
    generation: int = 0
    metageneration: int = 0
    size: int = 0
    crc32c: str = ""
    content_type: str = ""
    content_encoding: str | None = None
    cache_control: str | None = None
    custom_metadata: dict[str, str] = field(default_factory=dict)
    storage_class: str = ""
    kms_key_name: str | None = None
    md5_hash: str | None = None
    updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "metageneration": self.metageneration,
            "size": self.size,
            "crc32c": self.crc32c,
            "content_type": self.content_type,
            "content_encoding": self.content_encoding,
            "cache_control": self.cache_control,
            "custom_metadata": self.custom_metadata,
            "storage_class": self.storage_class,
            "kms_key_name": self.kms_key_name,
            "md5_hash": self.md5_hash,
            "updated": self.updated,
        }

    def to_json(self) -> str:
        """Serialize to canonical JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObjectDescriptor:
        return cls(
            generation=d.get("generation", 0),
            metageneration=d.get("metageneration", 0),
            size=d.get("size", 0),
            crc32c=d.get("crc32c", ""),
            content_type=d.get("content_type", ""),
            content_encoding=d.get("content_encoding"),
            cache_control=d.get("cache_control"),
            custom_metadata=d.get("custom_metadata", {}),
            storage_class=d.get("storage_class", ""),
            kms_key_name=d.get("kms_key_name"),
            md5_hash=d.get("md5_hash"),
            updated=d.get("updated"),
        )


# ── config hashing ────────────────────────────────────────────────────


def hash_config(config_path: str | Path) -> str:
    """Compute SHA-256 hash of a configuration file (first 16 hex chars)."""
    path = Path(config_path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


# ── event path helpers ────────────────────────────────────────────────


def event_dir_for_uri(recovery_root: str, uri: str) -> str:
    """Compute the event directory for a canonical URI.

    Layout: ``<recovery_root>/events/<sha256(uri)>/``
    """
    digest = hashlib.sha256(uri.encode()).hexdigest()
    return f"{recovery_root.rstrip('/')}/events/{digest}"


def event_path_for(
    recovery_root: str, uri: str, sequence: int, event_type: str,
) -> str:
    """Compute the path for an immutable event file.

    Layout: ``<recovery_root>/events/<sha256(uri)>/<sequence>-<event_type>.json``
    """
    base = event_dir_for_uri(recovery_root, uri)
    return f"{base}/{sequence:04d}-{event_type}.json"


# ── event construction ────────────────────────────────────────────────


def make_event(
    uri: str,
    run_id: str,
    sequence: int,
    event_type: str,
    config_hash: str,
    *,
    operation_id: str = "",
    prev_event_hash: str = "",
    generation_before: int | None = None,
    generation_after: int | None = None,
    metageneration_before: int | None = None,
    metageneration_after: int | None = None,
    crc32c: str | None = None,
    descriptor: ObjectDescriptor | None = None,
    details: dict[str, Any] | None = None,
) -> ImmutableEvent:
    """Create an immutable event record.

    Raises ``ValueError`` if *event_type* is not a known ``EventType``.
    """
    if event_type not in [e.value for e in EventType]:
        raise ValueError(f"Invalid event type: {event_type}")

    desc_json = descriptor.to_json() if descriptor is not None else None
    det_json = json.dumps(details, sort_keys=True) if details else None

    return ImmutableEvent(
        uri=uri,
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=_now_iso(),
        operation_id=operation_id,
        prev_event_hash=prev_event_hash,
        generation_before=generation_before,
        generation_after=generation_after,
        metageneration_before=metageneration_before,
        metageneration_after=metageneration_after,
        crc32c=crc32c,
        descriptor_json=desc_json,
        details_json=det_json,
        config_hash=config_hash,
    )


def event_content_hash(event: ImmutableEvent) -> str:
    """Deterministic content hash of a committed event."""
    canonical = json.dumps(event.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


# ── transition validation ─────────────────────────────────────────────


def validate_transition(current_status: str, next_event_type: str) -> list[str]:
    """Validate that a state transition is allowed.

    Returns a list of errors; empty means valid.
    """
    allowed = VALID_TRANSITIONS.get(current_status, [])
    if next_event_type not in allowed:
        return [
            f"Invalid transition: {current_status} -> {next_event_type}. "
            f"Allowed: {allowed}",
        ]
    return []


# ── event reducer ─────────────────────────────────────────────────────


def reduce_events(events: list[ImmutableEvent]) -> dict[str, Any]:
    """Reduce a sequence of events into the current state.

    Validates sequence continuity, transition legality, config/run
    consistency, operation integrity, and contradiction detection.

    Returns a dict with: ``status``, ``sequence``, ``last_event``,
    ``errors``, ``in_flight``, ``terminal``.
    """
    if not events:
        return {
            "status": "PENDING",
            "sequence": -1,
            "last_event": None,
            "errors": [],
            "in_flight": None,
            "terminal": False,
        }

    sorted_events = sorted(events, key=lambda e: e.sequence)
    errors: list[str] = []

    # ── sequence continuity and duplicates ────────────────────────────
    seen_sequences: dict[int, ImmutableEvent] = {}
    for event in sorted_events:
        if event.sequence in seen_sequences:
            errors.append(
                f"Duplicate sequence {event.sequence}: "
                f"existing={seen_sequences[event.sequence].event_type} "
                f"conflict={event.event_type}"
            )
        seen_sequences[event.sequence] = event

    # ── sequence gap detection ────────────────────────────────────────
    expected = 0
    for event in sorted_events:
        if event.sequence != expected:
            errors.append(
                f"Sequence gap: expected {expected}, got {event.sequence}"
            )
        expected = event.sequence + 1

    # ── config and run consistency ────────────────────────────────────
    first = sorted_events[0]
    base_config = first.config_hash
    base_run = first.run_id

    for event in sorted_events:
        if event.config_hash != base_config:
            errors.append(
                f"Config hash mismatch at seq {event.sequence}: "
                f"expected {base_config}, got {event.config_hash}"
            )
        if event.run_id != base_run:
            errors.append(
                f"Run ID mismatch at seq {event.sequence}: "
                f"expected {base_run}, got {event.run_id}"
            )

    # ── prev_event_hash chain ─────────────────────────────────────────
    for i, event in enumerate(sorted_events):
        if i == 0:
            if event.prev_event_hash:
                errors.append(
                    f"First event (seq {event.sequence}) has non-empty "
                    f"prev_event_hash"
                )
        else:
            expected_hash = event_content_hash(sorted_events[i - 1])
            if event.prev_event_hash != expected_hash:
                errors.append(
                    f"prev_event_hash mismatch at seq {event.sequence}: "
                    f"expected {expected_hash}, got {event.prev_event_hash}"
                )

    # ── transition validation ─────────────────────────────────────────
    current_status: str = "PENDING"
    last_event: ImmutableEvent | None = None
    in_flight: dict[str, Any] | None = None

    for event in sorted_events:
        transition_errors = validate_transition(current_status, event.event_type)
        for te in transition_errors:
            errors.append(f"seq {event.sequence}: {te}")

        current_status = event.event_type
        last_event = event

        # track in-flight operations
        op = event.operation_id
        if event.event_type in (
            EventType.ORIGINAL_CAPTURE_INTENT,
            EventType.PROMOTION_INTENT,
            EventType.ROLLBACK_INTENT,
        ):
            in_flight = {
                "operation_id": op,
                "event_type": event.event_type,
                "sequence": event.sequence,
            }
        elif event.event_type not in (EventType.BLOCKED,) and in_flight:
            if in_flight["operation_id"] == op:
                in_flight = None

    # ── contradiction detection ───────────────────────────────────────
    for event in sorted_events:
        if event.event_type == EventType.BLOCKED:
            errors.append(
                f"Blocked event at seq {event.sequence}: "
                f"{event.details_json or 'no details'}"
            )

    terminal = current_status in (EventType.CANONICAL_VERIFIED, EventType.BLOCKED)

    return {
        "status": current_status,
        "sequence": sorted_events[-1].sequence,
        "last_event": last_event,
        "errors": errors,
        "in_flight": in_flight,
        "terminal": terminal,
    }


# ── layout classification ─────────────────────────────────────────────


def compute_layout_signature(
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> str:
    """Compute a deterministic signature for a COG's layout issues."""
    combined = sorted(list(strict_errors) + list(strict_warnings))
    if not combined:
        return "CLEAN"
    return hashlib.sha256("|".join(combined).encode()).hexdigest()[:12]


class LayoutClass(StrEnum):
    STRICT_CLEAN = "strict_clean"
    MISSING_OVERVIEW = "missing_overview"
    HARD_LAYOUT = "hard_layout"
    UNEXPECTED = "unexpected"


def classify_layout(
    strict_valid: bool,
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> LayoutClass:
    """Classify COG layout quality from strict validation results."""
    if strict_valid and not strict_errors and not strict_warnings:
        return LayoutClass.STRICT_CLEAN

    if strict_errors:
        known_hard_patterns = [
            "The offset of the main IFD should be < 300",
            "The offset of the first block of overview of index",
            "The offset of the first block of the main resolution image",
            "The file is greater than 512xH or 512xW, but is not tiled",
        ]
        for error in strict_errors:
            for pattern in known_hard_patterns:
                if pattern in error:
                    return LayoutClass.HARD_LAYOUT
        return LayoutClass.UNEXPECTED

    if strict_warnings:
        if len(strict_warnings) == 1:
            warning = strict_warnings[0]
            if "internal overviews" in warning.lower():
                return LayoutClass.MISSING_OVERVIEW
        return LayoutClass.UNEXPECTED

    return LayoutClass.UNEXPECTED


# ── evidence classification ───────────────────────────────────────────


class EvidenceClass(StrEnum):
    UNTOUCHED = "untouched"
    AUDITED_MATCH = "audited_match"
    UNAUDITED_OVERWRITE = "unaudited_overwrite"
    CONTRADICTION = "contradiction"


def classify_evidence(
    was_overwritten: bool,
    has_legacy_audit: bool,
    legacy_crc_match: bool | None,
    legacy_generation_match: bool | None,
) -> EvidenceClass:
    """Classify the evidence quality for an asset.

    Returns ``EvidenceClass.CONTRADICTION`` on contradictory evidence.
    """
    if not was_overwritten:
        return EvidenceClass.UNTOUCHED

    if has_legacy_audit:
        if legacy_crc_match is True and legacy_generation_match is True:
            return EvidenceClass.AUDITED_MATCH
        if legacy_crc_match is False or legacy_generation_match is False:
            return EvidenceClass.CONTRADICTION
        return EvidenceClass.UNAUDITED_OVERWRITE

    return EvidenceClass.UNAUDITED_OVERWRITE


# ── strict validation result ──────────────────────────────────────────


class StrictCogResult:
    """Structured result from strict COG validation.

    ``valid=True`` only when there are zero errors AND zero warnings.
    """

    def __init__(
        self,
        valid: bool,
        errors: tuple[str, ...],
        warnings: tuple[str, ...],
        source: str = "",
    ):
        self.valid = valid and not errors and not warnings
        self.errors = errors
        self.warnings = warnings
        self.source = source
        self.layout_signature = compute_layout_signature(errors, warnings)
        self.layout_class = classify_layout(valid, errors, warnings)

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (
            f"StrictCogResult(valid={self.valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
            f"class={self.layout_class})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "source": self.source,
            "layout_signature": self.layout_signature,
            "layout_class": self.layout_class,
        }


# ── shared validation ─────────────────────────────────────────────────


def validate_strict_cog(uri: str) -> StrictCogResult:
    """Validate COG strict layout using rio-cogeo.

    Returns a ``StrictCogResult`` with ``valid=True`` ONLY when there are
    zero errors AND zero warnings.
    """
    from rio_cogeo.cogeo import cog_validate

    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    try:
        valid, errors, warnings = cog_validate(gdal_uri(uri), strict=True, quiet=True)
        return StrictCogResult(
            valid=valid,
            errors=tuple(f"COG strict: {e}" for e in errors),
            warnings=tuple(f"COG strict warning: {w}" for w in warnings),
            source=uri,
        )
    except FileNotFoundError:
        return StrictCogResult(
            valid=False,
            errors=("rio-cogeo not found in PATH",),
            warnings=(),
            source=uri,
        )
    except Exception as exc:
        return StrictCogResult(
            valid=False,
            errors=(f"COG strict validation failed: {exc}",),
            warnings=(),
            source=uri,
        )


def assert_raster_equivalent(
    source_uri: str,
    repaired_path: str | Path,
    *,
    layout_changed: bool = True,
) -> list[str]:
    """Compare source and repaired rasters for semantic equivalence.

    Requires identical base pixels and overview values, plus identical
    semantic metadata.  Layout fields are excluded when
    ``layout_changed=True``.

    Returns a list of errors; empty means equivalent.
    """
    import math

    import numpy as np
    import rasterio

    errors: list[str] = []

    try:
        with (
            rasterio.open(source_uri) as src,
            rasterio.open(str(repaired_path)) as rep,
        ):
            if src.width != rep.width or src.height != rep.height:
                errors.append(
                    f"Shape mismatch: source ({src.width}, {src.height}) "
                    f"vs repaired ({rep.width}, {rep.height})"
                )
                return errors

            if src.count != rep.count:
                errors.append(f"Band count mismatch: {src.count} vs {rep.count}")
                return errors

            src_crs = str(src.crs).upper() if src.crs else "None"
            rep_crs = str(rep.crs).upper() if rep.crs else "None"
            if src_crs != rep_crs:
                errors.append(f"CRS mismatch: {src_crs} vs {rep_crs}")

            if abs(src.transform.a - rep.transform.a) > 0.001:
                errors.append(
                    f"Resolution mismatch: {abs(src.transform.a)} vs "
                    f"{abs(rep.transform.a)}"
                )
            if abs(src.transform.c - rep.transform.c) > 0.01:
                errors.append(
                    f"X origin mismatch: {src.transform.c} vs {rep.transform.c}"
                )
            if abs(src.transform.f - rep.transform.f) > 0.01:
                errors.append(
                    f"Y origin mismatch: {src.transform.f} vs {rep.transform.f}"
                )

            # NoData
            if src.nodata is None and rep.nodata is not None:
                errors.append(f"NoData mismatch: None vs {rep.nodata}")
            elif src.nodata is not None and rep.nodata is None:
                errors.append(f"NoData mismatch: {src.nodata} vs None")
            elif src.nodata is not None and rep.nodata is not None:
                src_nan = isinstance(src.nodata, float) and math.isnan(src.nodata)
                rep_nan = isinstance(rep.nodata, float) and math.isnan(rep.nodata)
                if src_nan != rep_nan:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")
                elif not src_nan and src.nodata != rep.nodata:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")

            # Per-band metadata
            for i in range(1, src.count + 1):
                src_desc = src.descriptions[i - 1] if src.descriptions else ""
                rep_desc = rep.descriptions[i - 1] if rep.descriptions else ""
                if src_desc != rep_desc:
                    errors.append(
                        f"Band {i} description mismatch: {src_desc!r} vs {rep_desc!r}"
                    )

                if src.scales[i - 1] != rep.scales[i - 1]:
                    errors.append(
                        f"Band {i} scale mismatch: {src.scales[i - 1]} vs "
                        f"{rep.scales[i - 1]}"
                    )
                if src.offsets[i - 1] != rep.offsets[i - 1]:
                    errors.append(
                        f"Band {i} offset mismatch: {src.offsets[i - 1]} vs "
                        f"{rep.offsets[i - 1]}"
                    )

            # Base pixel comparison
            for i in range(1, src.count + 1):
                src_arr = src.read(i)
                rep_arr = rep.read(i)

                src_valid = (
                    ~np.isnan(src_arr)
                    if np.issubdtype(src_arr.dtype, np.floating)
                    else np.ones(src_arr.shape, dtype=bool)
                )
                rep_valid = (
                    ~np.isnan(rep_arr)
                    if np.issubdtype(rep_arr.dtype, np.floating)
                    else np.ones(rep_arr.shape, dtype=bool)
                )

                if not np.array_equal(src_valid, rep_valid):
                    errors.append(f"Band {i} NaN mask mismatch")
                    continue

                if not np.array_equal(src_arr[src_valid], rep_arr[rep_valid]):
                    errors.append(f"Band {i} base pixel values mismatch")
                    continue

            # Overview comparison
            src_overviews = src.overviews(1)
            rep_overviews = rep.overviews(1)
            if src_overviews and rep_overviews:
                for src_ov, rep_ov in zip(
                    src_overviews, rep_overviews, strict=True,
                ):
                    for i in range(1, src.count + 1):
                        src_h = src.height // src_ov
                        src_w = src.width // src_ov
                        rep_h = rep.height // rep_ov
                        rep_w = rep.width // rep_ov

                        src_ov_arr = src.read(i, out_shape=(src_h, src_w))
                        rep_ov_arr = rep.read(i, out_shape=(rep_h, rep_w))

                        if src_ov_arr.shape != rep_ov_arr.shape:
                            errors.append(
                                f"Band {i} overview {src_ov} shape mismatch: "
                                f"{src_ov_arr.shape} vs {rep_ov_arr.shape}"
                            )
                            continue

                        src_valid = (
                            ~np.isnan(src_ov_arr)
                            if np.issubdtype(src_ov_arr.dtype, np.floating)
                            else np.ones(src_ov_arr.shape, dtype=bool)
                        )
                        rep_valid = (
                            ~np.isnan(rep_ov_arr)
                            if np.issubdtype(rep_ov_arr.dtype, np.floating)
                            else np.ones(rep_ov_arr.shape, dtype=bool)
                        )

                        if not np.array_equal(src_valid, rep_valid):
                            errors.append(
                                f"Band {i} overview {src_ov} NaN mask mismatch"
                            )
                            continue

                        if not np.array_equal(
                            src_ov_arr[src_valid], rep_ov_arr[rep_valid],
                        ):
                            errors.append(
                                f"Band {i} overview {src_ov} values mismatch"
                            )

    except Exception as exc:
        errors.append(f"Comparison failed: {exc}")

    return errors


__all__ = [
    "SCHEMA_VERSION",
    "INVENTORY_SCHEMA_FIELDS",
    "EventType",
    "OperationType",
    "VALID_TRANSITIONS",
    "ImmutableEvent",
    "ObjectDescriptor",
    "LayoutClass",
    "EvidenceClass",
    "StrictCogResult",
    "hash_config",
    "make_operation_id",
    "event_dir_for_uri",
    "event_path_for",
    "make_event",
    "event_content_hash",
    "validate_transition",
    "reduce_events",
    "compute_layout_signature",
    "classify_layout",
    "classify_evidence",
    "validate_strict_cog",
    "assert_raster_equivalent",
]
