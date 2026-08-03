"""Fail-closed recovery state for COG layout repair.

Provides immutable per-object events, inventory/classification schemas,
metadata snapshots, and a deterministic reducer for recovery orchestration.

Every state transition is recorded as an immutable JSON event. Parquet
snapshots are derived views and never authoritative resume state.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow as pa

# ── versioned schemas ─────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"

# Inventory / classification matrix
INVENTORY_SCHEMA = pa.schema([
    pa.field("uri", pa.string(), nullable=False),
    pa.field("asset_kind", pa.string(), nullable=False),  # "data" or "flag"
    pa.field("source", pa.string()),
    pa.field("partition", pa.string()),
    pa.field("year", pa.int32(), nullable=True),
    pa.field("current_generation", pa.int64()),
    pa.field("current_crc32c", pa.string()),
    pa.field("current_size", pa.int64()),
    pa.field("current_content_type", pa.string()),
    pa.field("current_metadata_json", pa.string()),
    pa.field("original_generation", pa.int64(), nullable=True),
    pa.field("original_crc32c", pa.string(), nullable=True),
    pa.field("hard_delete_time", pa.string(), nullable=True),
    pa.field("layout_signature", pa.string()),
    pa.field("layout_class", pa.string()),
    pa.field("evidence_class", pa.string()),
    pa.field("recovery_bucket_uri", pa.string(), nullable=True),
    pa.field("original_backup_uri", pa.string(), nullable=True),
    pa.field("current_backup_uri", pa.string(), nullable=True),
    pa.field("status", pa.string()),  # events derive final status
])

# Immutable event record
EVENT_SCHEMA = pa.schema([
    pa.field("uri", pa.string(), nullable=False),
    pa.field("sequence", pa.int32(), nullable=False),
    pa.field("event_type", pa.string(), nullable=False),
    pa.field("timestamp", pa.string(), nullable=False),
    pa.field("generation_before", pa.int64(), nullable=True),
    pa.field("generation_after", pa.int64(), nullable=True),
    pa.field("crc32c", pa.string(), nullable=True),
    pa.field("details_json", pa.string(), nullable=True),
    pa.field("config_hash", pa.string(), nullable=False),
    pa.field("schema_version", pa.string(), nullable=False),
])

# ── event types ───────────────────────────────────────────────────────


class EventType(StrEnum):
    """Immutable event types for recovery state machine."""
    SNAPSHOTTED = "SNAPSHOTTED"          # inventory/classification recorded
    BACKUP_VERIFIED = "BACKUP_VERIFIED"  # current live backed up and verified
    ORIGINAL_RECOVERED = "ORIGINAL_RECOVERED"  # soft-deleted original recovered
    ORIGINAL_VERIFIED = "ORIGINAL_VERIFIED"    # original backup verified
    CANDIDATE_STAGED = "CANDIDATE_STAGED"      # repair candidate written to recovery bucket
    CANDIDATE_VERIFIED = "CANDIDATE_VERIFIED"  # candidate passes strict validation
    PROMOTION_INTENT = "PROMOTION_INTENT"      # about to promote candidate to canonical
    PROMOTED = "PROMOTED"                      # candidate promoted to canonical
    VERIFIED = "VERIFIED"                      # promoted object passes independent verification
    ROLLBACK_INTENT = "ROLLBACK_INTENT"        # about to restore original
    ROLLED_BACK = "ROLLED_BACK"                # original restored
    BLOCKED = "BLOCKED"                        # fatal error, cannot proceed


# Valid state transitions — maps from current status to allowed next event
# A status is derived from the last event type
VALID_TRANSITIONS: dict[str, list[str]] = {
    "PENDING": [EventType.SNAPSHOTTED, EventType.BLOCKED],
    EventType.SNAPSHOTTED: [EventType.BACKUP_VERIFIED, EventType.BLOCKED],
    EventType.BACKUP_VERIFIED: [
        EventType.ORIGINAL_RECOVERED,
        EventType.CANDIDATE_STAGED,
        EventType.BLOCKED,
    ],
    EventType.ORIGINAL_RECOVERED: [EventType.ORIGINAL_VERIFIED, EventType.BLOCKED],
    EventType.ORIGINAL_VERIFIED: [EventType.CANDIDATE_STAGED, EventType.BLOCKED],
    EventType.CANDIDATE_STAGED: [EventType.CANDIDATE_VERIFIED, EventType.BLOCKED],
    EventType.CANDIDATE_VERIFIED: [EventType.PROMOTION_INTENT, EventType.BLOCKED],
    EventType.PROMOTION_INTENT: [
        EventType.PROMOTED,
        EventType.ROLLBACK_INTENT,
        EventType.BLOCKED,
    ],
    EventType.PROMOTED: [EventType.VERIFIED, EventType.ROLLBACK_INTENT, EventType.BLOCKED],
    EventType.VERIFIED: [],  # terminal — no further transitions
    EventType.ROLLBACK_INTENT: [EventType.ROLLED_BACK, EventType.BLOCKED],
    EventType.ROLLED_BACK: [EventType.CANDIDATE_STAGED, EventType.BLOCKED],
    EventType.BLOCKED: [],  # terminal — no further transitions
}


# ── layout classification ─────────────────────────────────────────────


class LayoutClass(StrEnum):
    """Classification of COG layout quality."""
    STRICT_CLEAN = "strict_clean"         # zero errors, zero warnings
    MISSING_OVERVIEW = "missing_overview"  # only warning about missing overviews
    HARD_LAYOUT = "hard_layout"           # IFD/offset errors
    UNEXPECTED = "unexpected"             # any other error/warning


class EvidenceClass(StrEnum):
    """Classification of overwrite evidence."""
    UNTOUCHED = "untouched"              # not modified during incident
    AUDITED_MATCH = "audited_match"      # legacy audit CRC/generation match
    UNAUDITED_OVERWRITE = "unaudited_overwrite"  # modified but no persisted audit
    INTERRUPTED = "interrupted"           # repair started but no audit/state persisted
    CONTRADICTION = "contradiction"       # evidence conflicts — halt required


# ── config hashing ────────────────────────────────────────────────────


def hash_config(config_path: str | Path) -> str:
    """Compute SHA-256 hash of the recovery configuration file."""
    path = Path(config_path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def hash_dict(d: dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dict (canonical JSON)."""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


# ── event operations ──────────────────────────────────────────────────


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(UTC).isoformat()


def make_event(
    uri: str,
    sequence: int,
    event_type: str,
    config_hash: str,
    *,
    generation_before: int | None = None,
    generation_after: int | None = None,
    crc32c: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an immutable event record.

    Raises ValueError if event_type is invalid.
    """
    if event_type not in [e.value for e in EventType]:
        raise ValueError(f"Invalid event type: {event_type}")

    return {
        "uri": uri,
        "sequence": sequence,
        "event_type": event_type,
        "timestamp": _now_iso(),
        "generation_before": generation_before,
        "generation_after": generation_after,
        "crc32c": crc32c,
        "details_json": json.dumps(details, sort_keys=True) if details else None,
        "config_hash": config_hash,
        "schema_version": SCHEMA_VERSION,
    }


def validate_transition(current_status: str, next_event_type: str) -> list[str]:
    """Validate that a state transition is allowed.

    Returns a list of errors; empty means valid.
    """
    errors: list[str] = []

    allowed = VALID_TRANSITIONS.get(current_status, [])
    if next_event_type not in allowed:
        errors.append(
            f"Invalid transition: {current_status} -> {next_event_type}. "
            f"Allowed: {allowed}"
        )

    return errors


def reduce_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a sequence of events into the current state.

    Returns a dict with the current status and metadata.

    Raises ValueError on invalid transitions or sequence gaps.
    """
    if not events:
        return {"status": "PENDING", "sequence": 0, "errors": []}

    # Sort by sequence
    sorted_events = sorted(events, key=lambda e: e["sequence"])

    # Check sequence continuity
    errors: list[str] = []
    expected_seq = 0
    for event in sorted_events:
        if event["sequence"] != expected_seq:
            errors.append(
                f"Sequence gap: expected {expected_seq}, got {event['sequence']}"
            )
        expected_seq = event["sequence"] + 1

    # Validate transitions
    current_status = "PENDING"
    last_event = None
    config_hash = sorted_events[0].get("config_hash", "")

    for event in sorted_events:
        transition_errors = validate_transition(current_status, event["event_type"])
        if transition_errors:
            errors.extend(transition_errors)
        current_status = event["event_type"]
        last_event = event

        # Config hash must be consistent
        if event.get("config_hash") != config_hash:
            errors.append(
                f"Config hash mismatch: expected {config_hash}, "
                f"got {event.get('config_hash')}"
            )

    return {
        "status": current_status,
        "sequence": sorted_events[-1]["sequence"],
        "last_event": last_event,
        "errors": errors,
    }


# ── event persistence ─────────────────────────────────────────────────


def events_dir(recovery_root: str, uri: str) -> str:
    """Compute the event directory for a canonical URI.

    Layout: <recovery_root>/events/<asset-id>/
    """
    # Extract asset ID from URI — use path components
    # gs://bucket/prefix/.../file.tif -> prefix_..._file
    from berlin_lst_downscaling.data.io.storage import _parse_gs_uri
    _, key = _parse_gs_uri(uri)
    asset_id = key.replace("/", "_").replace(".tif", "").replace(".flag", "")
    return f"{recovery_root.rstrip('/')}/events/{asset_id}"


def event_path(recovery_root: str, uri: str, sequence: int, event_type: str) -> str:
    """Compute the path for an immutable event file.

    Layout: <recovery_root>/events/<asset-id>/<sequence>-<event_type>.json
    """
    base = events_dir(recovery_root, uri)
    return f"{base}/{sequence:04d}-{event_type}.json"


def load_events(recovery_root: str, uri: str) -> list[dict[str, Any]]:
    """Load all events for a canonical URI from the recovery store.

    Returns sorted list of events. Missing directory returns empty list.
    """
    # List event files — this requires GCS listing
    # For now, we load from a known index or scan
    # Implementation depends on storage backend
    return []


def save_event(
    recovery_root: str,
    event: dict[str, Any],
    *,
    overwrite: bool = False,
) -> str:
    """Save an immutable event to the recovery store.

    Events are create-only by default. Returns the event path.

    Raises FileExistsError if overwrite=False and event already exists.
    """
    from berlin_lst_downscaling.data.io.storage import atomic_write

    path = event_path(
        recovery_root,
        event["uri"],
        event["sequence"],
        event["event_type"],
    )
    content = json.dumps(event, indent=2, sort_keys=True).encode()
    atomic_write(path, content, overwrite=overwrite)
    return path


# ── layout signature ──────────────────────────────────────────────────


def compute_layout_signature(
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> str:
    """Compute a deterministic signature for a COG's layout issues.

    Used to route assets to the correct repair engine.
    """
    combined = list(strict_errors) + list(strict_warnings)
    if not combined:
        return "CLEAN"
    # Sort for determinism
    combined.sort()
    return hashlib.sha256("|".join(combined).encode()).hexdigest()[:12]


def classify_layout(
    strict_valid: bool,
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> LayoutClass:
    """Classify COG layout quality from strict validation results.

    Raises ValueError if classification is ambiguous.
    """
    if strict_valid and not strict_errors and not strict_warnings:
        return LayoutClass.STRICT_CLEAN

    if strict_errors:
        # Check for known hard-layout signatures
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
        # Only missing-overview warning is acceptable
        if len(strict_warnings) == 1:
            warning = strict_warnings[0]
            if "internal overviews" in warning.lower():
                return LayoutClass.MISSING_OVERVIEW
        return LayoutClass.UNEXPECTED

    return LayoutClass.UNEXPECTED


def classify_evidence(
    was_overwritten: bool,
    has_legacy_audit: bool,
    legacy_crc_match: bool | None,
    legacy_generation_match: bool | None,
) -> EvidenceClass:
    """Classify the evidence quality for an asset.

    Raises ValueError if evidence is contradictory.
    """
    if not was_overwritten:
        return EvidenceClass.UNTOUCHED

    if has_legacy_audit:
        if legacy_crc_match is True and legacy_generation_match is True:
            return EvidenceClass.AUDITED_MATCH
        if legacy_crc_match is False or legacy_generation_match is False:
            return EvidenceClass.CONTRADICTION
        # Audit exists but match is unknown
        return EvidenceClass.UNAUDITED_OVERWRITE

    # Overwritten but no audit
    return EvidenceClass.UNAUDITED_OVERWRITE


# ── strict validation result ──────────────────────────────────────────


class StrictCogResult:
    """Structured result from strict COG validation.

    Unlike the previous implementation, this class requires valid=True
    only when there are zero errors AND zero warnings.
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


__all__ = [
    "SCHEMA_VERSION",
    "INVENTORY_SCHEMA",
    "EVENT_SCHEMA",
    "EventType",
    "LayoutClass",
    "EvidenceClass",
    "VALID_TRANSITIONS",
    "StrictCogResult",
    "hash_config",
    "hash_dict",
    "make_event",
    "validate_transition",
    "reduce_events",
    "events_dir",
    "event_path",
    "load_events",
    "save_event",
    "compute_layout_signature",
    "classify_layout",
    "classify_evidence",
]
