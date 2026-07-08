"""Persistent per-scene ledger backed by PyArrow Parquet.

The ledger tracks the status of every scene processed by the ARD
pipeline.  Idempotency, resume, and QA reporting all depend on it.

Every ``upsert`` immediately persists via ``atomic_write``.  This
ensures crash consistency — there is no batch write at the end of
the pipeline run.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import atomic_write, exists, read_bytes


def pc_equal(a: object, b: object) -> Any:
    """Wrap ``pyarrow.compute.equal`` — avoids pyright stub gaps."""
    import pyarrow.compute as _pc

    return _pc.equal(a, b)  # type: ignore[attr-defined]


def pc_and(a: Any, b: Any) -> Any:
    """Wrap ``pyarrow.compute.and_``."""
    import pyarrow.compute as _pc

    return _pc.and_(a, b)  # type: ignore[attr-defined]


def pc_invert(a: Any) -> Any:
    """Wrap ``pyarrow.compute.invert``."""
    import pyarrow.compute as _pc

    return _pc.invert(a)  # type: ignore[attr-defined]

# ── schema ───────────────────────────────────────────────────────────

_STATUSES = {"pending", "exporting", "done", "failed", "skipped"}

_SCHEMA = pa.schema([
    pa.field("scene_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("year", pa.int32(), nullable=False),
    pa.field("path_cog", pa.string()),
    pa.field("path_stac", pa.string()),
    pa.field("status", pa.string(), nullable=False),
    pa.field("schema_hash", pa.string()),
    pa.field("schema_version", pa.int32()),
    pa.field("attempts", pa.int32()),
    pa.field("last_error", pa.string()),
    pa.field("run_id", pa.string()),
    pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    # AOI metrics (schema v3)
    pa.field("aoi_clear_px", pa.int32()),
    pa.field("aoi_cloudy_px", pa.int32()),
    pa.field("aoi_shadow_px", pa.int32()),
    pa.field("aoi_cirrus_px", pa.int32()),
    pa.field("aoi_saturated_px", pa.int32()),
    pa.field("aoi_fill_px", pa.int32()),
    pa.field("aoi_total_px", pa.int32()),
    pa.field("aoi_clear_frac", pa.float64()),
    # AOI overlap (schema v4): all pixels in COG∩AOI (including fill)
    pa.field("aoi_overlap_px", pa.int32()),
])

# Current schema version
SCHEMA_VERSION = 5

# ── row type ─────────────────────────────────────────────────────────


@dataclass
class LedgerRow:
    """A single row in the ARD processing ledger (schema v4)."""

    scene_id: str
    source: str
    year: int
    path_cog: str | None = None
    path_stac: str | None = None
    status: str = "pending"
    schema_hash: str | None = None
    schema_version: int = SCHEMA_VERSION
    attempts: int = 0
    last_error: str | None = None
    run_id: str | None = None
    updated_at: datetime | None = None
    # AOI metrics (schema v3)
    aoi_clear_px: int | None = None
    aoi_cloudy_px: int | None = None
    aoi_shadow_px: int | None = None
    aoi_cirrus_px: int | None = None
    aoi_saturated_px: int | None = None
    aoi_fill_px: int | None = None
    aoi_total_px: int | None = None
    aoi_clear_frac: float | None = None
    # AOI overlap (schema v4): all pixels in COG∩AOI (including fill)
    aoi_overlap_px: int | None = None

    def __post_init__(self) -> None:
        if self.updated_at is None:
            self.updated_at = datetime.now(UTC)
        if self.status not in _STATUSES:
            raise ValueError(f"Invalid status: {self.status!r}")


# ── ledger ───────────────────────────────────────────────────────────


class Ledger:
    """Read-write Parquet ledger for scene status tracking.

    Usage::

        ledger = Ledger.open("data/ard/ledger.parquet")
        ledger.upsert(LedgerRow(scene_id=..., source=..., ...))
        rows = ledger.scenes_for_source("landsat-c2-l2")
    """

    def __init__(self, path: str, table: pa.Table, schema_version: int = 1) -> None:
        self._path = path
        self._table = table
        self.schema_version = schema_version

    # ── factory ─────────────────────────────────────────────────

    @classmethod
    def open(cls, path: str) -> Ledger:
        """Open an existing Parquet ledger or create a new empty one."""
        if exists(path):
            raw = read_bytes(path)
            if len(raw) > 0:
                table = pq.read_table(io.BytesIO(raw))
            else:
                table = pa.Table.from_pylist([], schema=_SCHEMA)
        else:
            table = pa.Table.from_pylist([], schema=_SCHEMA)
        table = _legacy_schema_fill(table)
        return cls(path, table)

    # ── queries ─────────────────────────────────────────────────

    def scenes_for_source(self, source: str) -> list[LedgerRow]:
        """Return all rows for a given source."""
        if self._table.num_rows == 0:
            return []

        tbl = self._table.filter(pc_equal(self._table.column("source"), source))
        return _rows_from_table(tbl)

    def get(self, scene_id: str, source: str) -> LedgerRow | None:
        """Look up a specific scene row, or ``None``."""
        if self._table.num_rows == 0:
            return None

        mask = pc_and(
            pc_equal(self._table.column("scene_id"), scene_id),
            pc_equal(self._table.column("source"), source),
        )
        tbl = self._table.filter(mask)
        rows = _rows_from_table(tbl)
        return rows[0] if rows else None

    # ── mutations ───────────────────────────────────────────────

    def upsert(self, row: LedgerRow) -> None:
        """Insert or update a row identified by ``scene_id + source``.

        Persists immediately via atomic temp-file write.
        """
        # Auto-increment attempts from existing row (H4 fix)
        existing = self.get(row.scene_id, row.source)
        if existing is not None:
            row.attempts = min(existing.attempts + 1, 999)
        else:
            row.attempts = 1  # first attempt

        new_row = pa.Table.from_pylist(
            [_row_to_dict(row)], schema=_SCHEMA
        )

        if self._table.num_rows == 0:
            self._table = new_row
        else:
            existing_mask = pc_and(
                pc_equal(self._table.column("scene_id"), row.scene_id),
                pc_equal(self._table.column("source"), row.source),
            )
            self._table = pa.concat_tables(
                [self._table.filter(pc_invert(existing_mask)), new_row]
            )

        # Per-transition atomic write (H3 fix)
        self._write_atomic()

    # ── persistence ─────────────────────────────────────────────

    def _write_atomic(self) -> str:
        """Persist ledger via ``atomic_write``.

        The Parquet table is serialised to an in-memory buffer, then
        written atomically to the target URI (local or GCS).
        """
        buf = io.BytesIO()
        pq.write_table(self._table, buf)
        atomic_write(self._path, buf.getvalue(), overwrite=True)
        return self._path

    def write(self) -> str:
        """Persist the ledger to its Parquet path (batch convenience).

        With per-transition atomic writes enabled, this is a no-op
        unless called after manual ``self._table`` modification.
        """
        return self._write_atomic()

    def status_counts(self, source: str | None = None) -> dict[str, int]:
        """Return ``{status: count}``, optionally filtered by source."""
        if self._table.num_rows == 0:
            return {}

        tbl = self._table
        if source:
            tbl = tbl.filter(pc_equal(self._table.column("source"), source))
        if tbl.num_rows == 0:
            return {}

        counts: dict[str, int] = {}
        for s in _STATUSES:
            n = tbl.filter(pc_equal(tbl.column("status"), s)).num_rows
            if n > 0:
                counts[s] = n
        return counts

    # ── accessors ───────────────────────────────────────────────

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def path(self) -> str:
        return self._path


# ── helpers ──────────────────────────────────────────────────────────


def _row_to_dict(row: LedgerRow) -> dict:
    return {
        "scene_id": row.scene_id,
        "source": row.source,
        "year": row.year,
        "path_cog": row.path_cog,
        "path_stac": row.path_stac,
        "status": row.status,
        "schema_hash": row.schema_hash,
        "schema_version": row.schema_version,
        "attempts": row.attempts,
        "last_error": row.last_error,
        "run_id": row.run_id,
        "updated_at": row.updated_at,
        # AOI metrics (schema v3)
        "aoi_clear_px": row.aoi_clear_px,
        "aoi_cloudy_px": row.aoi_cloudy_px,
        "aoi_shadow_px": row.aoi_shadow_px,
        "aoi_cirrus_px": row.aoi_cirrus_px,
        "aoi_saturated_px": row.aoi_saturated_px,
        "aoi_fill_px": row.aoi_fill_px,
        "aoi_total_px": row.aoi_total_px,
        "aoi_clear_frac": row.aoi_clear_frac,
        "aoi_overlap_px": row.aoi_overlap_px,
    }


def _rows_from_table(tbl: pa.Table) -> list[LedgerRow]:
    rows: list[LedgerRow] = []
    for i in range(tbl.num_rows):
        row = tbl.slice(i, 1)
        d = row.to_pydict()
        rows.append(
            LedgerRow(
                scene_id=str(d["scene_id"][0]),
                source=str(d["source"][0]),
                year=int(d["year"][0]),
                path_cog=_opt_str(d, "path_cog"),
                path_stac=_opt_str(d, "path_stac"),
                status=str(d["status"][0]),
                schema_hash=_opt_str(d, "schema_hash"),
                schema_version=int(d["schema_version"][0]),
                attempts=int(d["attempts"][0]),
                last_error=_opt_str(d, "last_error"),
                run_id=_opt_str(d, "run_id"),
                updated_at=_opt_dt(d, "updated_at"),
                # AOI metrics (schema v3)
                aoi_clear_px=_opt_int(d, "aoi_clear_px"),
                aoi_cloudy_px=_opt_int(d, "aoi_cloudy_px"),
                aoi_shadow_px=_opt_int(d, "aoi_shadow_px"),
                aoi_cirrus_px=_opt_int(d, "aoi_cirrus_px"),
                aoi_saturated_px=_opt_int(d, "aoi_saturated_px"),
                aoi_fill_px=_opt_int(d, "aoi_fill_px"),
                aoi_total_px=_opt_int(d, "aoi_total_px"),
                aoi_clear_frac=_opt_float(d, "aoi_clear_frac"),
                aoi_overlap_px=_opt_int(d, "aoi_overlap_px"),
            )
        )
    return rows


def _legacy_schema_fill(tbl: pa.Table) -> pa.Table:
    """Add missing AOI columns with nulls for pre-v4 rows."""
    existing_names = {f.name for f in tbl.schema}
    missing = [f for f in _SCHEMA if f.name not in existing_names]
    if not missing:
        return tbl
    for field in missing:
        tbl = tbl.append_column(field, pa.nulls(tbl.num_rows, type=field.type))
    return tbl


def _opt_str(d: dict, key: str) -> str | None:
    val = d[key][0]
    return None if val is None else str(val)


def _opt_int(d: dict, key: str) -> int | None:
    val = d.get(key, [None])[0]
    if val is None or isinstance(val, (int, float)) and val != val:  # NaN guard
        return None
    return int(val)


def _opt_float(d: dict, key: str) -> float | None:
    val = d.get(key, [None])[0]
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN
        return None
    return float(val)


def _opt_dt(d: dict, key: str) -> datetime | None:
    val = d.get(key, [None])[0]
    if val is None:
        return None
    # PyArrow 24's to_pydict returns datetime objects directly
    if hasattr(val, "as_py"):
        val = val.as_py()
    return val  # type: ignore[return-value]


__all__ = [
    "Ledger",
    "LedgerRow",
]
