"""Persistent per-item secondary-data ledger backed by PyArrow Parquet.

Mirrors ``data.ard.ledger`` with a schema adapted for static/dynamic
secondary items: no scene-specific fields, but carries ``config_hash``,
``checksum``, ``output_uri``, and final-artifact URIs.

Every ``upsert`` immediately persists via ``atomic_write`` — crash
consistency is built in.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import atomic_write, exists, read_bytes

# pyarrow.compute stubs — every ``pc.<name>`` access needs the type-ignore
_pceq = pc.equal  # type: ignore[attr-defined]
_pcan = pc.and_  # type: ignore[attr-defined]
_pcinv = pc.invert  # type: ignore[attr-defined]

# ── schema ───────────────────────────────────────────────────────────

_STATUSES = {"pending", "exporting", "done", "failed", "skipped"}

# decision: nullable STAC/provenance/completion URIs let ``reconcile()``
# guard against partial publication.
_SCHEMA = pa.schema([
    pa.field("item_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("period_or_vintage", pa.string(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("run_id", pa.string()),
    pa.field("attempts", pa.int32()),
    pa.field("config_hash", pa.string()),
    pa.field("checksum", pa.string()),
    pa.field("output_uri", pa.string()),
    pa.field("stac_uri", pa.string()),
    pa.field("provenance_uri", pa.string()),
    pa.field("completion_uri", pa.string()),
    pa.field("last_error", pa.string()),
    pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    pa.field("role", pa.string()),
])

# ── row type ─────────────────────────────────────────────────────────


@dataclass
class SecondaryLedgerRow:
    """A single row in the secondary-data processing ledger."""

    item_id: str
    source: str
    period_or_vintage: str
    status: str = "pending"
    run_id: str | None = None
    attempts: int = 0
    config_hash: str | None = None
    checksum: str | None = None
    output_uri: str | None = None
    stac_uri: str | None = None
    provenance_uri: str | None = None
    completion_uri: str | None = None
    last_error: str | None = None
    updated_at: datetime | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if self.updated_at is None:
            self.updated_at = datetime.now(UTC)
        if self.status not in _STATUSES:
            raise ValueError(f"Invalid status: {self.status!r}")


# ── ledger ───────────────────────────────────────────────────────────


class SecondaryLedger:
    """Read-write Parquet ledger for secondary-data item status tracking.

    Usage::

        ledger = SecondaryLedger.open("data/secondary/ledger.parquet")
        ledger.upsert(SecondaryLedgerRow(item_id=..., source=..., status="done"))
        rows = ledger.items_for_source("versiegelung")
    """

    def __init__(self, path: str, table: pa.Table) -> None:
        self._path = path
        self._table = table

    # ── factory ─────────────────────────────────────────────────

    @classmethod
    def open(cls, path: str) -> SecondaryLedger:
        """Open an existing Parquet ledger or create a new empty one.

        Reads must match the current schema exactly — there is no
        version-migration path. Create-only callers receive an empty
        current-schema table.
        """
        if exists(path):
            raw = read_bytes(path)
            table = pq.read_table(io.BytesIO(raw))
            if not table.schema.equals(_SCHEMA, check_metadata=False):
                raise ValueError(
                    f"Secondary ledger schema mismatch at {path!r}: "
                    f"expected {_SCHEMA}, got {table.schema}"
                )
        else:
            table = pa.Table.from_pylist([], schema=_SCHEMA)
        return cls(path, table)

    # ── queries ─────────────────────────────────────────────────

    def get(self, item_id: str, source: str, period: str) -> SecondaryLedgerRow | None:
        """Look up a specific item row, or ``None``."""
        if self._table.num_rows == 0:
            return None
        mask = _pcan(
            _pcan(
                _pceq(self._table.column("item_id"), item_id),
                _pceq(self._table.column("source"), source),
            ),
            _pceq(self._table.column("period_or_vintage"), period),
        )
        tbl = self._table.filter(mask)
        rows = _rows_from_table(tbl)
        return rows[0] if rows else None

    def items_for_source(self, source: str) -> list[SecondaryLedgerRow]:
        """Return all rows for a given source."""
        if self._table.num_rows == 0:
            return []
        tbl = self._table.filter(_pceq(self._table.column("source"), source))
        return _rows_from_table(tbl)

    # ── mutations ───────────────────────────────────────────────

    def upsert(self, row: SecondaryLedgerRow) -> None:
        """Insert or update a row identified by ``item_id + source + period``.

        Persists immediately via atomic temp-file write.
        """
        existing = self.get(row.item_id, row.source, row.period_or_vintage)
        if existing is not None:
            row.attempts = min(existing.attempts + 1, 999)
        else:
            row.attempts = 1

        new_row = pa.Table.from_pylist(
            [_row_to_dict(row)], schema=_SCHEMA
        )

        if self._table.num_rows == 0:
            self._table = new_row
        else:
            existing_mask = _pcan(
                _pcan(
                    _pceq(self._table.column("item_id"), row.item_id),
                    _pceq(self._table.column("source"), row.source),
                ),
                _pceq(self._table.column("period_or_vintage"), row.period_or_vintage),
            )
            self._table = pa.concat_tables(
                [self._table.filter(_pcinv(existing_mask)), new_row]
            )

        self._write_atomic()

    # ── persistence ─────────────────────────────────────────────

    def _write_atomic(self) -> str:
        """Persist ledger via ``atomic_write``."""
        buf = io.BytesIO()
        pq.write_table(self._table, buf)
        atomic_write(self._path, buf.getvalue(), overwrite=True)
        return self._path

    def status_counts(self, source: str | None = None) -> dict[str, int]:
        """Return ``{status: count}``, optionally filtered by source."""
        if self._table.num_rows == 0:
            return {}
        tbl = self._table
        if source:
            tbl = tbl.filter(_pceq(self._table.column("source"), source))
        if tbl.num_rows == 0:
            return {}
        counts: dict[str, int] = {}
        for s in _STATUSES:
            n = tbl.filter(_pceq(tbl.column("status"), s)).num_rows
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


def _row_to_dict(row: SecondaryLedgerRow) -> dict:
    return {
        "item_id": row.item_id,
        "source": row.source,
        "period_or_vintage": row.period_or_vintage,
        "status": row.status,
        "run_id": row.run_id,
        "attempts": row.attempts,
        "config_hash": row.config_hash,
        "checksum": row.checksum,
        "output_uri": row.output_uri,
        "stac_uri": row.stac_uri,
        "provenance_uri": row.provenance_uri,
        "completion_uri": row.completion_uri,
        "last_error": row.last_error,
        "updated_at": row.updated_at,
        "role": row.role,
    }


def _rows_from_table(tbl: pa.Table) -> list[SecondaryLedgerRow]:
    rows: list[SecondaryLedgerRow] = []
    for i in range(tbl.num_rows):
        d = tbl.slice(i, 1).to_pydict()
        rows.append(SecondaryLedgerRow(
            item_id=str(d["item_id"][0]),
            source=str(d["source"][0]),
            period_or_vintage=str(d["period_or_vintage"][0]),
            status=str(d["status"][0]),
            run_id=_opt_str(d, "run_id"),
            attempts=int(d["attempts"][0]),
            config_hash=_opt_str(d, "config_hash"),
            checksum=_opt_str(d, "checksum"),
            output_uri=_opt_str(d, "output_uri"),
            stac_uri=_opt_str(d, "stac_uri"),
            provenance_uri=_opt_str(d, "provenance_uri"),
            completion_uri=_opt_str(d, "completion_uri"),
            last_error=_opt_str(d, "last_error"),
            updated_at=_opt_dt(d, "updated_at"),
            role=_opt_str(d, "role"),
        ))
    return rows


def _opt_str(d: dict, key: str) -> str | None:
    val = d.get(key, [None])[0]
    return None if val is None else str(val)


def _opt_dt(d: dict, key: str) -> datetime | None:
    val = d.get(key, [None])[0]
    if val is None:
        return None
    if hasattr(val, "as_py"):
        val = val.as_py()
    return val  # type: ignore[return-value]


__all__ = [
    "SecondaryLedger",
    "SecondaryLedgerRow",
]
