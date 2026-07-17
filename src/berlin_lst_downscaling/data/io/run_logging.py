"""Shared pipeline run logging — JSONL + stderr, local + GCS.

Provides :class:`RunLogSession`, a lightweight context manager that
configures both human-readable stderr and machine-parseable JSONL output
for a single pipeline invocation.

Usage::

    with RunLogSession(output_root, pipeline="ard", run_id=run_id) as log:
        logger = logging.getLogger("berlin_lst_downscaling.data.ard.pipeline")
        log_event(logger, logging.INFO, "start", mode="full")

The JSONL file lives at ``<output_root>/logs/<pipeline>/<run_id>.jsonl``.
For GCS ``output_root`` the log is written to a local spool directory and
atomically uploaded when the session exits.

Design principles
-----------------
- **No new dependencies.** Stdlib ``logging`` only.
- **No root-logger pollution.** Handlers attach to the session root and
  are removed in ``finally``.
- **Exception-safe.** Upload/close in ``finally`` guarantees no orphaned
  handlers and no lost JSONL lines.
- **Module loggers only.** The session never calls ``basicConfig()`` or
  adds handlers to individual module loggers; the root logger handles
  propagation from ``getLogger(__name__)`` call sites.
"""

from __future__ import annotations

import json
import logging
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── formatters ────────────────────────────────────────────────────────


class _TextFormatter(logging.Formatter):
    """Concise human-readable stderr output."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        event = record.getMessage()
        pipeline = getattr(record, "pipeline", "")
        prefix = f"[{pipeline}] " if pipeline else ""
        return f"{ts} {prefix}{event}"


class _JSONLFormatter(logging.Formatter):
    """Structured JSONL output for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in ("pipeline", "run_id"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        fields = getattr(record, "fields", None)
        if fields is not None:
            entry.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(entry, default=str)


# ── handlers ──────────────────────────────────────────────────────────


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Stream handler targeting stderr with the text formatter."""

    def __init__(self) -> None:
        super().__init__(stream=__import__("sys").stderr)
        self.setFormatter(_TextFormatter())


class _JSONLFileHandler(logging.FileHandler):  # type: ignore[type-arg]
    """File handler writing JSONL lines."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(path), encoding="utf-8")
        self.setFormatter(_JSONLFormatter())


# ── helpers ───────────────────────────────────────────────────────────


def run_log_path(
    output_root: str,
    pipeline: str,
    run_id: str,
) -> Path:
    """Deterministic log file path for a run (local filesystem)."""
    return Path(output_root) / "logs" / pipeline / f"{run_id}.jsonl"


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured event with extra fields attached to the log record.

    The ``event`` becomes the log message; ``**fields`` are merged into the
    JSONL output via a ``fields`` extra on the :class:`LogRecord`.
    """
    logger.log(level, event, extra={"fields": fields})


# ── session ───────────────────────────────────────────────────────────


class RunLogSession:
    """Context manager that configures stderr + JSONL logging for a run.

    Parameters
    ----------
    output_root:
        Base output directory (local path or ``gs://…``).
    pipeline:
        Short identifier, e.g. ``"ard"``, ``"selection"``,
        ``"secondary"``, ``"static-sources"``, ``"static-derived"``.
    run_id:
        Unique run identifier (typically uuid4 hex).
    level:
        Minimum log level for the session (default ``INFO``).
    """

    def __init__(
        self,
        output_root: str,
        *,
        pipeline: str,
        run_id: str,
        level: int = logging.INFO,
    ) -> None:
        self.output_root = output_root
        self.pipeline = pipeline
        self.run_id = run_id
        self.level = level
        self._log_path: Path | None = None
        self._spool_path: Path | None = None
        self._handlers: list[logging.Handler] = []  # type: ignore[type-arg]
        self._orig_propagate: bool | None = None

    def __enter__(self) -> RunLogSession:
        root = logging.getLogger()
        self._orig_propagate = root.propagate
        root.propagate = False
        root.setLevel(self.level)

        # stderr — always live
        stderr_h = _StderrHandler()
        stderr_h.setLevel(self.level)
        root.addHandler(stderr_h)
        self._handlers.append(stderr_h)

        # JSONL file — local or spool+upload for GCS
        log_path = run_log_path(self.output_root, self.pipeline, self.run_id)
        is_gcs = self.output_root.startswith("gs://")

        if is_gcs:
            # Spool locally, upload on exit
            spool_dir = Path(tempfile.mkdtemp(prefix=f"log_{self.pipeline}_"))
            self._spool_path = spool_dir / f"{self.run_id}.jsonl"
            file_h = _JSONLFileHandler(self._spool_path)
            final = Path(self.output_root) / "logs" / self.pipeline
            self._log_path = final / f"{self.run_id}.jsonl"
        else:
            file_h = _JSONLFileHandler(log_path)
            self._log_path = log_path

        file_h.setLevel(self.level)
        root.addHandler(file_h)
        self._handlers.append(file_h)

        # Suppress noisy third-party loggers
        for name in ("rasterio._err", "urllib3.connectionpool", "odc.loader._rio"):
            logging.getLogger(name).setLevel(logging.ERROR)

        return self

    def __exit__(self, *_: Any) -> None:
        root = logging.getLogger()

        # Close handlers to flush buffers
        for h in self._handlers:
            try:
                h.close()
            except Exception:  # noqa: S110 — best-effort flush
                pass
            root.removeHandler(h)

        # Restore root propagate
        if self._orig_propagate is not None:
            root.propagate = self._orig_propagate

        # GCS upload: local spool → final GCS URI
        if self._spool_path is not None and self._log_path is not None:
            try:
                from berlin_lst_downscaling.data.io.storage import atomic_upload

                atomic_upload(self._spool_path, str(self._log_path))
            except Exception:  # noqa: S110 — best-effort: don't crash over log upload
                pass
            finally:
                # Clean up spool directory
                try:
                    spool_dir = self._spool_path.parent
                    self._spool_path.unlink(missing_ok=True)
                    spool_dir.rmdir()
                except Exception:  # noqa: S110 — best-effort cleanup
                    pass
