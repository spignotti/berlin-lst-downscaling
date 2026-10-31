"""Wall-clock phase timing emitted as structured ``log_event`` records."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from berlin_lst_downscaling.data.io import log_event


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def timed_phase(
    phase: str,
    *,
    logger: logging.Logger,
    **fields: object,
) -> Iterator[None]:
    """Emit ``phase_start``/``phase_end`` around a block, re-raising failures.

    Nested phases overlap; consumers must not sum them indiscriminately.
    """
    started_at = _utc_now()
    started = time.perf_counter()
    log_event(logger, logging.INFO, "phase_start", phase=phase, started_at=started_at, **fields)
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        log_event(
            logger,
            logging.INFO,
            "phase_end",
            phase=phase,
            started_at=started_at,
            ended_at=_utc_now(),
            duration_seconds=time.perf_counter() - started,
            status=status,
            **fields,
        )


class EpochTimingRecorder(Callback):
    """Timing-only callback: per-epoch train, validation, and total wall time.

    Lightning runs in-epoch validation before ``on_train_epoch_end``, so the
    train-exclusive time is the epoch total minus the validation interval.
    Sanity validation is ignored; an epoch that never reaches
    ``on_train_epoch_end`` emits nothing. It reads no tensors, synchronises no
    device, and touches no loader or model state.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__()
        self._logger = logger
        self._clock = clock
        self._epoch_started: float | None = None
        self._validation_started: float | None = None
        self._validation_seconds = 0.0
        self.records: list[dict[str, float | int]] = []

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_started = self._clock()
        self._validation_started = None
        self._validation_seconds = 0.0

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._validation_started = self._clock()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or self._validation_started is None:
            return
        self._validation_seconds += self._clock() - self._validation_started
        self._validation_started = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._epoch_started is None or self._validation_started is not None:
            self._epoch_started = None
            return
        epoch_seconds = self._clock() - self._epoch_started
        record: dict[str, float | int] = {
            "epoch": int(trainer.current_epoch),
            "train_seconds": epoch_seconds - self._validation_seconds,
            "validation_seconds": self._validation_seconds,
            "epoch_seconds": epoch_seconds,
        }
        self._epoch_started = None
        self.records.append(record)
        log_event(self._logger, logging.INFO, "epoch_timing", **record)
