"""Mode translation for the ARD pipeline scripts.

Centralises the mapping from the user-facing ``mode`` flag (``plan``,
``smoke``, ``all``) to the internal ``dry_run``/``smoke`` flags that
``ard_export.py`` and ``ard_process.py`` actually use.

This collapses the old ``dry_run`` + ``smoke`` flag pair into a single
``mode`` flag at the user-facing surface, while keeping the internal
implementation unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Mode(StrEnum):
    """Pipeline run mode — drives scope and execution semantics."""

    PLAN = "plan"  # show what would run, no execution (dry_run=True)
    SMOKE = "smoke"  # 1 source × 1 year × 1 scene per source, then validate
    ALL = "all"  # all sources × all years, full run


def apply_mode(cfg: Any) -> Any:
    """Translate ``cfg.mode`` to ``cfg.dry_run`` + ``cfg.smoke`` in place.

    If ``cfg.mode`` is missing or ``None``, the existing ``dry_run`` and
    ``smoke`` values are kept unchanged — backward-compatible with scripts
    that pass those flags directly.

    Returns the same cfg (mutated).
    """
    mode = getattr(cfg, "mode", None)
    if mode is None:
        return cfg
    mode_str = str(mode)
    match mode_str:
        case Mode.PLAN:
            cfg.dry_run = True
            cfg.smoke = False
        case Mode.SMOKE:
            cfg.dry_run = False
            cfg.smoke = True
        case Mode.ALL:
            cfg.dry_run = False
            cfg.smoke = False
        case _:
            raise ValueError(
                f"Unknown mode: {mode_str!r} (expected {'|'.join(m.value for m in Mode)})"
            )
    return cfg
