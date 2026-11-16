"""Spatial holdout assignment over the patch-index anchors (add-on test, pm #553).

The v6 matrix splits by time only, so every model has seen the validation
anchors during training. The add-on test removes whole blocks of anchors from
training and scores them on validation 2024. This module holds the pure
assignment logic; it reads no data product and imports no training framework.

Rule
----
- Unit: the anchor positions of one patch index (global canonical 100 m
  ``(row, col)`` of each window's top-left cell; multiples of 16).
- Blocks: the fixed tessellation of the global anchor lattice into
  ``3 x 3`` anchors (4.8 km): ``block = (row // 48, col // 48)``. The origin
  is the canonical grid origin, so the tessellation has no free offset.
- Eligible: complete blocks only (all nine anchor positions are in the index).
- Draw: eligible blocks sorted by ``(block_row, block_col)``, permuted with
  ``numpy.random.default_rng(seed)``; whole blocks are taken in that order until
  the held-out share of all anchors reaches the lower bound. A share above the
  upper bound raises.
- Position: a held-out anchor is ``edge`` when one of its direct neighbours is
  a training anchor (not held out, with at least one train patch in the index),
  else ``interior``.

The committed file is the source of truth for a run; ``build_assignment`` must
reproduce it from the file's own anchor table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ASSIGNMENT_SCHEMA_VERSION = 1
ASSIGNMENT_KIND = "spatial_holdout_assignment"
# Anchor stride of the patch index in 100 m cells (data.training.patch_index
# PATCH_STRIDE_CELLS); literal so this module stays free of raster imports.
ANCHOR_STRIDE_CELLS = 16
BLOCK_ANCHORS = 3
MIN_FRACTION = 0.18
MAX_FRACTION = 0.22
POSITION_INTERIOR = "interior"
POSITION_EDGE = "edge"
NEIGHBOURHOODS = (4, 8)
ASSIGNMENT_DIR: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "modeling" / "spatial_holdout"
)

# The only configs that may enable the option, and the matrix arm each one repeats.
SPATIAL_HOLDOUT_CONFIG_ARMS: dict[str, str] = {
    "v6_l1_spatial_holdout": "L1",
    "v6_l4_spatial_holdout": "L4",
}
_CONFIG_KEYS = frozenset({"enabled", "assignment"})

Anchor = tuple[int, int]
# Columns of one row of the file's anchor table (index rows of train and validation).
_ANCHOR_COLUMNS = ("row", "col", "train", "validation")

_NEIGHBOUR_OFFSETS: dict[int, tuple[Anchor, ...]] = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}


class SpatialHoldoutError(ValueError):
    """Raised for an invalid anchor table, rule, or assignment file."""


@dataclass(frozen=True)
class AnchorCounts:
    """Index rows per anchor for the two splits a training run reads."""

    train: int
    validation: int


@dataclass(frozen=True)
class HoldoutAnchor:
    row: int
    col: int
    block: Anchor
    position: str


@dataclass(frozen=True)
class SpatialHoldoutAssignment:
    """A drawn assignment plus the anchor table and rule it was drawn from."""

    anchors: dict[Anchor, AnchorCounts]
    seed: int
    neighbourhood: int
    min_fraction: float
    max_fraction: float
    eligible_blocks: tuple[Anchor, ...]
    blocks: tuple[Anchor, ...]
    holdout: tuple[HoldoutAnchor, ...]
    patch_index: dict[str, str]

    @property
    def holdout_anchors(self) -> frozenset[Anchor]:
        return frozenset((a.row, a.col) for a in self.holdout)

    @property
    def fraction(self) -> float:
        return len(self.holdout) / len(self.anchors)

    def positions(self) -> dict[Anchor, str]:
        """Return ``anchor -> interior|edge`` for the held-out anchors."""
        return {(a.row, a.col): a.position for a in self.holdout}

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "kind": ASSIGNMENT_KIND,
            "patch_index": dict(sorted(self.patch_index.items())),
            "rule": {
                "anchor_stride_cells": ANCHOR_STRIDE_CELLS,
                "block_anchors": BLOCK_ANCHORS,
                "tessellation": "block = (row // 48, col // 48) on the global canonical lattice",
                "eligible_blocks": "complete: all nine anchor positions are in the index",
                "draw": (
                    "eligible blocks sorted by (block_row, block_col), permuted with "
                    "numpy.random.default_rng(seed); whole blocks until fraction >= min"
                ),
                "seed": self.seed,
                "min_fraction": self.min_fraction,
                "max_fraction": self.max_fraction,
                "position": (
                    "edge: a direct neighbour is a training anchor (not held out, "
                    "train patches > 0); else interior"
                ),
                "neighbourhood": self.neighbourhood,
            },
            "anchors_total": len(self.anchors),
            "holdout_anchors": len(self.holdout),
            "holdout_fraction": round(self.fraction, 6),
            "eligible_blocks": [list(b) for b in self.eligible_blocks],
            "blocks": [list(b) for b in self.blocks],
            "holdout": [
                {"row": a.row, "col": a.col, "block": list(a.block), "position": a.position}
                for a in self.holdout
            ],
            "anchor_columns": list(_ANCHOR_COLUMNS),
            "anchors": [
                [row, col, c.train, c.validation] for (row, col), c in sorted(self.anchors.items())
            ],
        }


def block_of(row: int, col: int) -> Anchor:
    """Return the tessellation block of an anchor on the global lattice."""
    span = ANCHOR_STRIDE_CELLS * BLOCK_ANCHORS
    return row // span, col // span


def anchor_counts(rows: Iterable[Mapping[str, object]]) -> dict[Anchor, AnchorCounts]:
    """Count train and validation index rows per anchor.

    Every anchor of the index is kept, also one that only carries rows of
    another split; such an anchor has zero train and zero validation rows.
    """
    counts: dict[Anchor, list[int]] = {}
    for record in rows:
        anchor = (int(record["row"]), int(record["col"]))  # type: ignore[call-overload]
        entry = counts.setdefault(anchor, [0, 0])
        split = str(record["split"])
        if split == "train":
            entry[0] += 1
        elif split == "validation":
            entry[1] += 1
    return {a: AnchorCounts(train=c[0], validation=c[1]) for a, c in counts.items()}


def _validate_anchors(anchors: Mapping[Anchor, AnchorCounts]) -> None:
    if not anchors:
        raise SpatialHoldoutError("the anchor table is empty")
    for (row, col), c in anchors.items():
        if row % ANCHOR_STRIDE_CELLS or col % ANCHOR_STRIDE_CELLS:
            raise SpatialHoldoutError(
                f"anchor ({row}, {col}) is not on the {ANCHOR_STRIDE_CELLS}-cell lattice"
            )
        if c.train < 0 or c.validation < 0:
            raise SpatialHoldoutError(f"anchor ({row}, {col}) has negative patch counts")


def complete_blocks(anchors: Iterable[Anchor]) -> tuple[Anchor, ...]:
    """Return the blocks whose nine anchor positions are all present, sorted."""
    per_block: dict[Anchor, int] = {}
    for row, col in set(anchors):
        block = block_of(row, col)
        per_block[block] = per_block.get(block, 0) + 1
    full = BLOCK_ANCHORS * BLOCK_ANCHORS
    return tuple(sorted(b for b, n in per_block.items() if n == full))


def label_positions(
    holdout: Iterable[Anchor],
    anchors: Mapping[Anchor, AnchorCounts],
    *,
    neighbourhood: int = 8,
) -> dict[Anchor, str]:
    """Label each held-out anchor ``edge`` or ``interior``."""
    if neighbourhood not in NEIGHBOURHOODS:
        raise SpatialHoldoutError(f"neighbourhood must be one of {NEIGHBOURHOODS}")
    held = set(holdout)
    step = ANCHOR_STRIDE_CELLS
    labels: dict[Anchor, str] = {}
    for row, col in sorted(held):
        edge = False
        for d_row, d_col in _NEIGHBOUR_OFFSETS[neighbourhood]:
            neighbour = (row + d_row * step, col + d_col * step)
            counts = anchors.get(neighbour)
            if neighbour not in held and counts is not None and counts.train > 0:
                edge = True
                break
        labels[(row, col)] = POSITION_EDGE if edge else POSITION_INTERIOR
    return labels


def build_assignment(
    anchors: Mapping[Anchor, AnchorCounts],
    *,
    seed: int = 0,
    neighbourhood: int = 8,
    min_fraction: float = MIN_FRACTION,
    max_fraction: float = MAX_FRACTION,
    patch_index: Mapping[str, str] | None = None,
) -> SpatialHoldoutAssignment:
    """Draw whole complete blocks with a fixed seed until the share is reached."""
    _validate_anchors(anchors)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SpatialHoldoutError(f"seed must be a non-negative int, got {seed!r}")
    if not 0.0 < min_fraction <= max_fraction < 1.0:
        raise SpatialHoldoutError(
            f"fractions must satisfy 0 < min <= max < 1, got {min_fraction}, {max_fraction}"
        )
    eligible = complete_blocks(anchors)
    order = np.random.default_rng(seed).permutation(len(eligible))
    total = len(anchors)
    per_block = BLOCK_ANCHORS * BLOCK_ANCHORS
    drawn: list[Anchor] = []
    for index in order:
        if len(drawn) * per_block / total >= min_fraction:
            break
        drawn.append(eligible[int(index)])
    fraction = len(drawn) * per_block / total
    if fraction < min_fraction:
        raise SpatialHoldoutError(
            f"only {len(eligible)} complete blocks: the held-out share {fraction:.4f} "
            f"cannot reach {min_fraction}"
        )
    if fraction > max_fraction:
        raise SpatialHoldoutError(
            f"held-out share {fraction:.4f} exceeds {max_fraction}; whole blocks cannot "
            "land inside the bounds"
        )
    blocks = tuple(sorted(drawn))
    block_set = set(blocks)
    held = sorted(a for a in anchors if block_of(*a) in block_set)
    labels = label_positions(held, anchors, neighbourhood=neighbourhood)
    return SpatialHoldoutAssignment(
        anchors=dict(sorted(anchors.items())),
        seed=seed,
        neighbourhood=neighbourhood,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
        eligible_blocks=eligible,
        blocks=blocks,
        holdout=tuple(
            HoldoutAnchor(row=row, col=col, block=block_of(row, col), position=labels[(row, col)])
            for row, col in held
        ),
        patch_index=dict(patch_index or {}),
    )


def assignment_bytes(assignment: SpatialHoldoutAssignment) -> bytes:
    """Canonical file bytes: sorted keys, one list element per line, trailing newline."""
    lines: list[str] = []
    for key, value in sorted(assignment.to_payload().items()):
        if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
            rows = ",\n".join(f"  {json.dumps(item, sort_keys=True)}" for item in value)
            lines.append(f" {json.dumps(key)}: [\n{rows}\n ]")
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(value, sort_keys=True)}")
    return ("{\n" + ",\n".join(lines) + "\n}\n").encode("utf-8")


def parse_assignment(payload: Mapping[str, Any]) -> SpatialHoldoutAssignment:
    """Parse an assignment payload and verify it against its own rule.

    The held-out anchors, blocks and positions are re-derived from the anchor
    table and the recorded rule; a file that disagrees with its own rule (a
    hand edit, or a changed random stream) raises.
    """
    if payload.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
        raise SpatialHoldoutError(f"unknown schema_version {payload.get('schema_version')!r}")
    if payload.get("kind") != ASSIGNMENT_KIND:
        raise SpatialHoldoutError("not a spatial holdout assignment")
    rule = payload.get("rule")
    if not isinstance(rule, Mapping):
        raise SpatialHoldoutError("assignment has no rule block")
    if (
        rule.get("anchor_stride_cells") != ANCHOR_STRIDE_CELLS
        or rule.get("block_anchors") != BLOCK_ANCHORS
    ):
        raise SpatialHoldoutError("assignment geometry differs from this code")
    if tuple(payload.get("anchor_columns") or ()) != _ANCHOR_COLUMNS:
        raise SpatialHoldoutError(f"anchor_columns must be {list(_ANCHOR_COLUMNS)}")
    raw_anchors = payload.get("anchors")
    if not isinstance(raw_anchors, Sequence) or not raw_anchors:
        raise SpatialHoldoutError("assignment has no anchors")
    anchors: dict[Anchor, AnchorCounts] = {}
    for entry in raw_anchors:
        if not isinstance(entry, Sequence) or len(entry) != len(_ANCHOR_COLUMNS):
            raise SpatialHoldoutError(f"anchor row {entry!r} is not {list(_ANCHOR_COLUMNS)}")
        row, col, train, validation = (int(v) for v in entry)
        if (row, col) in anchors:
            raise SpatialHoldoutError(f"duplicate anchor ({row}, {col})")
        anchors[(row, col)] = AnchorCounts(train=train, validation=validation)
    min_fraction, max_fraction = rule.get("min_fraction"), rule.get("max_fraction")
    if not isinstance(min_fraction, float) or not isinstance(max_fraction, float):
        raise SpatialHoldoutError("rule min_fraction and max_fraction must be floats")
    rebuilt = build_assignment(
        anchors,
        seed=rule.get("seed", -1),
        neighbourhood=rule.get("neighbourhood", 0),
        min_fraction=min_fraction,
        max_fraction=max_fraction,
        patch_index={str(k): str(v) for k, v in (payload.get("patch_index") or {}).items()},
    )
    if rebuilt.to_payload() != dict(payload):
        raise SpatialHoldoutError(
            "assignment does not match the one its own rule and anchor table produce"
        )
    return rebuilt


def load_assignment(path: str | Path) -> SpatialHoldoutAssignment:
    """Load and verify a committed assignment file."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"spatial holdout assignment not found: {file}")
    return parse_assignment(json.loads(file.read_text(encoding="utf-8")))


def assignment_sha256(path: str | Path) -> str:
    """SHA-256 of the assignment file bytes (recorded with every holdout run)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_assignment_path(name: str) -> Path:
    """Map a config value (``v1`` or a path) to the assignment file."""
    candidate = Path(name)
    if candidate.suffix == ".json":
        return candidate if candidate.is_absolute() else Path.cwd() / candidate
    return ASSIGNMENT_DIR / f"{name}.json"


def spatial_holdout_from_config(cfg: Any) -> str | None:
    """Return the assignment name when ``spatial_holdout.enabled`` is true, else ``None``.

    The block is absent from every base and matrix config: absent means off, so
    their resolved configs do not change. Unknown keys and wrong types raise.
    """
    block = cfg.get("spatial_holdout")
    if block is None:
        return None
    unknown = sorted(set(block) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown spatial_holdout key(s) {unknown}")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"spatial_holdout.enabled must be a bool, got {enabled!r}")
    if not enabled:
        return None
    name = block.get("assignment")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"spatial_holdout.assignment must be a non-empty string, got {name!r}")
    return name


def render_map(assignment: SpatialHoldoutAssignment) -> str:
    """ASCII map of the anchor lattice: ``I`` interior, ``E`` edge, ``o`` kept, ``.`` no anchor."""
    positions = assignment.positions()
    rows = sorted({a[0] for a in assignment.anchors})
    cols = sorted({a[1] for a in assignment.anchors})
    step = ANCHOR_STRIDE_CELLS
    lines: list[str] = []
    for row in range(rows[0], rows[-1] + step, step):
        cells: list[str] = []
        for col in range(cols[0], cols[-1] + step, step):
            if (row, col) not in assignment.anchors:
                cells.append(".")
            elif (row, col) in positions:
                cells.append("I" if positions[(row, col)] == POSITION_INTERIOR else "E")
            else:
                cells.append("o")
        lines.append(" ".join(cells))
    return "\n".join(lines)


__all__ = [
    "ANCHOR_STRIDE_CELLS",
    "ASSIGNMENT_DIR",
    "BLOCK_ANCHORS",
    "MAX_FRACTION",
    "MIN_FRACTION",
    "POSITION_EDGE",
    "POSITION_INTERIOR",
    "SPATIAL_HOLDOUT_CONFIG_ARMS",
    "Anchor",
    "AnchorCounts",
    "HoldoutAnchor",
    "SpatialHoldoutAssignment",
    "SpatialHoldoutError",
    "anchor_counts",
    "assignment_bytes",
    "assignment_sha256",
    "block_of",
    "build_assignment",
    "complete_blocks",
    "label_positions",
    "load_assignment",
    "parse_assignment",
    "render_map",
    "resolve_assignment_path",
    "spatial_holdout_from_config",
]
