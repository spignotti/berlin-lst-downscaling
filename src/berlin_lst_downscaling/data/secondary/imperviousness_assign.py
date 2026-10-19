"""Spatial sealing assignment for the Umweltatlas class rasters.

The class raster does not store a sealing percent for every code. The
rules below are the decided mapping. Pixel centers on the native 2.5 m
grid decide polygon membership (rasterio's default burn).

Priority for code 103, highest first:

1. Center inside a Stadtstruktur 2024 water polygon (``typ`` 100) → 0.
2. Center inside an Umweltatlas 2021 street segment → that segment's
   ``vg`` (track ballast counted as sealed). Overlapping segments at one
   center are averaged with equal weight. The 2026-10-19 extract had no
   overlaps.
3. Center inside another Stadtstruktur polygon → equal-area mean of donor
   pixels in that polygon.
4. Otherwise NaN (counted). This is the exterior of Berlin and a thin
   sliver between the boundary and the polygons.

Code 101 (unclassified shadow) uses the same block mean. A shadow pixel
outside every polygon takes the nearest block that has a finite donor
mean, within 100 m. A block with no donor pixels takes the equal-area
mean of donor pixels in a 100 m ring outside the polygon. That ring is
not a citywide default.

Code 110 is stored as 100. The rule set treats track ballast as 0 and
the map rendering treats it as 100. The decided value is the rendering.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rasterio.enums import MergeAlg
from rasterio.features import rasterize

# Codes whose own percent is a donor for block means. 110 contributes 100.
DONOR_CODES: frozenset[int] = frozenset(
    {0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 102, 110}
)
DEGREE_CODES: frozenset[int] = frozenset(range(5, 100, 10))
ALLOWED_CODES: frozenset[int] = DONOR_CODES | {101, 103, 255}
WATER_TYP: int = 100
# 40 native pixels × 2.5 m. Matches the empty-block ring.
NEIGHBOUR_PX: int = 40
EMPTY_BLOCK_RING_M: float = 100.0

MAPPING_RULES_ID = "imperviousness:v3"


def direct_percent(codes: np.ndarray) -> np.ndarray:
    """Map donor codes and code 110 to percent. 101, 103, and 255 stay NaN."""
    out = np.full(codes.shape, np.nan, dtype=np.float32)
    out[codes == 0] = 0.0
    for code in DEGREE_CODES:
        out[codes == code] = float(code)
    out[codes == 100] = 100.0
    out[codes == 102] = 100.0
    # Rule set: 0. Map rendering: 100. Decision: 100.
    out[codes == 110] = 100.0
    return out


def _burn(
    pairs: Iterable[tuple[object, float | int]],
    shape: tuple[int, int],
    transform,
    dtype: str,
    merge_alg,
) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    rasterize(
        list(pairs),
        out=out,
        transform=transform,
        merge_alg=merge_alg,
        dtype=dtype,
    )
    return out


def rasterize_streets(streets, shape: tuple[int, int], transform) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vg_sum, count)`` on the class grid. ``count`` is uint16."""
    geoms = list(streets.geometry)
    vg = streets["vg"].to_numpy(dtype=np.float32)
    if np.isnan(vg).any():
        raise ValueError("street layer has null vg values")
    vg_sum = _burn(zip(geoms, vg, strict=True), shape, transform, "float32", MergeAlg.add)
    count = _burn(
        ((geom, 1) for geom in geoms),
        shape,
        transform,
        "uint16",
        MergeAlg.add,
    )
    return vg_sum, count


def rasterize_blocks(
    blocks, shape: tuple[int, int], transform
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(water_mask, block_id, n_blocks)``.

    ``block_id`` is 0 outside polygons and 1..n inside non-water polygons.
    Water is any ``typ`` 100 polygon, including where it overlaps a street.
    """
    water = blocks[blocks["typ"] == WATER_TYP]
    built = blocks[blocks["typ"] != WATER_TYP]
    n_blocks = int(len(built))
    if n_blocks > np.iinfo(np.uint16).max:
        raise ValueError(f"block count {n_blocks} does not fit in uint16")
    water_mask = np.zeros(shape, dtype=bool)
    if len(water):
        burned = _burn(
            ((geom, 1) for geom in water.geometry),
            shape,
            transform,
            "uint8",
            MergeAlg.replace,
        )
        water_mask = burned > 0
    ids = np.arange(1, n_blocks + 1, dtype=np.uint16)
    block_id = _burn(
        zip(built.geometry, ids, strict=True),
        shape,
        transform,
        "uint16",
        MergeAlg.replace,
    )
    return water_mask, block_id, n_blocks


def donor_means(codes: np.ndarray, block_id: np.ndarray, n_blocks: int) -> np.ndarray:
    """Equal-area mean of donor pixels in each block. Index 0 is unused."""
    percent = direct_percent(codes)
    valid = np.isfinite(percent) & (block_id > 0) & (block_id <= n_blocks)
    sums = np.zeros(n_blocks + 1, dtype=np.float64)
    counts = np.zeros(n_blocks + 1, dtype=np.int64)
    if valid.any():
        np.add.at(sums, block_id[valid], percent[valid])
        np.add.at(counts, block_id[valid], 1)
    means = np.full(n_blocks + 1, np.nan, dtype=np.float64)
    ok = counts > 0
    means[ok] = sums[ok] / counts[ok]
    return means


def target_counts(
    codes: np.ndarray,
    block_id: np.ndarray,
    water: np.ndarray,
    street_count: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Count 101 pixels and residual 103 pixels per block."""
    in_block = (block_id > 0) & (block_id <= n_blocks)
    residual_103 = (codes == 103) & ~water & (street_count == 0) & in_block
    shadow = (codes == 101) & in_block
    counts = np.zeros(n_blocks + 1, dtype=np.int64)
    selected = residual_103 | shadow
    if selected.any():
        np.add.at(counts, block_id[selected], 1)
    return counts


def fill_empty_blocks_from_ring(
    codes: np.ndarray,
    built,
    shape: tuple[int, int],
    transform,
    means: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Fill blocks that have targets and no donor from a 100 m exterior ring.

    Returns the block ids that received a ring mean. Blocks whose ring
    also has no donor stay NaN.
    """
    empty = np.flatnonzero((targets > 0) & ~np.isfinite(means))
    if empty.size == 0:
        return np.asarray([], dtype=np.int64)
    buffered = []
    own = []
    for slot, block_index in enumerate(empty.tolist(), start=1):
        geom = built.geometry.iloc[int(block_index) - 1]
        buffered.append((geom.buffer(EMPTY_BLOCK_RING_M), slot))
        own.append((geom, slot))
    zone = _burn(buffered, shape, transform, "uint8", MergeAlg.replace)
    inside = _burn(own, shape, transform, "uint8", MergeAlg.replace)
    percent = direct_percent(codes)
    ring = (zone > 0) & (inside == 0) & np.isfinite(percent)
    sums = np.zeros(len(empty) + 1, dtype=np.float64)
    counts = np.zeros(len(empty) + 1, dtype=np.int64)
    if ring.any():
        np.add.at(sums, zone[ring], percent[ring])
        np.add.at(counts, zone[ring], 1)
    filled_ids: list[int] = []
    for slot, block_index in enumerate(empty.tolist(), start=1):
        if counts[slot] == 0:
            continue
        means[int(block_index)] = sums[slot] / counts[slot]
        filled_ids.append(int(block_index))
    return np.asarray(filled_ids, dtype=np.int64)


def apply_mapping(
    codes: np.ndarray,
    *,
    street_vg_sum: np.ndarray,
    street_count: np.ndarray,
    water: np.ndarray,
    block_id: np.ndarray,
    block_fill: np.ndarray,
    ring_blocks: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return sealing percent and native-grid statistics.

    ``ring_blocks`` marks block ids whose fill came from the 100 m ring,
    so those pixels are counted apart from in-polygon donors.
    """
    out = direct_percent(codes)
    street = street_count > 0
    vg = np.zeros(codes.shape, dtype=np.float32)
    np.divide(street_vg_sum, street_count, out=vg, where=street)
    is103 = codes == 103
    is101 = codes == 101
    street_103 = is103 & street & ~water
    out[street_103] = vg[street_103]
    water_103 = is103 & water
    out[water_103] = 0.0

    in_block = (block_id > 0) & (block_id < block_fill.shape[0])
    need = (is101 | is103) & np.isnan(out) & in_block
    if need.any():
        out[need] = block_fill[block_id[need]].astype(np.float32)

    unknown = ~np.isin(codes, np.fromiter(ALLOWED_CODES, dtype=np.uint16))
    assigned = need & np.isfinite(out)
    if ring_blocks.size:
        from_ring = assigned & np.isin(block_id, ring_blocks)
    else:
        from_ring = np.zeros(codes.shape, dtype=bool)
    in_polygon = assigned & ~from_ring
    stats: dict[str, float] = {
        "unknown_code_px": float(unknown.sum()),
        "nodata_255_px": float((codes == 255).sum()),
        "water_103_px": float(water_103.sum()),
        "street_103_px": float(street_103.sum()),
        "street_overlap_px": float((street_count > 1).sum()),
        "block_103_px": float((in_polygon & is103).sum()),
        "block_101_px": float((in_polygon & is101).sum()),
        "ring_103_px": float((from_ring & is103).sum()),
        "ring_101_px": float((from_ring & is101).sum()),
    }
    return out, stats


def sealing_coverage_stats(
    codes: np.ndarray,
    out: np.ndarray,
    water: np.ndarray,
    street_count: np.ndarray,
) -> dict[str, float]:
    """Area sums after every fill, including shadows borrowed from a neighbour block."""
    water_values = out[water]
    water_values = water_values[np.isfinite(water_values)]
    street = street_count > 0
    street_values = out[street]
    street_values = street_values[np.isfinite(street_values)]
    return {
        "unassigned_103_px": float(((codes == 103) & np.isnan(out)).sum()),
        "unassigned_101_px": float(((codes == 101) & np.isnan(out)).sum()),
        "water_finite_px": float(water_values.size),
        "water_sealing_sum": float(water_values.sum()) if water_values.size else 0.0,
        "street_finite_px": float(street_values.size),
        "street_sealing_sum": float(street_values.sum()) if street_values.size else 0.0,
    }


def fill_shadows_outside_blocks(
    codes: np.ndarray,
    out: np.ndarray,
    block_id: np.ndarray,
    block_fill: np.ndarray,
    *,
    search_px: int = NEIGHBOUR_PX,
) -> int:
    """Fill code 101 outside polygons from the nearest finite block within 100 m.

    Ties take the first pixel in row-major order inside the window.
    Returns the number of pixels filled.
    """
    # In-block shadows that the ring could not fill stay NaN. Only pixels
    # outside every polygon may borrow a neighbour.
    pending = np.argwhere((codes == 101) & (block_id == 0) & np.isnan(out))
    if pending.size == 0:
        return 0
    height, width = codes.shape
    filled = 0
    max_dist2 = search_px * search_px
    finite_blocks = np.isfinite(block_fill)
    for row, col in pending:
        y0 = max(0, int(row) - search_px)
        y1 = min(height, int(row) + search_px + 1)
        x0 = max(0, int(col) - search_px)
        x1 = min(width, int(col) + search_px + 1)
        window = block_id[y0:y1, x0:x1]
        occupied = window > 0
        if not occupied.any():
            continue
        ids = window[occupied]
        keep = finite_blocks[ids]
        if not keep.any():
            continue
        coords = np.argwhere(occupied)[keep]
        cy = int(row) - y0
        cx = int(col) - x0
        dist = (coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2
        nearest_i = int(np.argmin(dist))
        if int(dist[nearest_i]) > max_dist2:
            continue
        nearest = coords[nearest_i]
        bid = int(window[int(nearest[0]), int(nearest[1])])
        out[int(row), int(col)] = np.float32(block_fill[bid])
        filled += 1
    return filled


def built_frame(blocks):
    """Non-water polygons in the same order used to assign ``block_id``."""
    return blocks[blocks["typ"] != WATER_TYP].reset_index(drop=True)


__all__ = [
    "ALLOWED_CODES",
    "DONOR_CODES",
    "EMPTY_BLOCK_RING_M",
    "MAPPING_RULES_ID",
    "NEIGHBOUR_PX",
    "WATER_TYP",
    "apply_mapping",
    "built_frame",
    "direct_percent",
    "donor_means",
    "fill_empty_blocks_from_ring",
    "fill_shadows_outside_blocks",
    "rasterize_blocks",
    "rasterize_streets",
    "sealing_coverage_stats",
    "target_counts",
]
