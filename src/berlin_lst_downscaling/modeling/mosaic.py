"""Mean mosaic of the 10 m LST scenes that share one Landsat overpass.

An overpass is one sensor (``LC08`` or ``LC09``), one WRS path, and one
acquisition date. Rows of that path differ. Every input is on the canonical
EPSG:25833 10 m lattice. The mosaic is the union of those windows: a pixel
is the mean of the finite inputs, and a pixel with no finite input stays NaN.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from berlin_lst_downscaling.data.training.contracts import CANON_GRID_ORIGIN_X, CANON_GRID_ORIGIN_Y
from berlin_lst_downscaling.modeling.inference import (
    _assert_north_up_resolution,
    _exact_canonical_offset,
    _is_epsg25833,
)

_SCENE_RE = re.compile(
    r"^(?P<sensor>LC0[89])_L2SP_(?P<path>\d{3})(?P<row>\d{3})_(?P<date>\d{8})"
    r"(?:_(?P<suffix>[0-9A-Z_]+))?$"
)
_PRODUCT_SUFFIX = ".lst10m.tif"
_RES_M = 10.0


@dataclass(frozen=True)
class LandsatSceneId:
    """The parts of a Collection-2 scene id that define an overpass."""

    sensor: str
    path: str
    row: str
    date: str
    scene_id: str

    @property
    def overpass_id(self) -> str:
        """``LC09_193_20250608`` for every row of that path and day."""
        return f"{self.sensor}_{self.path}_{self.date}"

    @property
    def year(self) -> int:
        return int(self.date[:4])


@dataclass(frozen=True)
class OverpassMosaic:
    """The mean mosaic and the overlap QA that is written next to it."""

    overpass_id: str
    scene_ids: tuple[str, ...]
    array: np.ndarray
    transform: Affine
    qa: dict[str, object]


def parse_landsat_scene_id(scene_id: str) -> LandsatSceneId:
    """Parse ``LC09_L2SP_193023_20250608`` and the optional ``_02_T1`` tail.

    A full Collection-2 id (``…_20250610_02_T1``) is accepted. Anything else
    raises. The acquisition date is the first 8-digit token, not the
    processing date.
    """
    match = _SCENE_RE.match(scene_id)
    if match is None:
        raise ValueError(
            f"scene id {scene_id!r} is not an LC08/LC09 L2SP id "
            "(sensor, WRS path/row, acquisition date)"
        )
    return LandsatSceneId(
        sensor=match.group("sensor"),
        path=match.group("path"),
        row=match.group("row"),
        date=match.group("date"),
        scene_id=scene_id,
    )


def acquisition_year(scene_id: str) -> int:
    """Return the acquisition year of a Landsat scene id. Unparseable ids raise."""
    return parse_landsat_scene_id(scene_id).year


def group_scenes_by_overpass(scene_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Group scene ids by sensor, WRS path, and acquisition date.

    Members of a group are ordered by WRS row. Two scenes of the same row in
    one overpass raise: a row is one observation.
    """
    buckets: dict[str, list[LandsatSceneId]] = {}
    for scene_id in scene_ids:
        parsed = parse_landsat_scene_id(scene_id)
        buckets.setdefault(parsed.overpass_id, []).append(parsed)
    grouped: dict[str, tuple[str, ...]] = {}
    for overpass_id in sorted(buckets):
        members = buckets[overpass_id]
        rows = [item.row for item in members]
        if len(rows) != len(set(rows)):
            raise ValueError(f"overpass {overpass_id} has two scenes on the same WRS row")
        ordered = tuple(item.scene_id for item in sorted(members, key=lambda item: item.row))
        grouped[overpass_id] = ordered
    return grouped


def scene_id_from_lst_product(path: str) -> str:
    """Return the scene id of ``<scene_id>.lst10m.tif``."""
    name = Path(path).name
    if not name.endswith(_PRODUCT_SUFFIX):
        raise ValueError(f"{path} is not a <scene_id>.lst10m.tif product")
    return name[: -len(_PRODUCT_SUFFIX)]


def overpass_product_paths(out_dir: str, overpass_id: str) -> tuple[str, str]:
    """Return ``(<overpass>.lst10m.tif, <overpass>.json)`` under ``<out>/overpass``."""
    root = f"{out_dir.rstrip('/')}/overpass"
    return f"{root}/{overpass_id}.lst10m.tif", f"{root}/{overpass_id}.json"


def _union_transform(col0: int, row0: int) -> Affine:
    return Affine(
        _RES_M,
        0.0,
        CANON_GRID_ORIGIN_X + col0 * _RES_M,
        0.0,
        -_RES_M,
        CANON_GRID_ORIGIN_Y - row0 * _RES_M,
    )


def _read_product(path: str) -> tuple[LandsatSceneId, np.ndarray, int, int]:
    scene = parse_landsat_scene_id(scene_id_from_lst_product(path))
    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError(f"{path}: expected 1 band, got {src.count}")
        if not _is_epsg25833(src.crs):
            raise ValueError(f"{path}: CRS is {src.crs!r}, expected EPSG:25833")
        _assert_north_up_resolution(src.transform, _RES_M, label=path)
        col0, row0 = _exact_canonical_offset(src.transform, _RES_M)
        values = src.read(1).astype(np.float32)
    return scene, values, col0, row0


def mosaic_overpass(scene_tifs: Sequence[str]) -> OverpassMosaic:
    """Mean the valid pixels of one overpass onto the union of their windows.

    Inputs must already share the canonical 10 m lattice. After each scene is
    placed on the union, they share one transform and one shape. A pixel is
    the mean of the finite inputs. Overlap QA is one entry per unordered pair:
    the count of pixels where both are finite, and the mean and max absolute
    difference there.
    """
    if not scene_tifs:
        raise ValueError("an overpass mosaic needs at least one scene")
    loaded = [_read_product(path) for path in scene_tifs]
    overpass_ids = {scene.overpass_id for scene, _values, _col, _row in loaded}
    if len(overpass_ids) != 1:
        raise ValueError(f"scenes are not one overpass: {sorted(overpass_ids)}")
    overpass_id = next(iter(overpass_ids))
    rows = [scene.row for scene, _values, _col, _row in loaded]
    if len(rows) != len(set(rows)):
        raise ValueError(f"overpass {overpass_id} has two scenes on the same WRS row")
    order = sorted(range(len(loaded)), key=lambda index: loaded[index][0].scene_id)
    loaded = [loaded[index] for index in order]

    col0 = min(col for _scene, _values, col, _row in loaded)
    row0 = min(row for _scene, _values, _col, row in loaded)
    col1 = max(col + values.shape[1] for _scene, values, col, _row in loaded)
    row1 = max(row + values.shape[0] for _scene, values, _col, row in loaded)
    height = row1 - row0
    width = col1 - col0
    if height <= 0 or width <= 0:
        raise RuntimeError(f"overpass {overpass_id} union is empty")
    transform = _union_transform(col0, row0)
    placed = np.full((len(loaded), height, width), np.nan, dtype=np.float32)
    scene_ids: list[str] = []
    for index, (scene, values, origin_col, origin_row) in enumerate(loaded):
        scene_ids.append(scene.scene_id)
        local_row = origin_row - row0
        local_col = origin_col - col0
        placed[
            index,
            local_row : local_row + values.shape[0],
            local_col : local_col + values.shape[1],
        ] = values
    if placed.shape[1:] != (height, width):
        raise RuntimeError(f"aligned mosaic shape {placed.shape[1:]} != {(height, width)}")
    finite = np.isfinite(placed)
    count = finite.sum(axis=0)
    total = np.where(finite, placed, 0.0).sum(axis=0)
    mean64 = np.full((height, width), np.nan, dtype=np.float64)
    np.divide(total, count, out=mean64, where=count > 0)
    mean = mean64.astype(np.float32)
    pairs: list[dict[str, object]] = []
    for left, right in combinations(range(len(scene_ids)), 2):
        both = finite[left] & finite[right]
        overlap = int(both.sum())
        if overlap == 0:
            mean_abs: float | None = None
            max_abs: float | None = None
        else:
            diff = np.abs(
                placed[left][both].astype(np.float64) - placed[right][both].astype(np.float64)
            )
            mean_abs = float(diff.mean())
            max_abs = float(diff.max())
        pairs.append(
            {
                "scene_a": scene_ids[left],
                "scene_b": scene_ids[right],
                "overlap_count": overlap,
                "mean_abs_difference": mean_abs,
                "max_abs_difference": max_abs,
            }
        )
    qa: dict[str, object] = {
        "overpass_id": overpass_id,
        "scene_ids": scene_ids,
        "shape": [height, width],
        "pairs": pairs,
    }
    return OverpassMosaic(
        overpass_id=overpass_id,
        scene_ids=tuple(scene_ids),
        array=mean,
        transform=transform,
        qa=qa,
    )


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    if str(path).startswith("gs://"):
        from berlin_lst_downscaling.data.io import atomic_write

        atomic_write(path, body, overwrite=True)
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)


def write_overpass_mosaic(scene_tifs: Sequence[str], out_dir: str) -> tuple[str, str]:
    """Write ``<out>/overpass/<sensor>_<path>_<date>.lst10m.tif`` and its JSON."""
    from berlin_lst_downscaling.modeling.inference import write_lst_cog

    mosaic = mosaic_overpass(scene_tifs)
    tif_path, json_path = overpass_product_paths(out_dir, mosaic.overpass_id)
    write_lst_cog(mosaic.array, mosaic.transform, tif_path)
    _write_json(json_path, mosaic.qa)
    return tif_path, json_path


__all__ = [
    "LandsatSceneId",
    "OverpassMosaic",
    "acquisition_year",
    "group_scenes_by_overpass",
    "mosaic_overpass",
    "overpass_product_paths",
    "parse_landsat_scene_id",
    "scene_id_from_lst_product",
    "write_overpass_mosaic",
]
