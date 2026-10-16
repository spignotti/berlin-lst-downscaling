"""Read published inputs. Callers do not download a second copy of a raster."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from berlin_lst_downscaling.data.eda.rules import season_of_month
from berlin_lst_downscaling.data.io import read_bytes
from berlin_lst_downscaling.data.training.paths import (
    cells_parquet,
    manifest_parquet,
    patch_index_parquet,
    scaler_json,
)

_VINTAGES = (2020, 2021, 2022, 2023, 2024)


def read_frame(uri: str) -> pd.DataFrame:
    """Read one Parquet table from a local path or ``gs://`` URI."""
    path = Path(uri)
    if uri.startswith("gs://"):
        return pd.read_parquet(io.BytesIO(read_bytes(uri)))
    return pd.read_parquet(path)


def read_json(uri: str) -> dict[str, object]:
    """Read one JSON object from a local path or ``gs://`` URI."""
    if uri.startswith("gs://"):
        payload = json.loads(read_bytes(uri))
    else:
        payload = json.loads(Path(uri).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{uri} is not a JSON object")
    return payload


def training_manifest(root: str) -> pd.DataFrame:
    """Return the training-release scene manifest."""
    return read_frame(manifest_parquet(root))


def training_cells(root: str) -> pd.DataFrame:
    """Return the eligible-cell index."""
    return read_frame(cells_parquet(root))


def scaler_payload(root: str) -> dict[str, object]:
    """Return the published train-only scaler document."""
    return read_json(scaler_json(root))


def acquisition_calendar(manifest_uri: str) -> pd.DataFrame:
    """Return anchor scene id, month, and meteorological season."""
    manifest = read_frame(manifest_uri)
    anchors = manifest.loc[manifest["role"].astype(str) == "anchor"].copy()
    timestamps = pd.to_datetime(anchors["acquisition_datetime"], utc=True)
    months = [None if pd.isna(value) else int(value) for value in timestamps.dt.month]
    return pd.DataFrame(
        {
            "scene_id": anchors["scene_id"].astype(str).to_numpy(),
            "month": months,
            "season": [season_of_month(month) for month in months],
        }
    )


def stadtstruktur_shares(root: str) -> pd.DataFrame:
    """Return inside-AOI share rows for vintages 2020–2024."""
    frames: list[pd.DataFrame] = []
    base = root.rstrip("/")
    for year in _VINTAGES:
        shares = read_frame(f"{base}/grid/100m/{year}/shares.parquet")
        cells = read_frame(f"{base}/grid/100m/{year}/cells.parquet")
        inside = list(cells.loc[cells["inside_aoi"].astype(bool), "cell_id"].astype(str))
        kept = shares.loc[shares["cell_id"].astype(str).isin(inside)].copy()
        kept["vintage"] = year
        frames.append(kept)
    return pd.concat(frames, ignore_index=True)


def inside_cells(root: str) -> pd.DataFrame:
    """Return the 2024 inside-AOI lattice. The canonical grid is stable across vintages."""
    cells = read_frame(f"{root.rstrip('/')}/grid/100m/2024/cells.parquet")
    inside = cells.loc[cells["inside_aoi"].astype(bool), ["row", "col"]].copy()
    return inside.reset_index(drop=True)


def patch_index(root: str) -> pd.DataFrame:
    """Return accepted patch windows. An empty index is an empty frame."""
    return read_frame(patch_index_parquet(root))


__all__ = [
    "acquisition_calendar",
    "inside_cells",
    "patch_index",
    "read_frame",
    "read_json",
    "scaler_payload",
    "stadtstruktur_shares",
    "training_cells",
    "training_manifest",
]
