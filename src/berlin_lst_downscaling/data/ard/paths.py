"""Deterministic output paths for COG and STAC artefacts."""

from __future__ import annotations

from pathlib import Path


def _resolve(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    return Path(root) / source / str(year) / scene_id


def scene_dir(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    """Return the output directory for a scene.

    ``root`` is typically ``cfg.output_root`` (``data/ard`` or
    ``data/tmp/smoke_ard_<date>``).
    """
    return _resolve(root, source, year, scene_id)


def cog_path(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    """Return the full output path for the scene's COG.

    Example: ``<root>/landsat-c2-l2/2024/LC09_L2SP_193024_20240629_02_T1/…``
    """
    return scene_dir(root, source, year, scene_id) / f"{scene_id}.tif"


def stac_path(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    """Return the full output path for the scene's STAC item.

    Example: ``<root>/sentinel-2-l2a/2024/…/…stac.json``
    """
    return scene_dir(root, source, year, scene_id) / f"{scene_id}.stac.json"


def flag_path(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    """Return the output path for the scene's flag COG (uint8 bitmask).

    Example: ``<root>/sentinel-2-l2a/2024/…/…flag.tif``
    """
    return scene_dir(root, source, year, scene_id) / f"{scene_id}.flag.tif"


def tmp_dir(root: str | Path, source: str, year: int, scene_id: str) -> Path:
    """Return the temporary directory for atomic writes.

    Files are written here first, then ``os.replace``-ed to the target
    path.  Aborted runs leave only temp files, never half-baked COGs.
    """
    return _resolve(root, source, year, scene_id) / ".tmp"


__all__ = [
    "scene_dir",
    "cog_path",
    "flag_path",
    "stac_path",
    "tmp_dir",
]
