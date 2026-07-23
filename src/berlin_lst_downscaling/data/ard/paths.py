"""Deterministic output paths for COG and STAC artefacts."""

from __future__ import annotations


def scene_dir(root: str, source: str, year: int, scene_id: str) -> str:
    """Return the output directory for a scene.

    ``root`` is typically ``cfg.output_root`` (``data/ard``,
    ``data/smoke/primary/ard``, or ``gs://bucket/prefix``).
    """
    # NOTE: ``pathlib.Path`` would strip the double slash from
    # ``gs://bucket/...`` to ``gs:/bucket/...``, breaking GCS paths —
    # that's why we join with f-strings here.
    return f"{root.rstrip('/')}/{source}/{year}/{scene_id}"


def cog_path(root: str, source: str, year: int, scene_id: str) -> str:
    """Return the full output path for the scene's COG.

    Example: ``<root>/landsat-c2-l2/2024/LC09_L2SP_193024_20240629_02_T1/…``
    """
    return f"{scene_dir(root, source, year, scene_id)}/{scene_id}.tif"


def stac_path(root: str, source: str, year: int, scene_id: str) -> str:
    """Return the full output path for the scene's STAC item.

    Example: ``<root>/sentinel-2-l2a/2024/…/…stac.json``
    """
    return f"{scene_dir(root, source, year, scene_id)}/{scene_id}.stac.json"


def flag_path(root: str, source: str, year: int, scene_id: str) -> str:
    """Return the output path for the scene's flag COG (uint8 bitmask).

    Example: ``<root>/sentinel-2-l2a/2024/…/…flag.tif``
    """
    return f"{scene_dir(root, source, year, scene_id)}/{scene_id}.flag.tif"


__all__ = [
    "scene_dir",
    "cog_path",
    "flag_path",
    "stac_path",
]
