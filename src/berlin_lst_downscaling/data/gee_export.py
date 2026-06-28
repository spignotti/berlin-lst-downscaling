"""GEE batch export orchestration — submit and monitor export tasks.

All functions assume GEE has been initialized.
"""

import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import ee  # type: ignore[attr-defined]
from omegaconf import DictConfig

from berlin_lst_downscaling.data.gee_client import get_aoi_from_cfg
from berlin_lst_downscaling.data.gee_scenes import (
    list_landsat_scenes,
    list_sentinel2_scenes,
    prepare_landsat_collection,
    prepare_landsat_export_lst,
    prepare_sentinel2_collection_wrapped,
    prepare_sentinel2_export,
)

# GEE stubs don't export ee.batch — use a runtime-compatible alias
GeeTask = Any


def export_scenes_by_year(
    cfg: DictConfig,
    source: str = "landsat",
    year: int | None = None,
    dry_run: bool = True,
) -> list[GeeTask]:
    """Orchestrate GEE batch export for a given source and year.

    Args:
        cfg: Pipeline config.
        source: ``"landsat"`` or ``"sentinel2"``.
        year: Specific year, or ``None`` for all years in the config range.
        dry_run: If ``True``, print planned actions without submitting tasks.

    Returns:
        List of submitted ``GeeTask`` objects (empty in dry-run mode).
    """
    if source == "landsat":
        return _export_landsat(cfg, year, dry_run)
    elif source == "sentinel2":
        return _export_sentinel2(cfg, year, dry_run)
    else:
        msg = f"Unknown source: {source!r}. Expected 'landsat' or 'sentinel2'."
        raise ValueError(msg)


# ── Landsat ──────────────────────────────────────────────────────────────────


def _export_landsat(
    cfg: DictConfig,
    year: int | None,
    dry_run: bool,
) -> list[GeeTask]:
    years = _resolve_years(cfg, year)
    bucket = cfg.ard.output.bucket
    region = get_aoi_from_cfg(cfg)
    max_pixels = cfg.ard.gee.max_pixels
    tasks: list[GeeTask] = []

    for y in years:
        print(f"\n{'='*60}")
        print(f"Landsat {y}" + (" [DRY-RUN]" if dry_run else ""))
        print(f"{'='*60}")

        raw = list_landsat_scenes(cfg, year=y)
        processed = prepare_landsat_collection(raw, cfg)

        n_raw = processed.size().getInfo()
        n = int(n_raw) if n_raw else 0
        print(f"  Scenes: {n}")

        if n == 0:
            continue

        # Build the scene list once, batch-fetch properties (no pixel data)
        scene_list = processed.toList(n)
        props_list: list = processed.select([]).toList(n).getInfo() or []
        for i in range(n):
            props = props_list[i].get("properties", {})
            scene_id = str(props.get("system:index", f"scene_{i}"))
            time_ms = props.get("system:time_start")
            date = (
                datetime.fromtimestamp(time_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
                if time_ms is not None
                else "unknown"
            )

            image = ee.Image(scene_list.get(i))

            prefix = f"{cfg.landsat.export.prefix}/{y}/{scene_id}_LST"

            # ── LST export (100m) ──
            lst_img = prepare_landsat_export_lst(image, cfg)
            _submit_or_dry_run(
                lst_img,
                description=f"ard_landsat_lst_{scene_id}",
                bucket=bucket,
                prefix=prefix,
                scale=cfg.landsat.export.scale_lst,
                region=region,
                max_pixels=max_pixels,
                crs=cfg.ard.crs,
                dry_run=dry_run,
                label=f"  [{date}] LST @ {cfg.landsat.export.scale_lst}m",
                tasks=tasks,
            )

    return tasks


# ── Sentinel-2 ───────────────────────────────────────────────────────────────


def _export_sentinel2(
    cfg: DictConfig,
    year: int | None,
    dry_run: bool,
) -> list[GeeTask]:
    years = _resolve_years(cfg, year)
    bucket = cfg.ard.output.bucket
    region = get_aoi_from_cfg(cfg)
    max_pixels = cfg.ard.gee.max_pixels
    tasks: list[GeeTask] = []

    for y in years:
        print(f"\n{'='*60}")
        print(f"Sentinel-2 {y}" + (" [DRY-RUN]" if dry_run else ""))
        print(f"{'='*60}")

        raw = list_sentinel2_scenes(cfg, year=y)
        processed = prepare_sentinel2_collection_wrapped(raw, cfg)

        n_raw = processed.size().getInfo()
        n = int(n_raw) if n_raw else 0
        print(f"  Scenes: {n}")

        if n == 0:
            continue

        # Build the scene list once, batch-fetch properties (no pixel data)
        scene_list = processed.toList(n)
        props_list: list = processed.select([]).toList(n).getInfo() or []
        for i in range(n):
            props = props_list[i].get("properties", {})
            scene_id = str(props.get("system:index", f"scene_{i}"))
            time_ms = props.get("system:time_start")
            date = (
                datetime.fromtimestamp(time_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
                if time_ms is not None
                else "unknown"
            )

            image = ee.Image(scene_list.get(i))

            export_img = prepare_sentinel2_export(image, cfg)
            prefix = f"{cfg.sentinel2.export.prefix}/{y}/{scene_id}"

            _submit_or_dry_run(
                export_img,
                description=f"ard_sentinel2_{scene_id}",
                bucket=bucket,
                prefix=prefix,
                scale=cfg.sentinel2.export.scale,
                region=region,
                max_pixels=max_pixels,
                crs=cfg.ard.crs,
                dry_run=dry_run,
                label=f"  [{date}] S2 @ {cfg.sentinel2.export.scale}m",
                tasks=tasks,
            )

    return tasks


# ── Task submission ──────────────────────────────────────────────────────────


def _submit_or_dry_run(
    image: ee.Image,
    *,
    description: str,
    bucket: str,
    prefix: str,
    scale: float,
    region: ee.Geometry,
    max_pixels: int,
    crs: str,
    dry_run: bool,
    label: str,
    tasks: list,
) -> None:
    """Submit a GEE export task or print a dry-run message."""
    if dry_run:
        print(f"{label}  → gs://{bucket}/{prefix}.tif")
        return

    task = ee.batch.Export.image.toCloudStorage(  # type: ignore[reportPrivateImportUsage]
        image=image,
        description=description,
        bucket=bucket,
        fileNamePrefix=prefix,
        scale=scale,
        crs=crs,
        region=region,
        maxPixels=max_pixels,
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    tasks.append(task)
    print(f"{label}  → SUBMITTED ({task.status()['id']})")


# ── Monitoring ───────────────────────────────────────────────────────────────


def monitor_tasks(
    tasks: list[GeeTask],
    poll_interval_sec: int = 30,
    timeout_min: int = 1440,  # 24h for 1400+ exports
) -> tuple[list[GeeTask], list[GeeTask]]:
    """Poll a list of export tasks until all complete or fail.

    Args:
        tasks: List of ``GeeTask`` objects.
        poll_interval_sec: Seconds between status checks.
        timeout_min: Maximum total wait time (default 24h).

    Returns:
        ``(completed_tasks, failed_tasks)``.
    """
    if not tasks:
        return [], []

    start_time = time.time()
    deadline = start_time + timeout_min * 60
    completed = []
    failed = []
    pending = list(tasks)

    while pending and time.time() < deadline:
        still_pending = []
        for t in pending:
            state = t.status().get("state", "UNKNOWN")
            if state == "COMPLETED":
                completed.append(t)
            elif state == "FAILED":
                err = t.status().get("error_message", "unknown")
                desc = t.status().get("description", "?")
                print(f"  FAILED [{desc}]: {err}")
                failed.append(t)
            elif state in ("CANCELED",):
                failed.append(t)
            else:
                still_pending.append(t)

        if still_pending:
            states = Counter(
                t.status().get("state", "UNKNOWN") for t in still_pending
            )
            elapsed = int(time.time() - start_time)
            print(
                f"  Pending: {len(still_pending)} | "
                f"States: {dict(states)} | "
                f"Elapsed: {elapsed}s"
            )
            time.sleep(poll_interval_sec)

        pending = still_pending

    if pending:
        elapsed = int(time.time() - start_time)
        print(f"  WARNING: {len(pending)} tasks still pending after "
              f"{elapsed // 60}m (timeout={timeout_min}min).")

    return completed, failed


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_years(cfg: DictConfig, year: int | None) -> list[int]:
    """Return a list of years to process."""
    if year is not None:
        return [year]
    return list(range(cfg.ard.time.start_year, cfg.ard.time.end_year + 1))
