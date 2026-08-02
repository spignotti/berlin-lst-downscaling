#!/usr/bin/env python3
"""Isolated-scene dynamic pipeline runner.

Runs exactly one child process per scene via subprocess, keeping memory
bounded (~1.2 GB per scene). The parent owns the GCS run guard and the
child processes run without the guard; the shared ledger ensures
idempotency and resume-safety.

Usage:
    uv run python scripts/run_dynamic_isolated.py \\
        --manifest-uri gs://.../manifest.parquet \\
        --output-root gs://berlin-lst-data/dynamic/full \\
        --config-name full \\
        --years 2017 2025

    # 2026 inference:
    uv run python scripts/run_dynamic_isolated.py \\
        --manifest-uri gs://.../manifest.parquet \\
        --output-root gs://berlin-lst-data/dynamic/inference/2026 \\
        --config-name inference_2026 \\
        --years 2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Allow importing the manifest reader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dataclass
class SceneResult:
    scene_id: str
    ok: bool
    duration_s: float
    error: str | None = None


@dataclass
class RunSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[SceneResult] = field(default_factory=list)
    total_duration_s: float = 0.0


def load_scene_ids(manifest_uri: str, years: list[int] | None = None) -> list[str]:
    """Load scene IDs from manifest, filtered by year."""
    from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors

    report = load_landsat_anchors(manifest_uri, years=years)
    if not report.ok:
        raise RuntimeError(f"Manifest load failed: {report.errors}")
    return [s.scene_id for s in report.scenes]


def _all_sources_done(output_root: str, scene_id: str) -> bool:
    from berlin_lst_downscaling.data.dynamic.paths import ledger_path
    from berlin_lst_downscaling.data.io import exists
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    led = SecondaryLedger.open(ledger_path(output_root))
    for source in ("era5_land", "shadow_building", "shadow_vegetation"):
        row = led.get(f"{source}_{scene_id}", source, scene_id)
        if row is None or row.status != "done":
            return False
        output_ok = row.output_uri and exists(row.output_uri)
        completion_ok = row.completion_uri and exists(row.completion_uri)
        if not output_ok or not completion_ok:
            return False
    return True


def run_single_scene(
    scene_id: str,
    manifest_uri: str,
    output_root: str,
    config_name: str,
    dataset_role: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: int = 1800,
) -> SceneResult:
    """Run exactly one scene through run_dynamic.py as a subprocess."""
    script = Path(__file__).resolve().parent / "run_dynamic.py"
    cmd = [
        sys.executable,
        str(script),
        "--config-name",
        config_name,
        f"manifest_uri={manifest_uri}",
        f"output_root={output_root}",
        f"scene_ids=[{scene_id}]",
        "no_run_guard=true",
    ]
    if dataset_role:
        cmd.append(f"dataset_role={dataset_role}")
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(  # noqa: S603 — controlled script path
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        duration = time.perf_counter() - t0
        if result.returncode == 0:
            return SceneResult(scene_id=scene_id, ok=True, duration_s=duration)
        err_tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        return SceneResult(
            scene_id=scene_id,
            ok=False,
            duration_s=duration,
            error=f"exit {result.returncode}: {err_tail}",
        )
    except subprocess.TimeoutExpired as exc:
        stderr_tail = (exc.stderr or "")[-1000:]
        return SceneResult(
            scene_id=scene_id,
            ok=False,
            duration_s=time.perf_counter() - t0,
            error=f"timeout ({timeout_s}s); stderr tail: {stderr_tail}",
        )
    except Exception as exc:
        return SceneResult(
            scene_id=scene_id,
            ok=False,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated-scene dynamic pipeline runner",
    )
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-name", default="full")
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Year filter, e.g. 2017 2025",
    )
    parser.add_argument("--dataset-role", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip fully Done and validated scenes",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-child timeout in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scene list without executing",
    )
    args = parser.parse_args()

    print(f"[isolated] Loading manifest: {args.manifest_uri}", flush=True)
    scene_ids = load_scene_ids(args.manifest_uri, years=args.years)
    print(f"[isolated] {len(scene_ids)} scenes to process", flush=True)

    if args.dry_run:
        for sid in scene_ids:
            print(f"  {sid}")
        return 0

    if args.resume:
        try:
            before = len(scene_ids)
            scene_ids = [s for s in scene_ids if not _all_sources_done(args.output_root, s)]
            done_count = before - len(scene_ids)
            print(
                f"[isolated] Resume: {before} → {len(scene_ids)} scenes "
                f"({done_count} already done)",
                flush=True,
            )
        except Exception as exc:
            print(f"[isolated] Resume check failed: {exc}", flush=True)

    from berlin_lst_downscaling.data.dynamic.run_guard import (
        acquire_run_guard,
        release_run_guard,
    )

    run_id = f"iso-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    lease = acquire_run_guard(args.output_root, run_id)
    if lease is None:
        print("[isolated] ERROR: Cannot acquire run guard; another run is active.", flush=True)
        return 1

    summary = RunSummary(total=len(scene_ids))
    t_start = time.perf_counter()
    try:
        for i, scene_id in enumerate(scene_ids, 1):
            print(f"\n[isolated] [{i}/{len(scene_ids)}] {scene_id}", flush=True)
            result = run_single_scene(
                scene_id,
                args.manifest_uri,
                args.output_root,
                args.config_name,
                args.dataset_role,
                timeout_s=args.timeout_seconds,
            )
            summary.results.append(result)

            if result.ok:
                summary.succeeded += 1
                print(f"[isolated]   OK ({result.duration_s:.1f}s)", flush=True)
            else:
                summary.failed += 1
                print(f"[isolated]   FAILED ({result.duration_s:.1f}s): {result.error}", flush=True)

            if i % 10 == 0 or i == len(scene_ids):
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed * 60 if elapsed > 0 else 0.0
                print(
                    f"[isolated] Progress: {i}/{len(scene_ids)} "
                    f"({summary.succeeded} ok, {summary.failed} fail) "
                    f"rate={rate:.1f}/min elapsed={elapsed:.0f}s",
                    flush=True,
                )
    finally:
        release_run_guard(lease)

    summary.total_duration_s = time.perf_counter() - t_start

    print(f"\n{'=' * 60}", flush=True)
    print("[isolated] SUMMARY", flush=True)
    print(f"  Total:     {summary.total}", flush=True)
    print(f"  Succeeded: {summary.succeeded}", flush=True)
    print(f"  Failed:    {summary.failed}", flush=True)
    duration_min = summary.total_duration_s / 60
    print(
        f"  Duration:  {summary.total_duration_s:.0f}s ({duration_min:.1f}min)",
        flush=True,
    )

    if summary.failed > 0:
        print("\nFailed scenes:", flush=True)
        for r in summary.results:
            if not r.ok:
                print(f"  {r.scene_id}: {r.error}", flush=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    summary_path = Path(tempfile.gettempdir()) / f"isolated_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "output_root": args.output_root,
                "config_name": args.config_name,
                "total": summary.total,
                "succeeded": summary.succeeded,
                "failed": summary.failed,
                "duration_s": summary.total_duration_s,
                "failed_scenes": [
                    {"scene_id": r.scene_id, "error": r.error} for r in summary.results if not r.ok
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSummary saved: {summary_path}", flush=True)

    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
