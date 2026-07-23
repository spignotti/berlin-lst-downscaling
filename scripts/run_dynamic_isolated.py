#!/usr/bin/env python3
"""Isolated-scene dynamic pipeline runner.

Runs exactly one child process per scene via subprocess, keeping memory
bounded (~1.2 GB per scene). The shared output root and ledger handle
idempotency — re-running is safe.

Usage:
    uv run python scripts/run_dynamic_isolated.py \
        --manifest-uri gs://.../manifest.parquet \
        --output-root gs://.../dynamic/full/<run-id> \
        --config-name full \
        --years 2017-2025

    # 2026 inference:
    uv run python scripts/run_dynamic_isolated.py \
        --manifest-uri gs://.../manifest.parquet \
        --output-root gs://.../dynamic/inference/2026/<run-id> \
        --config-name inference_2026 \
        --years 2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def run_single_scene(
    scene_id: str,
    manifest_uri: str,
    output_root: str,
    config_name: str,
    dataset_role: str | None = None,
    extra_args: list[str] | None = None,
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
            timeout=600,  # 10 min max per scene
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        duration = time.perf_counter() - t0
        if result.returncode == 0:
            return SceneResult(scene_id=scene_id, ok=True, duration_s=duration)
        else:
            # Extract last few lines of stderr for error context
            err_tail = "\n".join(result.stderr.strip().splitlines()[-5:])
            return SceneResult(
                scene_id=scene_id,
                ok=False,
                duration_s=duration,
                error=f"exit {result.returncode}: {err_tail}",
            )
    except subprocess.TimeoutExpired:
        return SceneResult(
            scene_id=scene_id,
            ok=False,
            duration_s=time.perf_counter() - t0,
            error="timeout (600s)",
        )
    except Exception as e:
        return SceneResult(
            scene_id=scene_id,
            ok=False,
            duration_s=time.perf_counter() - t0,
            error=str(e),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated-scene dynamic pipeline runner")
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-name", default="full")
    parser.add_argument(
        "--years", nargs="*", type=int, default=None, help="Year range, e.g. 2017 2025"
    )
    parser.add_argument("--dataset-role", default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Skip scenes with done status in ledger"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print scene list without executing")
    args = parser.parse_args()

    # Load scene list
    print(f"[isolated] Loading manifest: {args.manifest_uri}", flush=True)
    scene_ids = load_scene_ids(args.manifest_uri, years=args.years)
    print(f"[isolated] {len(scene_ids)} scenes to process", flush=True)

    if args.dry_run:
        for sid in scene_ids:
            print(f"  {sid}")
        return 0

    # Optional: filter by ledger status for resume
    if args.resume:
        try:
            from berlin_lst_downscaling.data.dynamic.paths import ledger_path
            from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

            led = SecondaryLedger.open(ledger_path(args.output_root))
            done = {
                r.item_id.replace("era5_land_", "")
                for r in led.items_for_source("era5_land")
                if r.status == "done"
            }
            before = len(scene_ids)
            scene_ids = [s for s in scene_ids if s not in done]
            print(
                f"[isolated] Resume: {before} → {len(scene_ids)} scenes "
                f"({before - len(scene_ids)} already done)",
                flush=True,
            )
        except Exception as e:
            print(f"[isolated] Resume check failed: {e}", flush=True)

    # Run each scene in an isolated process
    summary = RunSummary(total=len(scene_ids))
    t_start = time.perf_counter()

    for i, scene_id in enumerate(scene_ids, 1):
        print(f"\n[isolated] [{i}/{len(scene_ids)}] {scene_id}", flush=True)
        result = run_single_scene(
            scene_id,
            args.manifest_uri,
            args.output_root,
            args.config_name,
            args.dataset_role,
        )
        summary.results.append(result)

        if result.ok:
            summary.succeeded += 1
            print(f"[isolated]   OK ({result.duration_s:.1f}s)", flush=True)
        else:
            summary.failed += 1
            print(f"[isolated]   FAILED ({result.duration_s:.1f}s): {result.error}", flush=True)

        # Progress every 10 scenes
        if i % 10 == 0 or i == len(scene_ids):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed * 60
            print(
                f"[isolated] Progress: {i}/{len(scene_ids)} "
                f"({summary.succeeded} ok, {summary.failed} fail) "
                f"rate={rate:.1f}/min elapsed={elapsed:.0f}s",
                flush=True,
            )

    summary.total_duration_s = time.perf_counter() - t_start

    # Final summary
    print(f"\n{'=' * 60}", flush=True)
    print("[isolated] SUMMARY", flush=True)
    print(f"  Total:     {summary.total}", flush=True)
    print(f"  Succeeded: {summary.succeeded}", flush=True)
    print(f"  Failed:    {summary.failed}", flush=True)
    print(
        f"  Duration:  {summary.total_duration_s:.0f}s ({summary.total_duration_s / 60:.1f}min)",
        flush=True,
    )

    if summary.failed > 0:
        print("\nFailed scenes:", flush=True)
        for r in summary.results:
            if not r.ok:
                print(f"  {r.scene_id}: {r.error}", flush=True)

    # Write summary JSON
    import tempfile

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
