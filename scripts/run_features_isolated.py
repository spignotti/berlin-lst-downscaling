# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Isolated-scene feature-stack runner.

Runs exactly one child process per assessable scene via subprocess, so
each scene gets a fresh interpreter and memory is released between
scenes. A single full-grid scene peaks at ~7-8 GB (24-band composition +
COG write); running in-process accumulated that across scenes and
OOM-killed the VM (n2-highmem-2, 16 GB) at scene 4. The shared features
ledger (reconcile + config hash) keeps the driver idempotent and
resume-safe — a re-run skips scenes already published with a matching
config hash.

Usage
-----
    # Full: one child process per assessable scene (324), GCS output.
    uv run python scripts/run_features_isolated.py --config-name full

    # Resume a previous run (skip scenes already done + complete).
    uv run python scripts/run_features_isolated.py --config-name full --resume

    # Dry run: print the scene list without executing.
    uv run python scripts/run_features_isolated.py --config-name full --dry-run

Exit code is non-zero when any assessable scene fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from berlin_lst_downscaling.data.qa.inventory import build_inventory

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_CONFIG_DIR = _REPO_ROOT / "configs" / "features"

_logger = logging.getLogger("features.isolated")


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
    results: list[SceneResult] = field(default_factory=list)
    total_duration_s: float = 0.0


def load_config(config_name: str) -> OmegaConf:
    """Load the feature config: _base merged with the named config."""
    base = OmegaConf.load(_CONFIG_DIR / "_base.yaml")
    over = OmegaConf.load(_CONFIG_DIR / f"{config_name}.yaml")
    return OmegaConf.merge(base, over)


def assessable_scene_ids(cfg: OmegaConf) -> list[str]:
    """Return the assessable Landsat anchor scene ids (2017-2025)."""
    inventory = build_inventory(
        manifest_uri=str(cfg.manifest_uri),
        ard_root=str(cfg.ard_root),
        static_sources_root=str(cfg.static_sources_root),
        static_derived_root=str(cfg.static_derived_root),
        dynamic_root=str(cfg.dynamic_root),
        geometry_mapping_uri=str(cfg.geometry_mapping_uri),
    )
    if not inventory.ok:
        raise RuntimeError(f"Feature inventory failed: {inventory.errors}")
    return sorted(s.scene_id for s in inventory.scenes if s.assessable)


def _scene_done(output_root: str, scene_id: str) -> bool:
    """True when the scene's ledger row is done and its artifacts exist."""
    from berlin_lst_downscaling.data.features.paths import ledger_path
    from berlin_lst_downscaling.data.io import exists
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    led = SecondaryLedger.open(ledger_path(output_root))
    row = led.get(f"feature_{scene_id}", "feature_stack", scene_id)
    if row is None or row.status != "done":
        return False
    output_ok = row.output_uri and exists(row.output_uri)
    completion_ok = row.completion_uri and exists(row.completion_uri)
    return output_ok and completion_ok


def _coverage_summary(output_root: str, scene_ids: list[str]) -> dict:
    """Aggregate published-provenance coverage across done scenes.

    Reads each scene's provenance ``coverage`` dict; scenes without a
    published provenance are skipped (their status is already surfaced by
    the per-scene ledger check). Returns ``{total_px, inside_aoi_px,
    outside_aoi_px, feature_valid_px}`` plus per-scene rows for the
    sparse-support diagnostic.
    """
    from berlin_lst_downscaling.data.features.paths import ledger_path
    from berlin_lst_downscaling.data.io import read_bytes
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger

    led = SecondaryLedger.open(ledger_path(output_root))
    agg = {"total_px": 0, "inside_aoi_px": 0, "outside_aoi_px": 0, "feature_valid_px": 0}
    per_scene: list[dict] = []
    for scene_id in scene_ids:
        row = led.get(f"feature_{scene_id}", "feature_stack", scene_id)
        if row is None or row.status != "done" or not row.provenance_uri:
            continue
        try:
            prov = json.loads(read_bytes(row.provenance_uri))
        except Exception as exc:  # coverage summary is best-effort, never fatal
            _logger.warning("coverage summary: could not read provenance for %s: %s",
                            scene_id, exc)
            continue
        cov = prov.get("coverage", {})
        for key in agg:
            agg[key] += int(cov.get(key, 0))
        inside = int(cov.get("inside_aoi_px", 0))
        valid = int(cov.get("feature_valid_px", 0))
        fraction = (valid / inside) if inside else 0.0
        per_scene.append(
            {
                "scene_id": scene_id,
                "feature_valid_px": valid,
                "inside_aoi_px": inside,
                "fraction": round(fraction, 6),
            }
        )
    return {"aggregate": agg, "per_scene": per_scene}


_SPARSE_FRAC = 0.01  # non-gating diagnostic threshold: <1% of AOI valid


def run_single_scene(
    scene_id: str,
    cfg: OmegaConf,
    config_name: str,
    *,
    output_root: str,
    timeout_s: int = 3600,
) -> SceneResult:
    """Run exactly one scene through run_features.py as a subprocess.

    A child exit code of 0 is necessary but not sufficient: the scene is
    only reported successful when the shared ledger row is ``done`` with a
    COG and completion marker present. A child that exits 0 without
    publishing is treated as a failure (prevents a false-positive success
    when finalisation is skipped or incomplete).
    """
    shared_overrides = [
        f"manifest_uri={cfg.manifest_uri}",
        f"ard_root={cfg.ard_root}",
        f"static_sources_root={cfg.static_sources_root}",
        f"static_derived_root={cfg.static_derived_root}",
        f"dynamic_root={cfg.dynamic_root}",
        f"geometry_mapping_uri={cfg.geometry_mapping_uri}",
        f"aoi_mask_uri={cfg.aoi_mask_uri}",
        f"output_root={cfg.output_root}",
        f"vegetation_carry_forward_vintage={cfg.vegetation_carry_forward_vintage}",
        f"scene_ids=[{scene_id}]",
    ]
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_features.py"),
        "--config-name",
        config_name,
        *shared_overrides,
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(  # noqa: S603 — controlled script path
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(_REPO_ROOT),
        )
        duration = time.perf_counter() - t0
        if result.returncode == 0:
            if _scene_done(output_root, scene_id):
                return SceneResult(scene_id=scene_id, ok=True, duration_s=duration)
            return SceneResult(
                scene_id=scene_id,
                ok=False,
                duration_s=duration,
                error="child exited 0 but ledger row not done with COG + complete marker",
            )
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
        description="Isolated-scene feature-stack runner",
    )
    parser.add_argument("--config-name", default="full")
    parser.add_argument("--output-root", default=None, help="Override feature output root")
    parser.add_argument("--resume", action="store_true", help="Skip done + complete scenes")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true", help="Print scene list only")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N scenes (local driver validation)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config_name)
    if args.output_root:
        cfg.output_root = args.output_root
    output_root = str(cfg.output_root)

    print(f"[isolated] Loading assessable scenes from {cfg.manifest_uri}", flush=True)
    scene_ids = assessable_scene_ids(cfg)
    print(f"[isolated] {len(scene_ids)} assessable scenes", flush=True)

    if args.dry_run:
        for sid in scene_ids:
            print(f"  {sid}")
        return 0

    if args.resume:
        try:
            before = len(scene_ids)
            scene_ids = [s for s in scene_ids if not _scene_done(output_root, s)]
            print(
                f"[isolated] Resume: {before} → {len(scene_ids)} scenes "
                f"({before - len(scene_ids)} already done)",
                flush=True,
            )
        except Exception as exc:
            print(f"[isolated] Resume check failed: {exc}", flush=True)

    if args.limit is not None:
        scene_ids = scene_ids[: args.limit]
        print(f"[isolated] Limited to first {len(scene_ids)} scenes", flush=True)

    summary = RunSummary(total=len(scene_ids))
    t_start = time.perf_counter()
    for i, scene_id in enumerate(scene_ids, 1):
        print(f"\n[isolated] [{i}/{len(scene_ids)}] {scene_id}", flush=True)
        result = run_single_scene(
            scene_id,
            cfg,
            args.config_name,
            output_root=output_root,
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
            eta_min = (len(scene_ids) - i) / rate if rate > 0 else 0.0
            print(
                f"[isolated] Progress: {i}/{len(scene_ids)} "
                f"({summary.succeeded} ok, {summary.failed} fail) "
                f"rate={rate:.2f}/min eta={eta_min:.0f}min elapsed={elapsed:.0f}s",
                flush=True,
            )

    summary.total_duration_s = time.perf_counter() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print("[isolated] SUMMARY", flush=True)
    print(f"  Total:     {summary.total}", flush=True)
    print(f"  Succeeded: {summary.succeeded}", flush=True)
    print(f"  Failed:    {summary.failed}", flush=True)
    duration_min = summary.total_duration_s / 60
    print(f"  Duration:  {summary.total_duration_s:.0f}s ({duration_min:.1f}min)", flush=True)
    if summary.failed > 0:
        print("\nFailed scenes:", flush=True)
        for r in summary.results:
            if not r.ok:
                print(f"  {r.scene_id}: {r.error}", flush=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    summary_dir = f"{output_root.rstrip('/')}/logs/features"

    # ── published-provenance coverage summary (all assessable scenes) ──
    coverage = _coverage_summary(output_root, scene_ids)
    agg = coverage["aggregate"]
    print(f"\n[isolated] COVERAGE SUMMARY ({len(coverage['per_scene'])} published)", flush=True)
    for key in ("total_px", "inside_aoi_px", "outside_aoi_px", "feature_valid_px"):
        print(f"  {key}: {agg[key]}", flush=True)

    # ── non-gating sparse-support diagnostic ────────────────────────────
    sparse = [r for r in coverage["per_scene"] if r["fraction"] < _SPARSE_FRAC]
    print(f"\n[isolated] SPARSE SUPPORT BELOW {_SPARSE_FRAC:.0%} "
          f"({len(sparse)} scenes, non-gating diagnostic)", flush=True)
    for r in sparse:
        print(
            f"  {r['scene_id']}: feature_valid_px={r['feature_valid_px']} "
            f"inside_aoi_px={r['inside_aoi_px']} fraction={r['fraction']:.4%}",
            flush=True,
        )

    summary_uri = f"{summary_dir}/isolated_summary_{ts}.json"
    summary_data = {
        "output_root": output_root,
        "config_name": args.config_name,
        "total": summary.total,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "duration_s": summary.total_duration_s,
        "failed_scenes": [
            {"scene_id": r.scene_id, "error": r.error}
            for r in summary.results
            if not r.ok
        ],
        "coverage": agg,
        "sparse_support_below_1pct": sparse,
    }
    from berlin_lst_downscaling.data.io.storage import atomic_write

    atomic_write(summary_uri, json.dumps(summary_data, indent=2, default=str).encode())
    print(f"\nSummary saved: {summary_uri}", flush=True)

    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
