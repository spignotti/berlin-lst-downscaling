#!/usr/bin/env python3
"""Smoke-test runner for GCP + GEE access — one command, all 5 checks.

Runs in order and prints a PASS/FAIL/SKIP table:

  1. rclone mount       — is the bucket mounted at ``~/.mnt/berlin-lst/``?
  2. rclone CLI         — does ``rclone ls`` work against the bucket?
  3. gcloud CLI         — does ``gcloud storage ls`` work?
  4. Python GCS         — does ``google.cloud.storage`` work via ADC?
  5. Python GEE         — does ``ee.Initialize`` work via service account?

Missing binaries (e.g. ``rclone`` not installed) are reported as SKIP, not
FAIL — they don't block the other checks. Exit code is non-zero if any
check FAILS.

Usage:
    uv run python scripts/ard_smoke.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.gcs_client import list_blobs
from berlin_lst_downscaling.data.gee_client import initialize

_MOUNT_POINT = Path.home() / ".mnt" / "berlin-lst"
_BUCKET = "berlin-lst-data"
_RCLONE_REMOTE = "gcs-masterarbeit:berlin-lst-data"
_GCLOUD_LS = ["gcloud", "storage", "ls", f"gs://{_BUCKET}/", "--project=masterarbeit-berlin-lst-v2"]
_RCLONE_LS = ["rclone", "ls", _RCLONE_REMOTE, "--max-depth", "1"]
_TIMEOUT_SEC = 10


class Status(StrEnum):
    PASS = "PASS"  # noqa: S105 — status label, not a credential
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""


# ── Individual checks ─────────────────────────────────────────────────────────


def _check_mount() -> CheckResult:
    """Is the rclone mount active and populated?"""
    if not _MOUNT_POINT.exists():
        return CheckResult("rclone mount", Status.FAIL, f"{_MOUNT_POINT} not mounted")
    try:
        entries = list(_MOUNT_POINT.iterdir())
    except OSError as exc:
        return CheckResult("rclone mount", Status.FAIL, f"list failed: {exc}")
    if not entries:
        return CheckResult("rclone mount", Status.FAIL, "mount point empty")
    return CheckResult("rclone mount", Status.PASS, f"{len(entries)} top-level entries")


def _run_subprocess(cmd: list[str]) -> tuple[Status, str]:
    """Run a CLI check; return (status, one-line detail). Missing binary → SKIP."""
    if shutil.which(cmd[0]) is None:
        return Status.SKIP, f"{cmd[0]} not installed"
    try:
        result = subprocess.run(  # noqa: S603 — argv is hardcoded module-level constants
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Status.FAIL, f"timeout after {_TIMEOUT_SEC}s"
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip().splitlines()
        return Status.FAIL, (err[0][:120] if err else f"exit {result.returncode}")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return Status.PASS, f"{len(lines)} line(s)"


def _check_rclone_cli() -> CheckResult:
    status, detail = _run_subprocess(_RCLONE_LS)
    return CheckResult("rclone CLI", status, detail)


def _check_gcloud_cli() -> CheckResult:
    status, detail = _run_subprocess(_GCLOUD_LS)
    return CheckResult("gcloud CLI", status, detail)


def _check_python_gcs() -> CheckResult:
    try:
        names = list_blobs(_BUCKET, max_results=5)
    except Exception as exc:  # ADC not set, network down, etc.
        return CheckResult("Python GCS", Status.FAIL, f"{type(exc).__name__}: {exc}")
    if not names:
        return CheckResult("Python GCS", Status.FAIL, "no blobs returned")
    return CheckResult("Python GCS", Status.PASS, f"{len(names)} blob(s): {names[0]}")


def _check_python_gee(cfg: DictConfig) -> CheckResult:
    try:
        import ee  # imported here so the previous checks don't pay the import cost
    except ImportError as exc:
        return CheckResult("Python GEE", Status.FAIL, f"earthengine-api: {exc}")
    try:
        initialize(cfg)
        bands = ee.Image("USGS/SRTMGL1_003").bandNames().getInfo()
    except Exception as exc:
        return CheckResult("Python GEE", Status.FAIL, f"{type(exc).__name__}: {exc}")
    return CheckResult("Python GEE", Status.PASS, f"SRTM bands: {bands}")


# ── Orchestration ─────────────────────────────────────────────────────────────


_CHECKS = [
    _check_mount,
    _check_rclone_cli,
    _check_gcloud_cli,
    _check_python_gcs,
]


@hydra.main(version_base=None, config_path="../configs/ard", config_name="ard_status")
def main(cfg: DictConfig) -> None:
    """Run all smoke checks and print a final table."""
    print("Running GCP/GEE smoke tests…\n")

    results: list[CheckResult] = []
    for check in _CHECKS:
        result = check()
        results.append(result)
        _print_line(result)

    # GEE check needs the config; run after the others
    gee_result = _check_python_gee(cfg)
    results.append(gee_result)
    _print_line(gee_result)

    # Summary table
    print()
    print(f"{'Check':<16} {'Status':<6} {'Detail'}")
    print(f"{'-' * 16} {'-' * 6} {'-' * 40}")
    for r in results:
        print(f"{r.name:<16} {r.status.value:<6} {r.detail}")

    passed = sum(1 for r in results if r.status is Status.PASS)
    failed = sum(1 for r in results if r.status is Status.FAIL)
    skipped = sum(1 for r in results if r.status is Status.SKIP)
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        print("\n  Some checks FAILED. See the skill for manual debugging:")
        print("    cat .opencode/skills/google-access/SKILL.md")
        sys.exit(1)


def _print_line(result: CheckResult) -> None:
    """Print a single in-progress check line, ANSI-coloured if available."""
    colour = {Status.PASS: "\033[32m", Status.FAIL: "\033[31m", Status.SKIP: "\033[33m"}.get(
        result.status, ""
    )
    reset = "\033[0m" if colour else ""
    print(f"  {colour}{result.status.value:<6}{reset} {result.name}: {result.detail}")


if __name__ == "__main__":
    main()
