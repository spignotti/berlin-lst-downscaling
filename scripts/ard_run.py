#!/usr/bin/env python3
"""ARD pipeline orchestrator — one entry point for the full pipeline.

Sub-commands:
  (default)   smoke     1 source × year × 1 scene per source, then visual QC
  plan        show what would run, no execution
  all         all sources × all years (cloud-ready)
  export      only stage 1: submit + wait for exports
  process     only stage 2: process all available scenes
  validate    only stage 3: visual QC on processed data
  doctor      infrastructure access check (alias for ard_smoke.py)
  boundary    refresh Berlin boundary from Geoportal WFS
  boundary    refresh Berlin boundary (Landesgrenze + 2 km buffer)

Typical usage:
    uv run python scripts/ard_run.py                  # smoke (default)
    uv run python scripts/ard_run.py plan             # show plan
    uv run python scripts/ard_run.py all              # full run
    uv run python scripts/ard_run.py --open           # open output in Finder
    uv run python scripts/ard_run.py --year 2024      # smoke a different year
    uv run python scripts/ard_run.py --source landsat # smoke one source only
    uv run python scripts/ard_run.py --verbose        # stream subprocess output
    uv run python scripts/ard_run.py boundary         # refresh Berlin boundary files
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMOKE_YEAR = 2023

# ── Command help text ────────────────────────────────────────────────────────

_COMMAND_HELP = {
    "plan":     "Show what would run, no execution.",
    "smoke":    "1 source × year × 1 scene per source, then visual QC.",
    "all":      "All sources × all years (cloud-ready).",
    "export":   "Stage 1 only: submit + wait for exports.",
    "process":  "Stage 2 only: process all available scenes (resume-aware).",
    "validate": "Stage 3 only: visual QC on processed data.",
    "doctor":   "Infrastructure access check (alias for ard_smoke.py).",
    "boundary": "Refresh Berlin boundary (Landesgrenze + 2 km buffer) from Geoportal WFS.",
}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI args and dispatch to the matching sub-command handler."""
    parser = argparse.ArgumentParser(
        prog="ard_run.py",
        description="ARD pipeline orchestrator — one entry point for the full pipeline.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for cmd, help_text in _COMMAND_HELP.items():
        sub.add_parser(cmd, help=help_text)

    parser.add_argument(
        "--year", type=int, default=None,
        help=f"Year (smoke default: {DEFAULT_SMOKE_YEAR}; other modes use config defaults).",
    )
    parser.add_argument(
        "--source", choices=["landsat", "sentinel2", "ecostress"], default=None,
        help="Single source (default: all 3).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open validation output dir in Finder at end (macOS only).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream subprocess output to terminal (default: status only).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress subprocess stdout (only show PASS/FAIL status).",
    )

    cli_args = parser.parse_args()
    command = cli_args.command or "smoke"
    handler = _HANDLERS[command]
    sys.exit(handler(cli_args))


# ── Sub-command handlers ─────────────────────────────────────────────────────


def _cmd_plan(args: argparse.Namespace) -> int:
    """Show export + process plan; no execution."""
    steps = [
        ("export plan",  ["uv", "run", "python", "scripts/ard_export.py",  "mode=plan"]),
        ("process plan", ["uv", "run", "python", "scripts/ard_process.py", "mode=plan"]),
    ]
    return _run_steps(steps, args.verbose, capture=args.quiet)


def _cmd_smoke(args: argparse.Namespace) -> int:
    """1 source × year × 1 scene per source, then visual QC.

    This is the default — the fast iteration loop used during dev.
    """
    year = args.year if args.year is not None else DEFAULT_SMOKE_YEAR
    source_args = [f"source={args.source}"] if args.source else []

    steps: list[tuple[str, list[str]]] = [
        ("access",   ["uv", "run", "python", "scripts/ard_smoke.py"]),
        ("export",   ["uv", "run", "python", "scripts/ard_export.py",
                      "mode=smoke", f"year={year}", *source_args]),
    ]
    # monitor is only meaningful for GEE sources (Landsat, S2). When the
    # user is iterating on ECOSTRESS alone, skip the GEE poll — there are
    # no GEE tasks to monitor and the step just adds Python startup time.
    if args.source != "ecostress":
        steps.append(("monitor", ["uv", "run", "python", "scripts/ard_monitor.py",
                                  "dry_run=false"]))
    steps.extend([
        ("process",  ["uv", "run", "python", "scripts/ard_process.py",
                      "mode=smoke", f"year={year}", *source_args]),
        ("validate", ["uv", "run", "python", "scripts/ard_smoke_validation.py",
                      "--year", str(year)]),
    ])
    last_output: Path | None = None
    for name, cmd in steps:
        rc, out = _run_step(name, cmd, verbose=args.verbose, capture=args.quiet)
        if rc != 0:
            return 1
        if name == "validate":
            last_output = _parse_output_dir(out)
    _finish_validation(last_output, open_after=args.open)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    """All sources × all years (cloud-ready)."""
    steps: list[tuple[str, list[str]]] = [
        ("access",  ["uv", "run", "python", "scripts/ard_smoke.py"]),
        ("export",  ["uv", "run", "python", "scripts/ard_export.py",  "mode=all"]),
        ("monitor", ["uv", "run", "python", "scripts/ard_monitor.py", "dry_run=false"]),
        ("process", ["uv", "run", "python", "scripts/ard_process.py", "mode=all"]),
    ]
    rc = _run_steps(steps, args.verbose, capture=args.quiet)
    if rc == 0:
        print(
            "\nFull run complete. Use `ard_run.py validate --year <latest>` for visual QC."
        )
    return rc


def _cmd_export(args: argparse.Namespace) -> int:
    """Stage 1 only: submit exports and wait for GEE tasks."""
    source_args = [f"source={args.source}"] if args.source else []
    year_args = [f"year={args.year}"] if args.year is not None else []
    steps: list[tuple[str, list[str]]] = [
        ("export",  ["uv", "run", "python", "scripts/ard_export.py",
                     "mode=all", *source_args, *year_args]),
    ]
    # Skip GEE monitor when only ECOSTRESS is selected — no GEE tasks to wait on.
    if args.source != "ecostress":
        steps.append(("monitor", ["uv", "run", "python", "scripts/ard_monitor.py",
                                  "dry_run=false"]))
    return _run_steps(steps, args.verbose, capture=args.quiet)


def _cmd_process(args: argparse.Namespace) -> int:
    """Stage 2 only: process all available scenes (resume-aware)."""
    source_args = [f"source={args.source}"] if args.source else []
    year_args = [f"year={args.year}"] if args.year is not None else []
    steps: list[tuple[str, list[str]]] = [
        ("process", ["uv", "run", "python", "scripts/ard_process.py",
                     "mode=all", *source_args, *year_args]),
    ]
    return _run_steps(steps, args.verbose, capture=args.quiet)


def _cmd_validate(args: argparse.Namespace) -> int:
    """Stage 3 only: visual QC on processed data."""
    year = args.year if args.year is not None else DEFAULT_SMOKE_YEAR
    rc, out = _run_step(
        "validate",
        ["uv", "run", "python", "scripts/ard_smoke_validation.py", "--year", str(year)],
        verbose=args.verbose,
        capture=args.quiet,
    )
    if rc != 0:
        return 1
    _finish_validation(_parse_output_dir(out), open_after=args.open)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Infrastructure access check (alias for ard_smoke.py)."""
    rc, _ = _run_step(
        "doctor", ["uv", "run", "python", "scripts/ard_smoke.py"],
        verbose=args.verbose,
        capture=args.quiet,
    )
    return rc


def _cmd_boundary(args: argparse.Namespace) -> int:
    """Refresh the Berlin boundary files from Geoportal Berlin WFS."""
    rc, _ = _run_step(
        "boundary", ["uv", "run", "python", "scripts/fetch_berlin_boundary.py"],
        verbose=args.verbose,
        capture=args.quiet,
    )
    return rc


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "plan":     _cmd_plan,
    "smoke":    _cmd_smoke,
    "all":      _cmd_all,
    "export":   _cmd_export,
    "process":  _cmd_process,
    "validate": _cmd_validate,
    "doctor":   _cmd_doctor,
    "boundary": _cmd_boundary,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_steps(
    steps: list[tuple[str, list[str]]], verbose: bool, capture: bool = False,
) -> int:
    """Run a sequence of steps; stop on first failure, return final exit code."""
    for name, cmd in steps:
        rc, _ = _run_step(name, cmd, verbose=verbose)
        if rc != 0:
            return 1
    return 0


def _run_step(
    name: str,
    cmd: list[str],
    *,
    verbose: bool = False,
    capture: bool = False,
) -> tuple[int, str]:
    """Run one pipeline step; print a one-line status; return (rc, stdout).

    By default (capture=False) the subprocess streams stdout+stderr live to
    the terminal so long waits (AppEEARS polls, GEE monitor) show progress.
    Pass ``capture=True`` (or ``--quiet``) to suppress subprocess output and
    just show the PASS/FAIL status line — useful for CI or chained scripts.
    """
    print(f"\n▶ {name}")
    if verbose:
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.monotonic()
    if not capture:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)  # noqa: S603
        stdout = ""
    else:
        result = subprocess.run(  # noqa: S603
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        stdout = result.stdout or ""
    elapsed = time.monotonic() - t0
    status = "PASS" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"  {status}  ({elapsed:.1f}s)")
    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"    {line}")
    return result.returncode, stdout


def _parse_output_dir(stdout: str) -> Path | None:
    """Extract the validation output dir from ``ard_smoke_validation.py`` stdout."""
    for line in stdout.splitlines():
        if "Output directory:" in line:
            return Path(line.split("Output directory:", 1)[1].strip().rstrip("/"))
    return None


def _finish_validation(output_dir: Path | None, *, open_after: bool) -> None:
    """Print the validation output dir and optionally open it in Finder."""
    if output_dir is None:
        print("\nValidation complete (no output dir found in stdout).")
        return
    print(f"\nValidation output: {output_dir}")
    if open_after and shutil.which("open"):
        subprocess.run(["open", str(output_dir)], check=False)  # noqa: S603,S607 — macOS only
    elif open_after:
        print("  (--open: 'open' command not available on this platform, skipping)")


if __name__ == "__main__":
    main()
