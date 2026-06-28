#!/usr/bin/env python3
"""Monitor ARD GEE export tasks — standalone, no submission.

Lists all pending GEE batch tasks with descriptions starting with ``ard_``,
waits for completion, and reports results. Can also monitor specific tasks
by ID.

Usage:
    # Monitor all outstanding ARD tasks
    uv run python scripts/ard_monitor.py

    # Monitor specific tasks by ID
    uv run python scripts/ard_monitor.py task_ids=TASK123,TASK456

    # Dry run: list pending tasks without waiting
    uv run python scripts/ard_monitor.py dry_run=true
"""

import sys

import ee  # type: ignore[attr-defined]
import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.gee_client import initialize


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    initialize(cfg)
    dry_run = cfg.get("dry_run", False)
    task_ids_override = cfg.get("task_ids", None)
    timeout_min = cfg.ard.gee.get("task_timeout_min", 1440)

    if task_ids_override is not None:
        # Monitor specific tasks by ID
        ids = [tid.strip() for tid in str(task_ids_override).split(",")]
        print(f"Looking up {len(ids)} specific task(s)...")
        all_tasks = list(ee.batch.Task.list())  # type: ignore[attr-defined]
        tasks = [t for t in all_tasks if t.status().get("id") in ids]
        found = len(tasks)
        if found < len(ids):
            print(f"  WARNING: {len(ids) - found} task ID(s) not found on GEE.")
        if found == 0:
            print("  No matching tasks found.")
            sys.exit(1)
    else:
        # List all pending ARD-prefix tasks
        print("Listing pending ARD export tasks...")
        tasks = _list_pending_ard_tasks()

    if not tasks:
        print("No pending ARD tasks.")
        return

    if dry_run:
        print(f"\nPending tasks ({len(tasks)}):")
        for t in sorted(tasks, key=lambda t: t.status().get("description", "")):
            s = t.status()
            print(f"  [{s.get('state', '?')}] {s.get('id', '?')} — {s.get('description', '?')}")
        print("\nUse `dry_run=false` to monitor until completion.")
        return

    # Monitor to completion
    from berlin_lst_downscaling.data.gee_export import monitor_tasks

    print(f"\nMonitoring {len(tasks)} task(s) (timeout={timeout_min}min)...")
    completed, failed = monitor_tasks(
        tasks,
        poll_interval_sec=cfg.ard.gee.task_poll_interval_sec,
        timeout_min=timeout_min,
    )

    print(f"\n{'=' * 60}")
    print(f"  Completed: {len(completed)}, Failed: {len(failed)}")
    if failed:
        for t in failed:
            s = t.status()
            print(f"  FAILED: {s.get('id', '?')} — {s.get('description', '?')}")
            err = s.get("error_message", "unknown")
            if err != "unknown":
                print(f"    Reason: {err}")
        sys.exit(1)


def _list_pending_ard_tasks() -> list:
    """List all GEE tasks with ``ard_`` prefix that are not in terminal state."""
    terminal = {"COMPLETED", "FAILED", "CANCELED"}
    pending = []
    for t in ee.batch.Task.list():
        s = t.status()
        desc = s.get("description", "")
        state = s.get("state", "UNKNOWN")
        if desc.startswith("ard_") and state not in terminal:
            pending.append(t)
    return pending


if __name__ == "__main__":
    main()
