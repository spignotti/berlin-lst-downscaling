#!/usr/bin/env python3
"""List all GEE export tasks for this project with their current state.

Uses the service-account init from ``berlin_lst_downscaling.data.gee_client``
(consistent with ``ard_list.py`` / ``ard_export.py`` / ``ard_monitor.py``).

Usage:
    uv run python scripts/ard_status.py
    uv run python scripts/ard_status.py show_all=true
"""

import sys
from collections import Counter

import ee
import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.gee_client import initialize


@hydra.main(version_base=None, config_path="../configs/ard", config_name="ard_status")
def main(cfg: DictConfig) -> None:
    """List GEE export tasks and their states."""
    initialize(cfg)
    show_all = bool(cfg.get("show_all", False))

    tasks = list(ee.batch.Task.list())
    states = Counter(t.status().get("state", "UNKNOWN") for t in tasks)

    print(f"Total tasks: {len(tasks)}")
    print(f"State summary: {dict(states)}")
    print()

    # Filter to show only relevant tasks (ard_ prefix)
    ard_tasks = [t for t in tasks if "ard_" in (t.status().get("description") or "")]

    relevant_states = {"READY", "RUNNING", "UNSUBMITTED"}
    if show_all:
        relevant_states = {"READY", "RUNNING", "UNSUBMITTED", "COMPLETED", "FAILED", "CANCELED"}

    for t in ard_tasks:
        status = t.status()
        state = status.get("state", "UNKNOWN")
        if state not in relevant_states:
            continue
        desc = status.get("description", "?")
        err = status.get("error_message", "")
        err_str = f"  ERROR: {err}" if err else ""
        print(f"  [{state:12s}] {desc}{err_str}")

    if not any(t.status().get("state") in relevant_states for t in ard_tasks):
        print("  (no matching tasks in relevant states)")

    # Non-zero exit if there are any failed tasks (helpful for monitoring)
    if states.get("FAILED", 0) > 0 and not show_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
