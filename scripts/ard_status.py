#!/usr/bin/env python3
"""List all GEE export tasks for this project with their current state.

Usage:
    uv run python scripts/ard_status.py
    uv run python scripts/ard_status.py --show-all    # include COMPLETED/FAILED
"""

import argparse
from collections import Counter

import ee


def main() -> None:
    parser = argparse.ArgumentParser(description="List GEE export tasks and their states.")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Include completed and failed tasks (default: only pending/running/ready)",
    )
    args = parser.parse_args()

    ee.Initialize(project="masterarbeit-berlin-lst-v2")

    tasks = ee.batch.Task.list()
    states = Counter(t.status().get("state", "UNKNOWN") for t in tasks)

    print(f"Total tasks: {len(tasks)}")
    print(f"State summary: {dict(states)}")
    print()

    # Filter to show only relevant tasks (ard_ prefix)
    ard_tasks = [t for t in tasks if "ard_" in (t.status().get("description") or "")]

    relevant_states = {"READY", "RUNNING", "UNSUBMITTED"}
    if args.show_all:
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


if __name__ == "__main__":
    main()
