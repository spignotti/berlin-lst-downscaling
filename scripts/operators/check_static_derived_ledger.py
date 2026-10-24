#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fail-closed ledger gate for static derived releases (read-only).

Exits 0 only when the release ledger exists and every row has
``status == done``. Any missing ledger, missing row, or non-done row
prints ``FAIL`` and exits 1. Used by the v6 launchers after the remote
run so a partially failed release can never report SUCCESS.

Usage
-----
    uv run python scripts/operators/check_static_derived_ledger.py \\
        --derived-root gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2
"""

from __future__ import annotations

import argparse

from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
from berlin_lst_downscaling.data.secondary.paths import derived_ledger_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--derived-root", required=True, help="Static derived release root")
    args = parser.parse_args(argv)

    ledger_uri = derived_ledger_path(args.derived_root)
    try:
        led = SecondaryLedger.open(ledger_uri)
    except Exception as exc:
        print(f"FAIL: cannot open static derived ledger at {ledger_uri}: {exc}")
        return 1

    rows = led.table.to_pylist()
    if not rows:
        print(f"FAIL: static derived ledger at {ledger_uri} has no rows.")
        return 1

    bad = [
        f"{r.get('source')}/{r.get('period_or_vintage')}: status={r.get('status')} "
        f"run_id={r.get('run_id')} last_error={r.get('last_error')}"
        for r in rows
        if r.get("status") != "done"
    ]
    if bad:
        print(f"FAIL: static derived ledger not all done ({len(bad)}/{len(rows)}).")
        for line in bad:
            print(f"  - {line}")
        return 1

    print(f"OK: static derived ledger all done ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
