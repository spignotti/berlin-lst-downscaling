#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fail-closed ledger gate for static sources releases (read-only).

Exits 0 only when the release ledger exists and every row has
``status == done``. Any missing ledger, missing row, or non-done row
prints ``FAIL`` and exits 1. Used by the v6 launcher after the remote
run so a partially failed release can never report SUCCESS.

Usage
-----
    uv run python scripts/operators/check_static_sources_ledger.py \\
        --source-root gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2
"""

from __future__ import annotations

import argparse

from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", required=True, help="Static sources release root")
    args = parser.parse_args(argv)

    ledger_uri = f"{args.source_root.rstrip('/')}/ledger.parquet"
    try:
        led = SecondaryLedger.open(ledger_uri)
    except Exception as exc:
        print(f"FAIL: cannot open static sources ledger at {ledger_uri}: {exc}")
        return 1

    rows = led.table.to_pylist()
    if not rows:
        print(f"FAIL: static sources ledger at {ledger_uri} has no rows.")
        return 1

    bad = [
        f"{r.get('item_id')}: status={r.get('status')} "
        f"run_id={r.get('run_id')} last_error={r.get('last_error')}"
        for r in rows
        if r.get("status") != "done"
    ]
    if bad:
        print(f"FAIL: static sources ledger not all done ({len(bad)}/{len(rows)}).")
        for line in bad:
            print(f"  - {line}")
        return 1

    print(f"OK: static sources ledger all done ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
