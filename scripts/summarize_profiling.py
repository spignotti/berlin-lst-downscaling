#!/usr/bin/env python3
"""Render concise Notion Markdown from a validated profiling summary.json.

Reads the validated summary artifact and emits deterministic Markdown
for direct paste into a Notion page.

Usage::

    uv run python scripts/summarize_profiling.py \\
        --summary gs://berlin-lst-data/profiling/wb2c-1/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys

from berlin_lst_downscaling.data.io.storage import read_bytes


def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Notion Markdown from profiling summary")
    parser.add_argument("--summary", required=True, help="GCS URI to summary.json")
    args = parser.parse_args()

    try:
        raw = read_bytes(args.summary)
        summary = json.loads(raw)
    except Exception as exc:
        print(f"ERROR: Cannot read summary: {exc}", file=sys.stderr)
        return 1

    total = summary.get("total_assets", 0)
    hard = summary.get("hard_failures", 0)
    sc = summary.get("structural_checks", {})
    cc = summary.get("contract_checks", {})
    comp = summary.get("manifest_ledger_completeness", {})
    coverage = summary.get("dynamic_coverage", [])
    by_source = summary.get("by_source", {})
    by_partition = summary.get("by_partition", {})

    lines: list[str] = []
    lines.append("# WB2c-1 Data Profiling — Summary")
    lines.append("")
    lines.append(f"**Total assets:** {total}")
    lines.append(f"**Hard failures:** {hard}")
    lines.append("")

    lines.append("## Partition split")
    lines.append("")
    for part, count in sorted(by_partition.items()):
        lines.append(f"- {part}: {count}")
    lines.append("")

    lines.append("## Structural checks")
    lines.append("")
    lines.append(f"- COG exists: {_fmt_pct(sc.get('cog_exists', 0), total)}")
    lines.append(f"- COG valid (strict + grid): {_fmt_pct(sc.get('cog_valid', 0), total)}")
    lines.append(f"- STAC exists: {_fmt_pct(sc.get('stac_exists', 0), total)}")
    lines.append(f"- Provenance exists: {_fmt_pct(sc.get('provenance_exists', 0), total)}")
    lines.append(f"- Completion marker exists: {_fmt_pct(sc.get('completion_exists', 0), total)}")
    lines.append("")

    lines.append("## Contract checks")
    lines.append("")
    lines.append(f"- dtype mismatches: {cc.get('dtype_mismatches', 0)}")
    lines.append(f"- nodata mismatches: {cc.get('nodata_mismatches', 0)}")
    lines.append(f"- channel order mismatches: {cc.get('channel_order_mismatches', 0)}")
    lines.append(
        f"- prose description absent (writer limitation): "
        f"{cc.get('prose_description_absent', 0)}"
    )
    lines.append(
        f"- unit absent (writer limitation): {cc.get('unit_absent', 0)}"
    )
    lines.append("")

    if comp:
        lines.append("## Manifest↔Ledger completeness")
        lines.append("")
        lines.append(f"- Manifest keys: {comp.get('manifest_key_count', 0)}")
        lines.append(f"- Ledger keys (done): {comp.get('ledger_key_count', 0)}")
        lines.append(f"- Status: {'✓ OK' if comp.get('ok', False) else '✗ issues detected'}")
        for label, key in [
            ("Missing in ledger", "missing_in_ledger"),
            ("Extra in ledger", "extra_in_ledger"),
            ("Duplicates", "duplicate_keys"),
        ]:
            vals = comp.get(key, [])
            if vals:
                lines.append(f"- {label}: {', '.join(vals)}")
        lines.append("")

    if coverage:
        lines.append("## Dynamic coverage (per partition)")
        lines.append("")
        for c in coverage:
            lines.append(
                f"- {c.get('partition')}: found {c.get('found')}/{c.get('expected')} "
                f"({'✓ OK' if c.get('ok', False) else '✗ issues'})"
            )
        lines.append("")

    lines.append("## By source")
    lines.append("")
    for src, info in sorted(by_source.items()):
        lines.append(
            f"- {src}: {info['total']} assets, "
            f"{info['valid']} valid, "
            f"{info['failures']} failures"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Quantiles (p1–p99) are histogram-CDF estimates, not exact percentiles.")
    lines.append("- Aggregate mean/std are valid-pixel weighted across scenes per group.")
    lines.append("- Histogram underflow/overflow counts are reported explicitly.")
    lines.append("- Prose description and unit metadata are not persisted by the writer; "
                 "channel order, dtype, and nodata are validated against contracts.")
    lines.append("")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
