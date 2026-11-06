"""Post-run provenance gate for VM launchers (RF and naive baseline).

Rejects a run whose report or run context does not record the deployed git SHA
on a clean tree. Stdlib only; run with ``python3``.

Usage:
    python3 scripts/validators/validate_run_provenance.py \
        --report REPORT.json --expected-sha SHA --context-dir DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _fail(reason: str) -> int:
    print(f"FAIL: {reason}")
    return 1


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--context-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    expected: str = args.expected_sha
    if not _SHA_RE.fullmatch(expected):
        return _fail(f"expected sha is not 40 lowercase hex characters: {expected!r}")

    try:
        report = _load_json(args.report)
    except (OSError, ValueError) as exc:
        return _fail(f"cannot read report {args.report}: {exc}")
    if not isinstance(report, dict):
        return _fail(f"report {args.report} is not a JSON object")
    revision = report.get("git_revision")
    if not isinstance(revision, str) or not revision:
        return _fail("report git_revision missing or empty")
    if revision != expected:
        return _fail(f"report git_revision {revision!r} != expected {expected}")

    if not args.context_dir.is_dir():
        return _fail(f"context dir not found: {args.context_dir}")
    contexts = sorted(args.context_dir.glob("*.context.json"))
    if len(contexts) != 1:
        return _fail(
            f"expected exactly one *.context.json in {args.context_dir}, found {len(contexts)}"
        )
    try:
        context = _load_json(contexts[0])
    except (OSError, ValueError) as exc:
        return _fail(f"cannot read context {contexts[0]}: {exc}")
    if not isinstance(context, dict):
        return _fail(f"context {contexts[0]} is not a JSON object")
    if context.get("git_commit") != expected:
        return _fail(f"context git_commit {context.get('git_commit')!r} != expected {expected}")
    if context.get("git_dirty") is not False:
        return _fail(f"context git_dirty is {context.get('git_dirty')!r}, expected False")

    print(f"OK: provenance {expected} clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
