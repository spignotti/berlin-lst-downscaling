"""Assemble and upload the metadata-only Vertex acceptance evidence record.

Run inside the Vertex worker after a successful bounded smoke. The record is a
small JSON object — no checkpoints, credentials, environment values, or the
local run directory. It is uploaded create-only (``if_generation_match=0``) so a
retained evidence object is never overwritten.

Usage (worker-side, called by ``vertex_entrypoint.sh``):
    uv run python scripts/operators/vertex_evidence.py \
        --run-root <local-output-root> --evidence-uri gs://<bucket>/<prefix>/evidence.json \
        --run-label <label> --source-sha <sha> --image-digest sha256:<digest>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage

# Bounded, non-secret stdout tail kept as execution evidence.
_RESULT_TAIL_LINES = 20


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, warning (never silently) on an unreadable/corrupt file."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def _run_context(run_root: Path) -> dict:
    contexts = sorted((run_root / "logs" / "modeling").glob("*.context.json"))
    if not contexts:
        return {}
    return _read_json(contexts[-1]) or {}


def _result_tail(result_file: Path | None) -> list[str]:
    if result_file is None or not result_file.is_file():
        return []
    lines = result_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-_RESULT_TAIL_LINES:]


def build_record(
    run_root: Path,
    *,
    run_label: str,
    source_sha: str,
    image_digest: str,
    result_file: Path | None,
) -> dict:
    """Build the compact, non-secret evidence record."""
    scope = _read_json(run_root / "data_scope.json") or {}
    context = _run_context(run_root)
    return {
        "run_label": run_label,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "pipeline": context.get("pipeline"),
            "run_id": context.get("run_id"),
            "git_commit": context.get("git_commit"),
            "git_dirty": context.get("git_dirty"),
        },
        "data_scope": {
            "mode": scope.get("mode"),
            "patches_per_split": scope.get("patches_per_split"),
            "skipped_per_split": scope.get("skipped_per_split"),
            "exclusions": scope.get("exclusions"),
            "patch_ids": scope.get("patch_ids"),
        },
        "result_tail": _result_tail(result_file),
    }


def _upload_create_only(uri: str, payload: bytes) -> None:
    if not uri.startswith("gs://"):
        raise ValueError(f"evidence URI must be a gs:// path, got {uri!r}")
    bucket_name, _, object_name = uri[len("gs://") :].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"evidence URI is missing a bucket or object path: {uri!r}")
    blob = storage.Client().bucket(bucket_name).blob(object_name)
    blob.upload_from_string(
        payload, content_type="application/json", if_generation_match=0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--result-file", type=Path, default=None)
    args = parser.parse_args()

    record = build_record(
        args.run_root,
        run_label=args.run_label,
        source_sha=args.source_sha,
        image_digest=args.image_digest,
        result_file=args.result_file,
    )
    payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8")
    _upload_create_only(args.evidence_uri, payload)
    print(f"evidence uploaded: {args.evidence_uri}")


if __name__ == "__main__":
    main()
