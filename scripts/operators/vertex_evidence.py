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
    """Read a JSON object, warning (never silently) on a missing/unreadable file."""
    value = _read_json_any(path)
    return value if isinstance(value, dict) else None


def _read_json_any(path: Path) -> object:
    """Read any JSON value, warning (never silently) on a missing/unreadable file."""
    if not path.is_file():
        print(f"WARNING: {path} not found", file=sys.stderr)
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


def _run_block(context: dict, source_sha: str) -> dict:
    return {
        "pipeline": context.get("pipeline"),
        "run_id": context.get("run_id"),
        # The image excludes .git, so the container's context git fields are
        # null; the operator-supplied source SHA is the authoritative revision.
        "git_commit": context.get("git_commit") or source_sha,
        "git_dirty": context.get("git_dirty"),
    }


def build_record(
    run_root: Path,
    *,
    run_label: str,
    source_sha: str,
    image_digest: str,
    result_file: Path | None,
    profile: str = "smoke",
) -> dict:
    """Build the compact, non-secret evidence record.

    ``profile="probe"`` retains the probe's cohort bounds, per-epoch metrics and
    summary — the numbers a go/no-go decision needs — instead of a raw stdout
    tail. ``profile="smoke"`` keeps the original bounded-tail record.
    """
    scope = _read_json(run_root / "data_scope.json") or {}
    context = _run_context(run_root)
    record: dict = {
        "run_label": run_label,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "generated_at": datetime.now(UTC).isoformat(),
        "run": _run_block(context, source_sha),
        "data_scope": {
            "mode": scope.get("mode"),
            "requested_per_split": scope.get("requested_per_split"),
            "patches_per_split": scope.get("patches_per_split"),
            "skipped_per_split": scope.get("skipped_per_split"),
            "scenes_per_split": scope.get("scenes_per_split"),
            "years_per_split": scope.get("years_per_split"),
            "exclusions": scope.get("exclusions"),
            "patch_ids": scope.get("patch_ids"),
        },
    }
    if profile == "probe":
        epochs = _read_json_any(run_root / "epoch_metrics.json")
        summary = _read_json(run_root / "probe_summary.json")
        record["profile"] = "probe"
        record["probe"] = {
            "epoch_metrics": epochs if isinstance(epochs, list) else [],
            "summary": summary or {},
            # A probe is only a valid learning signal if every requested epoch
            # ran; surface a short run explicitly rather than as a pass.
            "epochs_complete": isinstance(epochs, list)
            and bool(summary)
            and len(epochs) == int((summary or {}).get("max_epochs", -1)),
        }
    else:
        record["profile"] = "smoke"
        record["result_tail"] = _result_tail(result_file)
    return record


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
    parser.add_argument("--profile", choices=["smoke", "probe"], default="smoke")
    parser.add_argument("--result-file", type=Path, default=None)
    args = parser.parse_args()

    record = build_record(
        args.run_root,
        run_label=args.run_label,
        source_sha=args.source_sha,
        image_digest=args.image_digest,
        result_file=args.result_file,
        profile=args.profile,
    )
    payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8")
    _upload_create_only(args.evidence_uri, payload)
    print(f"evidence uploaded: {args.evidence_uri}")


if __name__ == "__main__":
    main()
