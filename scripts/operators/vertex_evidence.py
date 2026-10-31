"""Assemble and upload the Vertex acceptance evidence record.

Run inside the Vertex worker after a successful bounded run. The record is a
small JSON object — no credentials, environment values, or the local run
directory. For the full profile it also retains the epoch curve, summary, test
scope, and a create-only reference to the selected checkpoint. It is uploaded
create-only (``if_generation_match=0``) so a retained evidence object is never
overwritten.

Usage (worker-side, called by ``vertex_entrypoint.sh``):
    uv run python scripts/operators/vertex_evidence.py \
        --run-root <local-output-root> --evidence-uri gs://<bucket>/<prefix>/evidence.json \
        --run-label <label> --source-sha <sha> --image-digest sha256:<digest>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.modeling.timing import timed_phase

# Bounded, non-secret stdout tail kept as execution evidence.
_RESULT_TAIL_LINES = 20


_logger = logging.getLogger(__name__)

_PATCH_CACHE_SCOPE_KEYS = (
    "patch_cache_source",
    "patch_cache_prefix",
    "patch_cache_key",
    "prebuilt_download_seconds",
    "prebuilt_verify_seconds",
    "prebuilt_load_seconds",
    "prebuilt_bytes",
    "prebuilt_shards",
    "prebuilt_commit_match",
    "prebuilt_build_commit",
    "prebuilt_running_commit",
)

_UNOBSERVABLE_PHASES = {
    "provision_to_container": "Vertex provisioning precedes the container; not observable inside",
    "platform_teardown": "Vertex teardown follows the container exit; not observable inside it",
    "evidence_upload": "its own duration cannot be part of the immutable evidence payload; "
    "it is logged to the vertex run log after the upload returns",
}


@contextmanager
def _timed_upload(phase: str, durations: dict[str, float]) -> Iterator[None]:
    """Time an upload with ``timed_phase`` and keep its duration for the record."""
    started = time.perf_counter()
    with timed_phase(phase, logger=_logger):
        yield
    durations[phase] = time.perf_counter() - started


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"WARNING: {path}:{number} is not valid JSON: {exc}", file=sys.stderr)
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _modeling_log(run_root: Path, context: dict) -> Path | None:
    logs = run_root / "logs" / "modeling"
    run_id = context.get("run_id")
    if run_id:
        candidate = logs / f"{run_id}.jsonl"
        return candidate if candidate.is_file() else None
    candidates = sorted(logs.glob("*.jsonl"))
    return candidates[-1] if candidates else None


def build_phase_timings(
    run_root: Path,
    context: dict,
    scope: dict,
    uploader: dict[str, float] | None,
) -> dict:
    """Collect wall-clock phases of the modeling run and this uploader invocation.

    Nested phases overlap and must not be summed. An interval that cannot be
    observed is ``null`` with a reason, never ``0``.
    """
    unavailable: dict[str, dict] = {
        name: {"duration_seconds": None, "reason": reason}
        for name, reason in _UNOBSERVABLE_PHASES.items()
    }
    timings: dict = {
        "available": False,
        "reason": None,
        "modeling_log": None,
        "phases": {},
        "epochs": [],
        "data_prep": {
            key: scope[key]
            for key in (
                "source_setup_seconds",
                "admission_seconds",
                "cache_build_seconds",
                "cache_build_workers",
                "cache_reused",
                *_PATCH_CACHE_SCOPE_KEYS,
            )
            if key in scope
        },
        "uploader": {name: seconds for name, seconds in (uploader or {}).items()},
        "unavailable": unavailable,
    }
    log = _modeling_log(run_root, context)
    if log is None:
        timings["reason"] = "modeling JSONL log not found"
        return timings
    timings["modeling_log"] = log.name
    for entry in _read_jsonl(log):
        event = entry.get("event")
        phase = entry.get("phase")
        if event == "phase_end" and isinstance(phase, str):
            timings["phases"][phase] = {
                key: entry.get(key)
                for key in ("started_at", "ended_at", "duration_seconds", "status")
            }
        elif event == "phase_unavailable" and isinstance(phase, str):
            unavailable[phase] = {"duration_seconds": None, "reason": entry.get("reason")}
        elif event == "epoch_timing":
            timings["epochs"].append(
                {
                    key: entry.get(key)
                    for key in ("epoch", "train_seconds", "validation_seconds", "epoch_seconds")
                }
            )
    timings["available"] = True
    return timings


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


def lr_schedule_from_summary(summary: dict | None) -> dict:
    """Return the recorded LR schedule; a summary without the field is constant."""
    recorded = (summary or {}).get("lr_schedule")
    if recorded is None:
        return {"name": "constant"}
    if not isinstance(recorded, dict) or recorded.get("name") not in ("constant", "cosine"):
        raise ValueError(
            f"summary lr_schedule must be a dict with name constant|cosine: {recorded!r}"
        )
    return recorded


def full_run_status(epochs: object, summary: dict | None) -> tuple[bool, bool]:
    """Return ``(epochs_complete, test_skipped)`` for a full-profile run.

    An early-stopped run is complete when the curve ends at the recorded
    ``stopped_epoch``. A skipped test pass is accepted only when the summary
    records ``evaluate_test=false`` in its resolved config; the two must agree.
    A summary without ``evaluate_test`` counts as ``evaluate_test=true``.
    """
    summary = summary or {}
    resolved = summary.get("resolved_config")
    evaluate_test = resolved.get("evaluate_test", True) if isinstance(resolved, dict) else True
    test_skipped = summary.get("test") == {"skipped": True}
    if test_skipped and evaluate_test is not False:
        raise ValueError("test pass skipped but evaluate_test=false is not recorded")
    if evaluate_test is False and not test_skipped:
        raise ValueError("evaluate_test=false is recorded but the test pass is not marked skipped")
    stopped = summary.get("stopped_epoch")
    max_epochs = int(summary.get("max_epochs", -1))
    if stopped is not None:
        early = summary.get("early_stopping")
        if (
            isinstance(stopped, bool)
            or not isinstance(stopped, int)
            or not 1 <= stopped <= max_epochs
            or not isinstance(early, dict)
            or early.get("enabled") is not True
        ):
            raise ValueError(
                f"stopped_epoch={stopped!r} needs early_stopping.enabled=true and "
                f"1 <= stopped_epoch <= max_epochs ({max_epochs})"
            )
    expected = max_epochs if stopped is None else stopped
    epochs_complete = isinstance(epochs, list) and bool(summary) and len(epochs) == expected
    return epochs_complete, test_skipped


def build_record(
    run_root: Path,
    *,
    run_label: str,
    source_sha: str,
    image_digest: str,
    result_file: Path | None,
    profile: str = "smoke",
    checkpoint: dict | None = None,
    efficiency_slot: int | None = None,
    efficiency_role: str | None = None,
    uploader_timings: dict[str, float] | None = None,
) -> dict:
    """Build the compact, non-secret evidence record.

    ``profile="probe"`` (historical #45) and ``profile="probe-residual"``
    (issue #47 recovery) retain the probe's cohort bounds, per-epoch metrics and
    summary — the numbers a go/no-go decision needs — instead of a raw stdout
    tail. ``profile="efficiency"`` retains bounded performance and learning
    evidence. ``profile="full"`` (issue #53) retains the full-run epoch curve,
    selection/reload summary, one-shot test scope, and the create-only
    checkpoint reference. ``profile="smoke"`` keeps the original bounded-tail
    record.
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
            "requested_patch_ids": scope.get("requested_patch_ids"),
            "patches_per_split": scope.get("patches_per_split"),
            "skipped_per_split": scope.get("skipped_per_split"),
            "skipped_refs": scope.get("skipped_refs"),
            "scenes_per_split": scope.get("scenes_per_split"),
            "years_per_split": scope.get("years_per_split"),
            "partial_mask_patches_per_split": scope.get("partial_mask_patches_per_split"),
            "exclusions": scope.get("exclusions"),
            "patch_ids": scope.get("patch_ids"),
            **{key: scope[key] for key in _PATCH_CACHE_SCOPE_KEYS if key in scope},
            "prebuilt_build_provenance": scope.get("prebuilt_build_provenance"),
        },
    }
    if profile in ("probe", "probe-residual"):
        epochs = _read_json_any(run_root / "epoch_metrics.json")
        summary = _read_json(run_root / "probe_summary.json")
        record["profile"] = profile
        record["probe"] = {
            "epoch_metrics": epochs if isinstance(epochs, list) else [],
            "summary": summary or {},
            "lr_schedule": lr_schedule_from_summary(summary),
            # A probe is only a valid learning signal if every requested epoch
            # ran; surface a short run explicitly rather than as a pass.
            "epochs_complete": isinstance(epochs, list)
            and bool(summary)
            and len(epochs) == int((summary or {}).get("max_epochs", -1)),
        }
    elif profile == "full":
        epochs = _read_json_any(run_root / "epoch_metrics.json")
        summary = _read_json(run_root / "full_summary.json")
        test_scope = _read_json(run_root / "test_scope.json")
        record["profile"] = "full"
        epochs_complete, test_skipped = full_run_status(epochs, summary)
        record["full"] = {
            "epoch_metrics": epochs if isinstance(epochs, list) else [],
            "summary": summary or {},
            "lr_schedule": lr_schedule_from_summary(summary),
            "test_scope": test_scope or {},
            # A full run is only decidable when every requested epoch ran (or
            # the recorded early stop was reached) and the selected checkpoint
            # plus its test scope were retained.
            "epochs_complete": epochs_complete,
        }
        if isinstance(summary, dict) and "evaluate_test" in summary:
            record["full"]["test_skipped"] = test_skipped
        if checkpoint is not None:
            record["checkpoint"] = checkpoint
    elif profile == "verification":
        verification = _read_json(run_root / "verification_results.json") or {}
        record["profile"] = "verification"
        record["verification"] = verification
        record["complete"] = verification.get("status") == "complete"
        record["test_access"] = False
        if checkpoint is not None:
            record["checkpoint"] = checkpoint
    elif profile == "efficiency":
        results = _read_json(run_root / "efficiency_results.json")
        learning_required = efficiency_role in ("baseline", "final")
        cache_required = efficiency_role in ("cache", "diagnostic", "final")
        cache_manifest = (
            _read_json(run_root / "cache_manifest.json")
            if (run_root / "cache_manifest.json").is_file()
            else None
        )
        learning_cache_manifest = (
            _read_json(run_root / "learning_cache_manifest.json")
            if (run_root / "learning_cache_manifest.json").is_file()
            else None
        )
        learning_root = run_root / "learning"
        learning_scope = (
            _read_json(learning_root / "data_scope.json") if learning_required else None
        )
        learning_epochs = (
            _read_json_any(learning_root / "epoch_metrics.json") if learning_required else None
        )
        learning_summary = (
            _read_json(learning_root / "probe_summary.json") if learning_required else None
        )
        learning_complete = (
            isinstance(learning_epochs, list)
            and bool(learning_summary)
            and len(learning_epochs) == int((learning_summary or {}).get("max_epochs", -1))
        )
        checkpoint_complete = (
            isinstance(checkpoint, dict)
            and str(checkpoint.get("uri", "")).endswith(".ckpt")
            and len(str(checkpoint.get("sha256", ""))) == 64
            and int(checkpoint.get("bytes", 0)) > 0
        )
        record["profile"] = "efficiency"
        record["efficiency"] = {
            "slot": efficiency_slot,
            "role": efficiency_role,
            "results": results or {},
            "complete": (
                isinstance(results, dict)
                and results.get("status") == "complete"
                and (not learning_required or learning_complete)
                and (not learning_required or checkpoint_complete)
                and (not cache_required or isinstance(cache_manifest, dict))
                and (efficiency_role != "final" or isinstance(learning_cache_manifest, dict))
            ),
            "test_access": False,
            "cache_manifest": cache_manifest or {},
            "learning_cache_manifest": learning_cache_manifest or {},
            "learning": {
                "data_scope": learning_scope or {},
                "epoch_metrics": learning_epochs if isinstance(learning_epochs, list) else [],
                "summary": learning_summary or {},
                "complete": learning_complete,
            },
        }
        if checkpoint is not None:
            record["checkpoint"] = checkpoint
    else:
        record["profile"] = "smoke"
        record["result_tail"] = _result_tail(result_file)
    record["phase_timings"] = build_phase_timings(run_root, context, scope, uploader_timings)
    return record


def _upload_checkpoint_create_only(path: Path, uri: str) -> dict:
    """Upload the selected checkpoint create-only and return its reference.

    The checkpoint is uploaded before the evidence manifest, so a manifest that
    references it can never precede it. ``if_generation_match=0`` keeps a
    retained checkpoint from being overwritten.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"checkpoint URI must be a gs:// path, got {uri!r}")
    data = path.read_bytes()
    bucket_name, _, object_name = uri[len("gs://") :].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"checkpoint URI is missing a bucket or object path: {uri!r}")
    blob = storage.Client().bucket(bucket_name).blob(object_name)
    blob.upload_from_string(data, content_type="application/octet-stream", if_generation_match=0)
    return {
        "uri": uri,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _upload_create_only(uri: str, payload: bytes) -> None:
    if not uri.startswith("gs://"):
        raise ValueError(f"evidence URI must be a gs:// path, got {uri!r}")
    bucket_name, _, object_name = uri[len("gs://") :].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"evidence URI is missing a bucket or object path: {uri!r}")
    blob = storage.Client().bucket(bucket_name).blob(object_name)
    blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)


def _upload_epoch_metrics(run_root: Path, uri: str, *, required: bool) -> None:
    """Upload ``epoch_metrics.json`` create-only, before the evidence manifest."""
    path = run_root / "epoch_metrics.json"
    if not path.is_file():
        if required:
            raise SystemExit(f"ERROR: epoch metrics not found: {path}")
        print(f"note: {path} not found; epoch curve not uploaded")
        return
    _upload_create_only(uri, path.read_bytes())
    print(f"epoch metrics uploaded: {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument(
        "--profile",
        choices=["smoke", "probe", "probe-residual", "full", "efficiency", "verification"],
        default="smoke",
    )
    parser.add_argument("--efficiency-slot", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument(
        "--efficiency-role", choices=["baseline", "cache", "diagnostic", "final"], default=None
    )
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="selected checkpoint to upload create-only before the evidence manifest",
    )
    parser.add_argument(
        "--checkpoint-uri",
        default=None,
        help="gs:// destination for --checkpoint-path (required with it)",
    )
    parser.add_argument(
        "--epoch-metrics-uri",
        default=None,
        help="gs:// destination for run-root/epoch_metrics.json (create-only)",
    )
    parser.add_argument(
        "--epoch-metrics-only",
        action="store_true",
        help="upload only the epoch curve (failed runs); requires --epoch-metrics-uri",
    )
    args = parser.parse_args()

    if args.epoch_metrics_only and not args.epoch_metrics_uri:
        raise SystemExit("ERROR: --epoch-metrics-only requires --epoch-metrics-uri")
    with RunLogSession(str(args.run_root), pipeline="vertex", run_id=uuid.uuid4().hex[:8]):
        _publish(args)


def _publish(args: argparse.Namespace) -> None:
    uploader: dict[str, float] = {}
    if args.epoch_metrics_uri:
        with _timed_upload("epoch_curve_upload", uploader):
            _upload_epoch_metrics(
                args.run_root, args.epoch_metrics_uri, required=not args.epoch_metrics_only
            )
        if args.epoch_metrics_only:
            return

    checkpoint = None
    if args.checkpoint_path is not None:
        if not args.checkpoint_uri:
            raise SystemExit("ERROR: --checkpoint-path requires --checkpoint-uri")
        if not args.checkpoint_uri.endswith(".ckpt"):
            raise SystemExit("ERROR: --checkpoint-uri must end in '.ckpt'")
        if not args.checkpoint_path.is_file():
            raise SystemExit(f"ERROR: checkpoint not found: {args.checkpoint_path}")
        # Upload the checkpoint before the manifest that references it.
        with _timed_upload("checkpoint_hash_upload", uploader):
            checkpoint = _upload_checkpoint_create_only(args.checkpoint_path, args.checkpoint_uri)
        print(f"checkpoint uploaded: {checkpoint['uri']} ({checkpoint['bytes']} bytes)")

    record = build_record(
        args.run_root,
        run_label=args.run_label,
        source_sha=args.source_sha,
        image_digest=args.image_digest,
        result_file=args.result_file,
        profile=args.profile,
        checkpoint=checkpoint,
        efficiency_slot=args.efficiency_slot,
        efficiency_role=args.efficiency_role,
        uploader_timings=uploader,
    )
    payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8")
    with timed_phase("evidence_upload", logger=_logger):
        _upload_create_only(args.evidence_uri, payload)
    print(f"evidence uploaded: {args.evidence_uri}")


if __name__ == "__main__":
    main()
