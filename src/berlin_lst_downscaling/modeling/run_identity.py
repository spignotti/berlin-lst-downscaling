"""Run identity and optional run-lock enforcement (issue #80)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.io import exists, read_bytes, run_context_path
from berlin_lst_downscaling.data.training.contracts import training_policy_hash
from berlin_lst_downscaling.data.training.paths import (
    manifest_parquet,
    patch_index_completion,
    patch_index_qa,
    release_completion,
    scaler_json,
)
from berlin_lst_downscaling.modeling.aggregation import resolve_aggregation_method
from berlin_lst_downscaling.modeling.patches import RealSourceConfig, patch_index_fingerprints

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentFingerprint:
    """Bounded read-once SHA-256 over concrete small metadata artifacts."""

    artifacts: dict[str, str | None]
    composite_sha256: str
    provenance: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunIdentity:
    """Resolved identity for one modeling invocation."""

    git_commit: str
    vertex_source_sha: str | None
    image_digest: str | None
    data_release_id: str
    training_root: str | None
    patch_index_root: str | None
    training_policy_hash: str
    patch_index_fingerprints: dict[str, str]
    training_content_fingerprint: ContentFingerprint | None
    patch_index_content_fingerprint: ContentFingerprint | None
    patch_index_source_release_root: str | None
    aggregation: str
    seed: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.training_content_fingerprint is not None:
            payload["training_content_fingerprint"] = self.training_content_fingerprint.to_payload()
        if self.patch_index_content_fingerprint is not None:
            payload["patch_index_content_fingerprint"] = (
                self.patch_index_content_fingerprint.to_payload()
            )
        return payload


def _git_commit_from_context(output_root: str, pipeline: str, run_id: str) -> str:
    uri = run_context_path(output_root, pipeline, run_id)
    if uri.startswith("gs://"):
        if not exists(uri):
            return _git_commit_local()
        try:
            payload = json.loads(read_bytes(uri))
        except Exception:
            return _git_commit_local()
    else:
        path = Path(uri)
        if not path.is_file():
            return _git_commit_local()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return _git_commit_local()
    commit = str(payload.get("git_commit") or "").strip()
    return commit or _git_commit_local()


def _git_commit_local() -> str:
    try:
        result = subprocess.run(  # noqa: S607
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: S110 — best-effort git context
        pass
    return "unknown"


def _sha256_artifact(uri: str) -> str | None:
    if not uri or not exists(uri):
        return None
    return sha256_bytes(read_bytes(uri))


def training_content_fingerprint(training_root: str) -> ContentFingerprint:
    """Fingerprint completion marker, manifest, and scaler (read once each)."""
    artifacts = {
        "completion_marker": _sha256_artifact(release_completion(training_root)),
        "manifest_parquet": _sha256_artifact(manifest_parquet(training_root)),
        "scaler_json": _sha256_artifact(scaler_json(training_root)),
    }
    composite = sha256_bytes(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return ContentFingerprint(
        artifacts=artifacts,
        composite_sha256=composite,
        provenance="training release completion marker + manifest.parquet + scaler.json",
    )


def patch_index_content_fingerprint(patch_index_root: str) -> ContentFingerprint:
    """Fingerprint patch-index completion marker and QA report."""
    artifacts = {
        "completion_marker": _sha256_artifact(patch_index_completion(patch_index_root)),
        "patch_index_qa_json": _sha256_artifact(patch_index_qa(patch_index_root)),
    }
    composite = sha256_bytes(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return ContentFingerprint(
        artifacts=artifacts,
        composite_sha256=composite,
        provenance="patch-index completion marker + patch_index_qa.json",
    )


def _training_policy_from_root(training_root: str | None) -> str:
    if not training_root:
        return training_policy_hash()
    marker_uri = release_completion(training_root)
    if not exists(marker_uri):
        return training_policy_hash()
    try:
        marker = json.loads(read_bytes(marker_uri))
        published = str(marker.get("policy_hash") or "").strip()
        if published:
            return published
    except Exception:  # noqa: S110 — marker read is best-effort
        pass
    return training_policy_hash()


def _patch_index_source_root(patch_index_root: str) -> str | None:
    qa_uri = patch_index_qa(patch_index_root)
    if not exists(qa_uri):
        return None
    try:
        qa = json.loads(read_bytes(qa_uri))
        source = qa.get("source") or {}
        return str(source.get("release_root") or "").strip() or None
    except Exception:
        return None


def resolve_run_identity(
    cfg: DictConfig,
    *,
    run_id: str,
    output_root: str,
    pipeline: str = "modeling",
    source: RealSourceConfig | None = None,
) -> RunIdentity:
    """Resolve Git, image, data fingerprints, and aggregation from config + env."""
    aggregation = resolve_aggregation_method(cfg.get("aggregation"))
    training_root = str(cfg.training_root) if cfg.get("training_root") else None
    patch_index_root = str(cfg.patch_index_root) if cfg.get("patch_index_root") else None
    if source is not None:
        training_root = source.training_root
        patch_index_root = source.patch_index_root

    patch_fps: dict[str, str] = {}
    if patch_index_root:
        patch_fps = patch_index_fingerprints(patch_index_root)

    train_fp: ContentFingerprint | None = None
    index_fp: ContentFingerprint | None = None
    index_source: str | None = None
    if training_root:
        train_fp = training_content_fingerprint(training_root)
    if patch_index_root:
        index_fp = patch_index_content_fingerprint(patch_index_root)
        index_source = _patch_index_source_root(patch_index_root)
        if training_root and index_source and index_source.rstrip("/") != training_root.rstrip("/"):
            raise ValueError(
                f"patch-index QA source release {index_source!r} != training root {training_root!r}"
            )

    vertex_sha = os.environ.get("VERTEX_SOURCE_SHA")
    vertex_sha = vertex_sha.strip() if vertex_sha else None
    image = os.environ.get("VERTEX_IMAGE_DIGEST")
    image = image.strip() if image else None

    return RunIdentity(
        git_commit=_git_commit_from_context(output_root, pipeline, run_id),
        vertex_source_sha=vertex_sha,
        image_digest=image,
        data_release_id=str(cfg.get("data_release_id") or ""),
        training_root=training_root,
        patch_index_root=patch_index_root,
        training_policy_hash=_training_policy_from_root(training_root),
        patch_index_fingerprints=patch_fps,
        training_content_fingerprint=train_fp,
        patch_index_content_fingerprint=index_fp,
        patch_index_source_release_root=index_source,
        aggregation=aggregation,
        seed=_seed_from_config(cfg),
    )


def _seed_from_config(cfg: DictConfig) -> int | None:
    raw = cfg.get("seed")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"seed must be an int or null, got {raw!r}")
    return raw


def _active_expectations(expect: dict[str, object]) -> dict[str, str]:
    active: dict[str, str] = {}
    for key, value in expect.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            active[key] = text
    return active


def enforce_run_lock(cfg: DictConfig, identity: RunIdentity) -> list[str]:
    """Compare ``identity`` to optional ``run_lock.expect``.

    Without configured expectations, record identity only. When any expectation
    is set, mismatches abort by default; ``run_lock.mode: warn`` opts into warnings.
    Missing actual values for expected fields count as mismatches.
    """
    lock = cfg.get("run_lock")
    if lock is None:
        return []
    expect_raw = lock.get("expect") or {}
    if isinstance(expect_raw, DictConfig):
        expect = cast(
            dict[str, object],
            OmegaConf.to_container(expect_raw, resolve=True) or {},
        )
    elif isinstance(expect_raw, dict):
        expect = cast(dict[str, object], expect_raw)
    else:
        expect = {}
    active = _active_expectations(expect)
    if not active:
        return []

    mode = str(lock.get("mode") or "error").lower()
    if mode in ("off", "false", "none", ""):
        mode = "error"
    mismatches: list[str] = []

    def _check(label: str, actual: str | None, expected: str) -> None:
        got = (actual or "").strip()
        if not got:
            mismatches.append(f"{label}: expected {expected!r}, actual unavailable")
        elif got != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {got!r}")

    if "git_commit" in active:
        _check("git_commit", identity.git_commit, active["git_commit"])
    if "vertex_source_sha" in active:
        _check("vertex_source_sha", identity.vertex_source_sha, active["vertex_source_sha"])
    if "image_digest" in active:
        _check("image_digest", identity.image_digest, active["image_digest"])
    if "data_release_id" in active:
        _check("data_release_id", identity.data_release_id, active["data_release_id"])
    if "training_policy_hash" in active:
        _check(
            "training_policy_hash",
            identity.training_policy_hash,
            active["training_policy_hash"],
        )
    if "aggregation" in active:
        _check("aggregation", identity.aggregation, active["aggregation"])
    if "index_policy_hash" in active:
        index_hash = identity.patch_index_fingerprints.get("index_policy_hash")
        _check("index_policy_hash", index_hash, active["index_policy_hash"])
    if "source_policy_hash" in active:
        source_hash = identity.patch_index_fingerprints.get("source_policy_hash")
        _check("source_policy_hash", source_hash, active["source_policy_hash"])
    if "training_content_fingerprint" in active and identity.training_content_fingerprint:
        _check(
            "training_content_fingerprint",
            identity.training_content_fingerprint.composite_sha256,
            active["training_content_fingerprint"],
        )
    elif "training_content_fingerprint" in active:
        mismatches.append("training_content_fingerprint: expected value, actual unavailable")
    if "patch_index_content_fingerprint" in active and identity.patch_index_content_fingerprint:
        _check(
            "patch_index_content_fingerprint",
            identity.patch_index_content_fingerprint.composite_sha256,
            active["patch_index_content_fingerprint"],
        )
    elif "patch_index_content_fingerprint" in active:
        mismatches.append("patch_index_content_fingerprint: expected value, actual unavailable")

    if not mismatches:
        return []
    message = "run_lock mismatch: " + "; ".join(mismatches)
    if mode == "warn":
        _logger.warning(message)
        return mismatches
    raise ValueError(message)


def persist_run_identity(output_root: Path, identity: RunIdentity) -> Path:
    """Write ``run_identity.json`` under the run output root."""
    path = output_root / "run_identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_payload(), indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "ContentFingerprint",
    "RunIdentity",
    "enforce_run_lock",
    "patch_index_content_fingerprint",
    "persist_run_identity",
    "resolve_run_identity",
    "training_content_fingerprint",
]
