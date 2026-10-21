"""Run identity and optional run-lock enforcement (issue #80).

Records the Git revision, container image digest, published data fingerprints,
and the resolved aggregation method beside a modeling run. Vertex workers may
also set ``VERTEX_SOURCE_SHA`` and ``VERTEX_IMAGE_DIGEST``; when ``run_lock``
expectations are configured, mismatches warn or abort before training reads
data.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.io import exists, read_bytes, run_context_path
from berlin_lst_downscaling.data.training.contracts import training_policy_hash
from berlin_lst_downscaling.data.training.paths import release_completion
from berlin_lst_downscaling.modeling.aggregation import resolve_aggregation_method
from berlin_lst_downscaling.modeling.patches import RealSourceConfig, patch_index_fingerprints

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunIdentity:
    """Resolved identity for one modeling invocation."""

    git_commit: str
    vertex_source_sha: str | None
    image_digest: str | None
    data_release_id: str
    training_policy_hash: str
    patch_index_fingerprints: dict[str, str]
    aggregation: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


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
    patch_fps: dict[str, str] = {}
    if source is not None:
        patch_fps = patch_index_fingerprints(source.patch_index_root)
    elif cfg.get("patch_index_root"):
        patch_fps = patch_index_fingerprints(str(cfg.patch_index_root))

    vertex_sha = os.environ.get("VERTEX_SOURCE_SHA")
    vertex_sha = vertex_sha.strip() if vertex_sha else None
    image = os.environ.get("VERTEX_IMAGE_DIGEST")
    image = image.strip() if image else None

    return RunIdentity(
        git_commit=_git_commit_from_context(output_root, pipeline, run_id),
        vertex_source_sha=vertex_sha,
        image_digest=image,
        data_release_id=str(cfg.get("data_release_id") or ""),
        training_policy_hash=_training_policy_from_root(
            str(cfg.training_root) if cfg.get("training_root") else None
        ),
        patch_index_fingerprints=patch_fps,
        aggregation=aggregation,
    )


def enforce_run_lock(cfg: DictConfig, identity: RunIdentity) -> list[str]:
    """Compare ``identity`` to optional ``run_lock.expect``; warn or raise."""
    lock = cfg.get("run_lock")
    if lock is None:
        return []
    mode = str(lock.get("mode") or "off").lower()
    if mode in ("off", "false", "none", ""):
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
        expect: dict[str, object] = {}
    mismatches: list[str] = []

    def _check(label: str, actual: str | None, expected: object) -> None:
        if expected is None:
            return
        exp = str(expected).strip()
        if not exp:
            return
        got = (actual or "").strip()
        if got != exp:
            mismatches.append(f"{label}: expected {exp!r}, got {got!r}")

    _check("git_commit", identity.git_commit, expect.get("git_commit"))
    _check("vertex_source_sha", identity.vertex_source_sha, expect.get("vertex_source_sha"))
    _check("image_digest", identity.image_digest, expect.get("image_digest"))
    _check("data_release_id", identity.data_release_id, expect.get("data_release_id"))
    _check(
        "training_policy_hash",
        identity.training_policy_hash,
        expect.get("training_policy_hash"),
    )
    _check("aggregation", identity.aggregation, expect.get("aggregation"))
    index_hash = identity.patch_index_fingerprints.get("index_policy_hash")
    _check("index_policy_hash", index_hash, expect.get("index_policy_hash"))
    source_hash = identity.patch_index_fingerprints.get("source_policy_hash")
    _check("source_policy_hash", source_hash, expect.get("source_policy_hash"))

    if not mismatches:
        return []
    message = "run_lock mismatch: " + "; ".join(mismatches)
    if mode == "error":
        raise ValueError(message)
    _logger.warning(message)
    return mismatches


def persist_run_identity(output_root: Path, identity: RunIdentity) -> Path:
    """Write ``run_identity.json`` under the run output root."""
    path = output_root / "run_identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_payload(), indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "RunIdentity",
    "enforce_run_lock",
    "persist_run_identity",
    "resolve_run_identity",
]
