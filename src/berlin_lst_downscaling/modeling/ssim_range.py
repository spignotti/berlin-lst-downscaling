"""Fixed SSIM data range L from a committed, versioned, train-only artifact.

``C1 = (K1 * L)^2`` and ``C2 = (K2 * L)^2`` use one fixed ``L = p99.5 - p0.5``
of the valid 100 m targets of the train split (2017-2023). The value lives in
``configs/modeling/ssim_data_range/v<N>.yaml``; ``SSIM_DATA_RANGE_ARTIFACT``
names the active version. While that artifact is a placeholder every production
path raises; there is no fallback to the Kelvin plausibility range (250 K).

An artifact is bound to the patch index it was derived from (two fingerprints).
Loading it against a different index, or when either side lacks a fingerprint,
raises. The fix is a new version file written by the derivation script; a
published version is never overwritten. See
``configs/modeling/ssim_data_range/README.md``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.training.contracts import split_for_year

SSIM_RANGE_P_LOW: float = 0.5
SSIM_RANGE_P_HIGH: float = 99.5
SSIM_RANGE_PERCENTILE_METHOD: str = "linear"
SSIM_RANGE_SPLIT: str = "train"
SSIM_RANGE_TRAIN_YEARS: tuple[int, ...] = tuple(range(2017, 2024))
SSIM_DATA_RANGE_DIR: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "modeling" / "ssim_data_range"
)
# Module global, read at call time so tests can monkeypatch it.
SSIM_DATA_RANGE_ARTIFACT: Path = SSIM_DATA_RANGE_DIR / "v1.yaml"
SSIM_RANGE_DERIVATION_SCRIPT: str = "scripts/operators/derive_ssim_data_range.py"
SSIM_RANGE_DERIVATION_COMMAND: str = (
    "uv run python scripts/operators/derive_ssim_data_range.py --config-name v6_l5_locked"
)

_TOLERANCE_K = 1e-6
_FINGERPRINT_KEYS: tuple[str, ...] = ("index_policy_hash", "source_policy_hash")
_ARTIFACT_FILE_RE = re.compile(r"^v([1-9][0-9]*)\.yaml$")
_ARTIFACT_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")


def is_artifact_version(name: str) -> bool:
    """Return True for ``v<N>`` with N a positive integer without leading zeros."""
    return isinstance(name, str) and _ARTIFACT_VERSION_RE.fullmatch(name) is not None


def next_artifact_version(directory: Path | str) -> str:
    """Return the next free ``v<N>`` name from the ``v<N>.yaml`` files in ``directory``."""
    highest = 0
    root = Path(directory)
    if root.is_dir():
        for entry in root.iterdir():
            match = _ARTIFACT_FILE_RE.fullmatch(entry.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
    return f"v{highest + 1}"


def _missing_fingerprint(value: object) -> bool:
    return value is None or str(value).strip() in ("", "None")


def _mismatch_message(
    key: str, artifact_path: Path | str | None, version: str | None, artifact: str, run: str
) -> str:
    directory = Path(artifact_path).parent if artifact_path else SSIM_DATA_RANGE_DIR
    next_version = next_artifact_version(directory)
    return (
        f"SSIM data-range artifact {artifact_path} (version {version!r}) was derived on a "
        f"different patch index: {key} artifact={artifact!r} run={run!r}. The artifact is bound "
        f"to the patch index it was derived from. Derive a NEW artifact version (next: "
        f"{next_version}) with `{SSIM_RANGE_DERIVATION_COMMAND}`; never overwrite or reuse the "
        f"existing version. Then point SSIM_DATA_RANGE_ARTIFACT at the new file in a reviewed "
        "change."
    )


def _check_fingerprints(
    artifact_fp: Mapping[str, Any],
    run_fp: Mapping[str, Any],
    *,
    artifact_path: Path | str | None,
    version: str | None,
) -> None:
    """Fail closed: a missing fingerprint on either side never counts as a match."""
    for key in _FINGERPRINT_KEYS:
        for side, values in (("artifact", artifact_fp), ("run patch index", run_fp)):
            if _missing_fingerprint(values.get(key)):
                raise RuntimeError(
                    f"SSIM data-range fingerprint {key} is missing on the {side} side "
                    f"(artifact {artifact_path}, version {version!r}); a missing fingerprint "
                    "never counts as a match."
                )
        a, r = str(artifact_fp[key]), str(run_fp[key])
        if a != r:
            raise RuntimeError(_mismatch_message(key, artifact_path, version, a, r))


@dataclass(frozen=True)
class SsimRangeSample:
    """One train patch: 100 m target and validity mask."""

    patch_id: str
    scene_id: str
    split: str
    year: int
    target_100m: np.ndarray
    mask_100m: np.ndarray


@dataclass(frozen=True)
class SsimRangeStats:
    """Percentile range and the counts behind it."""

    low_k: float
    high_k: float
    data_range_k: float
    n_cells: int
    n_patches: int
    n_patches_empty: int
    n_nonfinite_masked_cells: int
    n_scenes: int
    years: tuple[int, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "low_k": self.low_k,
            "high_k": self.high_k,
            "data_range_k": self.data_range_k,
            "n_cells": self.n_cells,
            "n_patches": self.n_patches,
            "n_patches_empty": self.n_patches_empty,
            "n_nonfinite_masked_cells": self.n_nonfinite_masked_cells,
            "n_scenes": self.n_scenes,
            "years": list(self.years),
        }


def derive_ssim_data_range(
    samples: Iterable[SsimRangeSample],
    *,
    p_low: float = SSIM_RANGE_P_LOW,
    p_high: float = SSIM_RANGE_P_HIGH,
) -> SsimRangeStats:
    """Return the train-only percentile range of valid 100 m targets.

    Raises on any non-train sample (never skips). Valid values are held as
    float64: all train patches are about 5523 x 256 cells, roughly 11 MB.
    """
    chunks: list[np.ndarray] = []
    scenes: set[str] = set()
    years: set[int] = set()
    n_patches = 0
    n_empty = 0
    n_nonfinite = 0
    for sample in samples:
        if sample.split != SSIM_RANGE_SPLIT or split_for_year(sample.year) != SSIM_RANGE_SPLIT:
            raise ValueError(
                f"SSIM range derivation is train-only; patch {sample.patch_id!r} has "
                f"split={sample.split!r}, year={sample.year}"
            )
        target = np.asarray(sample.target_100m)
        mask = np.asarray(sample.mask_100m)
        if target.shape != mask.shape:
            raise ValueError(
                f"patch {sample.patch_id!r}: target shape {target.shape} != mask shape {mask.shape}"
            )
        masked = mask == 1
        finite = np.isfinite(target)
        n_nonfinite += int(np.count_nonzero(masked & ~finite))
        valid = masked & finite
        if not valid.any():
            n_empty += 1
            continue
        chunks.append(target[valid].astype(np.float64))
        n_patches += 1
        scenes.add(sample.scene_id)
        years.add(int(sample.year))
    if not chunks:
        raise ValueError("no valid train target cell: cannot derive the SSIM data range")
    values = np.concatenate(chunks)
    low, high = (float(v) for v in np.percentile(values, [p_low, p_high], method="linear"))
    data_range = high - low
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError(f"derived SSIM data range {data_range!r} is not finite and positive")
    return SsimRangeStats(
        low_k=low,
        high_k=high,
        data_range_k=data_range,
        n_cells=int(values.size),
        n_patches=n_patches,
        n_patches_empty=n_empty,
        n_nonfinite_masked_cells=n_nonfinite,
        n_scenes=len(scenes),
        years=tuple(sorted(years)),
    )


@dataclass(frozen=True)
class SsimDataRange:
    """Resolved SSIM data range with its provenance."""

    data_range_k: float
    low_k: float | None
    high_k: float | None
    source: str  # "artifact" | "test_override"
    version: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    index_policy_hash: str | None
    source_policy_hash: str | None
    derived_at: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "data_range_k": self.data_range_k,
            "low_k": self.low_k,
            "high_k": self.high_k,
            "source": self.source,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "index_policy_hash": self.index_policy_hash,
            "source_policy_hash": self.source_policy_hash,
            "derived_at": self.derived_at,
            "p_low": SSIM_RANGE_P_LOW,
            "p_high": SSIM_RANGE_P_HIGH,
            "percentile_method": SSIM_RANGE_PERCENTILE_METHOD,
            "clamp_range_k": list(LST_RANGE_K),
        }


def _is_positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_ssim_data_range(
    path: str | Path | None = None,
    *,
    expected_fingerprints: Mapping[str, str] | None = None,
) -> SsimDataRange:
    """Load and validate the committed SSIM data-range artifact (fail closed)."""
    artifact = Path(SSIM_DATA_RANGE_ARTIFACT if path is None else path)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"SSIM data-range artifact {artifact} is missing; derive it with "
            f"`{SSIM_RANGE_DERIVATION_COMMAND}` and commit it"
        )
    raw = artifact.read_bytes()
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict):
        raise RuntimeError(f"SSIM data-range artifact {artifact} is not a mapping")
    if (
        doc.get("status") != "filled"
        or doc.get("data_range_k") is None
        or doc.get("low_k") is None
        or doc.get("high_k") is None
    ):
        raise RuntimeError(
            f"SSIM data-range artifact {artifact} is a placeholder (status="
            f"{doc.get('status')!r}); run `{SSIM_RANGE_DERIVATION_COMMAND}` and commit the "
            "result. L5 must not start before the artifact is filled and committed."
        )
    if doc.get("split") != SSIM_RANGE_SPLIT:
        raise RuntimeError(
            f"SSIM data-range artifact split must be 'train', got {doc.get('split')!r}"
        )
    years = doc.get("years")
    if (
        not isinstance(years, list)
        or any(not isinstance(y, int) or isinstance(y, bool) for y in years)
        or years != list(SSIM_RANGE_TRAIN_YEARS)
    ):
        raise RuntimeError(
            f"SSIM data-range artifact years must be {list(SSIM_RANGE_TRAIN_YEARS)}, got {years!r}"
        )
    for key, expected in (
        ("p_low", SSIM_RANGE_P_LOW),
        ("p_high", SSIM_RANGE_P_HIGH),
        ("percentile_method", SSIM_RANGE_PERCENTILE_METHOD),
    ):
        if doc.get(key) != expected:
            raise RuntimeError(
                f"SSIM data-range artifact {key}={doc.get(key)!r} differs from {expected!r}"
            )
    data_range = doc["data_range_k"]
    low, high = doc["low_k"], doc["high_k"]
    if not _is_positive_finite(data_range):
        raise RuntimeError(f"SSIM data_range_k {data_range!r} is not finite and positive")
    if not (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and math.isfinite(low)
        and math.isfinite(high)
    ):
        raise RuntimeError(f"SSIM low_k/high_k {low!r}/{high!r} are not finite numbers")
    if abs((high - low) - data_range) > _TOLERANCE_K:
        raise RuntimeError(f"SSIM data_range_k {data_range!r} != high_k - low_k ({high - low!r})")
    if doc.get("version") != artifact.stem:
        raise RuntimeError(
            f"SSIM data-range artifact {artifact} declares version {doc.get('version')!r}, "
            f"which differs from its file stem {artifact.stem!r}"
        )
    fingerprints = doc.get("patch_index_fingerprints")
    fp = fingerprints if isinstance(fingerprints, dict) else {}
    if expected_fingerprints is not None:
        _check_fingerprints(
            fp,
            expected_fingerprints,
            artifact_path=artifact,
            version=None if doc.get("version") is None else str(doc["version"]),
        )
    return SsimDataRange(
        data_range_k=float(data_range),
        low_k=float(low),
        high_k=float(high),
        source="artifact",
        version=None if doc.get("version") is None else str(doc["version"]),
        artifact_path=str(artifact),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        index_policy_hash=None
        if fp.get("index_policy_hash") is None
        else str(fp["index_policy_hash"]),
        source_policy_hash=None
        if fp.get("source_policy_hash") is None
        else str(fp["source_policy_hash"]),
        derived_at=None if doc.get("derived_at") is None else str(doc["derived_at"]),
    )


def production_ssim_data_range_k() -> float:
    """Return the fixed production L; raises while the artifact is a placeholder."""
    return load_ssim_data_range().data_range_k


def test_override_ssim_data_range(k: float) -> SsimDataRange:
    """Return a test/smoke-only data range (never for real runs)."""
    if not _is_positive_finite(k):
        raise ValueError(f"SSIM data-range override must be finite and positive, got {k!r}")
    return SsimDataRange(
        data_range_k=float(k),
        low_k=None,
        high_k=None,
        source="test_override",
        version=None,
        artifact_path=None,
        artifact_sha256=None,
        index_policy_hash=None,
        source_policy_hash=None,
    )


def resolve_task_ssim_data_range(
    data_range_k: float | None, provenance: Mapping[str, Any] | None
) -> tuple[float, dict[str, Any]]:
    """Resolve L and its provenance for a task; re-check the artifact on checkpoint reload."""
    if provenance is None:
        if data_range_k is None:
            r = load_ssim_data_range()
        else:
            r = test_override_ssim_data_range(data_range_k)
        return r.data_range_k, r.payload()
    source = provenance.get("source")
    if source == "artifact":
        current = load_ssim_data_range()
        saved = provenance.get("artifact_sha256")
        if current.artifact_sha256 != saved:
            raise RuntimeError(
                "checkpoint was trained with a different SSIM data-range artifact: "
                f"checkpoint sha256={str(saved)[:12]} current sha256="
                f"{str(current.artifact_sha256)[:12]}"
            )
        if data_range_k is not None and abs(data_range_k - current.data_range_k) > _TOLERANCE_K:
            raise RuntimeError(
                f"ssim_data_range_k {data_range_k!r} differs from the artifact value "
                f"{current.data_range_k!r}"
            )
        return current.data_range_k, dict(provenance)
    if source == "test_override":
        if data_range_k is None or abs(data_range_k - float(provenance["data_range_k"])) > (
            _TOLERANCE_K
        ):
            raise RuntimeError(
                f"test_override SSIM data range mismatch: got {data_range_k!r}, provenance "
                f"{provenance.get('data_range_k')!r}"
            )
        return float(data_range_k), dict(provenance)
    raise RuntimeError(f"unknown SSIM data-range provenance source {source!r}")


def check_ssim_range_fingerprints(
    ssim_range: SsimDataRange, fingerprints: Mapping[str, str]
) -> None:
    """Raise if an artifact-sourced range was derived on a different patch index."""
    if ssim_range.source != "artifact":
        return
    _check_fingerprints(
        {key: getattr(ssim_range, key) for key in _FINGERPRINT_KEYS},
        fingerprints,
        artifact_path=ssim_range.artifact_path or SSIM_DATA_RANGE_ARTIFACT,
        version=ssim_range.version,
    )


# Not a test: stops pytest from collecting the helper when imported into a test module.
test_override_ssim_data_range.__test__ = False  # type: ignore[attr-defined]


def artifact_payload(
    stats: SsimRangeStats,
    *,
    patch_index_root: str,
    training_root: str,
    fingerprints: Mapping[str, str],
    config_name: str,
    git_commit: str,
    derived_at: str,
    exclusions: Mapping[str, int],
    n_refs_requested: int,
    version: str,
) -> dict[str, Any]:
    """Return the filled artifact document."""
    if not is_artifact_version(version):
        raise ValueError(f"artifact version must look like 'v<N>', got {version!r}")
    return {
        "schema_version": 1,
        "version": version,
        "status": "filled",
        "quantity": "ssim_data_range",
        "split": SSIM_RANGE_SPLIT,
        "years": list(stats.years),
        "p_low": SSIM_RANGE_P_LOW,
        "p_high": SSIM_RANGE_P_HIGH,
        "percentile_method": SSIM_RANGE_PERCENTILE_METHOD,
        "low_k": stats.low_k,
        "high_k": stats.high_k,
        "data_range_k": stats.data_range_k,
        "n_cells": stats.n_cells,
        "n_patches": stats.n_patches,
        "n_patches_empty": stats.n_patches_empty,
        "n_nonfinite_masked_cells": stats.n_nonfinite_masked_cells,
        "n_scenes": stats.n_scenes,
        "n_refs_requested": int(n_refs_requested),
        "exclusions": {str(k): int(v) for k, v in exclusions.items()},
        "patch_index_root": patch_index_root,
        "patch_index_fingerprints": {str(k): str(v) for k, v in fingerprints.items()},
        "training_root": training_root,
        "config_name": config_name,
        "script": SSIM_RANGE_DERIVATION_SCRIPT,
        "derivation_command": SSIM_RANGE_DERIVATION_COMMAND,
        "git_commit": git_commit,
        "derived_at": derived_at,
    }


def check_artifact_writable(path: Path | str) -> None:
    """Refuse an existing path or a stem that is not a ``v<N>`` version."""
    target = Path(path)
    if not is_artifact_version(target.stem):
        raise ValueError(
            f"SSIM data-range artifact file name must be 'v<N>.yaml', got {target.name!r}"
        )
    if target.exists():
        raise FileExistsError(
            f"SSIM data-range artifact {target} exists; published artifact versions are never "
            f"overwritten. Derive the next version ({next_artifact_version(target.parent)}) "
            "instead."
        )


def write_artifact(payload: dict[str, Any], path: Path | str) -> Path:
    """Create a new artifact version file locally (never remote, never overwriting)."""
    # Path("gs://x") collapses to "gs:/x", so check the raw string and that form.
    raw = str(path)
    if "://" in raw or raw.startswith("gs:/"):
        raise ValueError(f"SSIM data-range artifact must be written locally, got {str(path)!r}")
    out = Path(path)
    check_artifact_writable(out)
    if payload.get("version") != out.stem:
        raise ValueError(
            f"artifact file stem {out.stem!r} differs from payload version "
            f"{payload.get('version')!r}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Fixed SSIM data range L (train split only). Generated by\n"
        f"# {SSIM_RANGE_DERIVATION_SCRIPT}; immutable once published. A different patch\n"
        "# index needs a new version file; activate it via SSIM_DATA_RANGE_ARTIFACT.\n"
    )
    with out.open("x", encoding="utf-8") as handle:
        handle.write(header + yaml.safe_dump(payload, sort_keys=False))
    return out


__all__ = [
    "SSIM_DATA_RANGE_ARTIFACT",
    "SSIM_DATA_RANGE_DIR",
    "SSIM_RANGE_DERIVATION_COMMAND",
    "SSIM_RANGE_DERIVATION_SCRIPT",
    "SSIM_RANGE_PERCENTILE_METHOD",
    "SSIM_RANGE_P_HIGH",
    "SSIM_RANGE_P_LOW",
    "SSIM_RANGE_SPLIT",
    "SSIM_RANGE_TRAIN_YEARS",
    "SsimDataRange",
    "SsimRangeSample",
    "SsimRangeStats",
    "artifact_payload",
    "check_artifact_writable",
    "check_ssim_range_fingerprints",
    "derive_ssim_data_range",
    "is_artifact_version",
    "load_ssim_data_range",
    "next_artifact_version",
    "production_ssim_data_range_k",
    "resolve_task_ssim_data_range",
    "test_override_ssim_data_range",
    "write_artifact",
]
