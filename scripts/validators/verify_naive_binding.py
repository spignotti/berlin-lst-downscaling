"""Verify bound naive baseline evidence before an RF run starts (fail closed, offline).

The expected dataset fingerprints come from the committed SSIM data-range pin
(``configs/modeling/ssim_data_range/v1.yaml``); the pin's ``patch_index_root`` must equal the
composed config's ``patch_index_root``. No GCS read happens here.

``--sha256`` is the sha256 of ``naive_evidence_manifest.sha256`` next to ``--report``
(which must be named ``baseline_report.json``). The manifest is ``sha256sum`` output with exactly
3 entries: baseline_report.json, validation.txt and one logs/baseline/<name>.context.json.
Every covered file is re-hashed (also after the provenance check); no unlisted context may exist.

Usage:
    uv run --locked python scripts/validators/verify_naive_binding.py \
        --report PATH --sha256 HEX --config-name random_forest_v6_val_smoke \
        --expected-splits validation
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling import ssim_range

NAIVE_METHOD = "naive_prior_expand"
FINGERPRINT_KEYS = ("index_policy_hash", "source_policy_hash", "patches_total")
MANIFEST_NAME = "naive_evidence_manifest.sha256"
REPORT_NAME = "baseline_report.json"
VALIDATION_NAME = "validation.txt"
CONTEXT_PREFIX = "logs/baseline/"
CONTEXT_SUFFIX = ".context.json"
_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"
_PROVENANCE_SCRIPT = Path(__file__).resolve().parent / "validate_run_provenance.py"


class BindingError(Exception):
    """The bound naive evidence does not match the configured dataset or is not trustworthy."""


def expected_fingerprints(
    config_name: str, *, config_dir: Path = _CONFIG_DIR, pin_path: Path | None = None
) -> dict[str, str]:
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    root = str(cfg.patch_index_root)
    pin = yaml.safe_load(Path(pin_path or ssim_range.SSIM_DATA_RANGE_ARTIFACT).read_text())
    if not isinstance(pin, dict):
        raise BindingError("SSIM pin is not a mapping")
    if pin.get("patch_index_root") != root:
        raise BindingError(f"pin root {pin.get('patch_index_root')} != config root {root}")
    fps = pin.get("patch_index_fingerprints")
    if not isinstance(fps, dict):
        raise BindingError("SSIM pin has no patch_index_fingerprints")
    out: dict[str, str] = {}
    for key in FINGERPRINT_KEYS:
        value = str(fps.get(key, "")).strip()
        if not value:
            raise BindingError(f"SSIM pin fingerprint {key} is empty")
        out[key] = value
    return out


def _run_provenance(report_path: Path, revision: str) -> bool:
    spec = importlib.util.spec_from_file_location("_validate_run_provenance", _PROVENANCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise BindingError("cannot load validate_run_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main(
        [
            "--report",
            str(report_path),
            "--expected-sha",
            revision,
            "--context-dir",
            str(report_path.parent / "logs" / "baseline"),
        ]
    )
    return rc == 0


def parse_manifest(data: bytes, evidence_dir: Path) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BindingError("manifest is not UTF-8") from exc
    root = evidence_dir.resolve()
    entries: dict[str, str] = {}
    for line in text.splitlines():
        m = _MANIFEST_LINE_RE.fullmatch(line)
        if m is None:
            raise BindingError(f"malformed manifest line {line!r}")
        name = m.group(2)
        posix = PurePosixPath(name)
        if (
            "\\" in name
            or posix.is_absolute()
            or ".." in posix.parts
            or not (evidence_dir / name).resolve().is_relative_to(root)
        ):
            raise BindingError(f"manifest entry {name!r} points outside the evidence dir")
        if name in entries:
            raise BindingError(f"duplicate manifest entry {name!r}")
        entries[name] = m.group(1)
    return entries


def _required_entries(entries: Mapping[str, str]) -> str:
    contexts = [
        k
        for k in entries
        if k.startswith(CONTEXT_PREFIX) and k.endswith(CONTEXT_SUFFIX) and k.count("/") == 2
    ]
    if len(contexts) != 1:
        raise BindingError(
            f"manifest must list exactly one provenance context file, found {len(contexts)}"
        )
    expected = {REPORT_NAME, VALIDATION_NAME, contexts[0]}
    if set(entries) != expected:
        raise BindingError(f"manifest entries {sorted(entries)} != required {sorted(expected)}")
    return contexts[0]


def _verify_entries(evidence_dir: Path, entries: Mapping[str, str]) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for rel, exp in entries.items():
        path = evidence_dir / rel
        if not path.is_file():
            raise BindingError(f"covered file missing: {rel}")
        data = path.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != exp:
            raise BindingError(f"covered file changed: {rel} sha256 {got} != manifest {exp}")
        blobs[rel] = data
    return blobs


def _check_no_unlisted_contexts(evidence_dir: Path, ctx_rel: str) -> None:
    ctx_dir = evidence_dir / "logs" / "baseline"
    on_disk = sorted(p.name for p in ctx_dir.glob("*" + CONTEXT_SUFFIX))
    if on_disk != [PurePosixPath(ctx_rel).name]:
        raise BindingError(f"unlisted provenance context file(s) in logs/baseline: {on_disk}")


def check_binding(
    report_path: Path,
    sha256: str,
    expected: Mapping[str, str],
    expected_splits: Sequence[str],
) -> dict[str, object]:
    if not _SHA256_RE.fullmatch(sha256):
        raise BindingError("bound sha256 must be 64 lowercase hex")
    if report_path.name != REPORT_NAME:
        raise BindingError(f"report must be named {REPORT_NAME}, got {report_path.name}")
    if not report_path.is_file():
        raise BindingError(f"naive evidence not found: {report_path}")
    evidence_dir = report_path.parent
    manifest = evidence_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise BindingError(f"evidence manifest missing: {manifest}")
    manifest_bytes = manifest.read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != sha256:
        raise BindingError(f"manifest sha256 {actual} != bound {sha256}")
    entries = parse_manifest(manifest_bytes, evidence_dir)
    ctx = _required_entries(entries)
    blobs = _verify_entries(evidence_dir, entries)
    _check_no_unlisted_contexts(evidence_dir, ctx)
    try:
        report = json.loads(blobs[REPORT_NAME].decode("utf-8"))
    except ValueError as exc:
        raise BindingError(f"naive evidence is not valid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise BindingError("naive evidence is not a JSON object")
    if report.get("method") != NAIVE_METHOD:
        raise BindingError(f"method {report.get('method')!r} != {NAIVE_METHOD!r}")
    splits = report.get("splits")
    if not isinstance(splits, dict):
        raise BindingError("naive evidence has no splits mapping")
    if "test" in splits and "test" not in expected_splits:
        raise BindingError("naive evidence contains the test split")
    if set(splits) != set(expected_splits):
        raise BindingError(f"splits {sorted(splits)} != expected {sorted(expected_splits)}")
    for split in expected_splits:
        if int(splits[split]["valid_cells"]) <= 0:
            raise BindingError(f"split {split} has no valid cells")
    fingerprints = report.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise BindingError("naive evidence has no fingerprints")
    for key in FINGERPRINT_KEYS:
        got = str(fingerprints.get(key, ""))
        if got != expected[key]:
            raise BindingError(
                f"fingerprint {key}: evidence {got!r} != configured dataset {expected[key]!r}"
            )
    lines = blobs[VALIDATION_NAME].decode("utf-8").splitlines()
    for split in expected_splits:
        if not any(f"selection {split}: full" in line for line in lines):
            raise BindingError(f"validation.txt lacks 'selection {split}: full'")
    if any(line.startswith("FAIL") for line in lines):
        raise BindingError("validation.txt contains a FAIL line")
    revision = report.get("git_revision")
    if not isinstance(revision, str) or not _GIT_RE.fullmatch(revision):
        raise BindingError("git provenance missing")
    if not _run_provenance(report_path, revision):
        raise BindingError("run provenance check failed")
    _verify_entries(evidence_dir, entries)
    _check_no_unlisted_contexts(evidence_dir, ctx)
    return {
        "path": str(report_path),
        "sha256": sha256,
        "manifest": str(manifest),
        "report_sha256": entries[REPORT_NAME],
        "splits": sorted(splits),
        "fingerprints": dict(expected),
        "git_revision": revision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--expected-splits", required=True, help="comma list, e.g. test,validation")
    args = parser.parse_args(argv)
    splits = [s for s in str(args.expected_splits).split(",") if s]
    try:
        exp = expected_fingerprints(str(args.config_name))
        summary = check_binding(args.report, str(args.sha256), exp, splits)
    except (BindingError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: naive evidence binding: {exc}")
        return 1
    print(
        f"OK: naive evidence bound: {summary['path']} manifest_sha256={summary['sha256']} "
        f"report_sha256={summary['report_sha256']} "
        f"splits={summary['splits']} fingerprints={summary['fingerprints']} "
        f"git={summary['git_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
