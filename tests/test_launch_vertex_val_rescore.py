"""Launcher and entrypoint wiring for the v6 validation-only rescore (offline)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "scripts" / "operators" / "launch_vertex_modeling.py"
_ENTRYPOINT = _ROOT / "scripts" / "operators" / "vertex_entrypoint.sh"
_DIGEST = "sha256:" + "a" * 64
_IMAGE = "example.invalid/repo/image@" + _DIGEST
_SHA = "0" * 40
_MODE = "v6-val-rescore"
_LABEL = "v6m-070b33c-val-rescore-notest-r2"
_PREFIX = f"gs://berlin-lst-training-data/qa/modeling/vertex-smoke/{_LABEL}"
_MANIFEST = f"{_PREFIX}/manifest.json"


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    if importlib.util.find_spec("google.cloud.aiplatform_v1") is None:
        stub = MagicMock()
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1", stub)
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1.types", stub.types)
    spec = importlib.util.spec_from_file_location("launch_vertex_modeling_rescore_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _no_client(region: str) -> None:
        raise AssertionError("the Vertex client must not be created in these tests")

    monkeypatch.setattr(module, "_evidence_exists", lambda uri: uri.endswith("/best.ckpt"))
    monkeypatch.setattr(module, "_client", _no_client)
    return module


def _argv(*extra: str, preflight: bool = True, rate: bool = True) -> list[str]:
    argv = [
        "launch_vertex_modeling.py",
        "--mode",
        _MODE,
        "--image-uri",
        _IMAGE,
        "--source-sha",
        _SHA,
        "--service-account",
        "<service-account>",
        "--infisical-identity",
        "<identity>",
        "--infisical-project",
        "<project>",
    ]
    if preflight:
        argv.append("--preflight")
    if rate:
        argv += ["--hourly-rate-usd", "1.00"]
    return [*argv, *extra]


def test_preflight_prints_the_full_plan(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv())
    assert launcher.main() == 0
    out = capsys.readouterr().out
    for needle in (
        "evaluate_test=false",
        "validation 2024 only",
        "n1-standard-4",
        "NVIDIA_TESLA_T4",
        "europe-west3",
        "job-local rebuild once",
        "jobs           : 1",
        f"destination    : {_PREFIX}/",
        f"manifest       : {_MANIFEST}",
        "18000s",
        "~$5.17",
        "070b33ce4323834b7813db4609b5052702c4505f",
        "preflight only: no job submitted",
    ):
        assert needle in out
    for run in launcher.rescore_inventory():
        assert f"{run.checkpoint_uri} -> {run.export_uri}" in out
    assert out.count("/best.ckpt ->") == 27
    assert out.count("eval_validation_patches.parquet") == 27


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--skip-test"], "validation-only by construction"),
        (["--seed", "0"], "--seed"),
        (["--max-epochs", "40"], "--max-epochs"),
        (["--lr-schedule", "cosine"], "--lr-schedule"),
        (["--patch-cache", "prebuilt"], "--patch-cache"),
        (["--allow-job-local-fallback"], "--allow-job-local-fallback"),
        (["--early-stopping-patience", "3"], "--early-stopping-patience"),
        (["--projected-noncompute-total-usd", "1"], "--projected-noncompute-total-usd"),
        (["--evidence-prefix", "gs://berlin-lst-training-data/qa/modeling/o"], "evidence-prefix"),
        (["--run-label", "other"], "--run-label"),
        (["--hourly-rate-usd", "1.01"], "cap"),
        (["--timeout-seconds", "21601"], "ceiling"),
        (["--max-exposure-usd", "8"], "cannot exceed"),
        (["--max-wait-seconds", "1801"], "--max-wait-seconds"),
        (["--image-uri", "not-a-digest-uri"], "image"),
        (["--source-sha", "abc123"], "40-character"),
    ],
)
def test_rejections(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, extra: list[str], match: str
) -> None:
    # Later duplicates of a flag win in argparse, so the override replaces the default.
    monkeypatch.setattr(sys, "argv", _argv(*extra))
    with pytest.raises(SystemExit, match=match):
        launcher.main()


def test_missing_rate_is_refused(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", _argv(rate=False))
    with pytest.raises(SystemExit, match="hourly-rate-usd"):
        launcher.main()


def test_matching_run_label_is_accepted(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("--run-label", _LABEL))
    assert launcher.main() == 0


def test_missing_checkpoint_is_refused(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "_evidence_exists", lambda uri: False)
    monkeypatch.setattr(sys, "argv", _argv())
    with pytest.raises(SystemExit, match="missing checkpoints"):
        launcher.main()


@pytest.mark.parametrize("which", ["export", "manifest"])
def test_used_destination_is_refused(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    target = "/eval_validation_patches.parquet" if which == "export" else "/manifest.json"
    monkeypatch.setattr(
        launcher, "_evidence_exists", lambda uri: uri.endswith(("/best.ckpt", target))
    )
    monkeypatch.setattr(sys, "argv", _argv())
    with pytest.raises(SystemExit, match="create-only destination already used"):
        launcher.main()


def test_env_has_no_training_or_cache_variables(launcher: ModuleType) -> None:
    args = MagicMock(
        source_sha=_SHA,
        image_uri=_IMAGE,
        infisical_identity="<identity>",
        infisical_project="<project>",
        infisical_env="dev",
        infisical_path="/vertex",
    )
    env = dict(launcher._val_rescore_env(args))
    assert env == {
        "VERTEX_RUN_LABEL": _LABEL,
        "VERTEX_SOURCE_SHA": _SHA,
        "VERTEX_IMAGE_DIGEST": _DIGEST,
        "VERTEX_EVIDENCE_URI": _MANIFEST,
        "VERTEX_PROFILE": "val-rescore",
        "VERTEX_OUTPUT_ROOT": f"data/runs/v6-val-rescore/{_LABEL}",
        "INFISICAL_MACHINE_IDENTITY_ID": "<identity>",
        "INFISICAL_PROJECT_ID": "<project>",
        "INFISICAL_ENV": "dev",
        "INFISICAL_SECRET_PATH": "/vertex",
    }


def test_submit_path(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_submit(client: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return "projects/p/locations/europe-west3/customJobs/1"

    monkeypatch.setattr(launcher, "_client", lambda region: object())
    monkeypatch.setattr(launcher, "_submit", _fake_submit)
    monkeypatch.setattr(
        launcher,
        "_poll_until_terminal",
        lambda *a, **k: (launcher.JobState.JOB_STATE_SUCCEEDED, False),
    )
    monkeypatch.setattr(sys, "argv", _argv(preflight=False))
    assert launcher.main() == 0
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["timeout_seconds"] == 18000
    assert kwargs["profile"] == _MODE
    assert kwargs["display_name"] == f"{_MODE}-{_LABEL}"
    env = dict(kwargs["env"])  # type: ignore[arg-type]
    assert env["VERTEX_PROFILE"] == "val-rescore"
    assert env["VERTEX_EVIDENCE_URI"] == _MANIFEST
    assert not any(k in env for k in ("VERTEX_SEED", "VERTEX_SKIP_TEST", "VERTEX_CONFIG_NAME"))


def test_existing_modes_are_unchanged(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _MODE not in launcher.MODE_CONFIG_NAME
    assert _MODE not in launcher.MODE_EVIDENCE_PROFILE
    monkeypatch.setattr(launcher, "_evidence_exists", lambda uri: False)
    argv = _argv()
    argv[argv.index(_MODE)] = "v6-l4"
    argv += ["--run-label", "unit"]
    monkeypatch.setattr(sys, "argv", argv)
    assert launcher.main() == 0
    assert "v6-l4 / v6_l4_locked" in capsys.readouterr().out


# --- entrypoint ---------------------------------------------------------------


def _fake_bin(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    (bin_dir / "infisical").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "login" ]]; then echo dummy-token; exit 0; fi\n'
        'while [[ "$1" != "--" ]]; do shift; done\nshift\nexec "$@"\n'
    )
    (bin_dir / "uv").write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\nexit 0\n')
    for name in ("infisical", "uv"):
        path = bin_dir / name
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir, log


def _run_entrypoint(tmp_path: Path, **extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    bin_dir, log = _fake_bin(tmp_path)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "VERTEX_RUN_LABEL": _LABEL,
        "VERTEX_SOURCE_SHA": _SHA,
        "VERTEX_IMAGE_DIGEST": _DIGEST,
        "VERTEX_EVIDENCE_URI": _MANIFEST,
        "VERTEX_PROFILE": "val-rescore",
        "VERTEX_OUTPUT_ROOT": str(tmp_path / "out" / "x"),
        "INFISICAL_MACHINE_IDENTITY_ID": "id",
        "INFISICAL_PROJECT_ID": "proj",
        "INFISICAL_ENV": "dev",
        "INFISICAL_SECRET_PATH": "/vertex",
        **extra,
    }
    proc = subprocess.run(  # noqa: S603
        [bash, str(_ENTRYPOINT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    return proc, log


def test_entrypoint_runs_rescore_only(tmp_path: Path) -> None:
    proc, log = _run_entrypoint(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    logged = log.read_text()
    assert "scripts/operators/rescore_v6_validation.py" in logged
    assert "--device cuda" in logged
    assert f"--manifest-uri {_MANIFEST}" in logged
    assert "vertex_evidence.py" not in logged


@pytest.mark.parametrize(
    ("name", "value"),
    [("VERTEX_SKIP_TEST", "1"), ("VERTEX_PATCH_CACHE_SOURCE", "prebuilt"), ("VERTEX_SEED", "0")],
)
def test_entrypoint_rejects_training_variables(tmp_path: Path, name: str, value: str) -> None:
    proc, log = _run_entrypoint(tmp_path, **{name: value})
    assert proc.returncode == 2
    assert name in proc.stderr
    assert not log.exists()
