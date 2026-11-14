"""Offline regressions for the operator's TESSERA capacity watcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runners/vertex/retry_tessera_seeds.sh"


def _job(state: str, message: str = "", code: int = 14) -> dict:
    return {
        "state": state,
        "error": {"code": code, "message": message},
        "createTime": "2026-01-01T00:00:00Z",
    }


def _run(tmp_path: Path, jobs: list[dict], *, lookup_exit: int = 0):
    if shutil.which("jq") is None:
        pytest.skip("jq is required by the operator watcher")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(
        '#!/bin/bash\n[[ "$*" == *"error.code,error.message"* ]] || exit 2\n'
        f"cat <<'JSON'\n{json.dumps(jobs)}\nJSON\nexit {lookup_exit}\n"
    )
    (bin_dir / "uv").write_text('#!/bin/bash\nprintf "%s\\n" "$*" >>"$CALLS"\nexit 1\n')
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    calls = tmp_path / "calls"
    state_dir = tmp_path / "state"
    result = subprocess.run(
        ["/bin/bash"],
        input=SCRIPT.read_text(),
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TESSERA_REPO": str(tmp_path),
            "TESSERA_STATE_DIR": str(state_dir),
            "TESSERA_LOG": str(tmp_path / "watcher.log"),
            "TESSERA_AUTH": "user-adc",
            "CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, state_dir, calls


def test_nested_capacity_error_is_retried(tmp_path: Path):
    result, state_dir, calls = _run(
        tmp_path, [_job("JOB_STATE_FAILED", "Resources are insufficient in region: europe-west3.")]
    )
    assert result.returncode == 0
    assert not (state_dir / "STOP").exists()
    command = calls.read_text()
    assert command.count("--mode tessera") == 1
    assert "--skip-test" in command
    assert "--timeout-seconds 17400 --max-wait-seconds 600 --max-exposure-usd 5" in command
    assert "no capacity" in (state_dir / "status.txt").read_text()


@pytest.mark.parametrize(
    "jobs",
    [
        [_job("JOB_STATE_FAILED", "training crashed", 13)],
        [_job("JOB_STATE_FAILED")],
        [_job("JOB_STATE_CANCELLED", "Resources are insufficient")],
        [_job("UNRECOGNIZED")],
        [
            _job("JOB_STATE_FAILED", "training crashed", 13),
            _job("JOB_STATE_FAILED", "Resources are insufficient"),
        ],
    ],
)
def test_other_failures_never_submit(tmp_path: Path, jobs: list[dict]):
    result, state_dir, calls = _run(tmp_path, jobs)
    assert result.returncode == 1
    assert (state_dir / "STOP").exists()
    assert not calls.exists()


def test_failed_lookup_never_submits(tmp_path: Path):
    result, state_dir, calls = _run(tmp_path, [], lookup_exit=1)
    assert result.returncode == 1
    assert "lookup failed" in (state_dir / "STOP").read_text()
    assert not calls.exists()


def test_any_active_attempt_prevents_duplicate(tmp_path: Path):
    result, state_dir, calls = _run(
        tmp_path,
        [
            _job("JOB_STATE_RUNNING"),
            _job("JOB_STATE_FAILED", "Resources are insufficient"),
        ],
    )
    assert result.returncode == 0
    assert "active" in (state_dir / "status.txt").read_text()
    assert not calls.exists()


def test_successful_seeds_are_skipped(tmp_path: Path):
    result, state_dir, calls = _run(tmp_path, [_job("JOB_STATE_SUCCEEDED")])
    assert result.returncode == 0
    assert (state_dir / "status.txt").read_text() == "all three seeds succeeded\n"
    assert not calls.exists()
