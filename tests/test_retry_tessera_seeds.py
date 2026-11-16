"""Offline regressions for the operator's TESSERA capacity watcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runners/vertex/retry_tessera_seeds.sh"
REGISTRY = "europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex"
IMAGE = f"{REGISTRY}@sha256:{'a' * 64}"
RETIRED_IMAGE = (
    f"{REGISTRY}@sha256:e5aa497c215290343842011ca58b3da99dc22562e9795453b2bf5ed80a3e2c9f"
)
SOURCE_SHA = "b" * 40
SEQUENCE = {
    "TESSERA_IMAGE": IMAGE,
    "TESSERA_SOURCE_SHA": SOURCE_SHA,
    "TESSERA_LABEL": "tessera-test-r1",
}


def _job(state: str, message: str = "", code: int = 14) -> dict:
    return {
        "state": state,
        "error": {"code": code, "message": message},
        "createTime": "2026-01-01T00:00:00Z",
    }


def _run(
    tmp_path: Path,
    jobs: list[dict],
    *,
    lookup_exit: int = 0,
    sequence: dict[str, str] | None = None,
    jobs_after: list[dict] | None = None,
    uv_script: str = "exit 1",
):
    if shutil.which("jq") is None:
        pytest.skip("jq is required by the operator watcher")
    after = jobs if jobs_after is None else jobs_after
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(
        '#!/bin/bash\n[[ "$*" == *"error.code,error.message"* ]] || exit 2\n'
        # After the launcher stub has run, the lookup returns jobs_after.
        f"if [[ -e \"$CALLS\" ]]; then\ncat <<'JSON'\n{json.dumps(after)}\nJSON\n"
        f"else\ncat <<'JSON'\n{json.dumps(jobs)}\nJSON\nfi\nexit {lookup_exit}\n"
    )
    (bin_dir / "uv").write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >>"$CALLS"\n{uv_script}\n')
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    calls = tmp_path / "calls"
    state_dir = tmp_path / "state"
    result = subprocess.run(
        ["/bin/bash"],
        input=SCRIPT.read_text(),
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith("TESSERA_")},
            **(SEQUENCE if sequence is None else sequence),
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
    jobs = [_job("JOB_STATE_FAILED", "Resources are insufficient in region: europe-west3.")]
    result, state_dir, calls = _run(tmp_path, jobs, jobs_after=jobs * 2)
    assert result.returncode == 0
    assert not (state_dir / "STOP").exists()
    command = calls.read_text()
    assert command.count("--mode tessera") == 1
    assert "--skip-test --patch-cache prebuilt" in command
    assert f"--image-uri {IMAGE} --source-sha {SOURCE_SHA} --run-label tessera-test-r1" in command
    assert "--timeout-seconds 20400 --max-wait-seconds 600 --max-exposure-usd 6.25" in command
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


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"TESSERA_IMAGE": ""}, "TESSERA_IMAGE must be pinned by digest"),
        ({"TESSERA_IMAGE": f"{REGISTRY}:latest"}, "TESSERA_IMAGE must be pinned by digest"),
        ({"TESSERA_IMAGE": RETIRED_IMAGE}, "retired window-read image"),
        ({"TESSERA_SOURCE_SHA": "5e39bfe"}, "TESSERA_SOURCE_SHA"),
        ({"TESSERA_LABEL": ""}, "TESSERA_LABEL"),
        ({"TESSERA_TIMEOUT_SECONDS": "4h"}, "TESSERA_TIMEOUT_SECONDS"),
        ({"TESSERA_MAX_WAIT_SECONDS": "-1"}, "TESSERA_MAX_WAIT_SECONDS"),
        ({"TESSERA_MAX_EXPOSURE_USD": "6,25"}, "TESSERA_MAX_EXPOSURE_USD"),
    ],
)
def test_incomplete_or_retired_sequence_never_submits(
    tmp_path: Path, override: dict[str, str], reason: str
):
    result, state_dir, calls = _run(tmp_path, [], sequence={**SEQUENCE, **override})
    assert result.returncode == 2
    assert reason in (state_dir / "status.txt").read_text()
    assert not (state_dir / "STOP").exists()
    assert not calls.exists()


def test_bound_overrides_reach_the_launcher(tmp_path: Path):
    result, _, calls = _run(
        tmp_path,
        [_job("JOB_STATE_FAILED", "Resources are insufficient in region: europe-west3.")],
        jobs_after=[_job("JOB_STATE_FAILED", "Resources are insufficient in region: europe-west3.")]
        * 2,
        sequence={
            **SEQUENCE,
            "TESSERA_TIMEOUT_SECONDS": "13500",
            "TESSERA_MAX_WAIT_SECONDS": "2100",
            "TESSERA_MAX_EXPOSURE_USD": "4.33",
        },
    )
    assert result.returncode == 0
    assert "--timeout-seconds 13500 --max-wait-seconds 2100 --max-exposure-usd 4.33" in (
        calls.read_text()
    )


def test_launcher_failure_without_a_new_job_stops(tmp_path: Path):
    # An earlier capacity failure exists; the launcher then fails before it
    # creates a job. That must not be reported as "no capacity".
    result, state_dir, calls = _run(
        tmp_path, [_job("JOB_STATE_FAILED", "Resources are insufficient in region: europe-west3.")]
    )
    assert result.returncode == 1
    assert calls.read_text().count("--mode tessera") == 1
    assert "before creating a job" in (state_dir / "STOP").read_text()


def test_stop_written_during_a_seed_prevents_the_next_submit(tmp_path: Path):
    result, state_dir, calls = _run(
        tmp_path, [], uv_script='echo owner >"$TESSERA_STATE_DIR/STOP"\nexit 0'
    )
    assert result.returncode == 0
    assert calls.read_text().count("--mode tessera") == 1
    assert (state_dir / "STOP").read_text() == "owner\n"
