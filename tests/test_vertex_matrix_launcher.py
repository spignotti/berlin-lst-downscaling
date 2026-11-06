"""Offline tests for scripts/runners/vertex/launch_v6_matrix.sh.

The script is copied into a throwaway git repo with a fake credential wrapper and a fake
`gcloud`, so no real credentials, network, or Vertex client is ever reached.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "runners" / "vertex" / "launch_v6_matrix.sh"
BASH = shutil.which("bash")
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(BASH is None or GIT is None, reason="bash and git are required")

DIGEST = "a" * 64
IMAGE = f"europe-west3-docker.pkg.dev/p/r/img@sha256:{DIGEST}"
PREFIX = "v6m-test"
DN = f"v6-l4-{PREFIX}-l4-seed0-ep40-cos"

FAKE_WRAPPER = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/wrapper.log"
if [[ "${FAKE_SUBMIT:-0}" == 1 ]]; then
  echo "submitted job: projects/p/locations/europe-west3/customJobs/123"
else
  echo "preflight only: no job submitted"
fi
"""

FAKE_GCLOUD = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/gcloud.log"
case "$*" in
  *"custom-jobs list"*) [[ -n "${FAKE_LIST:-}" ]] && printf '%b\\n' "$FAKE_LIST"; exit 0 ;;
  *"custom-jobs describe"*) printf '%b\\n' "${FAKE_DESCRIBE:-}"; exit 0 ;;
esac
exit 1
"""


@dataclass
class Env:
    root: Path
    script: Path
    sha: str
    run_dir: Path
    log_dir: Path
    base: dict[str, str]

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {**self.base, **extra}
        env = {k: v for k, v in env.items() if v != ""}
        assert BASH is not None
        return subprocess.run(  # noqa: S603
            [BASH, str(self.script), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def log(self, name: str) -> str:
        path = self.log_dir / name
        return path.read_text() if path.exists() else ""


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    assert GIT is not None
    fake = tmp_path / "fake"
    script = fake / "scripts" / "runners" / "vertex" / "launch_v6_matrix.sh"
    script.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, script)
    _write_exec(fake / "scripts" / "operators" / "with-gcp-operator.sh", FAKE_WRAPPER)
    git_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        [*ident, "commit", "-q", "-m", "init"],
    ):
        subprocess.run(  # noqa: S603
            [GIT, "-C", str(fake), *cmd], env=git_env, check=True, capture_output=True
        )
    sha = subprocess.run(  # noqa: S603
        [GIT, "-C", str(fake), "rev-parse", "HEAD"],
        env=git_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gcloud = tmp_path / "bin" / "gcloud"
    _write_exec(gcloud, FAKE_GCLOUD)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "FAKE_LOG_DIR": str(log_dir),
        "V6M_GCLOUD": str(gcloud),
        "V6M_IMAGE_URI": IMAGE,
        "V6M_SOURCE_SHA": sha,
        "V6M_LABEL_PREFIX": PREFIX,
        "V6M_RUN_DIR": str(run_dir),
        "V6M_POLL_SECONDS": "1",
        "V6M_LOOKUP_BACKOFF_SECONDS": "0",
        "V6M_LAUNCHER_GRACE": "5",
        "V6M_CREATE_WAIT": "20",
        "V6M_SUBMIT_ATTEMPTS": "1",
    }
    return Env(fake, script, sha, run_dir, log_dir, base)


def test_all_vertex_runner_scripts_parse() -> None:
    assert BASH is not None
    scripts = sorted((REPO / "scripts" / "runners" / "vertex").glob("*.sh"))
    assert SCRIPT in scripts
    for sh in scripts:
        res = subprocess.run(  # noqa: S603
            [BASH, "-n", str(sh)], capture_output=True, text=True, timeout=60, check=False
        )
        assert res.returncode == 0, f"{sh}: {res.stderr}"


@pytest.mark.parametrize("args", [(), ("help",), ("-h",), ("--help",)])
def test_help_touches_nothing(env: Env, args: tuple[str, ...]) -> None:
    bare = Env(env.root, env.script, env.sha, env.run_dir, env.log_dir, {"PATH": "/usr/bin:/bin"})
    res = bare.run(*args)
    assert res.returncode == 0
    assert "usage:" in res.stdout
    assert env.log("wrapper.log") == ""
    assert env.log("gcloud.log") == ""


def test_unknown_subcommand(env: Env) -> None:
    res = env.run("frobnicate")
    assert res.returncode == 2
    assert "usage:" in res.stderr
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


@pytest.mark.parametrize(
    "bad",
    [
        {"V6M_IMAGE_URI": ""},
        {"V6M_IMAGE_URI": "img:latest"},
        {"V6M_IMAGE_URI": "img@sha256:abc"},
        {"V6M_SOURCE_SHA": ""},
        {"V6M_SOURCE_SHA": "ABCDEF" + "0" * 34},
        {"V6M_SOURCE_SHA": "abc123"},
        {"V6M_LABEL_PREFIX": ""},
        {"V6M_LABEL_PREFIX": "Bad_Prefix"},
        {"V6M_MAX_EPOCHS": "0"},
        {"V6M_HOURLY_RATE_USD": "0.00"},
    ],
)
def test_preflight_invalid_config_fails_closed(env: Env, bad: dict[str, str]) -> None:
    res = env.run("preflight", "L4", "0", **bad)
    assert res.returncode != 0
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_preflight_single_pair(env: Env) -> None:
    res = env.run("preflight", "L4", "0", V6M_RUN_DIR="")
    assert res.returncode == 0, res.stderr
    lines = env.log("wrapper.log").splitlines()
    assert len(lines) == 1
    for token in ("--preflight", "--mode v6-l4", "--seed 0", "--max-epochs 40"):
        assert token in lines[0]
    assert f"--run-label {PREFIX}-l4" in lines[0]
    assert env.log("gcloud.log") == ""


def test_preflight_wrong_head(env: Env) -> None:
    res = env.run("preflight", "L4", "0", V6M_SOURCE_SHA="0" * 40)
    assert res.returncode != 0
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_preflight_dirty_tree(env: Env) -> None:
    (env.root / "dirty.txt").write_text("x")
    res = env.run("preflight", "L4", "0")
    assert res.returncode != 0
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_submit_requires_run_dir(env: Env) -> None:
    res = env.run("submit", "L4", "0", V6M_RUN_DIR="")
    assert res.returncode == 2
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_submit_rejects_run_dir_inside_repo(env: Env) -> None:
    inside = env.root / "logs"
    inside.mkdir()
    res = env.run("submit", "L4", "0", V6M_RUN_DIR=str(inside))
    assert res.returncode == 2
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_submit_skips_succeeded(env: Env) -> None:
    res = env.run("submit", "L4", "0", FAKE_LIST="111\\tJOB_STATE_SUCCEEDED")
    assert res.returncode == 0, res.stderr
    assert "skip" in res.stdout
    assert env.log("wrapper.log") == ""


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"])
def test_submit_stops_on_failed_or_cancelled(env: Env, state: str) -> None:
    res = env.run("submit", "L4", "0", FAKE_LIST=f"111\\t{state}")
    assert res.returncode == 1
    assert "no automatic resubmit" in res.stdout + res.stderr
    assert env.log("wrapper.log") == ""


def test_submit_waits_on_active_job(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_LIST="111\\tJOB_STATE_RUNNING",
        FAKE_DESCRIBE="JOB_STATE_SUCCEEDED\\t2026-01-01T00:00:00Z\\t2026-01-01T00:01:00Z",
    )
    assert res.returncode == 0, res.stderr
    assert env.log("wrapper.log") == ""
    assert "custom-jobs describe 111" in env.log("gcloud.log")


def test_state_filter_and_project(env: Env) -> None:
    res = env.run("state", "L4", "0", V6M_RUN_DIR="", V6M_IMAGE_URI="", V6M_SOURCE_SHA="")
    assert res.returncode == 0, res.stderr
    log = env.log("gcloud.log")
    assert f'--filter=displayName="{DN}"' in log
    assert "--project berlin-lst-training" in log
    assert env.log("wrapper.log") == ""


@pytest.mark.parametrize(
    "args", [("state", "NOPE", "0"), ("state", "L4", "3"), ("state", "L4", "x")]
)
def test_bad_arm_or_seed(env: Env, args: tuple[str, ...]) -> None:
    res = env.run(*args)
    assert res.returncode == 2
    assert env.log("gcloud.log") == ""


def test_resubmit_failed_never_adopts_the_old_job(env: Env) -> None:
    # The fake launcher creates no job; the follow-up lookup still lists only the old
    # FAILED job, which must not be mistaken for this attempt's job.
    res = env.run(
        "submit",
        "L4",
        "0",
        V6M_RESUBMIT_FAILED="1",
        FAKE_LIST="111\\tJOB_STATE_FAILED",
    )
    assert res.returncode == 1
    assert "submission failed before create" in res.stdout
    assert "custom-jobs describe" not in env.log("gcloud.log")
    assert len(env.log("wrapper.log").splitlines()) == 1


def test_submit_polls_new_job_independently(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_SUBMIT="1",
        FAKE_DESCRIBE="JOB_STATE_SUCCEEDED\\t2026-01-01T00:00:00Z\\t2026-01-01T00:01:00Z",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    wrapper = env.log("wrapper.log")
    assert len(wrapper.splitlines()) == 1
    assert "--preflight" not in wrapper
    assert "custom-jobs describe 123" in env.log("gcloud.log")
