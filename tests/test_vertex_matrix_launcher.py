"""Offline tests for scripts/runners/vertex/launch_v6_matrix.sh.

The script is copied into a throwaway git repo with a fake credential wrapper and a fake
`gcloud`, so no real credentials, network, or Vertex client is ever reached. The fake wrapper
logs every invocation to wrapper.log, logs launcher (`uv run ...`) invocations to launcher.log,
and execs any other command (the fake gcloud), like the real per-command wrapper.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
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

ARMS = ["L1", "L2", "L3", "L4", "L5", "LOO-M", "LOO-S", "NO-NDVI-NDWI", "NO-RED-NIR-IDX"]
MODES = {
    "L1": "v6-l1",
    "L2": "v6-l2",
    "L3": "v6-l3",
    "L4": "v6-l4",
    "L5": "v6-l5",
    "LOO-M": "v6-loo-morph",
    "LOO-S": "v6-loo-shadow",
    "NO-NDVI-NDWI": "v6-no-ndvi-ndwi",
    "NO-RED-NIR-IDX": "v6-no-red-nir-index",
}
PAIRS = [(arm, seed) for arm in ARMS for seed in (0, 1, 2)]

ON_TIME = "2026-01-01T00:00:00Z\\t2026-01-01T00:01:00Z"
LATE = "2026-01-01T00:00:00Z\\t2026-01-01T02:00:00Z"  # start 2h after create > 1800s

FAKE_WRAPPER = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/wrapper.log"
if [[ "${FAKE_WRAPPER_FAIL:-0}" == 1 ]]; then
  echo "fake wrapper: credential fetch failed" >&2; exit 1
fi
[[ "${1:-}" == -- ]] || { echo "fake wrapper: expected --" >&2; exit 2; }
shift
if [[ "${1:-}" == uv ]]; then
  printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/launcher.log"
  if [[ "${FAKE_SUBMIT:-0}" == 1 ]]; then
    echo "submitted job: projects/p/locations/europe-west3/customJobs/123"
  else
    echo "preflight only: no job submitted"
  fi
  exit 0
fi
exec "$@"
"""

# list: FAKE_LIST for every displayName, FAKE_LIST_ODD for FAKE_ODD_DN only.
# describe: FAKE_DESCRIBE entries separated by '|', one per call, the last one repeats.
FAKE_GCLOUD = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/gcloud.log"
case "$*" in
  *"custom-jobs list"*)
    dn=$(sed -n 's/.*--filter=displayName="\\([^"]*\\)".*/\\1/p' <<<"$*")
    if [[ -n "${FAKE_ODD_DN:-}" && "$dn" == "$FAKE_ODD_DN" ]]; then
      [[ -n "${FAKE_LIST_ODD:-}" ]] && printf '%b\\n' "$FAKE_LIST_ODD"; exit 0
    fi
    [[ -n "${FAKE_LIST:-}" ]] && printf '%b\\n' "$FAKE_LIST"; exit 0 ;;
  *"custom-jobs describe"*)
    n=$(cat "$FAKE_LOG_DIR/describe.n" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$FAKE_LOG_DIR/describe.n"
    IFS='|' read -r -a seq <<<"${FAKE_DESCRIBE:-}"
    (( n >= ${#seq[@]} )) && n=$(( ${#seq[@]} - 1 ))
    (( n < 0 )) && exit 0
    printf '%b\\n' "${seq[$n]}"; exit 0 ;;
  *"custom-jobs cancel"*) exit 0 ;;
esac
exit 1
"""


def dn(arm: str, seed: int) -> str:
    return f"{MODES[arm]}-{PREFIX}-{arm.lower()}-seed{seed}-ep40-cos"


def listed_dns(gcloud_log: str) -> list[str]:
    return re.findall(r'custom-jobs list .*--filter=displayName="([^"]*)"', gcloud_log)


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
    # only what a fake call would need to leave a trace, so the empty-log check is not vacuous
    keep = {k: env.base[k] for k in ("PATH", "HOME", "FAKE_LOG_DIR", "V6M_GCLOUD")}
    bare = Env(env.root, env.script, env.sha, env.run_dir, env.log_dir, keep)
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
    lines = env.log("launcher.log").splitlines()
    assert len(lines) == 1
    for token in (
        "--preflight",
        "--mode v6-l4",
        "--seed 0",
        "--max-epochs 40",
        "--project berlin-lst-training",
        "--region europe-west3",
    ):
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
    assert env.log("launcher.log") == ""


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"])
def test_submit_stops_on_failed_or_cancelled(env: Env, state: str) -> None:
    res = env.run("submit", "L4", "0", FAKE_LIST=f"111\\t{state}")
    assert res.returncode == 1
    assert "no automatic resubmit" in res.stdout + res.stderr
    assert env.log("launcher.log") == ""


def test_submit_waits_on_active_job(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_LIST="111\\tJOB_STATE_RUNNING",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{ON_TIME}",
    )
    assert res.returncode == 0, res.stderr
    assert env.log("launcher.log") == ""
    assert "custom-jobs describe 111" in env.log("gcloud.log")


def test_state_filter_and_project(env: Env) -> None:
    res = env.run("state", "L4", "0", V6M_RUN_DIR="", V6M_IMAGE_URI="", V6M_SOURCE_SHA="")
    assert res.returncode == 0, res.stderr
    log = env.log("gcloud.log")
    assert f'--filter=displayName="{DN}"' in log
    assert "--project berlin-lst-training" in log
    assert "--region europe-west3" in log
    assert env.log("launcher.log") == ""


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
    assert len(env.log("launcher.log").splitlines()) == 1


def test_submit_polls_new_job_independently(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_SUBMIT="1",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{ON_TIME}",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    launcher = env.log("launcher.log")
    assert len(launcher.splitlines()) == 1
    assert "--preflight" not in launcher
    assert "custom-jobs describe 123" in env.log("gcloud.log")


# --- operator wrapper / project pin -------------------------------------------------------


def test_every_gcloud_call_goes_through_operator_wrapper(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_SUBMIT="1",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{ON_TIME}",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    gcloud_calls = env.log("gcloud.log").splitlines()
    wrapper_calls = env.log("wrapper.log").splitlines()
    assert gcloud_calls
    # one wrapper call per gcloud call plus the one launcher call; never --activate
    assert len(wrapper_calls) == len(gcloud_calls) + 1
    assert all(line.startswith("-- ") for line in wrapper_calls)
    assert not any("--activate" in line for line in wrapper_calls)
    assert sum(" ai custom-jobs " in line for line in wrapper_calls) == len(gcloud_calls)
    for line in gcloud_calls:
        assert "--project berlin-lst-training" in line
        assert "--region europe-west3" in line


@pytest.mark.parametrize(
    "bad",
    [
        {"V6M_PROJECT": "other-project"},
        {"V6M_REGION": "us-central1"},
        {"V6M_PROJECT": "other-project", "V6M_REGION": "us-central1"},
    ],
)
@pytest.mark.parametrize(
    "args", [("state", "L4", "0"), ("preflight", "L4", "0"), ("submit", "L4", "0")]
)
def test_project_region_must_match_launcher_pin(
    env: Env, bad: dict[str, str], args: tuple[str, ...]
) -> None:
    res = env.run(*args, **bad)
    assert res.returncode == 2
    assert "launcher pin" in res.stderr
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


@pytest.mark.parametrize("args", [("state", "L4", "0"), ("submit", "L4", "0"), ("submit-all",)])
def test_missing_operator_wrapper_fails_closed(env: Env, args: tuple[str, ...]) -> None:
    (env.root / "scripts" / "operators" / "with-gcp-operator.sh").unlink()
    res = env.run(*args)
    assert res.returncode == 2
    assert "operator wrapper missing" in res.stderr
    assert env.log("gcloud.log") == "" and env.log("launcher.log") == ""


@pytest.mark.parametrize("args", [("state", "L4", "0"), ("submit", "L4", "0")])
def test_failing_operator_wrapper_fails_closed(env: Env, args: tuple[str, ...]) -> None:
    res = env.run(*args, FAKE_WRAPPER_FAIL="1", FAKE_LIST="111\\tJOB_STATE_SUCCEEDED")
    assert res.returncode == 1
    # lookup retried within its bounds, never fell back to ambient gcloud, never submitted
    assert len(env.log("wrapper.log").splitlines()) == 6
    assert env.log("gcloud.log") == "" and env.log("launcher.log") == ""


# --- queue deadline -----------------------------------------------------------------------


def test_on_time_success_continues(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_LIST="111\\tJOB_STATE_RUNNING",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{ON_TIME}",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "final: JOB_STATE_SUCCEEDED" in res.stdout
    assert "QUEUE_DEADLINE" not in res.stdout + res.stderr
    assert "custom-jobs cancel" not in env.log("gcloud.log")


def test_late_start_success_is_success_and_not_cancelled(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_LIST="111\\tJOB_STATE_RUNNING",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{LATE}",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "final: JOB_STATE_SUCCEEDED" in res.stdout
    assert "QUEUE_DEADLINE_VIOLATION" in res.stdout + res.stderr
    assert "QUEUE_DEADLINE_MISSED" not in res.stdout + res.stderr
    assert "custom-jobs cancel" not in env.log("gcloud.log")
    assert env.log("launcher.log") == ""


def test_submit_all_continues_after_late_start_success(env: Env) -> None:
    res = env.run(
        "submit-all",
        FAKE_LIST="999\\tJOB_STATE_SUCCEEDED",
        FAKE_ODD_DN=dn("L2", 1),
        FAKE_LIST_ODD="111\\tJOB_STATE_RUNNING",
        FAKE_DESCRIBE=f"JOB_STATE_SUCCEEDED\\t{LATE}",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "QUEUE_DEADLINE_VIOLATION" in res.stdout + res.stderr
    assert "submit-all done: 27 pair(s)" in res.stdout
    assert listed_dns(env.log("gcloud.log")) == [dn(a, s) for a, s in PAIRS]
    assert "custom-jobs cancel" not in env.log("gcloud.log")
    assert env.log("launcher.log") == ""


def test_not_started_past_deadline_is_cancelled_and_stops(env: Env) -> None:
    res = env.run(
        "submit",
        "L4",
        "0",
        FAKE_LIST="111\\tJOB_STATE_PENDING",
        FAKE_DESCRIBE="JOB_STATE_PENDING\\t2026-01-01T00:00:00Z",
    )
    assert res.returncode == 1
    assert "QUEUE_DEADLINE_VIOLATION" in res.stdout + res.stderr
    assert "final: QUEUE_DEADLINE_MISSED" in res.stdout
    assert "custom-jobs cancel 111" in env.log("gcloud.log")
    assert env.log("launcher.log") == ""


def test_submit_all_stops_after_queue_deadline_cancel(env: Env) -> None:
    res = env.run(
        "submit-all",
        FAKE_LIST="999\\tJOB_STATE_SUCCEEDED",
        FAKE_ODD_DN=dn("L3", 0),
        FAKE_LIST_ODD="111\\tJOB_STATE_QUEUED",
        FAKE_DESCRIBE="JOB_STATE_QUEUED\\t2026-01-01T00:00:00Z",
    )
    assert res.returncode == 1
    assert "submit-all stopped at L3 seed0" in res.stdout
    log = env.log("gcloud.log")
    assert "custom-jobs cancel 111" in log
    expected = [dn(a, s) for a, s in PAIRS[: PAIRS.index(("L3", 0)) + 1]]
    assert listed_dns(log) == expected
    assert env.log("launcher.log") == ""


def test_queued_and_running_within_deadline_keep_polling(env: Env) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    seq = "|".join(
        [
            f"JOB_STATE_QUEUED\\t{now}",
            f"JOB_STATE_RUNNING\\t{now}\\t{now}",
            f"JOB_STATE_RUNNING\\t{now}\\t{now}",
            f"JOB_STATE_SUCCEEDED\\t{now}\\t{now}",
        ]
    )
    res = env.run("submit", "L4", "0", FAKE_LIST="111\\tJOB_STATE_QUEUED", FAKE_DESCRIBE=seq)
    assert res.returncode == 0, res.stdout + res.stderr
    log = env.log("gcloud.log")
    assert log.count("custom-jobs describe 111") == 4
    assert "custom-jobs cancel" not in log
    assert "QUEUE_DEADLINE" not in res.stdout + res.stderr


# --- matrix order -------------------------------------------------------------------------


def test_preflight_all_runs_27_pairs_in_order(env: Env) -> None:
    res = env.run("preflight", V6M_RUN_DIR="")
    assert res.returncode == 0, res.stderr
    lines = env.log("launcher.log").splitlines()
    got = [re.findall(r"--mode (\S+) --seed (\d)", line) for line in lines]
    assert got == [[(MODES[a], str(s))] for a, s in PAIRS]
    assert len(lines) == 27
    assert all("--preflight" in line for line in lines)
    assert env.log("gcloud.log") == ""


def test_submit_all_order(env: Env) -> None:
    res = env.run("submit-all", FAKE_LIST="999\\tJOB_STATE_SUCCEEDED")
    assert res.returncode == 0, res.stdout + res.stderr
    assert listed_dns(env.log("gcloud.log")) == [dn(a, s) for a, s in PAIRS]
    assert env.log("launcher.log") == ""


@pytest.mark.parametrize("start", [("L1", 0), ("LOO-S", 2), ("NO-RED-NIR-IDX", 2)])
def test_submit_all_from(env: Env, start: tuple[str, int]) -> None:
    arm, seed = start
    res = env.run("submit-all", "--from", arm, str(seed), FAKE_LIST="999\\tJOB_STATE_SUCCEEDED")
    assert res.returncode == 0, res.stdout + res.stderr
    expected = [dn(a, s) for a, s in PAIRS[PAIRS.index(start) :]]
    assert listed_dns(env.log("gcloud.log")) == expected


@pytest.mark.parametrize(
    "args",
    [
        ("--from",),
        ("--from", "L4"),
        ("--from", "NOPE", "0"),
        ("--from", "L4", "3"),
        ("--from", "l4", "0"),
        ("bogus",),
    ],
)
def test_submit_all_invalid_from_fails_closed(env: Env, args: tuple[str, ...]) -> None:
    res = env.run("submit-all", *args, FAKE_LIST="999\\tJOB_STATE_SUCCEEDED")
    assert res.returncode == 2
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"])
def test_submit_all_halts_on_existing_failure(env: Env, state: str) -> None:
    res = env.run(
        "submit-all",
        FAKE_LIST="999\\tJOB_STATE_SUCCEEDED",
        FAKE_ODD_DN=dn("L5", 1),
        FAKE_LIST_ODD=f"111\\t{state}",
    )
    assert res.returncode == 1
    assert "submit-all stopped at L5 seed1" in res.stdout
    expected = [dn(a, s) for a, s in PAIRS[: PAIRS.index(("L5", 1)) + 1]]
    assert listed_dns(env.log("gcloud.log")) == expected
    assert env.log("launcher.log") == ""


def test_submit_all_halts_when_new_job_fails(env: Env) -> None:
    # L1 seed0 has no job: one submit, the new job fails, and nothing else is listed or submitted.
    res = env.run("submit-all", FAKE_SUBMIT="1", FAKE_DESCRIBE=f"JOB_STATE_FAILED\\t{ON_TIME}")
    assert res.returncode == 1
    assert "submit-all stopped at L1 seed0" in res.stdout
    assert len(env.log("launcher.log").splitlines()) == 1
    assert listed_dns(env.log("gcloud.log")) == [dn("L1", 0)]
