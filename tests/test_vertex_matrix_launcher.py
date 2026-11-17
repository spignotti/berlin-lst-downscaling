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
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_spatial_holdout_cv import load_launcher

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
  mode=$(sed -n 's/.*--mode \\([^ ]*\\).*/\\1/p' <<<"$*")
  if [[ " ${FAKE_LAUNCH_FAIL_MODES:-} " == *" $mode "* ]]; then
    echo "${FAKE_LAUNCH_FAIL_MSG:-ERROR: launcher failed}" >&2; exit 1
  fi
  if [[ "${FAKE_SUBMIT:-0}" == 1 ]]; then
    id=123; [[ "${FAKE_UNIQUE_IDS:-0}" == 1 ]] && id=$$
    echo "$mode" > "$FAKE_LOG_DIR/job.$id"
    echo "+$id" >> "$FAKE_LOG_DIR/events.log"
    echo "submitted job: projects/p/locations/europe-west3/customJobs/$id"
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
    nl=$(cat "$FAKE_LOG_DIR/listn.$dn" 2>/dev/null || echo 0)
    echo $((nl + 1)) > "$FAKE_LOG_DIR/listn.$dn"
    if (( nl >= 1 )) && [[ -f "$FAKE_LOG_DIR/list2.$dn" ]]; then
      cat "$FAKE_LOG_DIR/list2.$dn"; exit 0
    fi
    if [[ -f "$FAKE_LOG_DIR/list.$dn" ]]; then cat "$FAKE_LOG_DIR/list.$dn"; exit 0; fi
    if [[ -n "${FAKE_ODD_DN:-}" && "$dn" == "$FAKE_ODD_DN" ]]; then
      [[ -n "${FAKE_LIST_ODD:-}" ]] && printf '%b\\n' "$FAKE_LIST_ODD"; exit 0
    fi
    [[ -n "${FAKE_LIST:-}" ]] && printf '%b\\n' "$FAKE_LIST"; exit 0 ;;
  *"custom-jobs describe"*)
    if [[ "${FAKE_UNIQUE_IDS:-0}" == 1 ]]; then
      id=$(sed -n 's/.*custom-jobs describe \\([0-9]*\\).*/\\1/p' <<<"$*")
      mode=$(cat "$FAKE_LOG_DIR/job.$id" 2>/dev/null)
      now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      if [[ -n "$mode" && " ${FAKE_DIE_MODES:-} " == *" $mode "* ]]; then
        # kill the batch's worker for this run: the second-to-last script process up the chain
        # (GNU timeout is a process of its own between gcloud and the script; the macOS shim execs)
        chain=(); p=$PPID
        while (( p > 1 )); do
          if [[ "$(ps -o command= -p "$p")" == *launch_v6_matrix.sh* ]]; then
            chain+=("$p")
          elif (( ${#chain[@]} > 0 )); then
            break
          fi
          p=$(ps -o ppid= -p "$p" | tr -d ' ')
        done
        kill -9 "${chain[$(( ${#chain[@]} - 2 ))]}"; exit 0
      fi
      if [[ -n "$mode" && " ${FAKE_PENDING_MODES:-} " == *" $mode "* ]]; then
        printf 'JOB_STATE_PENDING\\t2026-01-01T00:00:00Z\\n'; exit 0
      fi
      k=$(cat "$FAKE_LOG_DIR/desc.$id" 2>/dev/null || echo 0)
      echo $((k + 1)) > "$FAKE_LOG_DIR/desc.$id"
      polls=${FAKE_RUN_POLLS:-2}
      if [[ -n "$mode" && " ${FAKE_SLOW_MODES:-} " == *" $mode "* ]]; then
        polls=${FAKE_SLOW_POLLS:-8}
      fi
      if (( k < polls )); then
        printf 'JOB_STATE_RUNNING\\t%s\\t%s\\n' "$now" "$now"; exit 0
      fi
      state=JOB_STATE_SUCCEEDED
      if [[ -n "$mode" && " ${FAKE_FAILED_MODES:-} " == *" $mode "* ]]; then
        state=JOB_STATE_FAILED
      fi
      (( k == polls )) && echo "-$id" >> "$FAKE_LOG_DIR/events.log"
      printf '%s\\t%s\\t%s\\n' "$state" "$now" "$now"; exit 0
    fi
    n=$(cat "$FAKE_LOG_DIR/describe.n" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$FAKE_LOG_DIR/describe.n"
    IFS='|' read -r -a seq <<<"${FAKE_DESCRIBE:-}"
    (( n >= ${#seq[@]} )) && n=$(( ${#seq[@]} - 1 ))
    (( n < 0 )) && exit 0
    printf '%b\\n' "${seq[$n]}"; exit 0 ;;
  *"custom-jobs cancel"*) [[ "${FAKE_CANCEL_FAIL:-0}" == 1 ]] && exit 1; exit 0 ;;
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
        timeout = int(extra.pop("TEST_TIMEOUT", "60"))
        env = {**self.base, **extra}
        env = {k: v for k, v in env.items() if v != ""}
        assert BASH is not None
        return subprocess.run(  # noqa: S603
            [BASH, str(self.script), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def log(self, name: str) -> str:
        path = self.log_dir / name
        return path.read_text() if path.exists() else ""


def _gnu_tool_path(tmp_path: Path) -> str:
    """PATH for the script under test: ``/usr/bin:/bin``, plus shims where they lack GNU tools.

    The script needs GNU ``timeout`` and ``date -d`` (a Linux operator machine, and CI). On a Mac
    those are missing; the shims below only cover what the script calls: ``timeout [-k N] SEC cmd``
    runs ``cmd``, and ``date -d <ISO time> +%s`` prints the epoch seconds.
    """
    base = "/usr/bin:/bin"
    probe = subprocess.run(  # noqa: S603
        ["/bin/sh", "-c", "command -v timeout >/dev/null && date -d @0 +%s >/dev/null 2>&1"],
        env={"PATH": base},
        check=False,
    )
    if probe.returncode == 0:
        return base
    shims = tmp_path / "gnu-shims"
    _write_exec(shims / "timeout", '#!/bin/bash\n[[ "$1" == -k ]] && shift 2\nshift\nexec "$@"\n')
    _write_exec(
        shims / "date",
        '#!/bin/bash\nif [[ "${1:-}" == -d ]]; then\n'
        f"  exec {sys.executable} -c 'import sys, datetime as d; "
        't = sys.argv[1].replace("Z", "+00:00"); '
        'print(int(d.datetime.fromisoformat(t).timestamp()))\' "$2"\n'
        'fi\nexec /bin/date "$@"\n',
    )
    return f"{shims}:{base}"


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
        "PATH": _gnu_tool_path(tmp_path),
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


# --- spatial holdout batch ----------------------------------------------------------------

HOLDOUT_FLAGS = ("--max-total-usd", "100", "--include-stage1")


def hdn(arm: str, seed: int) -> str:
    return f"{MODES[arm]}-spatial-holdout-{PREFIX}-{arm.lower()}-seed{seed}-ep40-cos-notest"


def launcher_calls(env: Env) -> list[str]:
    return env.log("launcher.log").splitlines()


def launched(env: Env) -> list[tuple[str, int]]:
    out = []
    for line in launcher_calls(env):
        mode = re.search(r"--mode (\S+)", line)
        seed = re.search(r"--seed (\d)", line)
        assert mode and seed
        arm = next(a for a in ARMS if f"{MODES[a]}-spatial-holdout" == mode.group(1))
        out.append((arm, int(seed.group(1))))
    return out


def set_listed(env: Env, arm: str, seed: int, row: str) -> None:
    (env.log_dir / f"list.{hdn(arm, seed)}").write_text(row + "\n")


def max_in_flight(env: Env) -> int:
    live, peak, started = 0, 0, set()
    for event in env.log("events.log").split():
        if event.startswith("+"):
            started.add(event[1:])
            live += 1
            peak = max(peak, live)
        elif event[1:] in started:
            live -= 1
    assert live == 0
    return peak


BATCH = {"FAKE_SUBMIT": "1", "FAKE_UNIQUE_IDS": "1", "V6M_BATCH_TICK_SECONDS": "1"}


def test_holdout_constants_equal_the_launcher_caps() -> None:
    source = (REPO / "scripts" / "operators" / "launch_vertex_modeling.py").read_text()
    timeout = re.search(r"^SPATIAL_HOLDOUT_TIMEOUT_SECONDS = (\d+)$", source, re.M)
    exposure = re.search(r"^SPATIAL_HOLDOUT_MAX_EXPOSURE_USD = ([\d.]+)$", source, re.M)
    assert timeout and exposure
    script = SCRIPT.read_text()
    assert f"HOLDOUT_TIMEOUT_SECONDS={timeout.group(1)}\n" in script
    assert f"HOLDOUT_EXPOSURE_CENTS={round(float(exposure.group(1)) * 100)}\n" in script


def test_holdout_dry_run_prints_the_plan_and_touches_no_cloud(env: Env) -> None:
    res = env.run(
        "holdout",
        "--dry-run",
        "--max-total-usd",
        "100",
        "--include-stage1",
        V6M_IMAGE_URI="",
        V6M_SOURCE_SHA="",
        V6M_RUN_DIR="",
    )
    assert res.returncode == 0, res.stderr
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    assert len(plan) == 27
    for arm in ARMS:
        for seed in (0, 1, 2):
            assert any(
                f"jobname={hdn(arm, seed)}" in line
                and f"label={PREFIX}-{arm.lower()} " in line
                and f"mode={MODES[arm]}-spatial-holdout" in line
                for line in plan
            )
    # seeds first: the whole of seed 0 precedes seed 1
    assert [line.split()[2] for line in plan] == ["seed0"] * 9 + ["seed1"] * 9 + ["seed2"] * 9
    assert "summed exposure caps of all 27 run(s): USD 87.75" in res.stdout
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""


def test_holdout_dry_run_prices_the_requested_runs_against_the_limit(env: Env) -> None:
    args = ("holdout", "--dry-run", "--arms", "L5,LOO-M", "--seeds", "0,1")
    res = env.run(*args, "--max-total-usd", "13.00")
    assert res.returncode == 0 and "USD 13.00" in res.stdout
    assert [line.split()[1] for line in res.stdout.splitlines() if line.startswith("PLAN")] == [
        "L5",
        "LOO-M",
        "L5",
        "LOO-M",
    ]
    over = env.run(*args, "--max-total-usd", "12.99")
    assert over.returncode == 2 and "REFUSED" in over.stdout
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


@pytest.mark.parametrize(
    "args",
    [
        ("holdout",),  # --max-total-usd is required
        ("holdout", "--dry-run"),
        ("holdout", "--max-total-usd", "x", "--dry-run"),
        ("holdout", "--max-total-usd", "1.234", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--max-parallel", "5", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--max-parallel", "0", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--arms", "L9", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--arms", "L1,L1", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--seeds", "3", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--seeds", "", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--bogus", "--dry-run"),
        ("holdout", "--max-total-usd", "10", "--dry-run", "--preflight"),
    ],
)
def test_holdout_bad_arguments_fail_closed(env: Env, args: tuple[str, ...]) -> None:
    res = env.run(*args)
    assert res.returncode == 2
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""


@pytest.mark.parametrize(
    "bad",
    [
        {"V6M_TIMEOUT_SECONDS": "10801"},
        {"V6M_MAX_WAIT_SECONDS": "1800"},  # 10800 s + 1800 s would project USD 3.50
        {"V6M_HOURLY_RATE_USD": "1.01"},
    ],
)
def test_holdout_keeps_the_launcher_caps(env: Env, bad: dict[str, str]) -> None:
    res = env.run("holdout", "--dry-run", *HOLDOUT_FLAGS, **bad)
    assert res.returncode == 2
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


def test_holdout_preflight_runs_the_launcher_preflight_only(env: Env) -> None:
    res = env.run(
        "holdout", "--preflight", "--arms", "L5,NO-RED-NIR-IDX", "--seeds", "0", V6M_RUN_DIR=""
    )
    assert res.returncode == 0, res.stderr
    lines = launcher_calls(env)
    assert len(lines) == 2
    assert all("--preflight" in line and "--skip-test" in line for line in lines)
    assert "--mode v6-l5-spatial-holdout" in lines[0]
    assert "--mode v6-no-red-nir-index-spatial-holdout" in lines[1]
    assert env.log("gcloud.log") == ""


def test_holdout_submits_with_the_holdout_arguments_and_skips_what_exists(env: Env) -> None:
    set_listed(env, "L1", 0, "111\tJOB_STATE_SUCCEEDED")
    set_listed(env, "L2", 0, "222\tJOB_STATE_RUNNING")
    res = env.run("holdout", *HOLDOUT_FLAGS, "--arms", "L1,L2,L5", "--seeds", "0", **BATCH)
    assert res.returncode == 0, res.stdout + res.stderr
    assert launched(env) == [("L5", 0)]  # L1 succeeded, L2 is attached to, only L5 is submitted
    (line,) = launcher_calls(env)
    for token in (
        "--mode v6-l5-spatial-holdout",
        "--seed 0",
        "--max-epochs 40",
        "--lr-schedule cosine",
        "--skip-test",
        "--timeout-seconds 10800",
        "--max-wait-seconds 600",
        f"--run-label {PREFIX}-l5",
        "--project berlin-lst-training",
    ):
        assert token in line
    assert "--patch-cache" not in line and "--preflight" not in line
    assert "custom-jobs describe 222" in env.log("gcloud.log")
    assert "2 succeeded, 1 skipped" in res.stdout
    # every lookup is by the exact holdout displayName
    assert listed_dns(env.log("gcloud.log"))[:3] == [hdn("L1", 0), hdn("L2", 0), hdn("L5", 0)]


def test_holdout_everything_done_submits_nothing(env: Env) -> None:
    res = env.run(
        "holdout", *HOLDOUT_FLAGS, "--seeds", "0", FAKE_LIST="999\tJOB_STATE_SUCCEEDED", **BATCH
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert launcher_calls(env) == [] and "9 skipped" in res.stdout
    assert "custom-jobs describe" not in env.log("gcloud.log")


def test_holdout_refuses_before_any_submission_when_the_caps_exceed_the_limit(env: Env) -> None:
    arms = ("--arms", "L1,L2,L3", "--seeds", "0", "--include-stage1")
    res = env.run("holdout", "--max-total-usd", "9.74", *arms, **BATCH)
    assert res.returncode == 2 and "REFUSED" in res.stdout
    assert launcher_calls(env) == []
    assert len(listed_dns(env.log("gcloud.log"))) == 3  # looked up, nothing submitted
    # what already succeeded is not part of the sum: two runs left, 2 x 3.25 = 6.50
    set_listed(env, "L1", 0, "111\tJOB_STATE_SUCCEEDED")
    below = env.run("holdout", "--max-total-usd", "6.49", *arms, **BATCH)
    assert below.returncode == 2 and launcher_calls(env) == []
    ok = env.run("holdout", "--max-total-usd", "6.50", *arms, **BATCH)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert sorted(launched(env)) == [("L2", 0), ("L3", 0)]
    assert "summed exposure caps USD 6.50, limit USD 6.50" in ok.stdout


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"])
def test_holdout_refuses_a_pair_with_only_unsuccessful_jobs_unless_opted_in(
    env: Env, state: str
) -> None:
    set_listed(env, "L3", 0, f"333\t{state}")
    args = ("holdout", *HOLDOUT_FLAGS, "--arms", "L2,L3", "--seeds", "0")
    res = env.run(*args, **BATCH)
    assert res.returncode == 1 and "REFUSED: 1 pair(s)" in res.stdout and "BLOCKED" in res.stdout
    assert launcher_calls(env) == []  # not even the clean L2 run: the batch refuses as a whole
    wrong = "V6M_RESUBMIT_CANCELLED" if state == "JOB_STATE_FAILED" else "V6M_RESUBMIT_FAILED"
    assert env.run(*args, **{wrong: "1"}, **BATCH).returncode == 1
    right = "V6M_RESUBMIT_FAILED" if state == "JOB_STATE_FAILED" else "V6M_RESUBMIT_CANCELLED"
    ok = env.run(*args, **{right: "1"}, **BATCH)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert sorted(launched(env)) == [("L2", 0), ("L3", 0)]


@pytest.mark.parametrize(("parallel", "expected"), [(None, 4), ("2", 2)])
def test_holdout_keeps_at_most_n_jobs_in_flight(
    env: Env, parallel: str | None, expected: int
) -> None:
    flags = ["--max-parallel", parallel] if parallel else []
    res = env.run(
        "holdout",
        "--max-total-usd",
        "29.25",  # nine runs x 3.25: exact
        "--include-stage1",
        "--seeds",
        "0",
        *flags,
        FAKE_RUN_POLLS="3",
        TEST_TIMEOUT="150",
        **BATCH,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(launched(env)) == sorted((arm, 0) for arm in ARMS)
    assert max_in_flight(env) == expected
    assert "9 succeeded" in res.stdout


def test_holdout_continues_after_a_capacity_miss_and_exits_3(env: Env) -> None:
    res = env.run(
        "holdout",
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2,L3",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        FAKE_PENDING_MODES="v6-l1-spatial-holdout",
        **BATCH,
    )
    assert res.returncode == 3, res.stdout + res.stderr
    assert launched(env) == [("L1", 0), ("L2", 0), ("L3", 0)]
    assert "custom-jobs cancel" in env.log("gcloud.log")
    assert "1 without capacity [L1/seed0]" in res.stdout and "2 succeeded" in res.stdout


def test_holdout_treats_a_capacity_refusal_at_submission_as_capacity(env: Env) -> None:
    res = env.run(
        "holdout",
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2,L3",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        FAKE_LAUNCH_FAIL_MODES="v6-l2-spatial-holdout",
        FAKE_LAUNCH_FAIL_MSG="google.api_core.exceptions.ResourceExhausted: RESOURCE_EXHAUSTED",
        **BATCH,
    )
    assert res.returncode == 3, res.stdout + res.stderr
    assert launched(env) == [("L1", 0), ("L2", 0), ("L3", 0)]
    assert "1 without capacity [L2/seed0]" in res.stdout


@pytest.mark.parametrize("how", ["launcher", "job"])
def test_holdout_stops_on_any_other_failure_and_leaves_running_jobs_alone(
    env: Env, how: str
) -> None:
    fail = (
        {"FAKE_LAUNCH_FAIL_MODES": "v6-l2-spatial-holdout"}
        if how == "launcher"
        else {"FAKE_FAILED_MODES": "v6-l2-spatial-holdout"}
    )
    res = env.run(
        "holdout",
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2,L3,L4",
        "--seeds",
        "0",
        "--max-parallel",
        "2",
        # L1 runs long, so no slot frees before L2's failure is known: nothing may start in between
        FAKE_RUN_POLLS="1",
        FAKE_SLOW_MODES="v6-l1-spatial-holdout",
        FAKE_SLOW_POLLS="8",
        **fail,
        **BATCH,
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert ("L1", 0) in launched(env) and ("L2", 0) in launched(env)
    assert ("L3", 0) not in launched(env) and ("L4", 0) not in launched(env)
    assert "1 failed [L2/seed0]" in res.stdout and "not started [L3/seed0 L4/seed0]" in res.stdout
    assert "custom-jobs cancel" not in env.log("gcloud.log")
    assert "1 succeeded" in res.stdout  # the job in flight was waited for


# --- review fixes: attach slots, unconfirmed cancel, lock, stage 1, dead workers ---------------


def run_batch(env: Env, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return env.run("holdout", *args, **{**BATCH, **extra})


def test_holdout_default_skips_the_two_stage_one_runs(env: Env) -> None:
    res = env.run("holdout", "--dry-run", "--max-total-usd", "81.25")
    assert res.returncode == 0, res.stderr
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    assert len(plan) == 25 and "USD 81.25" in res.stdout
    stage_one = [ln for ln in plan if ln.split()[2] == "seed0" and ln.split()[1] in ("L1", "L4")]
    assert stage_one == []
    assert "NOTE L1 seed0 is a stage-1 run" in res.stdout
    assert "v6-l4-spatial-holdout-v6sh-99a71cf-l4-seed0-ep40-cos-notest" in res.stdout
    only = env.run(
        "holdout", "--dry-run", "--max-total-usd", "9.75", "--arms", "L1,L4", "--seeds", "0"
    )
    assert only.returncode == 2 and "only stage-1 runs" in only.stderr
    full = env.run("holdout", "--dry-run", "--max-total-usd", "87.75", "--include-stage1")
    assert full.returncode == 0 and "USD 87.75" in full.stdout
    assert len([ln for ln in full.stdout.splitlines() if ln.startswith("PLAN")]) == 27


def test_holdout_active_jobs_hold_their_slots_before_anything_new_starts(env: Env) -> None:
    # Ctrl-C then resume: L1-L4 are still running on Vertex, L5 was cancelled before
    for arm in ("L1", "L2", "L3", "L4"):
        set_listed(env, arm, 0, f"11{arm[1]}\tJOB_STATE_RUNNING")
    set_listed(env, "L5", 0, "555\tJOB_STATE_CANCELLED")
    args = (*HOLDOUT_FLAGS, "--arms", "L1,L2,L3,L4,L5", "--seeds", "0", "--max-parallel", "4")
    blocked = run_batch(env, *args)
    assert blocked.returncode == 1 and "BLOCKED" in blocked.stdout
    assert launcher_calls(env) == []
    res = run_batch(env, *args, V6M_RESUBMIT_CANCELLED="1", FAKE_RUN_POLLS="3")
    assert res.returncode == 0, res.stdout + res.stderr
    assert launched(env) == [("L5", 0)]
    out = res.stdout
    assert "4 active job(s) to watch first" in out
    # L5 waits for a slot: it starts only after a watched job has ended
    assert out.index("started L5") > out.index("final: JOB_STATE_SUCCEEDED")
    assert "summed exposure caps USD 3.25" in out


def test_holdout_more_active_jobs_than_slots_start_nothing_new_until_they_finish(
    env: Env,
) -> None:
    for arm in ("L2", "L3", "L5"):
        set_listed(env, arm, 0, f"11{arm[1]}\tJOB_STATE_RUNNING")
    res = run_batch(
        env,
        *HOLDOUT_FLAGS,
        "--arms",
        "L2,L3,L5,LOO-M",
        "--seeds",
        "0",
        "--max-parallel",
        "2",
        FAKE_RUN_POLLS="2",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert launched(env) == [("LOO-M", 0)]
    started = [m.group(1) for m in re.finditer(r"started (\S+) seed0", res.stdout)]
    assert started == ["L2", "L3", "L5", "LOO-M"]  # watched jobs first, the new run last


def test_holdout_attached_pair_never_submits_when_its_job_ended_meanwhile(env: Env) -> None:
    set_listed(env, "L1", 0, "111\tJOB_STATE_RUNNING")
    set_listed(env, "L2", 0, "222\tJOB_STATE_RUNNING")
    # at its own turn (second lookup) L2's job is FAILED; the opt-in must not make it a new job
    (env.log_dir / f"list2.{hdn('L2', 0)}").write_text("222\tJOB_STATE_FAILED\n")
    res = run_batch(
        env,
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        V6M_RESUBMIT_FAILED="1",
        V6M_RESUBMIT_CANCELLED="1",
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert launcher_calls(env) == []
    assert "planned as attached to an active job" in res.stdout
    assert "1 failed [L2/seed0]" in res.stdout


def test_holdout_unconfirmed_cancel_is_a_failure_and_names_the_job(env: Env) -> None:
    res = run_batch(
        env,
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2,L3",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        FAKE_PENDING_MODES="v6-l1-spatial-holdout",
        FAKE_CANCEL_FAIL="1",
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert launched(env) == [("L1", 0)]  # nothing was started in its place
    assert "CANCEL_UNCONFIRMED" in res.stdout
    assert re.search(r"cancel request for \d+ FAILED", res.stdout + res.stderr)
    assert "1 failed [L1/seed0]" in res.stdout and "0 without capacity" in res.stdout


def test_holdout_second_batch_in_the_same_run_directory_is_refused(env: Env) -> None:
    lock = env.run_dir / "holdout.lock"
    lock.mkdir()
    res = run_batch(env, *HOLDOUT_FLAGS, "--arms", "L2", "--seeds", "0")
    assert res.returncode == 2
    assert str(lock) in res.stderr and "lock" in res.stderr
    assert env.log("gcloud.log") == "" and launcher_calls(env) == []
    assert lock.is_dir()  # someone else's lock is left alone
    lock.rmdir()
    ok = run_batch(env, *HOLDOUT_FLAGS, "--arms", "L2", "--seeds", "0")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert not lock.exists()  # released on exit
    refused = run_batch(env, "--arms", "L2", "--seeds", "0", "--max-total-usd", "1")
    assert refused.returncode == 2 and not lock.exists()  # also released when it refuses


def test_holdout_dry_run_and_preflight_take_no_lock(env: Env) -> None:
    (env.run_dir / "holdout.lock").mkdir()
    assert env.run("holdout", "--dry-run", *HOLDOUT_FLAGS).returncode == 0
    assert env.run("holdout", "--preflight", "--arms", "L2", "--seeds", "0").returncode == 0


def test_holdout_dead_worker_without_a_result_is_a_failure_not_a_held_slot(env: Env) -> None:
    res = run_batch(
        env,
        *HOLDOUT_FLAGS,
        "--arms",
        "L1,L2,L3",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        FAKE_DIE_MODES="v6-l2-spatial-holdout",
        TEST_TIMEOUT="90",
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "ended without a result" in res.stdout
    assert "1 failed [L2/seed0]" in res.stdout and "not started [L3/seed0]" in res.stdout
    assert ("L3", 0) not in launched(env)


# --- spatial holdout batch: cross-validation folds ----------------------------------------

FOLDS = [f"cv5-fold{k}" for k in range(5)]
FOLD_FLAGS = ("--assignments", ",".join(FOLDS), "--arms", "l1,l4", "--seeds", "0")


def fdn(arm: str, fold: str) -> str:
    return f"{MODES[arm]}-spatial-holdout-{PREFIX}-{arm.lower()}-{fold}-seed0-ep40-cos-notest"


def launched_folds(env: Env) -> list[tuple[str, str]]:
    out = []
    for line in launcher_calls(env):
        mode = re.search(r"--mode (\S+)", line)
        fold = re.search(r"--assignment (\S+)", line)
        assert mode and fold
        arm = next(a for a in ARMS if f"{MODES[a]}-spatial-holdout" == mode.group(1))
        out.append((arm, fold.group(1)))
    return out


def test_cv_constants_equal_the_python_side() -> None:
    from berlin_lst_downscaling.modeling import spatial_holdout as sh

    script = SCRIPT.read_text()
    assert f"CV_ASSIGNMENTS=({' '.join(sh.SPATIAL_HOLDOUT_CV_ASSIGNMENTS)})\n" in script
    assert f"CV_ARMS=({' '.join(sh.SPATIAL_HOLDOUT_CV_ARMS)})\n" in script
    assert FOLDS == list(sh.SPATIAL_HOLDOUT_CV_ASSIGNMENTS)


def test_holdout_fold_dry_run_prints_ten_distinct_labels_at_the_caps(env: Env) -> None:
    res = env.run("holdout", "--dry-run", "--max-total-usd", "32.50", *FOLD_FLAGS)
    assert res.returncode == 0, res.stderr
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    assert len(plan) == 10 and "summed exposure caps of all 10 run(s): USD 32.50" in res.stdout
    labels = [re.search(r"label=(\S+)", line).group(1) for line in plan]  # type: ignore[union-attr]
    assert len(set(labels)) == 10
    assert f"{PREFIX}-l1-cv5-fold0" in labels and f"{PREFIX}-l4-cv5-fold4" in labels
    for arm in ("L1", "L4"):
        for fold in FOLDS:
            assert any(
                f"jobname={fdn(arm, fold)} " in line
                and f"label={PREFIX}-{arm.lower()}-{fold} " in line
                for line in plan
            )
    # the stage-1 skip is for v1 only: seed-0 L1 and L4 are planned under every fold
    assert "NOTE" not in res.stdout
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""
    over = env.run("holdout", "--dry-run", "--max-total-usd", "32.49", *FOLD_FLAGS)
    assert over.returncode == 2 and "REFUSED" in over.stdout


def test_holdout_v1_and_folds_in_one_batch_keep_the_stage_one_skip_for_v1_only(env: Env) -> None:
    res = env.run(
        "holdout",
        "--dry-run",
        "--max-total-usd",
        "6.50",
        "--assignments",
        "v1,cv5-fold0",
        "--arms",
        "L1,L4",
        "--seeds",
        "0",
    )
    assert res.returncode == 0, res.stderr
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    assert [line.split()[1] for line in plan] == ["L1", "L4"]
    assert all("-cv5-fold0-seed0" in line for line in plan)
    assert res.stdout.count("NOTE") == 2  # the two v1 runs are skipped as stage 1


@pytest.mark.parametrize(
    "args",
    [
        ("--assignments", "cv5-fold5", "--arms", "L1", "--seeds", "0"),
        ("--assignments", "cv5-fold0", "--arms", "L1,L2", "--seeds", "0"),
        ("--assignments", "cv5-fold0", "--arms", "all", "--seeds", "0"),
        ("--assignments", "cv5-fold0", "--arms", "L1", "--seeds", "0,1"),
        ("--assignments", "cv5-fold0", "--arms", "L1", "--seeds", "1"),
        ("--assignments", "cv5-fold0,cv5-fold0", "--arms", "L1", "--seeds", "0"),
        ("--assignments", "", "--arms", "L1", "--seeds", "0"),
        ("--assignments", "v2", "--arms", "L1", "--seeds", "0"),
        ("--assignments", "cv5-fold0,v1", "--arms", "L1,L5", "--seeds", "0"),
    ],
)
def test_holdout_folds_are_refused_for_other_assignments_arms_and_seeds(
    env: Env, args: tuple[str, ...]
) -> None:
    res = env.run("holdout", "--dry-run", "--max-total-usd", "100", *args)
    assert res.returncode == 2, res.stdout
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""


def test_holdout_fold_preflight_passes_the_assignment_to_the_launcher(env: Env) -> None:
    res = env.run("holdout", "--preflight", *FOLD_FLAGS, V6M_RUN_DIR="")
    assert res.returncode == 0, res.stderr
    lines = launcher_calls(env)
    assert len(lines) == 10
    assert all("--preflight" in line and "--skip-test" in line for line in lines)
    assert sorted(launched_folds(env)) == sorted((a, f) for a in ("L1", "L4") for f in FOLDS)
    assert all(f"--run-label {PREFIX}-" in line and "--seed 0" in line for line in lines)
    # a v1 run never gets the flag
    env.run("holdout", "--preflight", "--arms", "L5", "--seeds", "0", V6M_RUN_DIR="")
    assert "--assignment" not in launcher_calls(env)[-1]


def test_holdout_fold_jobnames_are_the_launchers_display_names(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    python_launcher = load_launcher(monkeypatch)
    # Batch idempotency looks a run up by displayName: the script must build the launcher's name.
    res = env.run("holdout", "--dry-run", "--max-total-usd", "32.50", *FOLD_FLAGS)
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    assert len(plan) == 10
    for line in plan:
        arm = line.split()[1]
        label = re.search(r"label=(\S+)", line).group(1)  # type: ignore[union-attr]
        jobname = re.search(r"jobname=(\S+)", line).group(1)  # type: ignore[union-attr]
        fold = re.search(r"(cv5-fold\d)$", label).group(1)  # type: ignore[union-attr]
        argv = [
            "launch_vertex_modeling.py",
            "--mode",
            f"{MODES[arm]}-spatial-holdout",
            "--image-uri",
            IMAGE,
            "--source-sha",
            "0" * 40,
            "--run-label",
            label.removesuffix(f"-{fold}"),
            "--service-account",
            "sa",
            "--infisical-identity",
            "i",
            "--infisical-project",
            "p",
            "--seed",
            "0",
            "--max-epochs",
            "40",
            "--lr-schedule",
            "cosine",
            "--skip-test",
            "--assignment",
            fold,
            "--hourly-rate-usd",
            "1.0",
            "--preflight",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        assert python_launcher.main() == 0
        run_label = re.search(r"run label +: (\S+)", capsys.readouterr().out).group(1)  # type: ignore[union-attr]
        assert jobname == f"{MODES[arm]}-spatial-holdout-{run_label}"


def test_holdout_fold_batch_submits_each_fold_once_within_four_in_flight(env: Env) -> None:
    res = env.run("holdout", "--max-total-usd", "32.50", *FOLD_FLAGS, **BATCH, TEST_TIMEOUT="120")
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(launched_folds(env)) == sorted((a, f) for a in ("L1", "L4") for f in FOLDS)
    assert max_in_flight(env) <= 4
    assert "10 succeeded, 0 skipped" in res.stdout
    looked_up = listed_dns(env.log("gcloud.log"))[:10]
    assert sorted(looked_up) == sorted(fdn(a, f) for a in ("L1", "L4") for f in FOLDS)
    # one log per run: no two folds share a result or submit log
    assert len({p.name for p in env.run_dir.glob("submit-*.log")}) == 10


def test_holdout_fold_batch_skips_a_succeeded_fold_and_prices_the_rest(env: Env) -> None:
    (env.log_dir / f"list.{fdn('L1', 'cv5-fold2')}").write_text("111\tJOB_STATE_SUCCEEDED\n")
    res = env.run("holdout", "--max-total-usd", "29.25", *FOLD_FLAGS, **BATCH, TEST_TIMEOUT="120")
    assert res.returncode == 0, res.stdout + res.stderr
    assert ("L1", "cv5-fold2") not in launched_folds(env) and len(launched_folds(env)) == 9
    over = env.run("holdout", "--max-total-usd", "29.24", *FOLD_FLAGS, **BATCH)
    assert over.returncode == 2 and "REFUSED" in over.stdout


def test_holdout_fold_attached_pair_never_submits_when_its_job_ended_meanwhile(env: Env) -> None:
    (env.log_dir / f"list.{fdn('L1', 'cv5-fold0')}").write_text("111\tJOB_STATE_RUNNING\n")
    (env.log_dir / f"list.{fdn('L4', 'cv5-fold0')}").write_text("222\tJOB_STATE_RUNNING\n")
    # at its own turn (second lookup) the L4 fold job is FAILED; the opt-in must not make it new
    (env.log_dir / f"list2.{fdn('L4', 'cv5-fold0')}").write_text("222\tJOB_STATE_FAILED\n")
    res = env.run(
        "holdout",
        "--max-total-usd",
        "100",
        "--assignments",
        "cv5-fold0",
        "--arms",
        "L1,L4",
        "--seeds",
        "0",
        "--max-parallel",
        "1",
        **BATCH,
        V6M_RESUBMIT_FAILED="1",
        V6M_RESUBMIT_CANCELLED="1",
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert launcher_calls(env) == []
    assert "planned as attached to an active job" in res.stdout
    assert "1 failed [L4/seed0/cv5-fold0]" in res.stdout


# --- GNU tools: fail at once, not after minutes of retries -----------------------------------

CLOUD_COMMANDS = [
    ("preflight",),
    ("preflight", "L1", "0"),
    ("submit", "L1", "0"),
    ("submit-all",),
    ("state", "L1", "0"),
    ("holdout", "--preflight", "--arms", "L1", "--seeds", "1"),
    ("holdout", "--max-total-usd", "100", "--arms", "L1", "--seeds", "1"),
]


def _bsd_date(env: Env) -> str:
    shim = env.root / "bsd-date"
    _write_exec(
        shim,
        '#!/bin/bash\n[[ "${1:-}" == -d ]] && { echo "illegal option -- d" >&2; exit 1; }\n'
        'exec /bin/date "$@"\n',
    )
    return str(shim)


@pytest.mark.parametrize("args", CLOUD_COMMANDS)
@pytest.mark.parametrize("missing", ["timeout", "date"])
def test_cloud_subcommands_fail_at_once_without_gnu_timeout_or_date(
    env: Env, args: tuple[str, ...], missing: str
) -> None:
    extra = (
        {"V6M_TIMEOUT_BIN": str(env.root / "no-such-timeout")}
        if missing == "timeout"
        else {"V6M_DATE_BIN": _bsd_date(env)}
    )
    started = time.monotonic()
    res = env.run(*args, **extra)
    assert time.monotonic() - started < 10
    assert res.returncode == 2, res.stdout + res.stderr
    assert "brew install coreutils" in res.stderr and "gnubin" in res.stderr
    assert ("GNU 'timeout'" if missing == "timeout" else "-d' is not GNU date") in res.stderr
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""  # nothing was looked up, launched or retried


def test_help_and_dry_run_need_no_gnu_tools(env: Env) -> None:
    gone = {"V6M_TIMEOUT_BIN": str(env.root / "no-such-timeout"), "V6M_DATE_BIN": _bsd_date(env)}
    assert env.run("help", **gone).returncode == 0
    res = env.run("holdout", "--dry-run", "--max-total-usd", "100", "--include-stage1", **gone)
    assert res.returncode == 0, res.stderr
    assert env.log("wrapper.log") == "" and env.log("gcloud.log") == ""


# --- TESSERA seeds in the holdout batch ---------------------------------------------------

TESSERA_ARMS = "TESSERA,TESSERA-TEMPORAL"


def tdn(arm: str, seed: int) -> str:
    mode = {"TESSERA": "tessera-spatial-holdout", "TESSERA-TEMPORAL": "tessera"}[arm]
    return f"{mode}-{PREFIX}-{arm.lower()}-seed{seed}-ep40-cos-notest"


def test_tessera_constants_equal_the_launcher_caps() -> None:
    source = (REPO / "scripts" / "operators" / "launch_vertex_modeling.py").read_text()
    timeout = re.search(r"^TESSERA_TIMEOUT_SECONDS = (\d+)$", source, re.M)
    exposure = re.search(r"^TESSERA_MAX_EXPOSURE_USD = ([\d.]+)$", source, re.M)
    assert timeout and exposure
    script = SCRIPT.read_text()
    assert f"TESSERA_TIMEOUT_SECONDS={timeout.group(1)}\n" in script
    assert f"TESSERA_EXPOSURE_CENTS={round(float(exposure.group(1)) * 100)}\n" in script


def test_tessera_dry_run_prices_three_temporal_and_three_holdout_seeds(env: Env) -> None:
    res = env.run(
        "holdout",
        "--dry-run",
        "--arms",
        TESSERA_ARMS,
        "--max-total-usd",
        "37.50",
        V6M_IMAGE_URI="",
        V6M_SOURCE_SHA="",
        V6M_RUN_DIR="",
    )
    assert res.returncode == 0, res.stderr
    plan = [line for line in res.stdout.splitlines() if line.startswith("PLAN")]
    names = [re.search(r"jobname=(\S+)", line).group(1) for line in plan]  # type: ignore[union-attr]
    assert len(plan) == len(set(names)) == 6
    assert set(names) == {tdn(a, s) for a in TESSERA_ARMS.split(",") for s in (0, 1, 2)}
    assert all("cap=USD 6.25" in line for line in plan)
    assert "summed exposure caps of all 6 run(s): USD 37.50; limit USD 37.50" in res.stdout
    for name in ("wrapper.log", "gcloud.log", "launcher.log"):
        assert env.log(name) == ""
    low = env.run("holdout", "--dry-run", "--arms", TESSERA_ARMS, "--max-total-usd", "37.49")
    assert low.returncode == 2 and "REFUSED" in low.stdout


def test_tessera_is_never_part_of_all_and_refuses_what_the_arm_does_not_have(env: Env) -> None:
    res = env.run("holdout", "--dry-run", *HOLDOUT_FLAGS)
    assert "TESSERA" not in res.stdout
    dry = ("holdout", "--dry-run", "--max-total-usd", "100", "--arms", "TESSERA")
    for args in (
        (*dry, "--assignments", "cv5-fold0"),
        ("submit", "TESSERA", "0"),
        ("state", "TESSERA-TEMPORAL", "0"),
    ):
        assert env.run(*args).returncode == 2, args
    over = env.run(*dry, V6M_TESSERA_TIMEOUT_SECONDS="20401")
    assert over.returncode == 2 and "TESSERA cap" in over.stderr


def test_tessera_preflight_uses_the_tessera_bounds(env: Env) -> None:
    pre = env.run("holdout", "--preflight", "--arms", TESSERA_ARMS, "--seeds", "0", V6M_RUN_DIR="")
    assert pre.returncode == 0, pre.stderr
    lines = launcher_calls(env)
    assert len(lines) == 2 and all("--preflight" in line for line in lines)
    for line, mode in zip(lines, ("tessera-spatial-holdout", "tessera"), strict=True):
        assert re.search(rf"--mode {mode} ", line) and "--skip-test" in line
        assert "--patch-cache prebuilt" in line and "--timeout-seconds 20400" in line
        assert "--max-wait-seconds 600" in line
    assert env.log("gcloud.log") == ""


def test_tessera_arms_join_a_batch_with_the_other_arms(env: Env) -> None:
    arms = ("--arms", f"L5,{TESSERA_ARMS}", "--seeds", "0")
    refused = env.run("holdout", "--max-total-usd", "15.74", *arms, **BATCH)
    assert refused.returncode == 2 and "REFUSED" in refused.stdout and launcher_calls(env) == []
    res = env.run("holdout", "--max-total-usd", "15.75", *arms, **BATCH)
    assert res.returncode == 0, res.stdout + res.stderr
    modes = {
        re.search(r"--mode (\S+)", line).group(1): line  # type: ignore[union-attr]
        for line in launcher_calls(env)
    }
    assert set(modes) == {"v6-l5-spatial-holdout", "tessera-spatial-holdout", "tessera"}
    assert "--timeout-seconds 10800" in modes["v6-l5-spatial-holdout"]
    assert "--patch-cache" not in modes["v6-l5-spatial-holdout"]
    for mode in ("tessera-spatial-holdout", "tessera"):
        assert "--timeout-seconds 20400" in modes[mode] and "--patch-cache prebuilt" in modes[mode]
    assert set(listed_dns(env.log("gcloud.log"))) >= {
        hdn("L5", 0),
        tdn("TESSERA", 0),
        tdn("TESSERA-TEMPORAL", 0),
    }
    assert max_in_flight(env) <= 4


def test_tessera_attaches_to_an_active_job_and_skips_a_succeeded_one(env: Env) -> None:
    (env.log_dir / f"list.{tdn('TESSERA', 0)}").write_text("222\tJOB_STATE_RUNNING\n")
    (env.log_dir / f"list.{tdn('TESSERA-TEMPORAL', 0)}").write_text("333\tJOB_STATE_SUCCEEDED\n")
    res = env.run(
        "holdout", "--max-total-usd", "0", "--arms", TESSERA_ARMS, "--seeds", "0", **BATCH
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert launcher_calls(env) == []  # watched and skipped, nothing is submitted
    assert "custom-jobs describe 222" in env.log("gcloud.log")
    assert "custom-jobs describe 333" not in env.log("gcloud.log")
    assert "1 skipped" in res.stdout
