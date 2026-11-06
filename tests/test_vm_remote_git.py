"""Offline tests for scripts/runners/vm/remote-git-lib.sh (real git, fake uv/stat/pgrep/git)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from _vm_remote_fixtures import (
    BASH,
    FOREIGN,
    HAVE_TOOLS,
    SKIP_REASON,
    UID,
    VM_DIR,
    Sandbox,
    foreign_owner_supported,
    gen,
    git,
    read_git_log,
    write_exec,
    write_fake_stat,
    write_fake_uv,
    write_logging_git,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason=SKIP_REASON)

OWNER_FN = ("vm_remote_app_preamble", "vm_remote_idle_probe_cmd", "vm_remote_provenance_cmd")


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


@pytest.fixture
def deployed(sb: Sandbox) -> Sandbox:
    res = sb.deploy()
    assert res.returncode == 0, res.stderr
    sb.uv_log.unlink(missing_ok=True)
    return sb


_GOOD = ["/home/u", "u", "/o/origin.git", ""]
_FUNCS = [
    "vm_remote_app_preamble",
    "vm_remote_idle_probe_cmd",
    "vm_remote_deploy_cmd",
    "vm_remote_provenance_cmd",
    "vm_remote_app_cd_cmd",
]
_BAD = [
    (0, "relative"),
    (0, "/"),
    (0, "/a/"),
    (0, "/a/../b"),
    (0, "/a b"),
    (0, ""),
    (1, "bad user"),
    (1, "-x"),
    (1, "a;b"),
    (1, ""),
    (2, "/o/'x"),
    (2, "/o/a b"),
    (2, ""),
    (3, "0"),
    (3, "abc"),
    (4, "-x"),
    (5, "abc"),
]


_UNSAFE_CASES = [
    (fn, idx, bad) for fn in _FUNCS for idx, bad in _BAD if idx < 4 or fn == "vm_remote_deploy_cmd"
]


@pytest.mark.parametrize(("fn", "idx", "bad"), _UNSAFE_CASES)
def test_generators_reject_unsafe_literals(fn: str, idx: int, bad: str) -> None:
    args = [*_GOOD, "feat/x", "a" * 40] if fn == "vm_remote_deploy_cmd" else list(_GOOD)
    good = gen(fn, *args)
    assert good.returncode == 0, good.stderr
    args[idx] = bad
    res = gen(fn, *args)
    assert res.returncode != 0
    assert res.stdout == ""


def test_bootstrap_clone_when_missing(sb: Sandbox) -> None:
    assert not sb.app.exists()
    res = sb.deploy()
    assert res.returncode == 0, res.stderr
    assert sb.head() == sb.origin.sha
    assert stat.S_IMODE(sb.app.stat().st_mode) == 0o700
    assert git(sb.app, "remote", "get-url", "origin") == str(sb.origin.path)
    assert sb.uv_log.read_text().split() == ["sync"]
    assert not list(sb.home.glob(".berlin-lst-app.clone.*"))
    assert not (sb.home / ".gitconfig").exists()


def test_bootstrap_clone_failure_leaves_nothing(sb: Sandbox) -> None:
    res = sb.deploy(origin=sb.base / "does-not-exist.git")
    assert res.returncode == 3
    assert not sb.app.exists()
    assert not list(sb.home.glob(".berlin-lst-app.clone.*"))
    assert not sb.uv_called


@pytest.mark.parametrize("how", ["set_url", "insteadof"])
def test_wrong_origin_url_fails_closed(deployed: Sandbox, how: str) -> None:
    before = deployed.head()
    new = deployed.origin.advance()
    if how == "set_url":
        git(deployed.app, "remote", "set-url", "origin", "/elsewhere")
    else:
        git(deployed.app, "config", "url./elsewhere/.insteadOf", str(deployed.origin.path))
    res = deployed.deploy(new)
    assert res.returncode == 3
    assert "origin" in res.stderr
    assert deployed.head() == before
    assert not deployed.uv_called


@pytest.mark.parametrize("kind", ["tracked", "untracked"])
def test_dirty_tree_aborts_before_reset(deployed: Sandbox, kind: str) -> None:
    before = deployed.head()
    target = deployed.app / ("a.txt" if kind == "tracked" else "new.txt")
    target.write_bytes(b"local work\x00bytes\n")
    new = deployed.origin.advance()
    write_logging_git(deployed.bin, deployed.git_log)
    res = deployed.deploy(new)
    assert res.returncode == 6, res.stderr
    assert target.read_bytes() == b"local work\x00bytes\n"
    assert deployed.head() == before
    assert not deployed.uv_called
    for argv in read_git_log(deployed.git_log):
        assert not {"reset", "checkout", "fetch"} & set(argv)


def test_ignored_outputs_do_not_block(deployed: Sandbox) -> None:
    (deployed.app / "data").mkdir()
    (deployed.app / "data" / "x").write_text("x\n")
    (deployed.app / "logs").mkdir()
    (deployed.app / "logs" / "y").write_text("y\n")
    new = deployed.origin.advance()
    res = deployed.deploy(new)
    assert res.returncode == 0, res.stderr
    assert deployed.head() == new


def test_status_probe_error_aborts(deployed: Sandbox) -> None:
    before = deployed.head()
    new = deployed.origin.advance()
    write_logging_git(deployed.bin, deployed.git_log, fail_on="status")
    res = deployed.deploy(new)
    assert res.returncode == 3
    assert deployed.head() == before
    assert not any("reset" in argv for argv in read_git_log(deployed.git_log))


@pytest.mark.parametrize(
    ("case", "mapping", "needle"),
    [
        ("consistent_foreign", [("/berlin-lst-app", 4242), ("/.git", 4242)], "uid 4242"),
        ("root", [("/berlin-lst-app", 0), ("/.git", 0)], "uid 0"),
        ("mismatch", [("/.git", 4242)], "uid 4242"),
        ("venv", [("/.venv", 4242)], "uid 4242"),
        ("data", [("/data", 4242)], "uid 4242"),
        ("logs", [("/logs", 4242)], "uid 4242"),
        ("home", [("/home", 4242)], "uid 4242"),
    ],
)
@pytest.mark.parametrize("fn", OWNER_FN)
def test_owner_checks_fail_before_any_git_call(
    deployed: Sandbox, fn: str, case: str, mapping: list[tuple[str, int]], needle: str
) -> None:
    for sub in (".venv", "data", "logs"):
        (deployed.app / sub).mkdir(exist_ok=True)
    write_logging_git(deployed.bin, deployed.git_log)
    write_fake_stat(deployed.bin, mapping)
    res = deployed.run(fn)
    if fn == "vm_remote_idle_probe_cmd":
        assert res.stdout.strip() == "broken", (case, res.stderr)
    else:
        assert res.returncode == 3, (case, res.stdout, res.stderr)
    assert needle in res.stderr
    assert read_git_log(deployed.git_log) == []


def test_owner_checks_on_deploy_fail_before_any_git_call(deployed: Sandbox) -> None:
    new = deployed.origin.advance()
    write_logging_git(deployed.bin, deployed.git_log)
    write_fake_stat(deployed.bin, [("/berlin-lst-app", 4242), ("/.git", 4242)])
    res = deployed.deploy(new)
    assert res.returncode == 3
    assert "uid 4242" in res.stderr
    assert read_git_log(deployed.git_log) == []
    assert not deployed.uv_called


def test_wrong_pin_or_user_fails_before_git(deployed: Sandbox) -> None:
    write_logging_git(deployed.bin, deployed.git_log)
    new = deployed.origin.advance()
    other_home = {"HOME": str(deployed.base)}
    cases: list[tuple[dict[str, str], dict[str, str]]] = [
        ({"pin": str(UID + 1)}, {}),
        ({"user": "nobody-x"}, {}),
        ({}, other_home),
        ({}, {"HOME": ""}),
    ]
    for kwargs, extra in cases:
        res = deployed.deploy(new, env=deployed.env(extra), **kwargs)
        assert res.returncode == 3, (kwargs, extra, res.stderr)
    assert read_git_log(deployed.git_log) == []
    assert not deployed.uv_called


def _break(sb: Sandbox, case: str) -> dict[str, str]:
    """Rearrange the deployed app dir for a symlink case; return extra args for the run."""
    home_kwargs: dict[str, str] = {}
    if case == "app_symlink":
        real = sb.base / "real-app"
        sb.app.rename(real)
        sb.app.symlink_to(real)
    elif case == "home_parent_symlink":
        (sb.base / "hl").symlink_to(sb.base)
        home_kwargs["home"] = str(sb.base / "hl" / "home")
    elif case == "dot_git_symlink":
        real = sb.base / "gitreal"
        (sb.app / ".git").rename(real)
        (sb.app / ".git").symlink_to(real)
    elif case == "dot_git_file":
        shutil.rmtree(sb.app / ".git")
        (sb.app / ".git").write_text(f"gitdir: {sb.base / 'elsewhere'}\n")
    elif case == "venv_symlink":
        real = sb.base / "venvreal"
        real.mkdir()
        (sb.app / ".venv").symlink_to(real)
    return home_kwargs


@pytest.mark.parametrize(
    "case",
    ["app_symlink", "home_parent_symlink", "dot_git_symlink", "dot_git_file", "venv_symlink"],
)
def test_symlink_cases_fail_closed(deployed: Sandbox, case: str) -> None:
    new = deployed.origin.advance()
    home_kwargs = _break(deployed, case)
    env = deployed.env({"HOME": home_kwargs["home"]}) if home_kwargs else None
    res = deployed.deploy(new, env=env, **home_kwargs)
    assert res.returncode == 3, res.stderr
    assert not deployed.uv_called


def test_realpath_mismatch(deployed: Sandbox) -> None:
    home = f"{deployed.base}/./home"
    generated = deployed.cmd("vm_remote_deploy_cmd", "feat/x", deployed.origin.sha, home=home)
    if generated.returncode != 0:
        assert generated.stdout == ""
        return
    res = deployed.deploy(env=deployed.env({"HOME": home}), home=home)
    assert res.returncode != 0
    assert not deployed.uv_called


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
@pytest.mark.parametrize("target", ["app", ".git", ".venv", "data", "logs"])
def test_not_writable_fails_closed(deployed: Sandbox, target: str) -> None:
    for sub in (".venv", "data", "logs"):
        (deployed.app / sub).mkdir(exist_ok=True)
    path = deployed.app if target == "app" else deployed.app / target
    before = deployed.head()
    new = deployed.origin.advance()
    path.chmod(0o500)
    try:
        res = deployed.deploy(new)
    finally:
        path.chmod(0o700)
    assert res.returncode == 3, res.stderr
    assert "not writable" in res.stderr
    assert not deployed.uv_called
    assert deployed.head() == before


def test_uv_missing_fails_before_mutation(sb: Sandbox) -> None:
    git_dir = str(Path(shutil.which("git") or "/usr/bin/git").parent)
    search = f"{git_dir}:/usr/bin:/bin"
    if shutil.which("uv", path=search):
        pytest.skip("a real uv is on the restricted PATH")
    empty = sb.base / "nouv"
    empty.mkdir()
    env = sb.env(path=f"{empty}:{search}")
    res = sb.deploy(env=env)
    assert res.returncode == 3
    assert "uv" in res.stderr
    assert not sb.app.exists()
    assert not list(sb.home.glob(".berlin-lst-app.clone.*"))

    assert sb.deploy().returncode == 0
    before = sb.head()
    new = sb.origin.advance()
    assert sb.deploy(new, env=env).returncode == 3
    assert sb.head() == before

    write_fake_uv(empty, sb.uv_log, version_ok=False)
    res = sb.deploy(new, env=sb.env(path=f"{empty}:{search}"))
    assert res.returncode == 3
    assert sb.head() == before


def test_probe_states(sb: Sandbox) -> None:
    assert sb.probe() == "absent"
    assert sb.deploy().returncode == 0
    assert sb.probe() == "idle"
    (sb.app / "new.txt").write_text("x\n")
    assert sb.probe() == "dirty"
    (sb.app / "new.txt").unlink()
    assert sb.probe() == "idle"

    write_fake_stat(sb.bin, [("/berlin-lst-app", 4242), ("/.git", 4242)])
    assert sb.probe() == "broken"
    (sb.bin / "stat").unlink()

    write_logging_git(sb.bin, sb.git_log, fail_on="status")
    assert sb.probe() == "broken"
    (sb.bin / "git").unlink()

    git(sb.app, "remote", "set-url", "origin", "/elsewhere")
    assert sb.probe() == "broken"
    git(sb.app, "remote", "set-url", "origin", str(sb.origin.path))
    assert sb.probe() == "idle"

    sb.git_log.unlink(missing_ok=True)
    write_logging_git(sb.bin, sb.git_log)
    write_exec(sb.bin / "pgrep", "#!/bin/sh\nexit 0\n")
    assert sb.probe() == "occupied"
    assert read_git_log(sb.git_log) == []


def test_no_safe_directory_anywhere(sb: Sandbox) -> None:
    write_logging_git(sb.bin, sb.git_log)
    assert sb.probe() == "absent"
    assert sb.deploy().returncode == 0
    assert sb.probe() == "idle"
    prov = sb.provenance()
    assert prov.returncode == 0, prov.stderr
    assert prov.stdout.strip() == sb.origin.sha
    calls = read_git_log(sb.git_log)
    assert calls
    banned = {"-c", "config", "--global", "--system"}
    for argv in calls:
        assert not banned & set(argv), argv
        assert "safe.directory" not in " ".join(argv)


def test_foreign_owner_git_refuses_without_trust(deployed: Sandbox) -> None:
    if not foreign_owner_supported(deployed.base, deployed.app):
        pytest.skip("installed git ignores GIT_TEST_ASSUME_DIFFERENT_OWNER")
    env = deployed.env(FOREIGN)
    before = deployed.head()
    new = deployed.origin.advance()
    assert deployed.probe(env) == "broken"
    assert deployed.deploy(new, env=env).returncode == 3
    assert deployed.head() == before
    prov = deployed.provenance(env)
    assert prov.returncode != 0
    assert prov.stdout == ""
    assert not (deployed.home / ".gitconfig").exists()


_FORBIDDEN = [
    r"safe\.directory",
    r"git config --global",
    r"git config --system",
    r"--global",
    r"\bsudo\b",
    r"\bchown\b",
    r"chmod\s+0?777",
    r"setfacl",
    r"add-iam-policy-binding",
    r"set-iam-policy",
]


def test_static_forbidden_ops() -> None:
    scripts = sorted(VM_DIR.glob("*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text()
        for pattern in _FORBIDDEN:
            assert not re.search(pattern, text), f"{script.name}: {pattern}"
        assert "/workspace/app" not in text, script.name
        res = subprocess.run(  # noqa: S603
            [BASH or "", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert res.returncode == 0, f"{script.name}: {res.stderr}"

    common = (VM_DIR / "vm-runner-common.sh").read_text()
    for needle in ("vm_remote_deploy_cmd", "vm_remote_provenance_cmd", "vm_resolve_app_dir"):
        assert needle in common
    identity = (VM_DIR / "vm-identity.sh").read_text()
    assert 'VM_APP_ORIGIN_URL="https://github.com/spignotti/berlin-lst-downscaling.git"' in identity
    assert '"/home/$SSH_USER"' in identity
    assert "berlin-lst-app" in identity


def test_all_app_dir_users_share_resolver() -> None:
    names = [
        "run-random-forest-vm.sh",
        "run-baseline-vm.sh",
        "status-dynamic-vm.sh",
        "cloud-monitor.sh",
        "sync-vm-env.sh",
    ]
    for name in names:
        lines = (VM_DIR / name).read_text().splitlines()
        code = [(i, ln) for i, ln in enumerate(lines) if not ln.lstrip().startswith("#")]
        resolver = [i for i, ln in code if "vm_resolve_app_dir" in ln or "vm_init_run" in ln]
        users = [i for i, ln in code if "APP_DIR" in ln]
        assert resolver, name
        assert users, name
        assert min(resolver) <= min(users), name
        assert not any(re.match(r"\s*APP_DIR=", ln) for _, ln in code), name
