"""Regression tests for the Vertex image build context.

`gcloud builds submit` once derived its upload filter from `.gitignore` and, not
anchoring `data/ard/` to the repo root, dropped `src/.../data/ard/` and
`src/.../data/dynamic/` from the image. These tests pin the layout and the
ignore files so a missing package fails here instead of on a Vertex worker.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "berlin_lst_downscaling"
SHIPPED_DIRS = ("src/", "scripts/", "configs/")
SHIPPED_FILES = ("pyproject.toml", "uv.lock", "Dockerfile.vertex")
MUST_EXCLUDE = (".env", ".env.local", "data/x", "data/ard/x.tif")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return [path for path in out.split("\0") if path]


def _shipped_files() -> list[str]:
    tracked = _tracked_files()
    shipped = [p for p in tracked if p.startswith(SHIPPED_DIRS)]
    assert shipped, "no tracked files under src/, scripts/, configs/"
    for name in SHIPPED_FILES:
        assert name in tracked, f"{name} is not tracked"
    return [*shipped, *SHIPPED_FILES]


def _patterns(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _glob_regex(glob: str) -> str:
    """Translate one path glob to a regex; `*` and `?` never cross `/`."""
    out: list[str] = []
    i = 0
    while i < len(glob):
        char = glob[i]
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = glob.index("]", i)
            out.append(glob[i : end + 1])
            i = end
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


def _gcloud_rule_matches(pattern: str, path: str) -> bool:
    """gcloudignore semantics for one (non-negated) pattern.

    A leading `/` anchors to the root; a trailing `/` matches directories only.
    Without a leading `/` the pattern matches at any depth, also when it holds a
    slash in the middle (this is where gcloud departs from git).
    """
    dir_only = pattern.endswith("/")
    body = pattern.rstrip("/")
    anchored = body.startswith("/")
    body = body.lstrip("/")
    prefix = "" if anchored else "(?:.*/)?"
    regex = re.compile(prefix + _glob_regex(body))
    parts = path.split("/")
    # Candidates: every ancestor directory, plus the file itself unless dir-only.
    candidates = ["/".join(parts[:n]) for n in range(1, len(parts))]
    if not dir_only:
        candidates.append(path)
    return any(regex.fullmatch(candidate) for candidate in candidates)


def _docker_rule_matches(pattern: str, path: str) -> bool:
    """`.dockerignore` semantics: root-relative, a match on a parent excludes it."""
    regex = re.compile(_glob_regex(pattern.strip("/")))
    parts = path.split("/")
    candidates = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
    return any(regex.fullmatch(candidate) for candidate in candidates)


def _excluded(patterns: list[str], path: str, *, docker: bool) -> bool:
    match = _docker_rule_matches if docker else _gcloud_rule_matches
    excluded = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if match(pattern.removeprefix("!"), path):
            excluded = not negated
    return excluded


def test_every_source_directory_is_a_regular_package() -> None:
    missing = sorted(
        str(directory.relative_to(REPO))
        for directory in {path.parent for path in PACKAGE.rglob("*.py")}
        if "__pycache__" not in directory.parts and not (directory / "__init__.py").is_file()
    )
    assert not missing, f"directories without __init__.py: {missing}"


def test_gcloudignore_patterns_are_root_anchored() -> None:
    path = REPO / ".gcloudignore"
    assert path.is_file(), ".gcloudignore is missing; gcloud would fall back to .gitignore"
    assert "#!include" not in path.read_text()

    components = {
        part
        for tracked in _tracked_files()
        if tracked.startswith(SHIPPED_DIRS)
        for part in tracked.split("/")
    }
    for pattern in _patterns(path):
        if pattern.startswith("!"):
            continue
        body = pattern.removesuffix("/")
        if "/" in body:
            assert body.startswith("/"), f"path pattern is not root-anchored: {pattern!r}"
            continue
        regex = re.compile(_glob_regex(body))
        hits = sorted(part for part in components if regex.fullmatch(part))
        assert not hits, f"bare pattern {pattern!r} matches shipped path components {hits}"


@pytest.mark.parametrize(
    ("ignore_file", "docker"), [(".gcloudignore", False), (".dockerignore", True)]
)
def test_build_context_ships_sources_and_drops_secrets(ignore_file: str, docker: bool) -> None:
    patterns = _patterns(REPO / ignore_file)

    dropped = [p for p in _shipped_files() if _excluded(patterns, p, docker=docker)]
    assert not dropped, f"{ignore_file} drops shipped files: {dropped[:20]}"

    kept = [p for p in MUST_EXCLUDE if not _excluded(patterns, p, docker=docker)]
    assert not kept, f"{ignore_file} does not exclude {kept}"
    assert not _excluded(patterns, ".env.example", docker=docker)


def test_gcloud_matcher_reproduces_the_gitignore_bug() -> None:
    # The pre-fix filter: an unanchored `data/ard/` also matches under src/.
    path = "src/berlin_lst_downscaling/data/ard/publish_guard.py"
    assert _excluded(["data/ard/"], path, docker=False)
    assert not _excluded(["/data/ard/"], path, docker=False)


@pytest.mark.parametrize(
    "module",
    [
        "berlin_lst_downscaling.data.ard.publish_guard",
        "berlin_lst_downscaling.data.dynamic",
        "berlin_lst_downscaling.data.selection",
        "berlin_lst_downscaling.data.training.patch_index",
        "berlin_lst_downscaling.modeling.run",
    ],
)
def test_vertex_entrypoint_modules_import(module: str) -> None:
    importlib.import_module(module)
