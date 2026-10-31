"""Prebuilt patch cache builder CLI: split guards, local-only, dry-run, immutability, commit pin."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf
from test_patch_cache_source_channels import _sample

from berlin_lst_downscaling.modeling.patch_cache import build_patch_cache
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    PREBUILT_MANIFEST,
    prebuilt_key_components,
    prebuilt_prefix,
    upload_prebuilt_cache,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "operators" / "build_prebuilt_patch_cache.py"
)


@pytest.fixture
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_prebuilt_patch_cache_cli", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeModule:
    """Stands in for ``RealPatchDataModule``: writes a real tiny cache on setup."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.provenance: dict[str, Any] = {
            "patch_index_fingerprints": {"index_policy_hash": "i", "source_policy_hash": "s"},
            "scaler_digest": "d",
            "channel_order": ["c0", "c1"],
            "channel_indices": [0, 1],
            "source_channels": 22,
        }

    def setup(self) -> None:
        splits = ["train", "train", "validation"]
        samples = [_sample(f"p{i}", 22, i, s) for i, s in enumerate(splits)]
        build_patch_cache(
            self.cache_root,
            iter(samples),
            patch_ids=[s.meta.patch_id for s in samples],
            active_channels=2,
            source_channels=22,
            provenance=self.provenance,
            max_bytes=1 << 30,
            channel_indices=(0, 1),
        )

    def prebuilt_components(self) -> dict[str, object]:
        return prebuilt_key_components(
            self.provenance,
            source_channel_order=[f"c{i}" for i in range(22)],
            splits=("train", "validation"),
        )

    def stats(self) -> dict[str, object]:
        return {"patches_per_split": {"train": 2, "validation": 1}}


@pytest.fixture
def wired(cli: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"build": [], "upload": [], "guard": []}
    cfg = OmegaConf.create(
        {"data": {"cache_max_bytes": 1 << 30, "splits": ["train", "validation"]}}
    )
    calls["cfg"] = cfg
    monkeypatch.setattr(cli, "_compose", lambda name, overrides: cfg)
    monkeypatch.setattr(cli, "guard_modeling_config", lambda c, n: calls["guard"].append(n))
    monkeypatch.setattr(cli, "ensure_gdal_gcs_auth", lambda: None)
    git = {"commit": "abc123", "dirty": False}
    calls["git"] = git
    monkeypatch.setattr(cli, "_git_commit", lambda: git["commit"])
    monkeypatch.setattr(cli, "_git_dirty", lambda: git["dirty"])

    def build(_cfg: object, *, cache_root: Path, cache_max_bytes: int, **overrides: Any) -> Any:
        calls["build"].append(overrides)
        return _FakeModule(cache_root)

    def upload(*args: Any, **kwargs: Any) -> None:
        calls["upload"].append((args, kwargs))

    monkeypatch.setattr(cli, "build_real_datamodule", build)
    monkeypatch.setattr(cli, "upload_prebuilt_cache", upload)
    return calls


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--config-name",
        "v6_l5_locked",
        "--local-dir",
        str(tmp_path / "work"),
        "--uri-root",
        str(tmp_path / "gcs"),
        *extra,
    ]


def test_test_split_without_flag_is_refused(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args(tmp_path, "--splits", "train,validation,test")) == 2
    assert "closed" in capsys.readouterr().out
    assert wired["build"] == []


def test_allow_test_split_still_closed(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args(tmp_path, "--splits", "train,validation,test", "--allow-test-split")) == 2
    assert "test split guard: closed" in capsys.readouterr().out
    assert wired["build"] == [] and wired["upload"] == []
    with pytest.raises(RuntimeError, match="closed"):
        cli._assert_test_split_open()


@pytest.mark.parametrize("splits", ["train", "validation", "train,train", "test", ""])
def test_other_split_sets_are_refused(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, splits: str
) -> None:
    assert cli.main(_args(tmp_path, "--splits", splits)) == 2
    assert wired["build"] == []


@pytest.mark.parametrize(
    "declared", [["train", "validation", "test"], None, ["train", "validation", "validation"]]
)
def test_config_splits_must_be_train_validation_before_any_read(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, declared: list[str] | None
) -> None:
    wired["cfg"].data.splits = declared
    assert cli.main(_args(tmp_path, "--local-only")) == 2
    assert wired["build"] == [] and wired["upload"] == []


def test_local_dir_must_be_local(cli: ModuleType, wired: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--config-name", "x", "--local-dir", "gs://bucket/x"])


def test_local_only_builds_and_packs_without_upload(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args(tmp_path, "--local-only", "--shard-max-bytes", "1")) == 0
    assert wired["upload"] == []
    assert wired["guard"] == ["v6_l5_locked"]
    assert wired["build"] == [{"cache_source": "job_local"}]
    manifest = json.loads((tmp_path / "work" / "prebuilt" / PREBUILT_MANIFEST).read_text())
    assert manifest["splits"]["train"]["count"] == 2
    assert manifest["provenance"]["config_name"] == "v6_l5_locked"
    summary = json.loads(capsys.readouterr().out)
    assert summary["key"] == manifest["key"]
    assert summary["shards"] == len(manifest["shards"])


def test_dry_run_neither_packs_nor_uploads(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args(tmp_path, "--dry-run")) == 0
    assert not (tmp_path / "work" / "prebuilt").exists()
    assert wired["upload"] == []
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True and summary["prefix"].startswith(str(tmp_path / "gcs"))


@pytest.mark.parametrize("extra", [(), ("--overwrite",)])
def test_published_prefix_is_refused_before_packing(
    cli: ModuleType,
    wired: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
) -> None:
    components = _FakeModule(tmp_path).prebuilt_components()
    prefix = Path(prebuilt_prefix(str(tmp_path / "gcs"), components))
    prefix.mkdir(parents=True)
    (prefix / PREBUILT_MANIFEST).write_text("{}")
    assert cli.main(_args(tmp_path, *extra)) == 2
    assert "published and immutable" in capsys.readouterr().out
    assert not (tmp_path / "work" / "prebuilt").exists()
    assert wired["upload"] == []
    assert (prefix / PREBUILT_MANIFEST).read_text() == "{}"


@pytest.mark.parametrize("overwrite", [False, True])
def test_upload_runs_and_overwrite_is_forwarded(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, overwrite: bool
) -> None:
    extra = ("--overwrite",) if overwrite else ()
    assert cli.main(_args(tmp_path, "--upload-workers", "3", *extra)) == 0
    ((args, kwargs),) = wired["upload"]
    assert args[0] == tmp_path / "work" / "prebuilt"
    assert kwargs == {"overwrite": overwrite, "max_workers": 3}


def test_overwrite_replaces_leftover_shards_without_a_manifest(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "upload_prebuilt_cache", upload_prebuilt_cache)
    components = _FakeModule(tmp_path).prebuilt_components()
    prefix = Path(prebuilt_prefix(str(tmp_path / "gcs"), components))
    (prefix / "shards").mkdir(parents=True)
    (prefix / "shards" / "features-00000.bin").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="--overwrite"):
        cli.main(_args(tmp_path))
    assert not (prefix / PREBUILT_MANIFEST).exists()
    shutil.rmtree(tmp_path / "work")
    assert cli.main(_args(tmp_path, "--overwrite")) == 0
    assert (prefix / PREBUILT_MANIFEST).exists()
    assert (prefix / "shards" / "features-00000.bin").read_bytes() != b"stale"


@pytest.mark.parametrize(
    ("commit", "dirty"), [("abc123", True), ("unknown", False), ("abc123", None)]
)
def test_dirty_or_unknown_commit_is_refused_without_allow_dirty(
    cli: ModuleType,
    wired: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    commit: str,
    dirty: bool | None,
) -> None:
    wired["git"].update(commit=commit, dirty=dirty)
    assert cli.main(_args(tmp_path)) == 2
    assert "--allow-dirty" in capsys.readouterr().out
    assert wired["build"] == [] and wired["upload"] == []
    assert cli.main(_args(tmp_path, "--allow-dirty")) == 0
    assert "WARNING" in capsys.readouterr().out
    assert len(wired["upload"]) == 1


def test_dirty_tree_is_allowed_for_local_only(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path
) -> None:
    wired["git"]["dirty"] = True
    assert cli.main(_args(tmp_path, "--local-only")) == 0
    assert wired["upload"] == []


def test_expected_commit_mismatch_fails(
    cli: ModuleType, wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_args(tmp_path, "--expected-commit", "def456")) == 2
    assert "--expected-commit" in capsys.readouterr().out
    assert wired["build"] == [] and wired["upload"] == []
    assert cli.main(_args(tmp_path, "--local-only", "--expected-commit", "def456")) == 2
    assert cli.main(_args(tmp_path, "--expected-commit", "abc123")) == 0
    assert len(wired["upload"]) == 1
