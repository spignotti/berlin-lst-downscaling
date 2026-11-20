"""Offline tests for the one-time v6 test-2025 patch export (mocks only, no GCS)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import DictConfig
from test_v6_val_rescore import (
    MAX_BYTES,
    N_V6,
    SSIM,
    _expected_mae,
    _full_cache,
    _loader,
    _module,
    _sample,
    _samples,
    _StubTask,
)

from berlin_lst_downscaling.analysis import scene_bootstrap as sb
from berlin_lst_downscaling.modeling import real_task
from berlin_lst_downscaling.modeling import v6_test_score as job
from berlin_lst_downscaling.modeling import v6_val_rescore as rescore
from berlin_lst_downscaling.modeling.channels import selection_from_config
from berlin_lst_downscaling.modeling.evaluation_exports import (
    PATCH_EXPORT_METADATA_KEY,
    EvaluationIdentity,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.metrics import EVAL_PREDICTION_CLAMP_K
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealSample,
    ScalerChannel,
    ScalerSpec,
)
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "operators" / "export_v6_test_patches.py"
LABELS = ("v6m-070b33c-l1-seed0-ep40-cos", "v6m-070b33c-l4-seed2-ep40-cos")
_IDENTITY = EvaluationIdentity(arm="L4", seed=0, run_id="run-a", checkpoint="gs://x/best.ckpt")


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _refs() -> list[PatchRef]:
    return [
        PatchRef("t0", "LC08_X", "S2_X", "train", 2018, 0, 0, 240, "m"),
        PatchRef("v0", "LC08_X", "S2_X", "validation", 2024, 0, 0, 240, "m"),
        PatchRef("v1", "LC08_Y", "S2_Y", "validation", 2024, 16, 0, 240, "m"),
        PatchRef("e0", "LC09_X", "S2_X", "test", 2025, 0, 0, 240, "m"),
        PatchRef("e1", "LC09_Y", "S2_Y", "test", 2025, 16, 0, 240, "m"),
    ]


class _FakeReader:
    def __init__(self, samples: dict[str, RealSample]) -> None:
        self.samples = samples
        self.read: list[str] = []
        self.exclusions: Counter[str] = Counter()
        names = tuple(f"v6_c{i}" for i in range(N_V6))
        self.scaler = ScalerSpec(
            channel_order=names,
            channels=tuple(
                ScalerChannel(index=i + 1, name=n, transform="none", mean=0.0, std=1.0)
                for i, n in enumerate(names)
            ),
            policy_hash="test",
        )

    def preload(self, scene_ids: object) -> None:
        return None

    def admission_reason(self, ref: PatchRef) -> str | None:
        return None

    def read_patch(self, ref: PatchRef) -> RealSample:
        self.read.append(ref.patch_id)
        return self.samples[ref.patch_id]


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeReader, list[tuple[str, ...]]]:
    samples = {
        "t0": _sample("t0", 0, split="train"),
        "v0": _sample("v0", 1),
        "v1": _sample("v1", 2, scene="LC08_Y"),
        "e0": _sample("e0", 3, scene="LC09_X", split="test"),
        "e1": _sample("e1", 4, scene="LC09_Y", split="test"),
    }
    reader = _FakeReader(samples)
    seen: list[tuple[str, ...]] = []

    def fake_refs(_source: object, *, splits: tuple[str, ...], scene_ids: object):
        seen.append(tuple(splits))
        return [ref for ref in _refs() if ref.split in splits]

    monkeypatch.setattr(real_task, "RealPatchReader", lambda _s: reader)
    monkeypatch.setattr(real_task, "load_patch_refs", fake_refs)
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: {"index": "x"})
    return reader, seen


def _cfg(tmp_path: Path) -> DictConfig:
    run = rescore.rescore_inventory()[0]
    cfg = rescore.compose_run_config(run, output_root=str(tmp_path / "runs" / run.run_label))
    cfg.data.cache_max_bytes = MAX_BYTES
    return cfg


# ── split parameter keeps the validation path ─────────────────────────


def test_validation_score_split_parameter_is_byte_identical(tmp_path: Path) -> None:
    samples = _samples()
    common = {
        "aggregation": "arithmetic",
        "ssim_range": SSIM,
        "device": torch.device("cpu"),
    }
    rows_default, mae_default = rescore.score_validation(
        _StubTask(5),  # type: ignore[arg-type]
        _loader(samples, 5),
        **common,  # type: ignore[arg-type]
    )
    rows_explicit, mae_explicit = rescore.score_validation(
        _StubTask(5),  # type: ignore[arg-type]
        _loader(samples, 5),
        split="validation",
        **common,  # type: ignore[arg-type]
    )
    assert rows_default == rows_explicit
    assert mae_default == mae_explicit
    path_default = write_patch_export(
        rows_default,
        tmp_path / "default.parquet",
        identity=_IDENTITY,
        split="validation",
        ssim_range=SSIM,
    )
    path_explicit = write_patch_export(
        rows_explicit,
        tmp_path / "explicit.parquet",
        identity=_IDENTITY,
        split="validation",
        ssim_range=SSIM,
    )
    assert path_default.read_bytes() == path_explicit.read_bytes()


def test_validation_cache_split_parameter_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    seen: list[object] = []

    def fake_build(cfg: DictConfig, **kwargs: object) -> RealPatchDataModule:
        seen.append(kwargs["splits"])
        splits = kwargs["splits"]
        assert isinstance(splits, tuple)
        return _module(splits=splits)

    monkeypatch.setattr(rescore, "build_real_datamodule", fake_build)
    cfg = _cfg(tmp_path)
    default = rescore.build_validation_cache(cfg, tmp_path / "default")
    explicit = rescore.build_validation_cache(cfg, tmp_path / "explicit", splits=("validation",))
    assert seen == [("validation",), ("validation",)]
    assert _file_bytes(default.root) == _file_bytes(explicit.root)
    assert default.patch_ids == ("v0", "v1")


def test_one_split_cache_matches_validation_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    old = _module()
    old.setup()
    old.build_validation_only_cache(tmp_path / "old", max_bytes=MAX_BYTES)
    new = _module()
    new.setup()
    new.build_one_split_cache(tmp_path / "new", max_bytes=MAX_BYTES, split="validation")
    assert _file_bytes(tmp_path / "old") == _file_bytes(tmp_path / "new")
    assert _file_bytes(tmp_path / "old")


def test_validation_only_cache_still_refuses_the_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    module = _module(splits=("test",))
    module.setup()
    with pytest.raises(RuntimeError, match="validation-only"):
        module.build_validation_only_cache(tmp_path / "cache", max_bytes=MAX_BYTES)


def test_shared_cache_refuses_train_before_building_a_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> RealPatchDataModule:
        raise AssertionError("datamodule built")

    monkeypatch.setattr(rescore, "build_real_datamodule", boom)
    cfg = _cfg(tmp_path)
    for splits in (("train",), ("train", "validation"), ("validation", "test")):
        with pytest.raises(ValueError, match="shared cache splits"):
            rescore.build_validation_cache(cfg, tmp_path / "c", splits=splits)


def test_test_cache_reads_the_test_split_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, seen = _install_fakes(monkeypatch)

    def fake_build(cfg: DictConfig, **kwargs: object) -> RealPatchDataModule:
        splits = kwargs["splits"]
        assert isinstance(splits, tuple)
        return _module(splits=splits)

    monkeypatch.setattr(rescore, "build_real_datamodule", fake_build)
    cache = rescore.build_validation_cache(_cfg(tmp_path), tmp_path / "cache", splits=("test",))
    assert seen == [("test",)]
    assert cache.patch_ids == ("e0", "e1")
    assert reader.read == ["e0", "e1"]
    assert set(cache.scope["patches_per_split"]) == {"test"}  # type: ignore[arg-type]


def test_score_validation_accepts_an_explicit_test_split() -> None:
    samples = [_sample(f"e{i}", 10 + i, split="test") for i in range(2)]
    rows, recomputed = rescore.score_validation(
        _StubTask(5),  # type: ignore[arg-type]
        _loader(samples, 5),
        aggregation="arithmetic",
        ssim_range=SSIM,
        device=torch.device("cpu"),
        split="test",
    )
    assert [row["split"] for row in rows] == ["test", "test"]
    assert recomputed == pytest.approx(_expected_mae(samples), rel=1e-5)
    with pytest.raises(RuntimeError, match="split"):
        rescore.score_validation(
            _StubTask(5),  # type: ignore[arg-type]
            _loader([_sample("v0", 1)], 5),
            aggregation="arithmetic",
            ssim_range=SSIM,
            device=torch.device("cpu"),
            split="test",
        )


# ── export job ────────────────────────────────────────────────────────


def _test_samples() -> list[RealSample]:
    return [_sample(f"e{i}", 100 + i, scene=f"LC09_{i % 2}", split="test") for i in range(3)]


class _Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.validation_samples = _samples()
        self.test_samples = _test_samples()
        self.validation_mae = _expected_mae(self.validation_samples)
        self.test_mae = _expected_mae(self.test_samples)
        assert self.validation_mae != pytest.approx(self.test_mae)
        self.built: list[tuple[str, ...]] = []
        self.fetched: list[str] = []
        self.bad_score = False
        self.out = tmp_path / "out"
        monkeypatch.setattr(job, "code_commit", lambda: "abc123")
        monkeypatch.setattr(rescore, "patch_index_fingerprints", lambda _root: {"index": "fp"})
        monkeypatch.setattr(rescore, "resolve_run_ssim_data_range", lambda *a, **k: SSIM)
        monkeypatch.setattr(rescore, "build_validation_cache", self._build_cache)
        monkeypatch.setattr(rescore, "fetch_checkpoint", self._fetch)
        monkeypatch.setattr(rescore, "checkpoint_selection_state", self._selection)
        monkeypatch.setattr(rescore, "load_rescore_task", self._load)
        monkeypatch.setattr(job, "ssim_range_for_checkpoint", lambda _path, ssim_range: ssim_range)

    def _build_cache(
        self, cfg: DictConfig, cache_root: Path, *, splits: tuple[str, ...] = rescore.RESCORE_SPLITS
    ) -> rescore.ValidationCache:
        self.built.append(tuple(splits))
        if splits == ("validation",):
            samples = self.validation_samples
        elif splits == ("test",):
            samples = self.test_samples
        else:
            raise AssertionError(f"unexpected cache splits {splits!r}")
        return _full_cache(cache_root, samples)

    def _fetch(self, uri: str, dest: Path) -> rescore.CheckpointRef:
        self.fetched.append(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"ckpt:{uri}".encode())
        data = dest.read_bytes()
        return rescore.CheckpointRef(uri, 7, len(data), hashlib.sha256(data).hexdigest())

    def _selection(self, path: Path) -> dict[str, object]:
        return {"best_model_score": self.validation_mae + (0.5 if self.bad_score else 0.0)}

    def _load(self, path: Path, cfg: DictConfig, ssim_range: object) -> _StubTask:
        return _StubTask(selection_from_config(cfg).n_active)

    def run(self, labels: tuple[str, ...] = LABELS) -> dict[str, object]:
        return job.run_test_export(run_labels=labels, output_root=self.out, device="cpu")


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def test_run_test_export_writes_test_rows_and_footer(harness: _Harness) -> None:
    manifest = harness.run()
    assert harness.built == [("test",), ("validation",)]
    assert harness.fetched == [run.checkpoint_uri for run in job.select_runs(LABELS)]
    assert not list((harness.out / "checkpoints").glob("*.ckpt"))
    assert not list(harness.out.glob("**/eval_validation_patches.parquet"))
    assert manifest["split"] == "test" and manifest["uploaded"] is False
    assert manifest["splits_read"] == ["test", "validation"]
    assert manifest["code_commit"] == "abc123"
    assert manifest["run_labels"] == list(LABELS)
    recheck = manifest["validation_recheck"]
    assert isinstance(recheck, dict)
    assert recheck["purpose"] == "identity" and recheck["split"] == "validation"
    on_disk = json.loads((harness.out / job.MANIFEST_NAME).read_text())
    assert on_disk["runs"] == manifest["runs"]
    entries = manifest["runs"]
    assert isinstance(entries, list) and len(entries) == 2
    for label, entry in zip(LABELS, entries, strict=True):
        assert entry["run_label"] == label
        assert entry["patches"] == 3 and entry["scenes"] == 2
        assert entry["recomputed_validation_mae_100m"] == pytest.approx(
            harness.validation_mae, rel=1e-5
        )
        assert entry["recomputed_test_mae_100m"] == pytest.approx(harness.test_mae, rel=1e-5)
        assert entry["checkpoint"]["sha256"]
        path = harness.out / entry["export"]["path"]
        assert path == harness.out / label / "eval_test_patches.parquet"
        assert entry["export"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        table = sb.load_patch_exports([path], split="test")
        assert table.column("split").to_pylist() == ["test"] * 3
        assert set(table.column("patch_id").to_pylist()) == {"e0", "e1", "e2"}
        footer = json.loads(pq.read_schema(path).metadata[PATCH_EXPORT_METADATA_KEY])
        assert footer["split"] == "test"
        assert footer["schema_version"] == 2
        assert footer["prediction_clamp_k"] == [float(v) for v in EVAL_PREDICTION_CLAMP_K]
        assert footer["run_id"] == label


def test_scene_bootstrap_refuses_a_test_export_requested_as_validation(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.run()
    path = harness.out / LABELS[0] / "eval_test_patches.parquet"

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("table must not be read")

    monkeypatch.setattr(sb.pq, "read_table", boom)
    with pytest.raises(sb.SceneBootstrapError, match="footer split"):
        sb.load_patch_exports([path], split="validation")


def test_run_test_export_is_create_only(harness: _Harness) -> None:
    existing = harness.out / LABELS[1] / "eval_test_patches.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"kept")
    with pytest.raises(FileExistsError, match="create-only"):
        harness.run()
    assert harness.built == [] and harness.fetched == []
    assert existing.read_bytes() == b"kept"

    manifest = harness.out / job.MANIFEST_NAME
    existing.unlink()
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="create-only"):
        harness.run()
    assert harness.built == [] and harness.fetched == []
    assert manifest.read_text() == "{}"


def test_selection_mismatch_writes_no_test_export(harness: _Harness) -> None:
    harness.bad_score = True
    with pytest.raises(RuntimeError, match="recomputed validation MAE"):
        harness.run()
    assert not (harness.out / job.MANIFEST_NAME).exists()
    assert not list(harness.out.glob("**/eval_test_patches.parquet"))


def test_unknown_label_does_no_work(harness: _Harness) -> None:
    with pytest.raises(ValueError, match="outside"):
        harness.run(("not-a-matrix-run",))
    assert harness.built == [] and harness.fetched == []


def test_test_score_does_not_upload() -> None:
    for path in (Path(job.__file__), _SCRIPT):
        text = path.read_text(encoding="utf-8")
        for token in ("create_only_upload", "upload_from_filename", "storage.", "delete_blob"):
            assert token not in text


# ── CLI ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_v6_test_patches", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_prints_the_27_labels_without_touching_gcs(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("export or GCS work")

    monkeypatch.setattr(cli, "run_test_export", boom)
    monkeypatch.setattr(rescore, "fetch_checkpoint", boom)
    monkeypatch.setattr(rescore, "build_validation_cache", boom)
    assert cli.main(["--list"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    labels = [line.split()[-1] for line in lines]
    assert labels == [run.run_label for run in rescore.rescore_inventory()]
    assert len(labels) == 27
    assert "v6m-070b33c-l4-seed0-ep40-cos" in labels


def test_all_passes_every_inventory_label(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def capture(*, run_labels: list[str], output_root: Path, device: str) -> dict[str, object]:
        seen["labels"] = tuple(run_labels)
        seen["device"] = device
        seen["root"] = output_root
        return {"runs": []}

    monkeypatch.setattr(cli, "run_test_export", capture)
    assert cli.main(["--all", "--output-root", str(tmp_path), "--device", "cpu"]) == 0
    assert seen["labels"] == tuple(run.run_label for run in rescore.rescore_inventory())
    assert seen["device"] == "cpu"


def test_all_and_run_labels_are_mutually_exclusive(cli: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["--all", "--run-labels", LABELS[0], "--output-root", str(tmp_path), "--device", "cpu"]
        )
    assert exc.value.code == 2
