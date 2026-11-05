"""Offline tests for the validation-only rescore of the frozen v6 matrix (mocks only)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from berlin_lst_downscaling.analysis.scene_bootstrap import load_patch_exports
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.modeling import real_task
from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling import v6_val_rescore as rescore
from berlin_lst_downscaling.modeling.channels import V6_ABLATION_ARMS, selection_from_config
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealBatch,
)
from berlin_lst_downscaling.modeling.evaluation_exports import (
    PATCH_EXPORT_METADATA_KEY,
    patch_export_filename,
)
from berlin_lst_downscaling.modeling.lr_schedule import lr_schedule_from_config
from berlin_lst_downscaling.modeling.metrics import EVAL_PREDICTION_CLAMP_K
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, build_patch_cache
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealSample,
    RealSampleMeta,
    RealSourceConfig,
    ScalerChannel,
    ScalerSpec,
    collate_real_batch,
)
from berlin_lst_downscaling.modeling.real_task import (
    CachedRealPatchDataset,
    RealLSTTask,
    RealPatchDataModule,
)
from berlin_lst_downscaling.modeling.ssim_range import test_override_ssim_data_range

N_V6 = 22
MAX_BYTES = 1 << 30
SSIM = test_override_ssim_data_range(100.0)
L4_CHECKPOINT = (
    "gs://berlin-lst-training-data/qa/modeling/vertex-smoke/v6m-070b33c-l4-seed0-ep40-cos/best.ckpt"
)
SOURCE_SHA = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64


# ── synthetic data helpers ────────────────────────────────────────────


def _sample(patch_id: str, seed: int, *, scene: str = "LC08_X", split: str = "validation"):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N_V6, REAL_PATCH_PX, REAL_PATCH_PX)).astype(np.float32)
    prior = np.full((1, REAL_PATCH_PX, REAL_PATCH_PX), 290.0, dtype=np.float32)
    target = (290.0 + rng.normal(size=(1, REAL_PATCH_CELLS, REAL_PATCH_CELLS))).astype(np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    mask[:, 0, :] = False
    meta = RealSampleMeta(
        patch_id=patch_id,
        scene_id=scene,
        split=split,
        year=2024,
        row=0,
        col=0,
        filled_feature_pixels=0,
        scene_date="2024-07-01",
    )
    return RealSample(meta, features, prior, target, mask)


def _samples(n: int = 4) -> list[RealSample]:
    return [_sample(f"v{i}", i, scene=f"LC08_{i % 2}") for i in range(n)]


def _provenance(indices: tuple[int, ...]) -> dict[str, object]:
    return {
        "channel_order": [f"c{i}" for i in indices],
        "channel_indices": list(indices),
        "source_channels": N_V6,
    }


def _build(root: Path, samples: list[RealSample], indices: tuple[int, ...]) -> dict[str, object]:
    provenance = _provenance(indices)
    build_patch_cache(
        root,
        iter(samples),
        patch_ids=[s.meta.patch_id for s in samples],
        active_channels=len(indices),
        source_channels=N_V6,
        provenance=provenance,
        max_bytes=MAX_BYTES,
        channel_indices=indices,
    )
    return provenance


def _full_cache(root: Path, samples: list[RealSample]) -> rescore.ValidationCache:
    provenance = _build(root, samples, tuple(range(N_V6)))
    cache = PatchCache(root, provenance)
    scope: dict[str, object] = {
        "patches_per_split": {"validation": len(samples)},
        "skipped_per_split": {"validation": 0},
        "exclusions": {},
    }
    return rescore.ValidationCache(root, provenance, cache.patch_ids, scope, 0.5, N_V6)


# ── 1-2 inventory ─────────────────────────────────────────────────────


def test_inventory_is_the_exact_27_run_matrix() -> None:
    runs = rescore.rescore_inventory()
    assert len(runs) == 27
    assert [(r.arm_id, r.seed) for r in runs] == [
        (arm, seed) for arm in rescore.EXPECTED_ARMS for seed in (0, 1, 2)
    ]
    by_key = {(r.arm_id, r.seed): r for r in runs}
    assert by_key[("L4", 0)].checkpoint_uri == L4_CHECKPOINT
    assert by_key[("LOO-M", 1)].run_label == "v6m-070b33c-loo-m-seed1-ep40-cos"
    assert by_key[("NO-RED-NIR-IDX", 2)].run_label == "v6m-070b33c-no-red-nir-idx-seed2-ep40-cos"
    for run in runs:
        assert run.config_name == V6_ABLATION_ARMS[run.arm_id].config_name
        assert run.export_uri.startswith(rescore.RESCORE_PREFIX + "/")
        assert run.export_uri.endswith("/eval_validation_patches.parquet")
        assert run.export_uri.split("/")[-1] == patch_export_filename("validation")
        assert run.hydra_overrides[-1] == "evaluate_test=false"


def _write_matrix(tmp_path: Path, edit: Callable[[dict], None]) -> Path:
    raw = OmegaConf.to_container(OmegaConf.load(rescore.MATRIX_FILE), resolve=True)
    assert isinstance(raw, dict)
    edit(raw)
    path = tmp_path / "matrix.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw["arms"].pop("L3"),
        lambda raw: raw["arms"]["L1"].update(seeds=[0, 1, 2, 3]),
        lambda raw: raw["arms"]["L1"].update(seeds=[0, 1]),
        lambda raw: raw.update(max_epochs=20),
        lambda raw: raw.update(lr_schedule="constant"),
        lambda raw: raw.update(early_stopping=True),
    ],
    ids=["missing-arm", "extra-seed", "dropped-seed", "epochs", "constant-lr", "early-stop"],
)
def test_inventory_rejects_a_drifted_matrix(tmp_path: Path, edit: Callable[[dict], None]) -> None:
    with pytest.raises(ValueError):
        rescore.rescore_inventory(_write_matrix(tmp_path, edit))


def test_validate_inventory_rejects_duplicates_drops_and_unknown_arms() -> None:
    runs = list(rescore.rescore_inventory())
    rescore.validate_inventory(runs)
    with pytest.raises(ValueError):
        rescore.validate_inventory(runs[:-1])
    with pytest.raises(ValueError):
        rescore.validate_inventory([*runs[:-1], runs[0]])
    unknown = dataclasses.replace(runs[-1], arm_id="L9")
    with pytest.raises(ValueError):
        rescore.validate_inventory([*runs[:-1], unknown])
    outside = dataclasses.replace(runs[-1], export_uri=runs[-1].checkpoint_uri)
    with pytest.raises(ValueError):
        rescore.validate_inventory([*runs[:-1], outside])


# ── 3 config composition ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def cfgs() -> dict[str, DictConfig]:
    return {
        run.run_label: rescore.compose_run_config(run, output_root=f"out/runs/{run.run_label}")
        for run in rescore.rescore_inventory()
    }


def test_compose_all_27_runs(cfgs: dict[str, DictConfig]) -> None:
    assert len(cfgs) == 27
    for run in rescore.rescore_inventory():
        cfg = cfgs[run.run_label]
        assert cfg.evaluate_test is False
        assert int(cfg.seed) == run.seed
        assert int(cfg.trainer.max_epochs) == 40
        assert lr_schedule_from_config(cfg).name == "cosine"
        assert cfg.ablation_arm == run.arm_id
        assert selection_from_config(cfg).names == V6_ABLATION_ARMS[run.arm_id].channels
        assert tuple(cfg.data.splits) == ("train", "validation")


def test_shared_data_identity(cfgs: dict[str, DictConfig]) -> None:
    rescore.assert_shared_data_identity(list(cfgs.values()))
    tampered = copy.deepcopy(list(cfgs.values()))
    tampered[3].data.batch_size = 8
    with pytest.raises(ValueError):
        rescore.assert_shared_data_identity(tampered)
    other_root = copy.deepcopy(list(cfgs.values()))
    other_root[5].features_root = "gs://elsewhere/features"
    with pytest.raises(ValueError):
        rescore.assert_shared_data_identity(other_root)


def test_compose_rejects_off_contract_config() -> None:
    run = rescore.rescore_inventory()[0]
    bad = rescore.RescoreRun(**{**run.__dict__, "arm_id": "L2"})
    with pytest.raises(ValueError):
        rescore.compose_run_config(bad, output_root="out/runs/x")


# ── 4 checkpoint selection state ──────────────────────────────────────


def _ckpt(tmp_path: Path, callbacks: object, name: str = "c.ckpt") -> Path:
    path = tmp_path / name
    torch.save({"callbacks": callbacks, "epoch": 7, "global_step": 70}, path)
    return path


def _mc(**over: object) -> dict[str, object]:
    state: dict[str, object] = {
        "monitor": "validation/mae_100m",
        "best_model_score": torch.tensor(1.25),
        "current_score": torch.tensor(1.25),
        "best_model_path": "/x/checkpoints/best-07-1.2500.ckpt",
    }
    state.update(over)
    return state


def test_selection_state_good(tmp_path: Path) -> None:
    path = _ckpt(tmp_path, {"ModelCheckpoint{'monitor': 'validation/mae_100m'}": _mc()})
    state = rescore.checkpoint_selection_state(path)
    assert state == {
        "best_model_score": 1.25,
        "best_model_path_name": "best-07-1.2500.ckpt",
        "epoch": 7,
        "global_step": 70,
    }


@pytest.mark.parametrize(
    "callbacks",
    [
        {},
        None,
        {"ModelCheckpoint{a}": _mc(), "ModelCheckpoint{b}": _mc()},
        {"ModelCheckpoint{a}": _mc(monitor="validation/loss")},
        {"ModelCheckpoint{a}": _mc(best_model_score=float("nan"), current_score=None)},
        {"ModelCheckpoint{a}": _mc(best_model_score=None)},
        {"ModelCheckpoint{a}": _mc(current_score=torch.tensor(1.5))},
        {"ModelCheckpoint{a}": _mc(best_model_path="/x/last.ckpt")},
    ],
    ids=["empty", "none", "two", "other-monitor", "nan", "no-score", "not-best", "last"],
)
def test_selection_state_rejects(tmp_path: Path, callbacks: object) -> None:
    with pytest.raises(RuntimeError):
        rescore.checkpoint_selection_state(_ckpt(tmp_path, callbacks))


# ── 5 task load ───────────────────────────────────────────────────────


def _stub_hparams(cfg: DictConfig) -> dict[str, object]:
    return {
        "n_active_channels": selection_from_config(cfg).n_active,
        "base_width": int(cfg.model.base_width),
        "depth": int(cfg.model.depth),
        "learning_rate": float(cfg.trainer.learning_rate),
        "weight_decay": float(cfg.trainer.weight_decay),
        "residual_prior": bool(cfg.stage1_residual_prior),
        "loss_name": str(cfg.contract.loss),
        "aggregation": "arithmetic",
        "lr_schedule": lr_schedule_from_config(cfg).payload(),
        "ssim_data_range": SSIM.payload(),
    }


class _LoadedStub:
    def __init__(self, hparams: dict[str, object]) -> None:
        self.hparams = hparams
        self.evaluated = False

    def eval(self) -> _LoadedStub:
        self.evaluated = True
        return self


def _patch_load(
    monkeypatch: pytest.MonkeyPatch, hparams: dict[str, object]
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake(path: str, **kwargs: object) -> _LoadedStub:
        calls.append({"path": path, **kwargs})
        return _LoadedStub(hparams)

    monkeypatch.setattr(RealLSTTask, "load_from_checkpoint", staticmethod(fake))
    return calls


def test_load_rescore_task_accepts_matching_hparams(
    monkeypatch: pytest.MonkeyPatch, cfgs: dict[str, DictConfig]
) -> None:
    cfg = cfgs["v6m-070b33c-l4-seed0-ep40-cos"]
    calls = _patch_load(monkeypatch, _stub_hparams(cfg))
    task = rescore.load_rescore_task(Path("x.ckpt"), cfg, SSIM)
    assert task.evaluated  # type: ignore[attr-defined]
    assert calls[0]["strict"] is True
    assert calls[0]["map_location"] == "cpu"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_active_channels", 18),
        ("lr_schedule", {"name": "constant"}),
        ("ssim_data_range", {"data_range_k": 1.0}),
        ("loss_name", "thermal_aware"),
        ("base_width", 16),
        ("residual_prior", False),
    ],
)
def test_load_rescore_task_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch, cfgs: dict[str, DictConfig], field: str, value: object
) -> None:
    cfg = cfgs["v6m-070b33c-l4-seed0-ep40-cos"]
    hparams = _stub_hparams(cfg)
    hparams[field] = value
    _patch_load(monkeypatch, hparams)
    with pytest.raises(RuntimeError, match=field):
        rescore.load_rescore_task(Path("x.ckpt"), cfg, SSIM)


# ── 6 cache equivalence ───────────────────────────────────────────────


@pytest.mark.parametrize("arm", ["L1", "NO-RED-NIR-IDX", "L3"])
def test_shared_full_width_cache_equals_the_arm_own_cache(tmp_path: Path, arm: str) -> None:
    samples = _samples()
    indices = V6_ABLATION_ARMS[arm]
    sel_indices = tuple(
        i
        for i, name in enumerate(feature_schema(FEATURE_SCHEMA_VERSION_V6).names)
        if name in indices.channels
    )
    n = len(sel_indices)
    full = _full_cache(tmp_path / "full", samples)
    own_prov = _build(tmp_path / "own", samples, sel_indices)

    shared = DataLoader(
        CachedRealPatchDataset(full.root, full.provenance, range(len(samples))),
        batch_size=3,
        collate_fn=lambda b: collate_real_batch(
            b, n_active_channels=n, channel_indices=sel_indices
        ),
    )
    own = DataLoader(
        CachedRealPatchDataset(tmp_path / "own", own_prov, range(len(samples))),
        batch_size=3,
        collate_fn=lambda b: collate_real_batch(b, n_active_channels=n),
    )
    batches = 0
    for left, right in zip(shared, own, strict=True):
        batches += 1
        assert torch.equal(left.features, right.features)
        assert torch.equal(left.lst_prior, right.lst_prior)
        assert torch.equal(left.target_100m, right.target_100m)
        assert torch.equal(left.mask_100m, right.mask_100m)
        assert left.metadata == right.metadata
    assert batches == 2


# ── 7 validation-only cache build ─────────────────────────────────────


class _FakeReader:
    def __init__(self, samples: dict[str, RealSample]) -> None:
        self.samples = samples
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
        return self.samples[ref.patch_id]


def _refs() -> list[PatchRef]:
    return [
        PatchRef("t0", "LC08_X", "S2_X", "train", 2018, 0, 0, 240, "m"),
        PatchRef("v0", "LC08_X", "S2_X", "validation", 2024, 0, 0, 240, "m"),
        PatchRef("v1", "LC08_Y", "S2_Y", "validation", 2024, 16, 0, 240, "m"),
    ]


@pytest.fixture
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    samples = {
        "t0": _sample("t0", 0, split="train"),
        "v0": _sample("v0", 1),
        "v1": _sample("v1", 2, scene="LC08_Y"),
    }
    seen: list[tuple[str, ...]] = []

    def fake_refs(_source: object, *, splits: tuple[str, ...], scene_ids: object):
        seen.append(tuple(splits))
        return [r for r in _refs() if r.split in splits]

    monkeypatch.setattr(real_task, "RealPatchReader", lambda _s: _FakeReader(samples))
    monkeypatch.setattr(real_task, "load_patch_refs", fake_refs)
    monkeypatch.setattr(real_task, "patch_index_fingerprints", lambda _root: {"index": "x"})
    return seen


def _module(
    *, splits: tuple[str, ...] = ("validation",), indices: tuple[int, ...] | None = None
) -> RealPatchDataModule:
    idx = tuple(range(N_V6)) if indices is None else indices
    return RealPatchDataModule(
        RealSourceConfig("local", "local", "local", "local"),
        batch_size=2,
        n_active_channels=len(idx),
        channel_indices=idx,
        mode="stream",
        splits=splits,
    )


def test_validation_only_cache_builds_and_reads_validation_refs_only(
    tmp_path: Path, fake_sources: list[tuple[str, ...]]
) -> None:
    dm = _module()
    dm.setup()
    provenance = dm.build_validation_only_cache(tmp_path / "cache", max_bytes=MAX_BYTES)
    assert fake_sources == [("validation",)]
    cache = PatchCache(tmp_path / "cache", provenance)
    assert cache.patch_ids == ("v0", "v1")
    assert cache.manifest["active_channels"] == N_V6
    assert cache.manifest["channel_indices"] == list(range(N_V6))
    assert dm.patch_cache_source == "job_local"
    assert dm.cache_build_seconds > 0.0
    assert set(dm.stats()["patches_per_split"]) == {"validation"}  # type: ignore[arg-type]


def test_validation_only_cache_refusals(
    tmp_path: Path, fake_sources: list[tuple[str, ...]]
) -> None:
    unset = _module()
    with pytest.raises(RuntimeError, match="setup"):
        unset.build_validation_only_cache(tmp_path / "a", max_bytes=MAX_BYTES)

    both = _module(splits=("train", "validation"))
    both.setup()
    with pytest.raises(RuntimeError, match="splits"):
        both.build_validation_only_cache(tmp_path / "b", max_bytes=MAX_BYTES)

    narrow = _module(indices=(0, 1, 2))
    narrow.setup()
    with pytest.raises(RuntimeError, match="identity channel indices"):
        narrow.build_validation_only_cache(tmp_path / "c", max_bytes=MAX_BYTES)

    with_cache = _module()
    with_cache.setup()
    with_cache.cache_root = tmp_path / "preset"
    with pytest.raises(RuntimeError, match="without a cache"):
        with_cache.build_validation_only_cache(tmp_path / "d", max_bytes=MAX_BYTES)

    existing = _module()
    existing.setup()
    (tmp_path / "e").mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        existing.build_validation_only_cache(tmp_path / "e", max_bytes=MAX_BYTES)

    wrong_split = _module()
    wrong_split.setup()
    wrong_split._admitted["validation"].append(_refs()[0])
    with pytest.raises(RuntimeError, match="non-validation"):
        wrong_split.build_validation_only_cache(tmp_path / "f", max_bytes=MAX_BYTES)

    empty = _module()
    empty.setup()
    empty._admitted["validation"] = []
    with pytest.raises(RuntimeError, match="empty"):
        empty.build_validation_only_cache(tmp_path / "g", max_bytes=MAX_BYTES)
    assert not any((tmp_path / n).exists() for n in "abcdfg")


def test_build_validation_cache_wires_the_datamodule(
    tmp_path: Path,
    fake_sources: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
    cfgs: dict[str, DictConfig],
) -> None:
    captured: dict[str, object] = {}

    def fake_build(cfg: DictConfig, **kwargs: object) -> RealPatchDataModule:
        captured.update(kwargs)
        return _module()

    monkeypatch.setattr(rescore, "build_real_datamodule", fake_build)
    cfg = copy.deepcopy(cfgs["v6m-070b33c-l4-seed0-ep40-cos"])
    cfg.data.cache_max_bytes = MAX_BYTES
    cache = rescore.build_validation_cache(cfg, tmp_path / "cache")
    assert captured["splits"] == ("validation",)
    assert captured["cache_root"] is None
    assert captured["cache_source"] == "job_local"
    assert captured["channel_indices"] == tuple(range(N_V6))
    assert captured["n_active_channels"] == N_V6
    assert captured["allow_job_local_fallback"] is False
    assert cache.patch_ids == ("v0", "v1")
    assert cache.source_channels == N_V6
    assert len(cache.provenance_sha256) == 64
    files = [p for p in (tmp_path / "cache").iterdir() if p.is_file()]
    assert files
    assert all(p.stat().st_mode & 0o222 == 0 for p in files)
    loader = rescore.validation_loader(cache, cfg)
    batch = next(iter(loader))
    assert tuple(batch.features.shape) == (2, N_V6, REAL_PATCH_PX, REAL_PATCH_PX)


# ── 8 scoring ─────────────────────────────────────────────────────────


class _StubTask(nn.Module):
    def __init__(self, n_active: int, value: float = 290.0) -> None:
        super().__init__()
        self.n_active_channels = n_active
        self.value = value
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, batch: RealBatch) -> torch.Tensor:
        b = batch.features.shape[0]
        return torch.full((b, 1, REAL_PATCH_PX, REAL_PATCH_PX), self.value)


def _loader(samples: list[RealSample], n_active: int) -> DataLoader[RealBatch]:
    return DataLoader(  # type: ignore[return-value]
        samples,  # type: ignore[arg-type]
        batch_size=3,
        collate_fn=lambda b: collate_real_batch(b, n_active_channels=n_active),
    )


def _expected_mae(samples: list[RealSample], value: float = 290.0) -> float:
    errors = [np.abs(value - s.target_100m[s.mask_100m]) for s in samples]
    return float(np.concatenate(errors).mean())


def test_score_validation_rows_and_recheck() -> None:
    samples = _samples()
    rows, recomputed = rescore.score_validation(
        _StubTask(5),  # type: ignore[arg-type]
        _loader(samples, 5),
        aggregation="arithmetic",
        ssim_range=SSIM,
        device=torch.device("cpu"),
    )
    assert [r["split"] for r in rows] == ["validation"] * 4
    assert recomputed == pytest.approx(_expected_mae(samples), rel=1e-5)


def test_score_validation_rejects_non_finite() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        rescore.score_validation(
            _StubTask(5, float("nan")),  # type: ignore[arg-type]
            _loader(_samples(), 5),
            aggregation="arithmetic",
            ssim_range=SSIM,
            device=torch.device("cpu"),
        )


def test_score_validation_rejects_non_validation_rows() -> None:
    with pytest.raises(RuntimeError, match="split"):
        rescore.score_validation(
            _StubTask(5),  # type: ignore[arg-type]
            _loader([_sample("t0", 0, split="train")], 5),
            aggregation="arithmetic",
            ssim_range=SSIM,
            device=torch.device("cpu"),
        )


# ── 9-10 end to end ───────────────────────────────────────────────────


class _FakeStore:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.existing = existing or set()
        self.uploads: list[str] = []

    def exists(self, uri: str) -> bool:
        return uri in self.objects or uri in self.existing

    def atomic_upload(
        self, local: Path, uri: str, overwrite: bool = True, if_generation_match: int | None = None
    ) -> str:
        assert overwrite is False and if_generation_match == 0
        if self.exists(uri):
            raise FileExistsError(uri)
        self.objects[uri] = Path(local).read_bytes()
        self.uploads.append(uri)
        return uri


class _Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.samples = _samples()
        self.expected_mae = _expected_mae(self.samples)
        self.cache_builds = 0
        self.fetches = 0
        self.bad_run: int | None = None
        self.nan_run: int | None = None
        self.store = _FakeStore()
        self.tmp_path = tmp_path
        self.n_active: dict[str, int] = {}
        self.monkeypatch = monkeypatch
        self._current_run = -1
        monkeypatch.setattr(rescore, "storage", self.store)
        monkeypatch.setattr(rescore, "patch_index_fingerprints", lambda _root: {"index": "fp"})
        monkeypatch.setattr(rescore, "resolve_run_ssim_data_range", lambda *a, **k: SSIM)
        monkeypatch.setattr(rescore, "build_validation_cache", self._build_cache)
        monkeypatch.setattr(rescore, "fetch_checkpoint", self._fetch)
        monkeypatch.setattr(rescore, "checkpoint_selection_state", self._selection)
        monkeypatch.setattr(rescore, "load_rescore_task", self._load)
        monkeypatch.setattr(
            run_module,
            "_score_test_once",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("test split scored")),
        )
        monkeypatch.setattr(
            RealPatchDataModule,
            "test_dataloader",
            lambda self: (_ for _ in ()).throw(AssertionError("test loader built")),
        )

    def _build_cache(self, cfg: DictConfig, cache_root: Path) -> rescore.ValidationCache:
        self.cache_builds += 1
        return _full_cache(cache_root, self.samples)

    def _fetch(self, uri: str, dest: Path) -> rescore.CheckpointRef:
        self._current_run = self.fetches
        self.fetches += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"ckpt:{uri}".encode())
        data = dest.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        return rescore.CheckpointRef(uri, 1000 + self.fetches, len(data), digest)

    def _selection(self, path: Path) -> dict[str, object]:
        best = self.expected_mae + (0.5 if self._current_run == self.bad_run else 0.0)
        return {
            "best_model_score": best,
            "best_model_path_name": "best-39-1.0000.ckpt",
            "epoch": 38,
            "global_step": 1234,
        }

    def _load(self, path: Path, cfg: DictConfig, ssim_range: object) -> _StubTask:
        value = float("nan") if self._current_run == self.nan_run else 290.0
        return _StubTask(selection_from_config(cfg).n_active, value)

    def run(self, *, upload: bool = True, root: Path | None = None) -> dict[str, object]:
        return rescore.run_rescore(
            output_root=root or self.tmp_path / "out",
            source_sha=SOURCE_SHA,
            image_digest=IMAGE_DIGEST,
            manifest_uri=rescore.RESCORE_MANIFEST_URI,
            device="cpu",
            upload=upload,
        )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def test_run_rescore_end_to_end(harness: _Harness) -> None:
    manifest = harness.run()
    out = harness.tmp_path / "out"
    runs = rescore.rescore_inventory()
    assert harness.cache_builds == 1
    assert (out / "data_scope.json").is_file()
    assert not list((out / "checkpoints").glob("*.ckpt"))

    for run in runs:
        path = out / "runs" / run.run_label / "eval_validation_patches.parquet"
        table = load_patch_exports([path], split="validation")
        assert table.num_rows == 4
        footer = json.loads(pq.read_schema(path).metadata[PATCH_EXPORT_METADATA_KEY])
        assert footer["arm"] == run.arm_id
        assert footer["seed"] == run.seed
        assert footer["run_id"] == run.run_label
        assert footer["checkpoint"] == run.checkpoint_uri
        assert footer["split"] == "validation"
        assert footer["ssim_data_range"] == SSIM.payload()
        assert footer["prediction_clamp_k"] == [float(v) for v in EVAL_PREDICTION_CLAMP_K]
        assert (out / "runs" / run.run_label / "rescore_run.json").is_file()

    assert manifest["evaluate_test"] is False
    assert manifest["splits_read"] == ["validation"]
    assert manifest["split"] == "validation"
    assert manifest["test_split"] == "not_read"
    entries = manifest["runs"]
    assert len(entries) == 27  # type: ignore[arg-type]
    for run, entry in zip(runs, entries, strict=True):  # type: ignore[arg-type]
        assert entry["arm"] == run.arm_id and entry["seed"] == run.seed
        assert entry["split"] == "validation" and entry["evaluate_test"] is False
        assert entry["checkpoint"]["uri"] == run.checkpoint_uri
        assert entry["checkpoint"]["generation"] > 0
        assert entry["recomputed_validation_mae_100m"] == pytest.approx(
            harness.expected_mae, rel=1e-5
        )
        assert entry["export"]["uri"] == run.export_uri
        assert entry["patches"] == 4 and entry["scenes"] == 2
        local = (out / entry["export"]["path"]).read_bytes()
        assert entry["export"]["sha256"] == hashlib.sha256(local).hexdigest()
        assert entry["export"]["bytes"] == len(local)

    expected_uploads = [run.export_uri for run in runs] + [rescore.RESCORE_MANIFEST_URI]
    assert harness.store.uploads == expected_uploads
    assert all(u.startswith(rescore.RESCORE_PREFIX + "/") for u in harness.store.uploads)
    assert json.loads(harness.store.objects[rescore.RESCORE_MANIFEST_URI]) == json.loads(
        (out / "manifest.json").read_text()
    )
    for entry in entries:  # type: ignore[attr-defined]
        assert (
            hashlib.sha256(harness.store.objects[entry["export"]["uri"]]).hexdigest()
            == (entry["export"]["sha256"])
        )


def test_run_rescore_without_upload_writes_a_local_manifest(harness: _Harness) -> None:
    manifest = harness.run(upload=False)
    assert harness.store.uploads == []
    assert manifest["uploaded"] is False
    assert (harness.tmp_path / "out" / "manifest.json").is_file()


def test_existing_destination_fails_before_any_work(harness: _Harness) -> None:
    runs = rescore.rescore_inventory()
    harness.store.existing.add(runs[13].export_uri)
    with pytest.raises(FileExistsError):
        harness.run()
    assert harness.cache_builds == 0 and harness.fetches == 0
    assert harness.store.uploads == []


def test_existing_manifest_fails_before_any_work(harness: _Harness) -> None:
    harness.store.existing.add(rescore.RESCORE_MANIFEST_URI)
    with pytest.raises(FileExistsError):
        harness.run()
    assert harness.cache_builds == 0 and harness.fetches == 0


def test_selection_mismatch_uploads_nothing(harness: _Harness) -> None:
    harness.bad_run = 4
    with pytest.raises(RuntimeError, match="recomputed validation MAE"):
        harness.run()
    assert harness.store.uploads == []
    assert not (harness.tmp_path / "out" / "manifest.json").exists()
    assert not list((harness.tmp_path / "out" / "checkpoints").glob("*.ckpt"))


def test_non_finite_prediction_uploads_nothing(harness: _Harness) -> None:
    harness.nan_run = 4
    with pytest.raises(RuntimeError, match="non-finite"):
        harness.run()
    assert harness.store.uploads == []


@pytest.mark.parametrize(
    ("source_sha", "digest", "manifest_uri"),
    [
        ("abc", IMAGE_DIGEST, rescore.RESCORE_MANIFEST_URI),
        (SOURCE_SHA, "sha256:abc", rescore.RESCORE_MANIFEST_URI),
        (SOURCE_SHA, IMAGE_DIGEST, "gs://berlin-lst-training-data/other/manifest.json"),
    ],
)
def test_run_rescore_validates_identifiers(
    harness: _Harness, source_sha: str, digest: str, manifest_uri: str
) -> None:
    with pytest.raises(ValueError):
        rescore.run_rescore(
            output_root=harness.tmp_path / "out",
            source_sha=source_sha,
            image_digest=digest,
            manifest_uri=manifest_uri,
            device="cpu",
        )
    assert harness.fetches == 0


def test_upload_create_only_refuses_outside_the_prefix(tmp_path: Path) -> None:
    local = tmp_path / "f.bin"
    local.write_bytes(b"x")
    with pytest.raises(ValueError):
        rescore.upload_create_only(local, rescore.MATRIX_EVIDENCE_PREFIX + "/other/best.ckpt")


def test_rescore_module_never_touches_the_test_split() -> None:
    source = Path(rescore.__file__).read_text(encoding="utf-8")
    for forbidden in ("_score_test_once", "test_dataloader", "evaluate_test=true"):
        assert forbidden not in source
    assert rescore.RESCORE_SPLITS == ("validation",)


def test_compose_appends_the_seed_segment_the_lock_requires(cfgs: dict[str, DictConfig]) -> None:
    cfg = cfgs["v6m-070b33c-l4-seed2-ep40-cos"]
    assert cfg.output_root == "out/runs/v6m-070b33c-l4-seed2-ep40-cos/seed-2"
