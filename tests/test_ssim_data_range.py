"""Fixed, train-only SSIM data range L: derivation, artifact, metric, and wiring."""

from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.modeling import metrics, real_task, ssim_range
from berlin_lst_downscaling.modeling.contracts import RealBatch, RealSampleMeta
from berlin_lst_downscaling.modeling.metrics import (
    METRIC_ONLY_SSIM_KEYS,
    SSIM_CLAMP_RANGE_K,
    SSIM_K1,
    SSIM_K2,
    SSIM_LEGACY_DATA_RANGE_K,
    masked_ssim_stats,
    masked_ssim_stats_patch_range,
    ssim_constants,
    thermal_objective_batch,
)
from berlin_lst_downscaling.modeling.patches import PatchRef, RealSourceConfig
from berlin_lst_downscaling.modeling.real_task import RealLSTTask
from berlin_lst_downscaling.modeling.run import resolve_run_ssim_data_range
from berlin_lst_downscaling.modeling.ssim_range import (
    SsimRangeSample,
    artifact_payload,
    derive_ssim_data_range,
    load_ssim_data_range,
    write_artifact,
)

_REPO = Path(__file__).resolve().parents[1]
_COMMITTED = _REPO / "configs" / "modeling" / "ssim_data_range" / "v1.yaml"
_FP = {"index_policy_hash": "idx-h", "source_policy_hash": "src-h"}


def _sample(
    patch_id: str,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    split: str = "train",
    year: int = 2018,
    scene_id: str = "scene-a",
) -> SsimRangeSample:
    return SsimRangeSample(
        patch_id=patch_id,
        scene_id=scene_id,
        split=split,
        year=year,
        target_100m=target,
        mask_100m=mask,
    )


# 1. train-only derivation


@pytest.mark.parametrize(
    ("split", "year"),
    [("validation", 2024), ("test", 2025), ("train", 2024), ("validation", 2018)],
)
def test_derivation_rejects_non_train(split: str, year: int) -> None:
    target = np.full((16, 16), 300.0, dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="train-only"):
        derive_ssim_data_range([_sample("p", target, mask, split=split, year=year)])


def test_derivation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        derive_ssim_data_range([_sample("p", np.zeros((16, 16)), np.ones((8, 8), dtype=bool))])


def test_derivation_without_valid_cells_raises() -> None:
    with pytest.raises(ValueError, match="no valid"):
        derive_ssim_data_range([_sample("p", np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))])


# 2. percentiles and counts


def test_derivation_percentiles_and_counts() -> None:
    rng = np.random.default_rng(0)
    samples = []
    expected: list[np.ndarray] = []
    n_nonfinite = 0
    for i in range(6):
        target = rng.normal(300.0, 8.0, size=(16, 16)).astype(np.float32)
        mask = rng.random((16, 16)) > 0.3
        if i == 0:
            target[1, 1], target[2, 2] = np.nan, np.inf
            mask[1, 1] = mask[2, 2] = True
            n_nonfinite += 2
            target[3, 3] = np.nan
            mask[3, 3] = False  # non-finite under mask 0: not counted
        valid = mask & np.isfinite(target)
        expected.append(target[valid].astype(np.float64))
        samples.append(_sample(f"p{i}", target, mask, year=2017 + i, scene_id=f"s{i % 3}"))
    samples.append(
        _sample("empty", np.full((16, 16), np.nan, np.float32), np.zeros((16, 16), bool))
    )
    stats = derive_ssim_data_range(samples)
    values = np.concatenate(expected)
    low, high = np.percentile(values, [0.5, 99.5], method="linear")
    assert stats.low_k == pytest.approx(float(low))
    assert stats.high_k == pytest.approx(float(high))
    assert stats.data_range_k == pytest.approx(float(high - low))
    assert stats.n_cells == values.size
    assert stats.n_patches == 6
    assert stats.n_patches_empty == 1
    assert stats.n_nonfinite_masked_cells == n_nonfinite
    assert stats.n_scenes == 3
    assert stats.years == tuple(range(2017, 2023))
    assert stats.payload()["years"] == list(range(2017, 2023))


# 3. script and artifact IO


def _load_script() -> ModuleType:
    path = _REPO / "scripts" / "operators" / "derive_ssim_data_range.py"
    spec = importlib.util.spec_from_file_location("derive_ssim_data_range_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ref(pid: str, split: str = "train", year: int = 2018) -> PatchRef:
    return PatchRef(
        patch_id=pid,
        scene_id="scene-a",
        s2_scene_id="s2",
        split=split,
        year=year,
        row=0,
        col=0,
        n_eligible=256,
        eligibility_mask="m",
    )


def _source() -> RealSourceConfig:
    return RealSourceConfig(
        patch_index_root="gs://example/index",
        training_root="gs://example/training",
        features_root="gs://example/features",
        ard_root="gs://example/ard",
    )


class _FakeReader:
    def __init__(self, _source: RealSourceConfig) -> None:
        self.exclusions: Counter[str] = Counter()
        self.preloaded: set[str] = set()
        self.calls = 0

    def preload(self, scene_ids: set[str]) -> None:
        self.preloaded = set(scene_ids)

    def read_target_and_mask(self, ref: PatchRef):  # noqa: ANN201
        self.calls += 1
        if ref.patch_id == "excluded":
            self.exclusions["empty_mask"] += 1
            return None
        rng = np.random.default_rng(abs(hash(ref.patch_id)) % 1000)
        return rng.normal(300.0, 5.0, (16, 16)).astype(np.float32), np.ones((16, 16), bool)


def test_derive_from_source_requests_train_only_and_matches() -> None:
    script = _load_script()
    seen: dict[str, object] = {}

    def refs_loader(_cfg: RealSourceConfig, **kwargs: object) -> list[PatchRef]:
        seen.update(kwargs)
        return [_ref("a"), _ref("b"), _ref("excluded")]

    readers: list[_FakeReader] = []

    def factory(cfg: RealSourceConfig) -> _FakeReader:
        readers.append(_FakeReader(cfg))
        return readers[-1]

    stats, exclusions, n_refs, fingerprints = script.derive_from_source(
        _source(),
        reader_factory=factory,
        refs_loader=refs_loader,
        fingerprints_loader=lambda _root: dict(_FP),
    )
    assert seen == {"splits": ("train",)}
    assert script.DERIVATION_SPLITS == ("train",)
    assert n_refs == 3
    assert exclusions == {"empty_mask": 1}
    assert fingerprints == _FP
    assert readers[0].preloaded == {"scene-a"}
    assert stats.n_patches == 2
    assert stats.n_cells == 512
    values = np.concatenate(
        [factory(_source()).read_target_and_mask(_ref(p))[0].ravel() for p in ("a", "b")]
    ).astype(np.float64)
    low, high = np.percentile(values, [0.5, 99.5], method="linear")
    assert stats.data_range_k == pytest.approx(float(high - low))


def test_derive_from_source_rejects_validation_ref() -> None:
    script = _load_script()
    with pytest.raises(RuntimeError, match="non-train"):
        script.derive_from_source(
            _source(),
            reader_factory=_FakeReader,
            refs_loader=lambda _c, **_k: [_ref("a"), _ref("v", "validation", 2024)],
            fingerprints_loader=lambda _r: dict(_FP),
        )


def _filled_payload() -> dict:
    stats = derive_ssim_data_range(
        [_sample("p", np.linspace(280, 320, 256).reshape(16, 16), np.ones((16, 16), bool))]
    )
    return artifact_payload(
        stats,
        patch_index_root="gs://example/index",
        training_root="gs://example/training",
        fingerprints=_FP,
        config_name="v6_l5_locked",
        git_commit="abc",
        derived_at="2026-01-01T00:00:00+00:00",
        exclusions={"empty_mask": 1},
        n_refs_requested=2,
    )


def test_write_artifact_refuses_remote_and_round_trips(tmp_path: Path) -> None:
    payload = _filled_payload()
    with pytest.raises(ValueError, match="locally"):
        write_artifact(payload, "gs://bucket/v1.yaml")
    out = write_artifact(payload, tmp_path / "sub" / "v1.yaml")
    loaded = load_ssim_data_range(out, expected_fingerprints=_FP)
    assert loaded.source == "artifact"
    assert loaded.data_range_k == pytest.approx(payload["data_range_k"])
    assert loaded.index_policy_hash == "idx-h"
    assert loaded.artifact_sha256 is not None and len(loaded.artifact_sha256) == 64
    assert loaded.payload()["clamp_range_k"] == list(LST_RANGE_K)
    with pytest.raises(RuntimeError, match="derive_ssim_data_range.py"):
        load_ssim_data_range(out, expected_fingerprints={**_FP, "source_policy_hash": "other"})


def test_load_rejects_inconsistent_artifact(tmp_path: Path) -> None:
    payload = _filled_payload()
    for key, value in (
        ("split", "validation"),
        ("p_low", 1.0),
        ("percentile_method", "nearest"),
        ("data_range_k", payload["data_range_k"] + 1.0),
        ("data_range_k", -1.0),
    ):
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump({**payload, key: value}), encoding="utf-8")
        with pytest.raises(RuntimeError):
            load_ssim_data_range(path)


# 4. committed placeholder fails closed


def _point_at_committed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ssim_range, "SSIM_DATA_RANGE_ARTIFACT", _COMMITTED)


def test_committed_artifact_is_placeholder_or_valid() -> None:
    doc = yaml.safe_load(_COMMITTED.read_text(encoding="utf-8"))
    assert doc["split"] == "train"
    if doc["status"] == "placeholder":
        assert doc["data_range_k"] is None
    else:
        assert load_ssim_data_range(_COMMITTED).data_range_k > 0


def test_placeholder_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = yaml.safe_load(_COMMITTED.read_text(encoding="utf-8"))
    if doc["status"] != "placeholder":
        pytest.skip("artifact already filled")
    _point_at_committed(monkeypatch)
    pred = torch.full((1, 1, 16, 16), 300.0)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    with pytest.raises(RuntimeError, match=r"derive_ssim_data_range\.py"):
        load_ssim_data_range()
    with pytest.raises(RuntimeError, match=r"derive_ssim_data_range\.py"):
        masked_ssim_stats(pred, pred + 1.0, mask)
    with pytest.raises(RuntimeError, match=r"derive_ssim_data_range\.py"):
        thermal_objective_batch(pred, pred + 1.0, mask)
    with pytest.raises(RuntimeError, match=r"derive_ssim_data_range\.py"):
        RealLSTTask(n_active_channels=1, depth=1, base_width=4)


def test_missing_artifact_names_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"derive_ssim_data_range\.py"):
        load_ssim_data_range(tmp_path / "missing.yaml")


# 5. constants and reference SSIM


def test_ssim_constants() -> None:
    assert ssim_constants(40.0) == ((0.01 * 40.0) ** 2, (0.03 * 40.0) ** 2)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ssim_constants(bad)


def _reference_mean_ssim(pred: np.ndarray, target: np.ndarray, length: float) -> tuple[float, int]:
    c1, c2 = (SSIM_K1 * length) ** 2, (SSIM_K2 * length) ** 2
    values = []
    h, w = pred.shape
    for i in range(h - 6):
        for j in range(w - 6):
            x = pred[i : i + 7, j : j + 7].astype(np.float64).ravel()
            y = target[i : i + 7, j : j + 7].astype(np.float64).ravel()
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cov = ((x - mx) * (y - my)).mean()
            values.append(
                ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
            )
    return float(np.sum(values)), len(values)


def _synthetic_pair(seed: int = 1, n: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    target = 300.0 + 5.0 * torch.randn((n, 1, 12, 12), generator=g)
    pred = target + 1.5 * torch.randn((n, 1, 12, 12), generator=g)
    return pred, target, torch.ones((n, 1, 12, 12), dtype=torch.bool)


def test_ssim_matches_float64_reference_and_default_uses_artifact() -> None:
    pred, target, mask = _synthetic_pair()
    for length in (40.0, 17.5):
        total, count = masked_ssim_stats(
            pred, target, mask, clamp_predictions=False, data_range_k=length
        )
        ref_sum, ref_n = _reference_mean_ssim(pred[0, 0].numpy(), target[0, 0].numpy(), length)
        assert int(count) == ref_n
        assert float(total) == pytest.approx(ref_sum, abs=1e-3)
    default = masked_ssim_stats(pred, target, mask)
    explicit = masked_ssim_stats(pred, target, mask, data_range_k=40.0)
    legacy = masked_ssim_stats(pred, target, mask, data_range_k=250.0)
    assert float(default[0]) == float(explicit[0])
    assert float(default[0]) != float(legacy[0])


# 6. clamp stays at LST_RANGE_K


def test_clamp_range_unchanged() -> None:
    assert SSIM_CLAMP_RANGE_K == LST_RANGE_K
    assert SSIM_LEGACY_DATA_RANGE_K == 250.0
    pred, target, mask = _synthetic_pair()
    target = target.clone()
    target[0, 0, 0, 0] = 900.0
    target[0, 0, 5, 5] = -20.0
    clamped = target.clamp(*LST_RANGE_K)
    a = masked_ssim_stats(pred, target, mask, data_range_k=40.0)
    b = masked_ssim_stats(pred, clamped, mask, data_range_k=40.0)
    assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-6)
    # Clamping to the artifact bounds (280, 320) would give a different result.
    narrow = target.clamp(280.0, 320.0)
    c = masked_ssim_stats(pred, narrow, mask, data_range_k=40.0)
    assert float(a[0]) != pytest.approx(float(c[0]), abs=1e-6)


# 7. bf16 autocast


def test_bf16_autocast_matches_fp32() -> None:
    pred, target, mask = _synthetic_pair(n=2)
    ref = masked_ssim_stats(pred, target, mask, clamp_predictions=False, data_range_k=40.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = masked_ssim_stats(
            pred.bfloat16(), target.bfloat16(), mask, clamp_predictions=False, data_range_k=40.0
        )
    assert math.isfinite(float(got[0]))
    assert float(got[1]) == float(ref[1])
    assert float(got[0]) / float(got[1]) == pytest.approx(float(ref[0]) / float(ref[1]), abs=0.05)


# 8. per-patch-L variant


def test_patch_range_variant_excludes_zero_range_patch() -> None:
    pred, target, mask = _synthetic_pair(seed=3, n=3)
    target[0] = 300.0  # zero-range patch
    mask[2] = False  # no valid cells: excluded, not counted as zero-range
    total, count, zero = masked_ssim_stats_patch_range(pred, target, mask)
    assert int(zero) == 1
    assert math.isfinite(float(total)) and float(count) > 0
    patch_l = float(target[1].clamp(*LST_RANGE_K).max() - target[1].clamp(*LST_RANGE_K).min())
    exp_total, exp_count = masked_ssim_stats(
        pred[1:2], target[1:2], mask[1:2], clamp_predictions=False, data_range_k=patch_l
    )
    assert float(count) == float(exp_count)
    assert float(total) == pytest.approx(float(exp_total), abs=1e-4)


# 9. variants never in loss or checkpoint selection


def _tiny_batch() -> RealBatch:
    meta = RealSampleMeta(
        patch_id="s:E0N0",
        scene_id="s",
        split="train",
        year=2018,
        row=0,
        col=0,
        filled_feature_pixels=0,
    )
    g = torch.Generator().manual_seed(0)
    return RealBatch(
        features=torch.zeros(1, 1, 160, 160),
        lst_prior=torch.zeros(1, 1, 160, 160),
        target_100m=300.0 + torch.randn((1, 1, 16, 16), generator=g),
        mask_100m=torch.ones(1, 1, 16, 16, dtype=torch.bool),
        metadata=[meta],
    )


def test_variants_never_enter_loss_or_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("metric-only SSIM variant used in the loss path")

    monkeypatch.setattr(metrics, "masked_ssim_stats_patch_range", boom)
    monkeypatch.setattr(real_task, "masked_ssim_stats_patch_range", boom)
    batch = _tiny_batch()
    pred = torch.full((1, 1, 16, 16), 300.0, requires_grad=True)
    thermal_objective_batch(pred, batch.target_100m, batch.mask_100m)
    task = RealLSTTask(
        loss_name="thermal_aware",
        n_active_channels=1,
        depth=1,
        base_width=4,
        ssim_data_range_k=40.0,
    )
    task.log = lambda *_a, **_k: None  # type: ignore[method-assign]
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert task.hparams["ssim_data_range_k"] == 40.0
    assert "validation/mae_100m" not in METRIC_ONLY_SSIM_KEYS
    run_source = (_REPO / "src/berlin_lst_downscaling/modeling/run.py").read_text()
    assert 'monitored = "validation/mae_100m"' in run_source


def test_validation_step_logs_variants() -> None:
    task = RealLSTTask(
        loss_name="masked_l1", n_active_channels=1, depth=1, base_width=4, ssim_data_range_k=40.0
    )
    logged: set[str] = set()
    task.log = lambda name, *_a, **_k: logged.add(name)  # type: ignore[method-assign]
    task.forward = lambda _b: torch.full((1, 1, 160, 160), 300.0)  # type: ignore[method-assign]
    task.validation_step(_tiny_batch(), 0)
    assert set(METRIC_ONLY_SSIM_KEYS) <= logged


# 10. run-level resolution


def test_resolve_run_ssim_data_range_override_rules() -> None:
    real = OmegaConf.create({"data": {"kind": "real"}, "ssim_data_range_test_override_k": 40.0})
    with pytest.raises(RuntimeError, match="rejected"):
        resolve_run_ssim_data_range(real, expected_fingerprints=None)
    smoke = OmegaConf.create(
        {"data": {"kind": "synthetic_contract"}, "ssim_data_range_test_override_k": 40.0}
    )
    got = resolve_run_ssim_data_range(smoke, expected_fingerprints=None)
    assert got.source == "test_override" and got.data_range_k == 40.0
    plain = OmegaConf.create({"data": {"kind": "real"}})
    from_artifact = resolve_run_ssim_data_range(
        plain,
        expected_fingerprints={
            "index_policy_hash": "test-index-policy-hash",
            "source_policy_hash": "test-source-policy-hash",
        },
    )
    assert from_artifact.source == "artifact" and from_artifact.data_range_k == 40.0
