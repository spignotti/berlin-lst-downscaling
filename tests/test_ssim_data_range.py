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
    check_artifact_writable,
    derive_ssim_data_range,
    load_ssim_data_range,
    resolve_task_ssim_data_range,
    test_override_ssim_data_range,
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
        [
            _sample(
                f"p{year}",
                np.linspace(280, 320, 256).reshape(16, 16),
                np.ones((16, 16), bool),
                year=year,
            )
            for year in range(2017, 2024)
        ]
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


def test_load_rejects_missing_years(tmp_path: Path) -> None:
    payload = _filled_payload()
    del payload["years"]
    path = tmp_path / "noyears.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="years"):
        load_ssim_data_range(path)


def test_load_rejects_inconsistent_artifact(tmp_path: Path) -> None:
    payload = _filled_payload()
    for key, value in (
        ("split", "validation"),
        ("years", None),
        ("years", [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]),
        ("years", [2018, 2019, 2020, 2021, 2022, 2023]),
        ("years", [True, 2018, 2019, 2020, 2021, 2022, 2023]),
        ("years", ["2017", 2018, 2019, 2020, 2021, 2022, 2023]),
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


# 11. checkpoint provenance and reload check


def _artifact_path() -> Path:
    return ssim_range.SSIM_DATA_RANGE_ARTIFACT


def test_task_hparams_carry_provenance() -> None:
    import hashlib

    task = RealLSTTask(n_active_channels=1, depth=1, base_width=4)
    prov = task.hparams["ssim_data_range"]
    assert prov["source"] == "artifact"
    assert prov["version"] == "v1"
    assert prov["artifact_sha256"] == hashlib.sha256(_artifact_path().read_bytes()).hexdigest()
    assert prov["index_policy_hash"] == "test-index-policy-hash"
    assert prov["source_policy_hash"] == "test-source-policy-hash"
    assert prov["derived_at"] == "2026-01-01T00:00:00+00:00"
    assert prov["data_range_k"] == 40.0
    assert task.hparams["ssim_data_range_k"] == 40.0
    yaml.safe_dump(dict(task.hparams))  # plain, serializable types only


def test_reload_detects_changed_artifact(tmp_path: Path, write_ssim_artifact) -> None:  # noqa: ANN001
    import lightning

    task = RealLSTTask(n_active_channels=1, depth=1, base_width=4)
    ckpt = tmp_path / "t.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": lightning.__version__,
            "state_dict": task.state_dict(),
            "hyper_parameters": dict(task.hparams),
        },
        ckpt,
    )
    loaded = RealLSTTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    assert loaded.ssim_data_range == task.ssim_data_range
    RealLSTTask(**dict(task.hparams))
    # Same L, different bytes: still a different artifact.
    doc = yaml.safe_load(_artifact_path().read_text(encoding="utf-8"))
    doc["derived_at"] = "2026-02-02T00:00:00+00:00"
    _artifact_path().write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different SSIM data-range artifact"):
        RealLSTTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    with pytest.raises(RuntimeError, match="different SSIM data-range artifact"):
        RealLSTTask(**dict(task.hparams))


def test_tessera_task_provenance_round_trip() -> None:
    from berlin_lst_downscaling.modeling.tessera_adapter import TesseraRealLSTTask

    task = TesseraRealLSTTask(n_kept_channels=2, tessera_adapter_width=2, base_width=4, depth=1)
    assert task.hparams["ssim_data_range"]["source"] == "artifact"
    assert task.hparams["ssim_data_range_k"] == 40.0
    TesseraRealLSTTask(**dict(task.hparams))


def test_override_provenance_round_trip_and_mismatch() -> None:
    task = RealLSTTask(n_active_channels=1, depth=1, base_width=4, ssim_data_range_k=33.0)
    assert task.hparams["ssim_data_range"]["source"] == "test_override"
    again = RealLSTTask(**dict(task.hparams))
    assert again.ssim_data_range_k == 33.0
    prov = test_override_ssim_data_range(33.0).payload()
    assert resolve_task_ssim_data_range(33.0, prov) == (33.0, prov)
    with pytest.raises(RuntimeError, match="mismatch"):
        resolve_task_ssim_data_range(34.0, prov)
    with pytest.raises(RuntimeError, match="mismatch"):
        resolve_task_ssim_data_range(None, prov)
    with pytest.raises(RuntimeError, match="unknown"):
        resolve_task_ssim_data_range(None, {"source": "other"})
    art = load_ssim_data_range().payload()
    with pytest.raises(RuntimeError, match="differs from the artifact"):
        resolve_task_ssim_data_range(41.0, art)


# 12. RF and baseline load the artifact once, up front


def _count_loader_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = ssim_range.load_ssim_data_range

    def counting(*a: object, **k: object):  # noqa: ANN202
        calls[0] += 1
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(ssim_range, "load_ssim_data_range", counting)
    monkeypatch.setattr(metrics, "production_ssim_data_range_k", counting)
    return calls


def test_baseline_uses_passed_range_and_never_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    from berlin_lst_downscaling.modeling import baseline

    calls = _count_loader_calls(monkeypatch)
    seen: list[float | None] = []
    real_stats = baseline.masked_ssim_stats

    def spy(*a: object, data_range_k: float | None = None, **k: object):  # noqa: ANN202
        seen.append(data_range_k)
        return real_stats(*a, data_range_k=data_range_k, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(baseline, "masked_ssim_stats", spy)
    rng = np.random.default_rng(0)

    class Reader:
        exclusions: Counter[str] = Counter()
        cfg = RealSourceConfig(
            patch_index_root="x", training_root="x", features_root="x", ard_root="x"
        )

        def read_patch(self, ref: PatchRef) -> real_task.RealSample:
            meta = RealSampleMeta(
                patch_id=ref.patch_id,
                scene_id="s",
                split=ref.split,
                year=ref.year,
                row=0,
                col=0,
                filled_feature_pixels=0,
            )
            return real_task.RealSample(
                meta=meta,
                features=np.zeros((1, 160, 160), np.float32),
                lst_prior_k=(300 + rng.normal(size=(1, 160, 160))).astype(np.float32),
                target_100m=(300 + rng.normal(size=(1, 16, 16))).astype(np.float32),
                mask_100m=np.ones((1, 16, 16), bool),
            )

    rng_range = test_override_ssim_data_range(17.0)
    records, _ = baseline.evaluate_baseline(
        Reader(),
        [_ref("a"), _ref("b")],
        ssim_range=rng_range,  # type: ignore[arg-type]
    )
    assert len(records) == 2
    assert seen == [17.0, 17.0]
    assert calls == [0]


def test_rf_score_stage_uses_passed_range_and_never_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from berlin_lst_downscaling.modeling import random_forest as rf

    calls = _count_loader_calls(monkeypatch)
    seen: list[float | None] = []
    real_stats = rf.masked_ssim_stats

    def spy(*a: object, data_range_k: float | None = None, **k: object):  # noqa: ANN202
        seen.append(data_range_k)
        return real_stats(*a, data_range_k=data_range_k, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(rf, "masked_ssim_stats", spy)
    monkeypatch.setattr(rf, "predict_10m", lambda *a, **k: np.zeros((160, 160), dtype=np.float32))
    feature = tmp_path / "f.npy"
    np.save(feature, np.zeros((1, 160, 160), np.float32))
    cache = rf.EvalSplit(
        split="validation",
        requested_patches=2,
        patch_ids=["a", "b"],
        feature_files=[str(feature)] * 2,
        prior_100m=np.full((2, 16, 16), 300.0, np.float32),
        target_100m=(300 + np.random.default_rng(0).normal(size=(2, 16, 16))).astype(np.float32),
        mask_100m=np.ones((2, 16, 16), bool),
        filled_feature_pixels=[0, 0],
        exclusions={},
        seconds=0.0,
    )
    stage = rf.StageSelection(stage=1, names=(), indices=(), feature_order="x")
    records = rf.score_stage(cache, None, stage, ssim_range=test_override_ssim_data_range(23.0))  # type: ignore[arg-type]
    assert len(records) == 2
    assert seen == [23.0, 23.0]
    assert calls == [0]


def _runner(name: str) -> ModuleType:
    path = _REPO / "scripts" / "runners" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["run_baseline", "run_random_forest"])
def test_runners_load_artifact_before_any_data_read(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import inspect

    module = _runner(name)
    cfg = OmegaConf.create(
        {
            k: "gs://x/" + k
            for k in ("patch_index_root", "training_root", "features_root", "ard_root")
        }
    )
    cfg.aggregation = "arithmetic"
    monkeypatch.setattr(
        "berlin_lst_downscaling.modeling.baseline.patch_index_fingerprints",
        lambda _r: {"index_policy_hash": "a", "source_policy_hash": "b"},
    )
    monkeypatch.setattr(
        "berlin_lst_downscaling.modeling.random_forest.patch_index_fingerprints",
        lambda _r: {"index_policy_hash": "a", "source_policy_hash": "b"},
    )
    monkeypatch.setattr(ssim_range, "SSIM_DATA_RANGE_ARTIFACT", tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError, match=r"derive_ssim_data_range\.py"):
        module._resolve_source_and_ssim(cfg)
    placeholder = tmp_path / "placeholder.yaml"
    placeholder.write_text(
        yaml.safe_dump({"status": "placeholder", "split": "train", "data_range_k": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ssim_range, "SSIM_DATA_RANGE_ARTIFACT", placeholder)
    with pytest.raises(RuntimeError, match="placeholder"):
        module._resolve_source_and_ssim(cfg)
    # In main, the loader call precedes every data read and the fit.
    src = inspect.getsource(module.main)
    first_read = min(
        src.index(token)
        for token in ("load_refs(", "load_patch_refs(", "collect_train_pool(", ".fit(")
        if token in src
    )
    assert src.index("_resolve_source_and_ssim(") < first_read


# 13. overwrite guard and GDAL credentials


def test_overwrite_guard(tmp_path: Path) -> None:
    payload = _filled_payload()
    path = tmp_path / "v1.yaml"
    write_artifact(payload, path)
    with pytest.raises(FileExistsError, match="--overwrite-filled"):
        write_artifact(payload, path)
    with pytest.raises(FileExistsError, match="--overwrite-filled"):
        check_artifact_writable(path, overwrite_filled=False)
    write_artifact(payload, path, overwrite_filled=True)
    garbage = tmp_path / "bad.yaml"
    garbage.write_text(": : [", encoding="utf-8")
    with pytest.raises(FileExistsError):
        check_artifact_writable(garbage, overwrite_filled=False)
    placeholder = tmp_path / "ph.yaml"
    placeholder.write_text(yaml.safe_dump({"status": "placeholder"}), encoding="utf-8")
    write_artifact(payload, placeholder)
    check_artifact_writable(tmp_path / "missing.yaml", overwrite_filled=False)


def test_cli_refuses_filled_before_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    out = write_artifact(_filled_payload(), tmp_path / "v1.yaml")

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("derivation must not start")

    monkeypatch.setattr(script, "derive_from_source", boom)
    monkeypatch.setattr(script, "ensure_gdal_gcs_auth", boom)
    with pytest.raises(FileExistsError, match="--overwrite-filled"):
        script.main(["--out", str(out)])


def test_cli_sets_gdal_credentials_before_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    script = _load_script()
    for var in (
        "GS_SECRET_ACCESS_KEY",
        "GS_ACCESS_KEY_ID",
        "GS_OAUTH2_REFRESH_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.setenv(var, "x")  # registers restore-to-unset for ensure_gdal_gcs_auth
        monkeypatch.delenv(var)
    monkeypatch.setenv("HOME", str(tmp_path))
    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}", encoding="utf-8")
    seen: dict[str, str | None] = {}
    stats = derive_ssim_data_range(
        [
            _sample(
                f"p{y}", np.linspace(280, 320, 256).reshape(16, 16), np.ones((16, 16), bool), year=y
            )
            for y in range(2017, 2024)
        ]
    )

    def fake_derive(_source: RealSourceConfig) -> tuple:
        seen["adc"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        return stats, {}, 7, dict(_FP)

    monkeypatch.setattr(script, "derive_from_source", fake_derive)
    monkeypatch.setattr(script, "real_source_config", lambda _cfg: _source())
    monkeypatch.setattr(script, "compose", lambda **_k: OmegaConf.create({}))
    out = tmp_path / "out" / "v1.yaml"
    assert script.main(["--out", str(out)]) == 0
    assert seen["adc"] == str(adc)
    assert str(adc) not in capsys.readouterr().out
    assert out.is_file()
