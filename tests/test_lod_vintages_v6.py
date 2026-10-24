"""Historical LoD vintages into the v6 roots (republish step 4b).

Covers task acceptance criteria 2 (runnable path reusing lod_vintages.py
with v6 contracts), 3 (fail-closed preflight for every vintage), 4 (the
4b launcher validates all four vintages), and 5 (runbook wiring), with
offline fakes.
"""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]

STATIC_SOURCES_V6 = "gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
STATIC_DERIVED_V6 = "gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
GEOMETRY_MAPPING = "gs://berlin-lst-training-data/static/geometry_vintages/v1/geometry_mapping.json"


def _import_preflight():
    path = _REPO / "scripts/operators/preflight_republish_v6.py"
    spec = importlib.util.spec_from_file_location("preflight_republish_v6_4b", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_lod_vintages_config():
    base = _REPO / "configs/lod_vintages/v6.yaml"
    return OmegaConf.load(base)


# ── 1. config wiring ────────────────────────────────────────────────


def test_lod_vintages_v6_config_points_at_v6_roots() -> None:
    cfg = _load_lod_vintages_config()
    assert str(cfg["source_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(cfg["derived_root"]).rstrip("/") == STATIC_DERIVED_V6
    assert str(cfg["raw_root"]).rstrip("/") == "gs://berlin-lst-training-data"
    assert [int(v) for v in list(cfg["vintages"])] == [2017, 2021, 2022]
    assert int(cfg["svf_max_radius"]) == 10
    assert int(cfg["svf_n_directions"]) == 16
    # Same derived geometry parameters as the 2024 v6 bundle.
    derived_base = OmegaConf.load(_REPO / "configs/static_derived/_base.yaml")
    derived = OmegaConf.merge(
        derived_base, OmegaConf.load(_REPO / "configs/static_derived/v6.yaml")
    )
    assert int(derived["svf_max_radius"]) == int(cfg["svf_max_radius"])
    assert int(derived["svf_n_directions"]) == int(cfg["svf_n_directions"])
    assert int(derived["horizon_max_radius_m"]) == int(cfg["horizon_max_radius_m"])
    assert str(derived["source_root"]).rstrip("/") == STATIC_SOURCES_V6
    assert str(derived["derived_root"]).rstrip("/") == STATIC_DERIVED_V6


def test_lod_vintages_v6_config_references_no_old_root() -> None:
    text = (_REPO / "configs/lod_vintages/v6.yaml").read_text()
    for old in (
        "static/sources/full",
        "static/derived/full",
        "dynamic/full",
        "static/sources/imperviousness-v3",
    ):
        assert old not in text


# ── 2. expected ledger items (pure) ─────────────────────────────────


def test_expected_items_cover_three_vintages_and_four_products() -> None:
    from berlin_lst_downscaling.data.secondary.lod_vintages_v6 import (
        expected_derived_items,
        expected_source_items,
    )

    cfg = OmegaConf.create({"vintages": [2017, 2021, 2022]})
    assert expected_source_items(cfg) == [
        ("lod2_morphology_2017", "lod2_morphology", "2017"),
        ("lod2_morphology_2021", "lod2_morphology", "2021"),
        ("lod2_morphology_2022", "lod2_morphology", "2022"),
    ]
    derived = expected_derived_items(cfg)
    assert len(derived) == 12
    geometries = {period for _, _, period in derived}
    assert geometries == {
        "dgm1-2021__lod2-2017__vh-2020",
        "dgm1-2021__lod2-2021__vh-2020",
        "dgm1-2021__lod2-2022__vh-2020",
    }
    products = {source for _, source, _ in derived}
    assert products == {"building_dsm", "combined_dsm", "horizon_building", "svf"}


# ── 3. geometry horizon root override ───────────────────────────────


def _fake_mapping_payload() -> bytes:
    year_to_vintage = {
        **{str(y): "2017" for y in range(2017, 2021)},
        "2021": "2021",
        "2022": "2022",
        "2023": "2022",
        **{str(y): "2024" for y in range(2024, 2027)},
    }
    vintages = {
        str(v): {"geometry_id": f"dgm1-2021__lod2-{v}__vh-2020"} for v in (2017, 2021, 2022, 2024)
    }
    return json.dumps(
        {
            "version": "v1",
            "rule": "carry-forward: scene_year → most-recent vintage with year ≤ scene_year",
            "year_to_vintage": year_to_vintage,
            "vintages": vintages,
        }
    ).encode()


def _patch_geometry(monkeypatch: pytest.MonkeyPatch, horizon_root: str):
    from berlin_lst_downscaling.data.dynamic import geometry as geometry_module

    mapping_uri = "mapping.json"
    prov_suffix = "provenance.json"

    def fake_exists(uri: str) -> bool:
        return str(uri).startswith(horizon_root) or str(uri) == mapping_uri

    def fake_read(uri: str) -> bytes:
        if str(uri) == mapping_uri:
            return _fake_mapping_payload()
        if str(uri).endswith(prov_suffix):
            return json.dumps({"config_hash": "veghash"}).encode()
        raise FileNotFoundError(uri)

    monkeypatch.setattr(geometry_module, "exists", fake_exists)
    monkeypatch.setattr(geometry_module, "read_bytes", fake_read)
    return geometry_module, mapping_uri


def test_geometry_defaults_to_legacy_full_root(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry_module, mapping_uri = _patch_geometry(
        monkeypatch, "gs://berlin-lst-training-data/static/derived/full"
    )
    report = geometry_module.load_geometry_mapping(mapping_uri)
    assert report.ok, report.errors
    assert report.mapping is not None
    assert all(
        uri.startswith("gs://berlin-lst-training-data/static/derived/full/")
        for uri in report.mapping.building_horizons.values()
    )
    assert report.mapping.vegetation_horizon_uri.startswith(
        "gs://berlin-lst-training-data/static/derived/full/"
    )


def test_geometry_resolves_horizons_under_v6_derived_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_module, mapping_uri = _patch_geometry(monkeypatch, STATIC_DERIVED_V6)
    report = geometry_module.load_geometry_mapping(mapping_uri, derived_root=STATIC_DERIVED_V6)
    assert report.ok, report.errors
    assert report.mapping is not None
    assert sorted(report.mapping.building_horizons) == [2017, 2021, 2022, 2024]
    for vintage, uri in report.mapping.building_horizons.items():
        geometry_id = f"dgm1-2021__lod2-{vintage}__vh-2020"
        assert uri == (
            f"{STATIC_DERIVED_V6}/ard/static/derived/horizon_building/"
            f"{geometry_id}/horizon_building_{geometry_id}.tif"
        )
    scene = geometry_module.resolve_scene_geometry(2019, report.mapping)
    assert scene.building_vintage == 2017
    assert scene.building_geometry_id == "dgm1-2021__lod2-2017__vh-2020"


def test_geometry_fails_closed_when_v6_horizon_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from berlin_lst_downscaling.data.dynamic import geometry as geometry_module

    mapping_uri = "mapping.json"

    def fake_exists(uri: str) -> bool:
        # Every 2017 horizon artifact is absent under the v6 root.
        if "lod2-2017" in str(uri):
            return False
        return True

    def fake_read(uri: str) -> bytes:
        if str(uri) == mapping_uri:
            return _fake_mapping_payload()
        return json.dumps({"config_hash": "veghash"}).encode()

    monkeypatch.setattr(geometry_module, "exists", fake_exists)
    monkeypatch.setattr(geometry_module, "read_bytes", fake_read)
    report = geometry_module.load_geometry_mapping(mapping_uri, derived_root=STATIC_DERIVED_V6)
    assert not report.ok
    assert any("2017" in err for err in report.errors)


# ── 4. preflight: new step + per-vintage gates ──────────────────────


def _patch_io(monkeypatch: pytest.MonkeyPatch, present: set[str]):
    import berlin_lst_downscaling.data.io as io

    monkeypatch.setattr(io, "exists", lambda uri: str(uri) in present)
    monkeypatch.setattr(io, "read_bytes", lambda uri: (_ for _ in ()).throw(FileNotFoundError(uri)))


def _full_v6_present(preflight) -> set[str]:
    present = {
        f"{STATIC_SOURCES_V6}/ledger.parquet",
        f"{STATIC_DERIVED_V6}/_state/static/derived/ledger.parquet",
        GEOMETRY_MAPPING,
        "gs://berlin-lst-training-data/manifests/v4/"
        "2017-2026-cutoff-20260717T235959Z/manifest.parquet",
        "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10/ledger.parquet",
        "gs://berlin-lst-training-data/dynamic/lod2-area-exact-v2/_state/dynamic/ledger.parquet",
        "gs://berlin-lst-training-data/features/v6/ledger.parquet",
        "gs://berlin-lst-training-data/lod_vintages/2017/LoD1_2017.zip",
        "gs://berlin-lst-training-data/lod_vintages/2021/LoD2_BE_1_33_2021.zip",
        "gs://berlin-lst-training-data/lod_vintages/2022/LoD2_2022.zip",
    }
    present |= {uri for uri, _ in preflight.v6_vintage_horizon_uris()}
    present |= {uri for uri, _ in preflight.v6_vintage_product_uris()}
    return present


def test_preflight_lod_vintages_v6_passes_with_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    _patch_io(monkeypatch, _full_v6_present(preflight))
    cfg = preflight._load_config("configs/lod_vintages", "v6")
    assert preflight._failures_for("lod_vintages_v6", cfg) == []


def test_preflight_lod_vintages_v6_pins_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    _patch_io(monkeypatch, _full_v6_present(preflight))
    cfg = preflight._load_config("configs/lod_vintages", "v6")
    cfg = OmegaConf.merge(cfg, {"svf_max_radius": 3})
    failures = preflight._failures_for("lod_vintages_v6", cfg)
    assert any("svf_max_radius" in f for f in failures)
    cfg = OmegaConf.merge(
        preflight._load_config("configs/lod_vintages", "v6"), {"vintages": [2024]}
    )
    failures = preflight._failures_for("lod_vintages_v6", cfg)
    assert any("vintages" in f for f in failures)


def test_preflight_dynamic_fails_closed_on_missing_vintage_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    present = _full_v6_present(preflight)
    missing = next(uri for uri, _ in preflight.v6_vintage_horizon_uris() if "lod2-2017" in uri)
    present.discard(missing)
    _patch_io(monkeypatch, present)
    cfg = preflight._load_config("configs/dynamic", "v6")
    failures = preflight._failures_for("dynamic_v6", cfg)
    assert any("2017" in f for f in failures), failures


def test_preflight_features_fails_closed_on_missing_raw_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    present = _full_v6_present(preflight)
    present.discard(f"{STATIC_SOURCES_V6}/ard/static/sources/lod_vintages/raw_manifest_2021.json")
    _patch_io(monkeypatch, present)
    cfg = preflight._load_config("configs/features", "v6")
    failures = preflight._failures_for("features_v6", cfg)
    assert any("raw_manifest_2021" in f for f in failures), failures


def test_preflight_training_fails_closed_on_missing_vintage_svf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    present = _full_v6_present(preflight)
    missing = next(uri for uri, _ in preflight.v6_vintage_product_uris() if "svf" in uri)
    present.discard(missing)
    _patch_io(monkeypatch, present)
    cfg = preflight._load_config("configs/training", "v3")
    failures = preflight._failures_for("training_v3", cfg)
    assert any("svf" in f for f in failures), failures


def test_preflight_vintage_gates_pass_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _import_preflight()
    present = _full_v6_present(preflight)
    markers = {
        "gs://berlin-lst-training-data/training/v3/complete.json": json.dumps(
            {"policy_hash": "abc123"}
        ),
    }
    import berlin_lst_downscaling.data.io as io

    monkeypatch.setattr(io, "exists", lambda uri: str(uri) in present)
    monkeypatch.setattr(io, "read_bytes", lambda uri: markers[str(uri)].encode())
    for step, config_dir, config_name in (
        ("lod_vintages_v6", "configs/lod_vintages", "v6"),
        ("dynamic_v6", "configs/dynamic", "v6"),
        ("features_v6", "configs/features", "v6"),
        ("training_v3", "configs/training", "v3"),
    ):
        cfg = preflight._load_config(config_dir, config_name)
        assert preflight._failures_for(step, cfg) == [], step


# ── 5. derived ledger gate ──────────────────────────────────────────


def _load_script(relative: str, name: str):
    import sys

    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(item_id: str, source: str, period: str, status: str, **kw):
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedgerRow

    return SecondaryLedgerRow(
        item_id=item_id,
        source=source,
        period_or_vintage=period,
        status=status,
        run_id="test",
        config_hash="hash",
        output_uri=kw.get("output_uri"),
        completion_uri=kw.get("completion_uri"),
        last_error=kw.get("last_error"),
    )


def test_check_static_derived_ledger_script(tmp_path: Path) -> None:
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
    from berlin_lst_downscaling.data.secondary.paths import derived_ledger_path

    check = _load_script(
        "scripts/operators/check_static_derived_ledger.py", "check_static_derived_ledger"
    )

    root = tmp_path / "derived"
    root.mkdir()
    # Missing ledger -> FAIL.
    assert check.main(["--derived-root", str(root)]) == 1

    led = SecondaryLedger.open(derived_ledger_path(str(root)))
    led.upsert(_row("building_dsm", "building_dsm", "g", "failed", last_error="boom"))
    assert check.main(["--derived-root", str(root)]) == 1

    led3_root = tmp_path / "derived3"
    led3_root.mkdir()
    cog = led3_root / "a.tif"
    comp = led3_root / "complete.json"
    cog.write_bytes(b"x")
    comp.write_bytes(b"{}")
    led3 = SecondaryLedger.open(derived_ledger_path(str(led3_root)))
    led3.upsert(
        _row(
            "building_dsm",
            "building_dsm",
            "g",
            "done",
            output_uri=str(cog),
            completion_uri=str(comp),
        )
    )
    assert check.main(["--derived-root", str(led3_root)]) == 0


# ── 6. runner orchestration (offline fakes) ─────────────────────────


def _runner_config(tmp_path: Path):
    return OmegaConf.create(
        {
            "raw_root": "gs://berlin-lst-training-data",
            "source_root": str(tmp_path / "sources"),
            "derived_root": str(tmp_path / "derived"),
            "vintages": [2017, 2021, 2022],
            "svf_max_radius": 10,
            "svf_n_directions": 16,
            "horizon_max_radius_m": 200,
        }
    )


class _Fakes:
    """Offline doubles for the GCS-bound lod_vintages.py entry points."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.manifests: list[int] = []
        self.morphology_calls: list[int] = []
        self.derive_calls: list[tuple[int, list[str]]] = []
        self.fail_derive: set[int] = set()

    @contextmanager
    def materialize(self, spec, raw_root):
        del raw_root
        yield SimpleNamespace(
            spec=spec,
            sha256="abc123",
            byte_count=10,
            member_count=2,
            member_names=["LoD2_001_0001_x.xml", "LoD2_001_0002_x.xml"],
        )

    def publish_manifest(self, manifest, source_root):
        del source_root
        self.manifests.append(int(manifest.vintage))
        return f"raw_manifest_{manifest.vintage}.json"

    def publish_morphology(
        self,
        vintage,
        source_root,
        run_id,
        *,
        lod1_mat=None,
        lod2_mat=None,
        raw_root=None,
        grid=None,
        smoke_tile_count=None,
    ):
        del source_root, run_id, lod1_mat, lod2_mat, raw_root, grid, smoke_tile_count
        from berlin_lst_downscaling.data.secondary.lod_vintages import MorphologyArtifacts

        self.morphology_calls.append(int(vintage))
        cog = self.tmp_path / f"lod2_morphology_{vintage}.tif"
        comp = self.tmp_path / f"lod2_morphology_{vintage}.complete.json"
        cog.write_bytes(b"x")
        comp.write_bytes(b"{}")
        return (
            MorphologyArtifacts(
                cog_uri=str(cog),
                stac_uri=str(self.tmp_path / f"lod2_morphology_{vintage}.stac.json"),
                provenance_uri=str(self.tmp_path / f"lod2_morphology_{vintage}.prov.json"),
                completion_uri=str(comp),
                archive_uri="archive",
                archive_sha256="abc123",
                archive_byte_count=10,
                raw_manifest_uri="manifest",
            ),
            SimpleNamespace(retained=0, rejected=0),
        )

    def derive(self, vintage, source_root, derived_root, run_id, **kwargs):
        from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
        from berlin_lst_downscaling.data.secondary.lod_vintages import vintage_geometry_id
        from berlin_lst_downscaling.data.secondary.paths import derived_ledger_path

        products = sorted(kwargs.get("products") or [])
        self.derive_calls.append((int(vintage), products))
        assert kwargs.get("svf_max_radius") == 10
        assert kwargs.get("upstream_source_root") == source_root
        assert kwargs.get("upstream_derived_root") == derived_root
        if int(vintage) in self.fail_derive:
            raise RuntimeError(f"boom-{vintage}")
        led = SecondaryLedger.open(derived_ledger_path(derived_root))
        for product in products:
            cog = self.tmp_path / f"{vintage}_{product}.tif"
            comp = self.tmp_path / f"{vintage}_{product}.complete.json"
            cog.write_bytes(b"x")
            comp.write_bytes(b"{}")
            led.upsert(
                _row(
                    product,
                    product,
                    vintage_geometry_id(int(vintage)),
                    "done",
                    output_uri=str(cog),
                    completion_uri=str(comp),
                )
            )
        return {"vintage": vintage, "artifacts": {}}


def _install_fakes(monkeypatch: pytest.MonkeyPatch, fakes: _Fakes):
    from berlin_lst_downscaling.data.secondary import lod_vintages_v6 as core

    monkeypatch.setattr(core, "materialize_vintage_archive", fakes.materialize)
    monkeypatch.setattr(core, "publish_raw_archive_manifest", fakes.publish_manifest)
    monkeypatch.setattr(core, "publish_vintage_morphology", fakes.publish_morphology)
    monkeypatch.setattr(core, "derive_vintage_products", fakes.derive)
    return core


def test_runner_builds_all_vintages_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = _Fakes(tmp_path)
    core = _install_fakes(monkeypatch, fakes)

    assert core.run_lod_vintages_v6(_runner_config(tmp_path), run_id="r1") == 0
    # 2017 publishes both the LoD1-2017 and the LoD2-2021 manifests.
    assert fakes.manifests == [2017, 2021, 2021, 2022]
    assert fakes.morphology_calls == [2017, 2021, 2022]
    assert [v for v, _ in fakes.derive_calls] == [2017, 2021, 2022]
    assert all(
        products == ["building_dsm", "combined_dsm", "horizon_building", "svf"]
        for _, products in fakes.derive_calls
    )
    # Morphology contracts match the 2024 v6 product: same builder, same
    # ranges, and the ledger rows carry the vintage config hash that
    # reconcile compares on rerun.
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
    from berlin_lst_downscaling.data.secondary.lod2 import (
        config_hash_for_vintage,
        contract_for_lod2_morphology,
    )

    assert contract_for_lod2_morphology().output_bands[0].valid_range == (0.0, 400.0)
    src_led = SecondaryLedger.open(str(tmp_path / "sources" / "ledger.parquet"))
    for vintage in (2017, 2021, 2022):
        row = src_led.get(f"lod2_morphology_{vintage}", "lod2_morphology", str(vintage))
        assert row is not None and row.status == "done"
        assert row.config_hash == config_hash_for_vintage(vintage)

    # Rerun is a no-op (ledger reconcile skips done rows).
    n_manifests = len(fakes.manifests)
    n_derive = len(fakes.derive_calls)
    assert core.run_lod_vintages_v6(_runner_config(tmp_path), run_id="r2") == 0
    assert len(fakes.manifests) == n_manifests
    assert fakes.morphology_calls == [2017, 2021, 2022]
    assert len(fakes.derive_calls) == n_derive


def test_runner_fails_closed_on_derive_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = _Fakes(tmp_path)
    fakes.fail_derive = {2022}
    core = _install_fakes(monkeypatch, fakes)

    assert core.run_lod_vintages_v6(_runner_config(tmp_path), run_id="r1") == 1

    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
    from berlin_lst_downscaling.data.secondary.paths import derived_ledger_path

    led = SecondaryLedger.open(derived_ledger_path(str(tmp_path / "derived")))
    row = led.get("svf", "svf", "dgm1-2021__lod2-2022__vh-2020")
    assert row is not None and row.status == "failed"


# ── 7. launcher + runbook wiring ────────────────────────────────────


def test_historical_launcher_follows_vm_runner_pattern() -> None:
    text = (_REPO / ".opencode/skills/google-access/scripts/run-lod-vintages-v6-vm.sh").read_text()
    assert 'source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"' in text
    assert "set -euo pipefail" in text
    assert "--step lod_vintages_v6" in text
    assert "--config-name ${CONFIG_NAME}" in text
    assert "check_static_sources_ledger.py" in text
    assert "check_static_derived_ledger.py" in text
    assert "--vintages 2017,2021,2022,2024" in text
    assert "POLL_MAX_SECONDS=172800" in text
    # No hard-coded run id: the wrapper mints WRAP_RUN_ID per launch.
    assert "WRAP_RUN_ID" in text
    assert "RUN_ID=" not in text.replace("WRAP_RUN_ID", "")


def test_republish_runbook_lists_historical_step() -> None:
    text = (_REPO / "docs/republish-v6.md").read_text()
    assert "run-lod-vintages-v6-vm.sh" in text
    assert "configs/lod_vintages/v6.yaml" in text
    assert "2017,2021,2022,2024" in text
