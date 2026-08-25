"""Validation sessions for the berlin-lst-downscaling project."""

import nox

from berlin_lst_downscaling.data.io import exists as exists_uri

nox.options.sessions = ["lint", "typecheck"]


# ── universal ──────────────────────────────────────────────────────────


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", ".", external=True)


# ── shared helpers ────────────────────────────────────────────────────


def _write_smoke_bundle(manifest_path: str) -> None:
    """Write the full 8-row smoke bundle (1 Landsat + 1 S2 + 6 ECOSTRESS).

    Uses the project's own :func:`write_bundle` so the smoke fixture
    satisfies the same three-artifact manifest contract as production
    (manifest.parquet + pairings.parquet + manifest_report.json).
    """
    import os
    from datetime import UTC, datetime

    from berlin_lst_downscaling.data.acquisition.ecostress import parse_granule_datetime
    from berlin_lst_downscaling.data.selection.manifest import write_bundle
    from berlin_lst_downscaling.data.selection.schema import ECOSTRESS_VALIDATION_IDS

    bundle_dir = os.path.dirname(manifest_path)
    os.makedirs(bundle_dir, exist_ok=True)

    anchor = {
        "scene_id": "LC09_L2SP_193024_20240629_02_T1",
        "year": 2024,
        "datetime": datetime(2024, 6, 29, 10, 20, 0, tzinfo=UTC),
        "item_href": (
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
            "landsat-c2-l2/items/LC09_L2SP_193024_20240629_02_T1"
        ),
        "aoi_clear_px": 5000,
        "aoi_total_px": 10000,
        "aoi_clear_frac": 0.5,
        "cloud_cover": None,
        "sun_azimuth": None,
        "sun_elevation": None,
    }
    s2 = {
        "scene_id": "S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907",
        "year": 2024,
        "datetime": datetime(2024, 6, 29, 11, 20, 0, tzinfo=UTC),
        "item_href": (
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
            "sentinel-2-l2a/items/S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907"
        ),
        "aoi_clear_px": 6000,
        "aoi_total_px": 10000,
        "aoi_clear_frac": 0.6,
        "cloud_cover": None,
    }
    coupled = [
        {
            "anchor": anchor,
            "s2": s2,
            "landsat_clear_px": 5000,
            "joint_clear_px": 4000,
            "joint_clear_frac": 0.8,
            "score": 0.7,
        }
    ]
    eco_granules = []
    for eco_id in sorted(ECOSTRESS_VALIDATION_IDS):
        acq = parse_granule_datetime(eco_id)
        eco_granules.append({"granule_id": eco_id, "year": acq.year, "datetime": acq})

    # Selection-policy fingerprint — used only for bundle metadata, not a gate.
    smoke_cfg = {
        "platforms": ["landsat-8", "landsat-9"],
        "years": [2024],
        "months": [6],
        "bbox": [13.08, 52.34, 13.76, 52.68],
        "landsat": {"collection": "landsat-c2-l2", "anchor": {"min_clear_frac": 0.05}},
        "sentinel2": {
            "collection": "sentinel-2-l2a",
            "min_clear_frac": 0.05,
            "window_days": 3,
            "score": {"lambda": 0.1},
        },
        "cutoff_utc": "2024-07-31T23:59:59Z",
    }
    write_bundle(
        coupled,
        [],
        eco_granules,
        manifest_path=manifest_path,
        pairings_path=f"{bundle_dir}/pairings.parquet",
        report_path=f"{bundle_dir}/manifest_report.json",
        cutoff_utc="2024-07-31T23:59:59Z",
        cfg=smoke_cfg,
    )
    print(f"Smoke bundle written: {bundle_dir}/")


@nox.session(venv_backend="none")
def format(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "format", ".", external=True)


@nox.session(venv_backend="none")
def fix(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", "--fix", ".", external=True)
    session.run("uv", "run", "ruff", "format", ".", external=True)


@nox.session(venv_backend="none")
def typecheck(session: nox.Session) -> None:
    session.run("uv", "run", "pyright", external=True)


# ── manifest-driven smoke test ────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-primary")
def smoke_primary(session: nox.Session) -> None:
    """Run manifest-driven smoke test for all 3 sources locally.

    Builds the 8-row smoke bundle (1 Landsat, 1 S2, 6 ECOSTRESS), runs
    the ARD pipeline twice to confirm ledger idempotency, then asserts
    that the ledger contains eight rows in status ``done`` and runs the
    standalone validator.
    """
    manifest_dir = "data/smoke/primary"
    manifest_path = f"{manifest_dir}/manifest.parquet"
    output_root = f"{manifest_dir}/ard"

    _write_smoke_bundle(manifest_path)

    for _ in range(2):
        # Run the unified ARD pipeline — ECOSTRESS is downloaded+staged
        # automatically by the pipeline's _process_ecostress_todo path.
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_ard.py",
            "--config-name",
            "smoke_primary",
            f"manifest_uri={manifest_path}",
            f"output_root={output_root}",
            external=True,
        )

    print(f"\nSmoke-primary output: {output_root}/ledger.parquet")
    print("Expected: 8 scenes with status=done")

    _assert_ard_done(session, output_root, expected=8)
    session.run(
        "uv",
        "run",
        "python",
        "scripts/validate_ard.py",
        f"--ledger={output_root}/ledger.parquet",
        f"--manifest={manifest_path}",
        external=True,
    )


def _assert_ard_done(session: nox.Session, output_root: str, expected: int) -> None:
    """Assert that *output_root*/ledger.parquet has *expected* ``done`` rows."""
    session.run(
        "uv",
        "run",
        "python",
        "-c",
        (
            "import sys, pyarrow.parquet as pq; "
            f"tbl = pq.read_table('{output_root}/ledger.parquet'); "
            "done = sum(1 for s in tbl.column('status').to_pylist() if s == 'done'); "
            "print(f'ledger done rows: {done}'); "
            "sys.exit(0 if done == " + str(expected) + " else 1)"
        ),
        external=True,
    )


def _assert_dynamic_done(session: nox.Session, output_root: str, expected: int) -> None:
    """Assert that *output_root* has *expected* done scenes across all dynamic sources."""
    session.run(
        "uv",
        "run",
        "python",
        "-c",
        (
            "import sys, pyarrow.parquet as pq; "
            f"tbl = pq.read_table('{output_root.rstrip('/')}/_state/dynamic/ledger.parquet'); "
            "from collections import Counter; "
            "rows = list(zip(tbl.column('source').to_pylist(), tbl.column('status').to_pylist())); "
            "counts = Counter(rows); "
            "for src in ('era5_land', 'shadow_building', 'shadow_vegetation'): "
            "    print(f'{src}: {counts.get((src, \"done\"), 0)}'); "
            "ok = all(counts.get((s, 'done'), 0) >= "
            + str(expected)
            + " for s in ('era5_land', 'shadow_building', 'shadow_vegetation')); "
            "sys.exit(0 if ok else 1)"
        ),
        external=True,
    )


# ── Szenen-Selektion ─────────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-selection-couple")
def smoke_selection_couple(session: nox.Session) -> None:
    """Run Szenen-Selektion coupling for the bounded smoke window.

    Produces a local couple-mode bundle in ``data/manifest_build/v3/smoke``
    — it is never published. The May-2017 window is chosen so the bundle
    contains the fixed Landsat anchor consumed by ``smoke-dynamic``.
    """
    session.run(
        "uv",
        "run",
        "python",
        "scripts/build_manifest.py",
        "output_root=data/manifest_build/v3/smoke",
        "checkpoint_dir=data/manifest_build/v3/smoke/checkpoints",
        "years=[2017]",
        "months=[5]",
        "cutoff_utc=2024-07-31T23:59:59Z",
        external=True,
    )


# ── Secondary-data pipeline ──────────────────────────────────────────


def _verify_local_artifacts(
    session: nox.Session,
    output_root: str,
    required_suffixes: tuple[str, ...],
) -> None:
    """Check that every required artifact is present under a local root."""
    session.run(
        "uv",
        "run",
        "python",
        "-c",
        f"""import sys
from pathlib import Path
required_suffixes = {required_suffixes!r}
root = Path({output_root!r})
if not root.exists():
    print(f'Missing output root: {{root}}')
    sys.exit(1)
all_paths = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
missing = []
for s in required_suffixes:
    if s == 'ledger.parquet':
        if not any(p == s for p in all_paths):
            missing.append(s)
    elif s == 'report.json':
        if not any(p.endswith('/report.json') for p in all_paths):
            missing.append(s)
    elif not any(p.endswith(s) for p in all_paths):
        missing.append(s)
print(f'Artifacts under {{root}}:')
for p in sorted(all_paths):
    print(f'  {{p}}')
if missing:
    print(f'Missing required artifacts: {{missing}}')
    sys.exit(1)
print('All required artifacts present.')
""",
        external=True,
    )


def _preflight_gcs(session: nox.Session) -> None:
    """Confirm ADC + the bucket are reachable before a cloud run."""
    session.run(
        "uv",
        "run",
        "python",
        "-c",
        (
            "from google.cloud import storage; "
            "client = storage.Client(); "
            "bucket = client.get_bucket('berlin-lst-data'); "
            "print('Bucket reachable:', bucket.name)"
        ),
        external=True,
    )


def _delete_gcs_prefix(prefix: str) -> bool:
    """Delete every blob under a GCS prefix; return True on full cleanup.

    A cleanup failure is reported by the caller — smoke prefixes are
    ephemeral and a leftover prefix must fail the session so the operator
    notices and removes it. The discovery and deletion are wrapped
    together so a listing failure is reported the same way as a deletion
    failure.
    """
    from google.api_core.exceptions import GoogleAPIError
    from google.cloud import storage

    client = storage.Client()
    bucket = client.get_bucket("berlin-lst-data")
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            print(f"  No blobs to clean under {prefix}")
            return True
        bucket.delete_blobs(blobs)
        print(f"  Removed {len(blobs)} blobs under {prefix}")
        return True
    except GoogleAPIError as exc:
        print(f"  ERROR: smoke cleanup failed under {prefix}: {exc}")
        return False


def _list_gcs_subdirs(prefix: str) -> list[str]:
    """Return the immediate child names under a GCS prefix (run dirs)."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.get_bucket("berlin-lst-data")
    it = bucket.list_blobs(prefix=prefix, delimiter="/")
    list(it)  # consume the iterator so .prefixes is populated
    return sorted(str(p).rstrip("/").split("/")[-1] for p in it.prefixes)


def _verify_gcs_artifacts(
    session: nox.Session,
    run_id: str,
    required_suffixes: tuple[str, ...],
    prefix: str = "secondary/smoke/{run_id}/",
) -> None:
    """Check that every required blob exists under a GCS run prefix."""
    prefix = prefix.format(run_id=run_id)
    session.run(
        "uv",
        "run",
        "python",
        "-c",
        f"""import sys
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('berlin-lst-data')
prefix = '{prefix}'
blobs = list(bucket.list_blobs(prefix=prefix))
print(f'Outputs in gs://berlin-lst-data/{{prefix}}')
print(f'  {{len(blobs)}} blob(s)')
for b in blobs:
    print(f'  {{b.name}} ({{b.size}} bytes)')
required_suffixes = {required_suffixes!r}
names = [b.name for b in blobs]
missing = []
for s in required_suffixes:
    if s == 'ledger.parquet':
        if not any(n.endswith('ledger.parquet') for n in names):
            missing.append(s)
    elif s == 'report.json':
        if not any(n.endswith('report.json') for n in names):
            missing.append(s)
    elif not any(n.endswith(s) for n in names):
        missing.append(s)
if missing:
    print(f'Missing required blobs: {{missing}}')
    sys.exit(1)
""",
        external=True,
    )


# ── static source pipeline ──────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-static-sources")
def smoke_static_sources(session: nox.Session) -> None:
    """Run Pipeline A locally with real data on a small aligned subset.

    Downloads all 4 source products (imperviousness, VH, DGM, LoD2) for
    a 2×2 km representative extent, writes final products, validates.
    Runs twice to confirm idempotency.
    """
    output_root = "data/static/sources/smoke"

    for _ in range(2):
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_static_sources.py",
            "--config-name",
            "smoke",
            f"source_root={output_root}",
            external=True,
        )

    _verify_local_artifacts(
        session,
        output_root,
        required_suffixes=(
            # imperviousness
            "ard/static/sources/imperviousness/2016/imperviousness_2016.tif",
            "ard/static/sources/imperviousness/2016/complete.json",
            "ard/static/sources/imperviousness/2021/imperviousness_2021.tif",
            "ard/static/sources/imperviousness/2021/complete.json",
            # vegetation height
            "ard/static/sources/vegetation_height/2020/vegetation_height_2020.tif",
            "ard/static/sources/vegetation_height/2020/complete.json",
            # terrain height
            "ard/static/sources/terrain_height/2021/terrain_height_2021.tif",
            "ard/static/sources/terrain_height/2021/complete.json",
            # LoD2 morphology
            "ard/static/sources/lod2_morphology/2024/lod2_morphology_2024.tif",
            "ard/static/sources/lod2_morphology/2024/complete.json",
            # report + ledger
            "report.json",
            "ledger.parquet",
        ),
    )


@nox.session(venv_backend="none", name="cloud-static-sources")
def cloud_static_sources(session: nox.Session) -> None:
    """Run Pipeline A against GCS with all source products.

    Requires ADC / Workload Identity. Creates a unique run prefix,
    processes all 4 source products, then verifies all artifacts via GCS.
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = f"stat-src-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    source_root = f"gs://berlin-lst-data/static/sources/smoke/{run_id}"

    _preflight_gcs(session)

    for _ in range(2):
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_static_sources.py",
            "--config-name",
            "smoke",
            f"source_root={source_root}",
            external=True,
        )

    _verify_gcs_artifacts(
        session,
        run_id,
        prefix=f"static/sources/smoke/{run_id}/",
        required_suffixes=(
            "ard/static/sources/imperviousness/2016/imperviousness_2016.tif",
            "ard/static/sources/imperviousness/2016/complete.json",
            "ard/static/sources/imperviousness/2021/imperviousness_2021.tif",
            "ard/static/sources/imperviousness/2021/complete.json",
            "ard/static/sources/vegetation_height/2020/vegetation_height_2020.tif",
            "ard/static/sources/vegetation_height/2020/complete.json",
            "ard/static/sources/terrain_height/2021/terrain_height_2021.tif",
            "ard/static/sources/terrain_height/2021/complete.json",
            "ard/static/sources/lod2_morphology/2024/lod2_morphology_2024.tif",
            "ard/static/sources/lod2_morphology/2024/complete.json",
            "report.json",
            "ledger.parquet",
        ),
    )


@nox.session(venv_backend="none", name="smoke-static-derived")
def smoke_static_derived(session: nox.Session) -> None:
    """Run Pipeline B locally against Pipeline-A smoke output.

    Consumes existing local Pipeline-A smoke products and produces
    building/vegetation/combined DSMs, horizons, and SVF.
    Runs twice to confirm idempotency.
    """
    output_root = "data/static/derived/smoke"

    for _ in range(2):
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_static_derived.py",
            "--config-name",
            "smoke",
            f"derived_root={output_root}",
            external=True,
        )

    _verify_local_artifacts(
        session,
        output_root,
        required_suffixes=(
            # Derived products — canonical layout: COG + completion marker
            # per product under <geometry_id>; ledger under _state/.
            "ard/static/derived/building_dsm/dgm1-2021__lod2-2024__vh-2020/"
            "building_dsm_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/building_dsm/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "ard/static/derived/vegetation_dsm/dgm1-2021__lod2-2024__vh-2020/"
            "vegetation_dsm_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/vegetation_dsm/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "ard/static/derived/combined_dsm/dgm1-2021__lod2-2024__vh-2020/"
            "combined_dsm_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/combined_dsm/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "ard/static/derived/horizon_building/dgm1-2021__lod2-2024__vh-2020/"
            "horizon_building_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/horizon_building/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "ard/static/derived/horizon_vegetation/dgm1-2021__lod2-2024__vh-2020/"
            "horizon_vegetation_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/horizon_vegetation/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "ard/static/derived/svf/dgm1-2021__lod2-2024__vh-2020/"
            "svf_dgm1-2021__lod2-2024__vh-2020.tif",
            "ard/static/derived/svf/dgm1-2021__lod2-2024__vh-2020/complete.json",
            "_state/static/derived/ledger.parquet",
            "report.json",
        ),
    )


@nox.session(venv_backend="none", name="cloud-static-derived")
def cloud_static_derived(session: nox.Session) -> None:
    """Run Pipeline B (derived geometry) against GCS.

    Consumes finalized Pipeline A source products and produces
    building/vegetation/combined DSMs, horizons, and SVF.
    Requires ADC / Workload Identity.

    Usage:
        uv run nox -s cloud-static-derived -- \
            gs://berlin-lst-data/static/sources/smoke/...
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    source_root = (
        session.posargs[0] if session.posargs else "gs://berlin-lst-data/static/sources/full"
    )

    run_id = f"stat-drv-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    derived_root = f"gs://berlin-lst-data/static/derived/smoke/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv",
        "run",
        "python",
        "scripts/run_static_derived.py",
        "--config-name",
        "smoke",
        f"source_root={source_root}",
        f"derived_root={derived_root}",
        external=True,
    )

    _verify_gcs_artifacts(
        session,
        run_id,
        prefix=f"static/derived/smoke/{run_id}/",
        required_suffixes=(
            "building_dsm",
            "vegetation_dsm",
            "combined_dsm",
            "horizon_building",
            "horizon_vegetation",
            "svf",
            "report.json",
            "ledger.parquet",
        ),
    )


# ── Dynamic scene pipeline ──────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-dynamic")
def smoke_dynamic(session: nox.Session) -> None:
    """Run dynamic pipeline smoke test locally.

    Processes a single fixed Landsat anchor (see configs/dynamic/smoke.yaml),
    runs the pipeline twice to confirm ledger idempotency, asserts all
    three dynamic sources reach status ``done``, and then runs the
    standalone validator.

    Requires:
    - Local static smoke products (run smoke-static-sources + smoke-static-derived first)
    - CDS API access (~/.cdsapirc or CDS_API_KEY env)
    - Manifest bundle that contains the smoke scene_id

    Usage:
        uv run nox -s smoke-dynamic -- \
            data/manifest_build/v3/smoke/manifest.parquet
    """
    manifest_uri = session.posargs[0] if session.posargs else ""
    output_root = "data/dynamic/smoke"

    for _ in range(2):
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_dynamic.py",
            "--config-name",
            "smoke",
            f"manifest_uri={manifest_uri}",
            f"output_root={output_root}",
            external=True,
        )

    _assert_dynamic_done(session, output_root, expected=1)
    session.run(
        "uv",
        "run",
        "python",
        "scripts/validate_dynamic.py",
        f"--output-root={output_root}",
        "--expected-role",
        "anchor",
        "--expected-scenes",
        "1",
        external=True,
    )


@nox.session(venv_backend="none", name="cloud-smoke-dynamic")
def cloud_smoke_dynamic(session: nox.Session) -> None:
    """Run a deterministic 1-scene dynamic smoke test against GCS.

    Uses cloud_smoke.yaml config with a fixed scene ID.
    Output goes to gs://berlin-lst-data/dynamic/smoke/<run_id>/.

    Requires:
    - ADC / Workload Identity
    - Published v3 manifest
    - Published static source + derived products
    - CDS API access for ERA5-Land download

    Usage:
        uv run nox -s cloud-smoke-dynamic -- \
            gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    manifest_uri = session.posargs[0] if session.posargs else ""

    run_id = f"dyn-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output_root = f"gs://berlin-lst-data/dynamic/smoke/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv",
        "run",
        "python",
        "scripts/run_dynamic.py",
        "--config-name",
        "cloud_smoke",
        f"manifest_uri={manifest_uri}",
        f"output_root={output_root}",
        external=True,
    )


@nox.session(venv_backend="none", name="cloud-dynamic")
def cloud_dynamic(session: nox.Session) -> None:
    """Run dynamic pipeline against GCS (all 324 scenes).

    Requires:
    - ADC / Workload Identity
    - Published v3 manifest
    - Published static source + derived products
    - CDS API access for ERA5-Land download

    Usage:
        uv run nox -s cloud-dynamic -- \
            gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    manifest_uri = session.posargs[0] if session.posargs else ""

    run_id = f"dyn-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output_root = f"gs://berlin-lst-data/dynamic/full/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv",
        "run",
        "python",
        "scripts/run_dynamic.py",
        "--config-name",
        "full",
        f"manifest_uri={manifest_uri}",
        f"output_root={output_root}",
        external=True,
    )


# ── Stage-1 raw-input QA gate ────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-qa-stage1")
def smoke_qa_stage1(session: nox.Session) -> None:
    """Run the Stage-1 raw QA gate on one deterministic pair (real GCS inputs).

    Reads published inputs over GCS (requires ADC), writes only local
    ephemeral output under ``data/smoke/qa-stage1/``, runs the gate twice,
    asserts deterministic aggregate metrics, runs the independent report
    validator on both runs, and removes the local smoke output in
    ``finally`` (never uploaded).
    """
    import glob
    import json
    import os
    import shutil

    session.env.setdefault("UV_ENV_FILE", ".env")

    output_root = "data/smoke/qa-stage1"

    def _run_dirs() -> list[str]:
        if not os.path.isdir(output_root):
            return []
        out = []
        for name in os.listdir(output_root):
            path = os.path.join(output_root, name)
            if os.path.isdir(path) and name != "logs":
                out.append(path)
        return sorted(out, key=lambda p: os.path.getmtime(p))

    try:
        for _ in range(2):
            session.run(
                "uv",
                "run",
                "python",
                "scripts/run_qa_stage1_raw.py",
                "--config-name",
                "stage1_raw_smoke",
                external=True,
            )

        run_dirs = _run_dirs()
        if len(run_dirs) != 2:
            session.error(f"expected 2 smoke run dirs, found {len(run_dirs)}: {run_dirs}")

        for run_dir in run_dirs:
            session.run(
                "uv",
                "run",
                "python",
                "scripts/validate_qa_stage1_raw.py",
                f"--run-prefix={run_dir}",
                external=True,
            )

        # Determinism: aggregate metrics must be identical across runs.
        keys = ("target_valid_cells", "all_100_cells", "full_support_cells")
        summaries = []
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        for key in keys:
            vals = [s["aggregate"][key] for s in summaries]
            if vals[0] != vals[1]:
                session.error(f"non-deterministic aggregate {key}: {vals}")

        # No-mask invariant: no raster artifact may be produced.
        for run_dir in run_dirs:
            masks = glob.glob(os.path.join(run_dir, "*.tif"))
            if masks:
                session.error(f"no-mask invariant violated under {run_dir}: {masks}")

        print(f"smoke-qa-stage1 OK — run dirs: {run_dirs}")
    finally:
        if os.path.isdir(output_root):
            shutil.rmtree(output_root)
            print(f"Removed local smoke output: {output_root}")


# ── Scene feature stacks ─────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-features")
def smoke_features(session: nox.Session) -> None:
    """Run the V3 feature-stack smoke on four deterministic scenes (real GCS).

    One scene per LoD vintage (2017, 2021, 2022, 2024) on a bounded
    canonical-aligned bbox that provably contains every semantic case
    (building, covered no-building, source gap, S2-clear, S2-flagged).
    Reads published inputs over GCS (requires ADC) and the local Berlin
    AOI mask; writes only local ephemeral output under
    ``data/smoke/features/``. Runs the pipeline twice, asserts
    deterministic aggregate metrics across both run reports, then runs
    the independent validators — feature stacks, LoD coverage, V2→V3
    release comparison — and the Stage-2 gate against the freshly
    produced stacks. Removes all local smoke output in ``finally``
    (never uploaded).
    """
    import glob
    import json
    import os
    import shutil

    import pyarrow.parquet as pq

    session.env.setdefault("UV_ENV_FILE", ".env")

    output_root = "data/smoke/features"
    stage2_root = "data/smoke/qa-stage2-v3"
    bbox = "13.4602832,52.4041766,13.6691668,52.4965573"
    scene_ids = [
        "LC08_L2SP_193023_20170720_02_T1",  # 2017
        "LC08_L2SP_192024_20210910_02_T1",  # 2021
        "LC09_L2SP_193023_20220624_02_T1",  # 2022
        "LC09_L2SP_193023_20240629_02_T1",  # 2024
    ]

    def _run_dirs() -> list[str]:
        qa_root = os.path.join(output_root, "qa", "features")
        if not os.path.isdir(qa_root):
            return []
        out = []
        for name in os.listdir(qa_root):
            path = os.path.join(qa_root, name)
            if os.path.isdir(path) and name != "logs":
                out.append(path)
        return sorted(out, key=lambda p: os.path.getmtime(p))

    try:
        for _ in range(2):
            session.run(
                "uv",
                "run",
                "python",
                "scripts/run_features.py",
                "--config-name",
                "smoke",
                external=True,
            )

        run_dirs = _run_dirs()
        if len(run_dirs) != 2:
            session.error(f"expected 2 smoke run dirs, found {len(run_dirs)}: {run_dirs}")

        # Determinism: run-level aggregates must be identical across runs.
        keys = ("feature_valid_px", "inside_aoi_px", "outside_aoi_px")
        summaries = []
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        for key in keys:
            vals = [s["aggregate_coverage"][key] for s in summaries]
            if vals[0] != vals[1]:
                session.error(f"non-deterministic aggregate {key}: {vals}")
        counts = [s["scenes"] for s in summaries]
        for field in ("processed", "failed", "excluded"):
            vals = [c[field] for c in counts]
            if vals[0] != vals[1]:
                session.error(f"non-deterministic scene count {field}: {vals}")

        # Independent feature-stack validator over the published products.
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_feature_stacks.py",
            f"--root={output_root}",
            "--expected-scenes",
            "4",
            external=True,
        )

        # Independent LoD source-coverage validator on the same window.
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_lod_coverage.py",
            f"--bbox={bbox}",
            external=True,
        )

        # V2 → V3 comparison: V2-valid pixels unchanged, newly valid
        # pixels have zero LoD bands. Runs only while a baseline release
        # exists (the retired features/v2 root is gone).
        baseline_ledger = "gs://berlin-lst-data/features/v2/_state/features/ledger.parquet"
        if exists_uri(baseline_ledger):
            session.run(
                "uv",
                "run",
                "python",
                "scripts/compare_feature_releases.py",
                "--baseline-root",
                "gs://berlin-lst-data/features/v2",
                "--candidate-root",
                output_root,
                "--scene-ids",
                ",".join(scene_ids),
                external=True,
            )
        else:
            print("Baseline release absent (retired) — skipping V2→V3 comparison.")

        # Exactly four published stacks (one scene dir per vintage).
        scene_dirs = glob.glob(os.path.join(output_root, "LC08*")) + glob.glob(
            os.path.join(output_root, "LC09*")
        )
        if len(scene_dirs) != 4:
            session.error(f"expected 4 feature-stack scene dirs, found {len(scene_dirs)}")

        # Stage-2 gate against the freshly produced local V3 stacks.
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_qa_stage2_features.py",
            "--config-name",
            "stage2_features_smoke",
            f"features_root={output_root}",
            f"output_root={stage2_root}",
            external=True,
        )
        stage2_dirs = [
            os.path.join(stage2_root, name)
            for name in os.listdir(stage2_root)
            if os.path.isdir(os.path.join(stage2_root, name)) and name != "logs"
        ]
        if len(stage2_dirs) != 1:
            session.error(f"expected 1 Stage-2 smoke run dir, found {len(stage2_dirs)}")
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_qa_stage2_features.py",
            f"--run-prefix={stage2_dirs[0]}",
            external=True,
        )

        # Profile totals: 4 assessed scenes × 28 channels.
        table = pq.read_table(os.path.join(stage2_dirs[0], "profiles.parquet"))
        if table.num_rows != 112:
            session.error(f"expected 112 profile rows, found {table.num_rows}")

        print(f"smoke-features OK — run dirs: {run_dirs}, Stage-2: {stage2_dirs[0]}")
    finally:
        for path in (output_root, stage2_root):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed local smoke output: {path}")


@nox.session(venv_backend="none", name="cloud-smoke-features")
def cloud_smoke_features(session: nox.Session) -> None:
    """Publish the four-vintage V3 smoke stacks to a unique GCS prefix.

    Runs the feature pipeline against real GCS inputs with the output
    rooted at a unique ``gs://berlin-lst-data/features/smoke/<run-id>/``
    prefix, then validates the published stacks with the independent
    validator, the LoD coverage validator, the V2→V3 comparison, and a
    bounded Stage-2 gate. Exercises the exact GCS/GDAL runtime path: data
    COG write, mask COG write into the same folder, and the same-process
    mask read (the GDAL directory-cache failure this gate guards against).

    The smoke prefixes are deleted in ``finally`` on success and failure;
    a leftover prefix fails the session.
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = f"feat-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output_root = f"gs://berlin-lst-data/features/smoke/{run_id}"
    prefix = f"features/smoke/{run_id}/"
    stage2_prefix = f"qa/smoke/stage2-v3/{run_id}/"
    stage2_root = f"gs://berlin-lst-data/{stage2_prefix.rstrip('/')}"
    bbox = "13.4602832,52.4041766,13.6691668,52.4965573"
    scene_ids = [
        "LC08_L2SP_193023_20170720_02_T1",  # 2017
        "LC08_L2SP_192024_20210910_02_T1",  # 2021
        "LC09_L2SP_193023_20220624_02_T1",  # 2022
        "LC09_L2SP_193023_20240629_02_T1",  # 2024
    ]

    _preflight_gcs(session)

    try:
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_features.py",
            "--config-name",
            "smoke",
            f"output_root={output_root}",
            external=True,
        )
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_feature_stacks.py",
            f"--root={output_root}",
            "--expected-scenes",
            "4",
            external=True,
        )
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_lod_coverage.py",
            f"--bbox={bbox}",
            external=True,
        )
        baseline_ledger = "gs://berlin-lst-data/features/v2/_state/features/ledger.parquet"
        if exists_uri(baseline_ledger):
            session.run(
                "uv",
                "run",
                "python",
                "scripts/compare_feature_releases.py",
                "--baseline-root",
                "gs://berlin-lst-data/features/v2",
                "--candidate-root",
                output_root,
                "--scene-ids",
                ",".join(scene_ids),
                external=True,
            )
        else:
            print("Baseline release absent (retired) — skipping V2→V3 comparison.")
        session.run(
            "uv",
            "run",
            "python",
            "scripts/run_qa_stage2_features.py",
            "--config-name",
            "stage2_features_smoke",
            f"features_root={output_root}",
            f"output_root={stage2_root}",
            external=True,
        )
        stage2_dirs = [d for d in _list_gcs_subdirs(stage2_prefix) if d != "logs"]
        if len(stage2_dirs) != 1:
            session.error(f"expected 1 Stage-2 smoke run dir, found {stage2_dirs}")
        session.run(
            "uv",
            "run",
            "python",
            "scripts/validate_qa_stage2_features.py",
            f"--run-prefix={stage2_root}/{stage2_dirs[0]}",
            external=True,
        )
        print(f"cloud-smoke-features OK — {output_root}")
    finally:
        ok_features = _delete_gcs_prefix(prefix)
        ok_stage2 = _delete_gcs_prefix(stage2_prefix)
        if not (ok_features and ok_stage2):
            session.error(
                f"cloud smoke cleanup failed: features={ok_features} stage2={ok_stage2} — "
                f"remove {prefix} and {stage2_prefix} manually"
            )


# ── Stage-2 feature-stack QA gate ────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-qa-stage2")
def smoke_qa_stage2(session: nox.Session) -> None:
    """Run the Stage-2 feature QA gate on one deterministic pair (real GCS).

    Reads published feature stacks over GCS (requires ADC), writes only
    local ephemeral output under ``data/smoke/qa-stage2/``, runs the gate
    twice, asserts deterministic support and profile metrics, runs the
    independent bundle validator on both runs, and removes the local smoke
    output in ``finally`` (never uploaded).
    """
    import glob
    import json
    import os
    import shutil

    session.env.setdefault("UV_ENV_FILE", ".env")

    output_root = "data/smoke/qa-stage2"

    def _run_dirs() -> list[str]:
        if not os.path.isdir(output_root):
            return []
        out = []
        for name in os.listdir(output_root):
            path = os.path.join(output_root, name)
            if os.path.isdir(path) and name != "logs":
                out.append(path)
        return sorted(out, key=lambda p: os.path.getmtime(p))

    try:
        for _ in range(2):
            session.run(
                "uv",
                "run",
                "python",
                "scripts/run_qa_stage2_features.py",
                "--config-name",
                "stage2_features_smoke",
                external=True,
            )

        run_dirs = _run_dirs()
        if len(run_dirs) != 2:
            session.error(f"expected 2 smoke run dirs, found {len(run_dirs)}: {run_dirs}")

        for run_dir in run_dirs:
            session.run(
                "uv",
                "run",
                "python",
                "scripts/validate_qa_stage2_features.py",
                f"--run-prefix={run_dir}",
                external=True,
            )

        # Determinism: support + profile metrics must be identical across runs.
        keys = (
            "feature_valid_px",
            "target_valid_cells",
            "all_100_cells",
            "full_support_cells",
            "support_histogram",
        )
        summaries = []
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        for key in keys:
            vals = [s["aggregate"][key] for s in summaries]
            if vals[0] != vals[1]:
                session.error(f"non-deterministic aggregate {key}: {vals}")

        # Profile totals: every assessed scene has exactly 28 profile rows
        # (4 smoke scenes → 112).
        for run_dir in run_dirs:
            import pyarrow.parquet as pq

            table = pq.read_table(os.path.join(run_dir, "profiles.parquet"))
            if table.num_rows != 112:
                session.error(f"expected 112 profile rows, found {table.num_rows}")

        # No-raster invariant: no .tif artifact under the run prefix.
        for run_dir in run_dirs:
            masks = glob.glob(os.path.join(run_dir, "*.tif"))
            if masks:
                session.error(f"no-raster invariant violated under {run_dir}: {masks}")

        print(f"smoke-qa-stage2 OK — run dirs: {run_dirs}")
    finally:
        if os.path.isdir(output_root):
            shutil.rmtree(output_root)
            print(f"Removed local smoke output: {output_root}")

