"""Validation sessions for the berlin-lst-downscaling project."""

import nox

nox.options.sessions = ["lint", "typecheck"]


# ── universal ──────────────────────────────────────────────────────────


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", ".", external=True)


# ── shared helpers ────────────────────────────────────────────────────


_SMOKE_ROWS = [
    {
        "scene_id": "LC09_L2SP_193024_20240629_02_T1",
        "source": "landsat-c2-l2",
        "role": "anchor",
        "platform": "landsat-9",
        "year": 2024,
        "item_href": "https://planetarycomputer.microsoft.com/api/stac/data/landsat-c2-l2/items/LC09_L2SP_193024_20240629_02_T1",
        "aoi_clear_px": 5000,
        "aoi_total_px": 10000,
        "aoi_clear_frac": 0.5,
    },
    {
        "scene_id": "S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907",
        "source": "sentinel-2-l2a",
        "role": "predictor",
        "platform": "sentinel-2",
        "year": 2024,
        "item_href": "https://planetarycomputer.microsoft.com/api/stac/data/sentinel-2-l2a/items/S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907",
        "aoi_clear_px": 6000,
        "aoi_total_px": 10000,
        "aoi_clear_frac": 0.6,
    },
    {
        "scene_id": "ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01",
        "source": "ecostress",
        "role": "validation",
        "platform": "ecostress",
        "year": 2018,
        "item_href": None,
        "aoi_clear_px": None,
        "aoi_total_px": None,
        "aoi_clear_frac": None,
    },
]

_SMOKE_PAIRINGS = [
    {
        "landsat_scene_id": "LC09_L2SP_193024_20240629_02_T1",
        "sentinel2_scene_id": "S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907",
        "dt_seconds": 3600,
        "landsat_clear_px": 5000,
        "joint_clear_px": 4000,
        "joint_clear_frac": 0.8,
        "score": 0.7,
    },
]


def _write_smoke_manifest(manifest_path: str) -> None:
    """Write the 3-row v3 smoke manifest to *manifest_path* (Parquet)."""
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    from berlin_lst_downscaling.data.selection.schema import MANIFEST_SCHEMA

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # Add missing datetime field (required by schema)
    from datetime import UTC, datetime

    for row in _SMOKE_ROWS:
        if "acquisition_datetime" not in row:
            row["acquisition_datetime"] = datetime(2024, 6, 29, 10, 20, 0, tzinfo=UTC)
        if "cloud_cover" not in row:
            row["cloud_cover"] = None
        if "solar_azimuth" not in row:
            row["solar_azimuth"] = None
        if "solar_elevation" not in row:
            row["solar_elevation"] = None

    table = pa.Table.from_pylist(_SMOKE_ROWS, schema=MANIFEST_SCHEMA)
    pq.write_table(table, manifest_path)
    print(f"Manifest written: {manifest_path}")


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

    Builds a 3-row manifest (1 Landsat, 1 S2, 1 ECOSTRESS), then runs the
    ARD pipeline.  The pipeline downloads + stages ECOSTRESS from CMR
    automatically via ``_process_ecostress_todo``.

    Final COGs land in ``data/smoke/primary/ard/``.
    """
    manifest_dir = "data/smoke/primary"
    manifest_path = f"{manifest_dir}/manifest.parquet"
    output_root = f"{manifest_dir}/ard"

    _write_smoke_manifest(manifest_path)

    # Run the unified ARD pipeline — ECOSTRESS is downloaded+staged
    # automatically by the pipeline's _process_ecostress_todo path.
    session.run(
        "uv", "run", "python", "scripts/run_ard.py",
        "--config-name", "smoke_primary",
        f"manifest_uri={manifest_path}",
        f"output_root={output_root}",
        "+ecostress.persist_stage=true",  # keep stage for inspection
        external=True,
    )

    print(f"\nSmoke-primary output: {output_root}/ledger.parquet")
    print("Expected: 3 scenes with status=done")


# ── Szenen-Selektion ─────────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-selection-2024")
def smoke_selection_2024(session: nox.Session) -> None:
    """Run Szenen-Selektion coupling on Mai–Sep 2024.

    Validates the coupling logic across the full configured season.
    Uses SCL-based cloud detection (the only method).
    Writes ``data/smoke/manifest_2024.parquet``.
    """
    session.run(
        "uv", "run", "python", "scripts/build_manifest.py",
        "--config-dir", "configs/selection",
        "--config-name", "smoke_2024_mai_sep",
        external=True,
    )


@nox.session(venv_backend="none", name="selection-scan")
def selection_scan(session: nox.Session) -> None:
    """Run full metadata-only volume scan (2017–2025, Mai–Sep).

    Writes ``data/ard/scan_report.{json,md}`` with counts and GB estimates.
    No pixel loads — PC STAC + CMR metadata only.
    """
    session.run(
        "uv", "run", "python", "scripts/build_manifest.py",
        "--config-dir", "configs/selection",
        "--config-name", "full_2017_2025",
        external=True,
    )


# ── cloud pilot ──────────────────────────────────────────────────────


@nox.session(venv_backend="none", name="cloud-pilot")
def cloud_pilot(session: nox.Session) -> None:
    """Run smoke-primary targeting GCS (requires ADC / Workload Identity).

    Requires ``GOOGLE_APPLICATION_CREDENTIALS`` to be set, or runs under
    a GCP Workload Identity in Cloud Run.
    """
    import uuid
    from datetime import UTC, datetime

    # uv ≥0.11 requires explicit opt-in to auto-load .env via UV_ENV_FILE.
    # Set it here so the cloud-pilot works on both uv 0.7 (auto) and ≥0.11 (opt-in).
    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = f"cp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    stage_base = "gs://berlin-lst-data/_staging/ecostress"
    eco_stage = f"{stage_base}/{run_id}"
    manifest_path = f"data/smoke/cloud_pilot_{run_id}/manifest.parquet"
    output_root = "gs://berlin-lst-data/ard/smoke"

    # ── pre-flight: GCS reachable ─────────────────────────────────────
    session.run(
        "uv", "run", "python", "-c",
        (
            "from google.cloud import storage; "
            "client = storage.Client(); "
            "bucket = client.get_bucket('berlin-lst-data'); "
            "print('Bucket reachable:', bucket.name)"
        ),
        external=True,
    )

    # Step 1: Build 3-row smoke manifest
    _write_smoke_manifest(manifest_path)

    # Step 2: Stage ECOSTRESS fixture to GCS
    session.run(
        "uv", "run", "python", "scripts/download_ecostress_fixture.py",
        "--tile", "33UUU",
        "--date", "2018-07-30",
        "--stage-dir", stage_base,
        "--run-id", run_id,
        external=True,
    )

    # Step 3: Run the unified ARD pipeline (reads ECOSTRESS from GCS stage,
    # AOI from GCS — exercises the full cloud-read path)
    session.run(
        "uv", "run", "python", "scripts/run_ard.py",
        "--config-name", "smoke_primary",
        f"manifest_uri={manifest_path}",
        f"output_root={output_root}/smoke_primary",
        f"ecostress.raw_dir={eco_stage}",
        "+ecostress.persist_stage=true",
        "aoi.mask_base=gs://berlin-lst-data/boundaries",
        external=True,
    )

    # Step 4: Clean up ECOSTRESS GCS stage
    session.run(
        "uv", "run", "python", "-c",
        f"""
import sys; sys.path.insert(0, 'src')
from berlin_lst_downscaling.data.io.staging import StageSession
with StageSession('{stage_base}', run_id='{run_id}', persist=False) as stage:
    print(f'Stage cleaned up: {{stage.uri}}')
""",
        external=True,
    )

    # Step 5: Verify final COGs landed in GCS
    session.run(
        "uv", "run", "python", "-c",
        (
            "import sys\n"
            "from google.cloud import storage\n"
            "client = storage.Client()\n"
            "bucket = client.get_bucket('berlin-lst-data')\n"
            "prefix = 'ard/smoke/smoke_primary/'\n"
            "blobs = list(bucket.list_blobs(prefix=prefix))\n"
            "print(f'Final COGs: {len(blobs)} blob(s) under gs://berlin-lst-data/{prefix}')\n"
            "for b in blobs[:6]:\n"
            "    print(' ', b.name)\n"
            "sys.exit(0 if blobs else 1)\n"
        ),
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
        "uv", "run", "python", "-c",
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
        "uv", "run", "python", "-c",
        (
            "from google.cloud import storage; "
            "client = storage.Client(); "
            "bucket = client.get_bucket('berlin-lst-data'); "
            "print('Bucket reachable:', bucket.name)"
        ),
        external=True,
    )


def _verify_gcs_artifacts(
    session: nox.Session,
    run_id: str,
    required_suffixes: tuple[str, ...],
    prefix: str = "secondary/smoke/{run_id}/",
) -> None:
    """Check that every required blob exists under a GCS run prefix."""
    prefix = prefix.format(run_id=run_id)
    session.run(
        "uv", "run", "python", "-c",
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
            "uv", "run", "python", "scripts/run_static_sources.py",
            "--config-name", "smoke",
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

    run_id = (
        f"stat-src-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    source_root = f"gs://berlin-lst-data/static/sources/smoke/{run_id}"

    _preflight_gcs(session)

    for _ in range(2):
        session.run(
            "uv", "run", "python", "scripts/run_static_sources.py",
            "--config-name", "smoke",
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
            "uv", "run", "python", "scripts/run_static_derived.py",
            "--config-name", "smoke",
            f"derived_root={output_root}",
            external=True,
        )

    _verify_local_artifacts(
        session,
        output_root,
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

    source_root = session.posargs[0] if session.posargs else "gs://berlin-lst-data/static/sources/full"

    run_id = (
        f"stat-drv-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    derived_root = f"gs://berlin-lst-data/static/derived/smoke/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv", "run", "python", "scripts/run_static_derived.py",
        "--config-name", "smoke",
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

    Requires:
    - Local static smoke products (run smoke-static-sources + smoke-static-derived first)
    - CDS API access (~/.cdsapirc or CDS_API_KEY env)

    Usage:
        uv run nox -s smoke-dynamic -- \
            data/ard/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet
    """
    manifest_uri = session.posargs[0] if session.posargs else ""

    session.run(
        "uv", "run", "python", "scripts/run_dynamic.py",
        "--config-name", "smoke",
        f"manifest_uri={manifest_uri}",
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
            gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    manifest_uri = session.posargs[0] if session.posargs else ""

    run_id = (
        f"dyn-smoke-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = f"gs://berlin-lst-data/dynamic/smoke/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv", "run", "python", "scripts/run_dynamic.py",
        "--config-name", "cloud_smoke",
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
            gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z/manifest.parquet
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    manifest_uri = session.posargs[0] if session.posargs else ""

    run_id = (
        f"dyn-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = f"gs://berlin-lst-data/dynamic/full/{run_id}"

    _preflight_gcs(session)

    session.run(
        "uv", "run", "python", "scripts/run_dynamic.py",
        "--config-name", "full",
        f"manifest_uri={manifest_uri}",
        f"output_root={output_root}",
        external=True,
    )
