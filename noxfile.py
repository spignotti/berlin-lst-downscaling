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
        "year": 2024,
        "status": "coupled",
        "date": "2024-06-29",
    },
    {
        "scene_id": "S2A_MSIL2A_20240629T102021_R065_T33UVU_20240629T161907",
        "source": "sentinel-2-l2a",
        "year": 2024,
        "status": "coupled",
        "date": "2024-06-29",
    },
    {
        "scene_id": "ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01",
        "source": "ecostress",
        "year": 2018,
        "status": "coupled",
        "date": "2018-07-30",
    },
]


def _write_smoke_manifest(manifest_path: str) -> None:
    """Write the 3-row smoke manifest to *manifest_path* (Parquet)."""
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    schema = pa.schema([
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("date", pa.string(), nullable=True),
    ])
    table = pa.Table.from_pylist(_SMOKE_ROWS, schema=schema)
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


@nox.session(venv_backend="none", name="smoke-secondary-all")
def smoke_secondary_all(session: nox.Session) -> None:
    """Run source-registered fixtures locally.

    Finalises a synthetic product for every registered source (imperviousness,
    vegetation_height), then runs again to confirm idempotency.  Validates
    the full contract: COG + STAC + provenance + completion marker +
    ledger + QA report.
    """
    output_root = "data/smoke/secondary/all"

    # First run — finalise fixtures
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_all",
        f"output_root={output_root}",
        external=True,
    )

    # Second run — must be idempotent
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_all",
        f"output_root={output_root}",
        external=True,
    )

    # Verify artifacts
    session.run(
        "uv", "run", "python", "-c",
        f"""import sys
from pathlib import Path
required_suffixes = (
    'imperviousness_2021.tif',
    'imperviousness_2021.stac.json',
    'imperviousness/2021/provenance.json',
    'imperviousness/2021/complete.json',
    'vegetation_height_2020.tif',
    'vegetation_height_2020.stac.json',
    'vegetation_height/2020/provenance.json',
    'vegetation_height/2020/complete.json',
    'report.json',
    'ledger.parquet',
)
root = Path('{output_root}')
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


@nox.session(venv_backend="none", name="cloud-secondary-all")
def cloud_secondary_all(session: nox.Session) -> None:
    """Run every configured secondary source against GCS.

    Requires ADC / Workload Identity (``GOOGLE_APPLICATION_CREDENTIALS``).
    Creates a unique run prefix on GCS, finalises imperviousness 2016/2021
    and vegetation height 2020, then verifies all four artifacts per
    product plus the aggregate QA report.
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = (
        f"sec-all-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = f"gs://berlin-lst-data/secondary/smoke/{run_id}"

    # Pre-flight: GCS reachable
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

    # First run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "full_all",
        f"output_root={output_root}",
        external=True,
    )

    # Idempotency run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "full_all",
        f"output_root={output_root}",
        external=True,
    )

    # Verify outputs
    session.run(
        "uv", "run", "python", "-c",
        f"""import sys
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('berlin-lst-data')
prefix = 'secondary/smoke/{run_id}/'
blobs = list(bucket.list_blobs(prefix=prefix))
print(f'Outputs in gs://berlin-lst-data/{{prefix}}')
print(f'  {{len(blobs)}} blob(s)')
for b in blobs:
    print(f'  {{b.name}} ({{b.size}} bytes)')
required_suffixes = (
    'imperviousness_2016.tif',
    'imperviousness_2016.stac.json',
    'imperviousness/2016/provenance.json',
    'imperviousness/2016/complete.json',
    'imperviousness_2021.tif',
    'imperviousness_2021.stac.json',
    'imperviousness/2021/provenance.json',
    'imperviousness/2021/complete.json',
    'vegetation_height_2020.tif',
    'vegetation_height_2020.stac.json',
    'vegetation_height/2020/provenance.json',
    'vegetation_height/2020/complete.json',
    'report.json',
    'ledger.parquet',
)
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


# ── imperviousness smoke ─────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-secondary-imperviousness")
def smoke_secondary_imperviousness(session: nox.Session) -> None:
    """Run imperviousness 2016 + 2021 end-to-end locally.

    Downloads both official ZIPs, converts class codes to percent,
    reprojects to the canonical 10 m grid, writes COGs, validates.
    Then runs again to confirm idempotency.
    """
    # First run — download + process both vintages
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_imperviousness",
        external=True,
    )

    # Second run — must be idempotent (no processing)
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_imperviousness",
        external=True,
    )


@nox.session(venv_backend="none", name="cloud-secondary-imperviousness")
def cloud_secondary_imperviousness(session: nox.Session) -> None:
    """Run imperviousness processing targeting GCS.

    Requires ADC / Workload Identity (``GOOGLE_APPLICATION_CREDENTIALS``).
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = (
        f"sec-imperv-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = f"gs://berlin-lst-data/secondary/smoke/{run_id}"

    # Pre-flight
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

    # Run imperviousness via smoke config
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_imperviousness",
        f"output_root={output_root}",
        external=True,
    )

    # Verify outputs
    session.run(
        "uv", "run", "python", "-c",
        f"""import sys
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('berlin-lst-data')
prefix = 'secondary/smoke/{run_id}/'
blobs = list(bucket.list_blobs(prefix=prefix))
print(f'Outputs in gs://berlin-lst-data/{{prefix}}')
print(f'  {{len(blobs)}} blob(s)')
for b in blobs:
    print(f'  {{b.name}} ({{b.size}} bytes)')
required_suffixes = (
    'imperviousness_2016.tif',
    'imperviousness_2016.stac.json',
    'imperviousness/2016/provenance.json',
    'imperviousness/2016/complete.json',
    'imperviousness_2021.tif',
    'imperviousness_2021.stac.json',
    'imperviousness/2021/provenance.json',
    'imperviousness/2021/complete.json',
    'report.json',
    'ledger.parquet',
)
names = [b.name for b in blobs]
missing = [s for s in required_suffixes if not any(n.endswith(s) for n in names)]
if missing:
    print(f'Missing required blobs: {{missing}}')
    sys.exit(1)
""",
        external=True,
    )


# ── vegetation-height smoke ─────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-secondary-vegetation-height")
def smoke_secondary_vegetation_height(session: nox.Session) -> None:
    """Run vegetation-height processing locally.

    Downloads the official Umweltatlas ZIP (~785 MB), extracts the inner
    GeoTIFF, reprojects to canonical 10 m, writes the four final
    artifacts (COG + STAC + provenance + complete), and validates.
    Then runs again to confirm ledger idempotency.
    """
    # First run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_vegetation_height",
        external=True,
    )

    # Idempotency run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_vegetation_height",
        external=True,
    )


@nox.session(venv_backend="none", name="cloud-secondary-vegetation-height")
def cloud_secondary_vegetation_height(session: nox.Session) -> None:
    """Run vegetation-height processing targeting GCS.

    Downloads the official Umweltatlas ZIP (~785 MB), extracts the inner
    GeoTIFF, reprojects to canonical 10 m, writes the COG, and
    validates.  Then runs again to confirm ledger idempotency.

    Requires ADC / Workload Identity (``GOOGLE_APPLICATION_CREDENTIALS``).
    """
    import uuid
    from datetime import UTC, datetime

    session.env.setdefault("UV_ENV_FILE", ".env")

    run_id = (
        f"sec-veght-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_root = f"gs://berlin-lst-data/secondary/smoke/{run_id}"

    # Pre-flight
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

    # First run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_vegetation_height",
        f"output_root={output_root}",
        external=True,
    )

    # Idempotency run
    session.run(
        "uv", "run", "python", "scripts/run_secondary.py",
        "--config-name", "smoke_vegetation_height",
        f"output_root={output_root}",
        external=True,
    )

    # Verify outputs (raw ZIP + COG + STAC + provenance + complete + ledger)
    session.run(
        "uv", "run", "python", "-c",
        f"""import sys
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('berlin-lst-data')
prefix = 'secondary/smoke/{run_id}/'
blobs = list(bucket.list_blobs(prefix=prefix))
print(f'Outputs in gs://berlin-lst-data/{{prefix}}')
print(f'  {{len(blobs)}} blob(s)')
for b in blobs:
    print(f'  {{b.name}} ({{b.size}} bytes)')
required_suffixes = (
    'veghoehe_2020.zip',
    'vegetation_height_2020.tif',
    'vegetation_height_2020.stac.json',
    'vegetation_height/2020/provenance.json',
    'vegetation_height/2020/complete.json',
    'report.json',
    'ledger.parquet',
)
names = [b.name for b in blobs]
missing = [s for s in required_suffixes if not any(n.endswith(s) for n in names)]
if missing:
    print(f'Missing required blobs: {{missing}}')
    sys.exit(1)
""",
        external=True,
    )


@nox.session(venv_backend="none", name="upload-manifest")
def upload_manifest(session: nox.Session) -> None:
    """Upload a local manifest.parquet to ``gs://berlin-lst-data/manifests/``.

    Usage::

        uv run nox -s upload-manifest -- data/ard/manifest.parquet

    Prints the GCS URI to pass as ``manifest_uri=...`` for a cloud full run.
    """
    from pathlib import Path

    args = session.posargs
    if not args:
        session.error(
            "Provide a local manifest path: "
            "nox -s upload-manifest -- data/ard/manifest.parquet",
        )
    local = Path(args[0])
    if not local.is_file():
        session.error(f"Manifest not found: {local}")

    dst = f"gs://berlin-lst-data/manifests/{local.name}"
    session.run("gcloud", "storage", "cp", str(local), dst, external=True)
    session.log(f"Uploaded: {dst}")
    session.log(f"Use: manifest_uri={dst}")
