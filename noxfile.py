"""Validation sessions for the berlin-lst-downscaling project."""

import nox

nox.options.sessions = ["lint", "typecheck"]


# ── universal ──────────────────────────────────────────────────────────


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", ".", external=True)


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


# ── smoke tests (per-source) ─────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke")
def smoke(session: nox.Session) -> None:
    """Run Landsat + Sentinel-2 smoke tests in sequence.

    Use ``smoke-landsat`` or ``smoke-sentinel2`` to run a single source.
    Use ``smoke-ecostress`` for the ECOSTRESS fixture (after downloading it).
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard_landsat.py", "--config-name", "smoke",
        external=True,
    )
    session.run(
        "uv", "run", "python", "scripts/run_ard_sentinel2.py", "--config-name", "smoke",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-landsat")
def smoke_landsat(session: nox.Session) -> None:
    """Run the Landsat ARD pipeline in smoke mode (local disk).

    Produces COGs, STAC items, ledger, and RGB visualisation PNGs
    under ``data/tmp/smoke_landsat_<date>/``.
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard_landsat.py", "--config-name", "smoke",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-sentinel2")
def smoke_sentinel2(session: nox.Session) -> None:
    """Run the Sentinel-2 ARD pipeline in smoke mode (local disk).

    Produces COGs, STAC items, ledger, and RGB visualisation PNGs
    under ``data/tmp/smoke_sentinel2_<date>/``.
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard_sentinel2.py", "--config-name", "smoke",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-ecostress")
def smoke_ecostress(session: nox.Session) -> None:
    """Run the ECOSTRESS ARD pipeline in smoke mode.

    Self-contained: stages the raw L2T granule to a local tmp directory,
    runs the pipeline, then deletes the stage.  Final COGs land in
    ``data/tmp/smoke_ecostress_<date>/``.

    Uses tile 33UUU (western Berlin, ~88% footprint overlap on 2018-07-30,
    ~293k valid LST pixels inside Berlin bbox).
    """
    import uuid
    from datetime import UTC, datetime

    run_id = f"eco-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    stage_base = "data/tmp/ecostress_stage"
    raw_dir = f"{stage_base}/{run_id}"

    # Download + stage
    session.run(
        "uv", "run", "python", "scripts/download_ecostress_fixture.py",
        "--tile", "33UUU",
        "--date", "2018-07-30",
        "--stage-dir", stage_base,
        "--run-id", run_id,
        external=True,
    )

    # Pipeline reads from stage; cleanup handled by nox session StageSession
    session.run(
        "uv", "run", "python", "scripts/run_ard_ecostress.py",
        "--config-name", "smoke",
        f"ecostress.raw_dir={raw_dir}",
        "ecostress.persist_stage=true",  # pipeline does NOT clean; nox owns cleanup
        external=True,
    )

    # Always clean up stage — even if pipeline fails, the nox session always
    # reaches the last step. StageSession is idempotent (no-op if already gone).
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


# ── Szenen-Selektion ─────────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-selection")
def smoke_selection(session: nox.Session) -> None:
    """Run Szenen-Selektion coupling on a single month (July 2024).

    Validates the coupling logic before running the full volume scan.
    Writes ``data/tmp/manifest_smoke.parquet``.
    """
    session.run(
        "uv", "run", "python", "scripts/build_manifest.py",
        "--config-dir", "configs/selection",
        "--config-name", "smoke_jul2024",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-selection-2024")
def smoke_selection_2024(session: nox.Session) -> None:
    """Run Szenen-Selektion coupling on Mai–Sep 2024.

    Validates the coupling logic across the full configured season.
    Writes ``data/tmp/manifest_smoke_2024.parquet``.
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


# ── cloud smoke tests ─────────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke-cloud")
def smoke_cloud(session: nox.Session) -> None:
    """Run Landsat + Sentinel-2 smoke tests targeting GCS (requires rclone mount).

    Mount the bucket first (``rclone mount``) or ensure Application Default
    Credentials are configured.
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard_landsat.py",
        "--config-name", "smoke",
        "output_root=gs://berlin-lst-data/ard/smoke/landsat",
        "viz=true",
        external=True,
    )
    session.run(
        "uv", "run", "python", "scripts/run_ard_sentinel2.py",
        "--config-name", "smoke",
        "output_root=gs://berlin-lst-data/ard/smoke/sentinel2",
        "viz=true",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-ecostress-cloud")
def smoke_ecostress_cloud(session: nox.Session) -> None:
    """Run ECOSTRESS smoke test targeting GCS.

    Pre-flight: verifies GCS bucket is reachable and ADC are configured.
    Stages raw L2T granule to ``gs://berlin-lst-data/_staging/ecostress/<run_id>/``,
    runs the pipeline, then deletes the stage.

    Requires ``GOOGLE_APPLICATION_CREDENTIALS`` to be set (service account JSON key).
    """
    import uuid
    from datetime import UTC, datetime

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

    run_id = f"eco-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    stage_base = "gs://berlin-lst-data/_staging/ecostress"
    raw_dir = f"{stage_base}/{run_id}"

    session.run(
        "uv", "run", "python", "scripts/download_ecostress_fixture.py",
        "--tile", "33UUU",
        "--date", "2018-07-30",
        "--stage-dir", stage_base,
        "--run-id", run_id,
        external=True,
    )

    # Pipeline reads from GCS stage; pipeline does NOT clean (nox owns cleanup)
    session.run(
        "uv", "run", "python", "scripts/run_ard_ecostress.py",
        "--config-name", "cloud",
        f"ecostress.raw_dir={raw_dir}",
        "output_root=gs://berlin-lst-data/ard/smoke/ecostress",
        "viz=true",
        "ecostress.persist_stage=true",
        external=True,
    )

    # Cleanup: always run even on failure
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

    # Verify final COGs landed
    session.run(
        "uv", "run", "python", "-c",
        (
            "from google.cloud import storage; "
            "client = storage.Client(); "
            "bucket = client.get_bucket('berlin-lst-data'); "
            "prefix = 'ard/smoke/ecostress/2018/'; "
            "blobs = list(bucket.list_blobs(prefix=prefix)); "
            "print(f'Final COGs: {len(blobs)} blob(s) under gs://berlin-lst-data/{prefix}'); "
            "for b in blobs[:5]: print(' ', b.name)"
        ),
        external=True,
    )
