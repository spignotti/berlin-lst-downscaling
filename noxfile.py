"""Validation sessions for the berlin-lst-downscaling project."""

from pathlib import Path

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

    Self-contained: downloads the fixture granule (if missing) then runs
    the pipeline and visualises the output.

    Uses tile 33UUU (western Berlin, ~88% footprint overlap on 2018-07-30,
    ~293k valid LST pixels inside Berlin bbox).
    Override with HYDRA_OVERRIDE env, e.g.:
        HYDRA_OVERRIDE="ecostress.tile=33UVU" nox -s smoke-ecostress
    """
    # Step 1: ensure fixture is present (download if missing or incomplete)
    fixture_root = Path("data/ecostress/fixtures")
    granule_id = "ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01"
    granule_dir = fixture_root / granule_id
    required_layers = {"LST", "cloud", "water", "QC"}
    fixture_complete = (
        granule_dir.is_dir()
        and required_layers.issubset({p.stem.split("_")[-1] for p in granule_dir.glob("*.tif")})
    )
    if not fixture_complete:
        session.run(
            "uv", "run", "python", "scripts/download_ecostress_fixture.py",
            "--tile", "33UUU", "--date", "2018-07-30",
            "--out", str(fixture_root),
            external=True,
        )

    # Step 2: run the pipeline
    session.run(
        "uv", "run", "python", "scripts/run_ard_ecostress.py", "--config-name", "smoke",
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
