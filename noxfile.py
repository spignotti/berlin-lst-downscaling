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
    """Run the ECOSTRESS ARD pipeline in smoke mode (local fixture).

    Requires ``scripts/download_ecostress_fixture.py`` to be run first
    to download the fixture granule to ``data/ecostress/fixtures/``.

    Produces COGs, STAC items, ledger, and visualisation PNGs
    under ``data/tmp/smoke_ecostress_<date>/``.
    """
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
