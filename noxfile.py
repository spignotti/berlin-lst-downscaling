"""Validation sessions for the berlin-lst-downscaling project."""

import nox

nox.options.sessions = ["lint", "typecheck"]


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


# ── smoke tests ────────────────────────────────────────────────────────


@nox.session(venv_backend="none", name="smoke")
def smoke(session: nox.Session) -> None:
    """Run the ARD pipeline in smoke mode (local disk).

    Produces COGs, STAC items, ledger, and RGB visualisation PNGs
    under ``data/tmp/smoke_ard_<date>/``.
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard.py", "--config-name", "smoke",
        external=True,
    )


@nox.session(venv_backend="none", name="smoke-cloud")
def smoke_cloud(session: nox.Session) -> None:
    """Smoke test targeting GCS (requires ``gs://berlin-lst-data/`` access).

    Mount the bucket first (``rclone mount``) or ensure Application Default
    Credentials are configured.  This session is **not** required for PR
    validation — it is slow and environment-dependent.
    """
    session.run(
        "uv", "run", "python", "scripts/run_ard.py",
        "--config-name", "smoke",
        "output_root=gs://berlin-lst-data/ard/smoke-test",
        "viz=true",
        external=True,
    )
