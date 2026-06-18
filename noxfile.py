import nox


nox.options.sessions = ["lint", "typecheck", "test"]


@nox.session
def lint(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", ".")


@nox.session
def format(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def fix(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", "--fix", ".")
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.run("uv", "run", "pyright")


@nox.session
def test(session: nox.Session) -> None:
    session.run("uv", "run", "pytest", "-v")
