# berlin-lst-downscaling

Reproducible deep-learning pipeline that downscales Landsat land surface temperature from ~100 m to 10 m for Berlin via a fixed 2D U-Net and a 5-stage feature ablation. 100 m thermal data is too coarse for block-level urban heat analysis; this produces a 10 m LST time series plus a published model.

## Repository Category

`portfolio` — public-facing, presentable, polished.

- feature branches preferred for meaningful work; direct commits to `main` acceptable for small changes
- conventional commits always
- README quality matters — keep it accurate, clear, and presentable
- no formal release process needed

## Tech Stack

- Python 3.12
- uv — package management
- ruff — linting and formatting
- pyright — type checking
- pytest — tests
- nox — validation entrypoint
- PyTorch, Lightning, TorchGeo, Hydra

## Project Type

`data-pipeline`

## Structure

```
src/berlin_lst_downscaling/    # main package
tests/            # tests
```

## Validation

- `uv run nox` — full validation gate; run before every commit
- `nox -s lint` — docs, config, comment-only changes
- `nox -s lint typecheck` — structural changes (new modules, imports, type signatures)
- `nox -s lint typecheck test` — logic or behavior changes

## Python Stack

- `uv` — package and environment management
- `ruff` — linting and formatting
- `pyright` — type checking
- `pytest` — tests
- `nox` — validation entrypoint; run `uv run nox` before every commit

## Conventions

<!-- filled by /python-init and updated over time -->
- follow existing patterns before introducing new ones
- keep the README honest and presentable — this is portfolio work

## Library Documentation

Context7 MCP is available in this project. When working with any external library, use it to fetch current, version-specific documentation rather than relying on training data. Invoke with the library name or a Context7 library ID (e.g. `/fastapi/fastapi`, `/pydantic/pydantic`).

## Known Constraints

- Storage: COG per scene, assembled into a multi-date Zarr; no pre-baked patches; pseudo-pairs on the fly. Bucket mounted locally via rclone (not gcsfuse — x86_64 macOS limitation) at `~/.mnt/berlin-lst/`. See `.opencode/skills/google-access/` for mount/access commands.
- Reproducibility: Hydra configs (nothing hardcoded), seed management, env lock (uv) plus Docker for Vertex AI, Git commit hash logged per W&B run.
- Secrets via ENV, never committed.
- GEE experiments stay manual (Silas). Data cleaning and repetitive engineering may be delegated.

## Notion Integration

Notion Page ID: 28c35645-1f66-8057-b647-db5aebf191a5
