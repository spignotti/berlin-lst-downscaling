# berlin-lst-downscaling

Cloud-native skeleton for Berlin LST downscaling (pre-implementation). Work-in-progress pipeline using Microsoft Planetary Computer STAC for data access, with future DL training via PyTorch/Lightning/TorchGeo.

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
- nox — validation entrypoint
- dvc[gs] — data versioning (GCS remote)
- wandb — experiment tracking
- pydantic-settings — env-based config
- google-cloud-storage — bucket access
- _planned (not yet used):_ pystac-client, odc-stac, rioxarray, zarr, PyTorch, Lightning, TorchGeo

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
- No test session — tests are opt-in. Quality validated via real-data QA gates (smoke, spike scripts), not unit tests.

## Python Stack

- `uv` — package and environment management
- `ruff` — linting and formatting
- `pyright` — type checking
- `nox` — validation entrypoint; run `uv run nox` before every commit
- `dvc[gs]` — data versioning
- `wandb` — experiment tracking
- `pydantic-settings` — env-based config

## Conventions

- follow existing patterns before introducing new ones
- keep the README honest and presentable — this is portfolio work
- **No tests unless explicitly requested** — QA is validated through real-data smoke/spike scripts, not unit tests
- **Build order:** Spike → Core → Framework (not the reverse — no premature scaffolding)

## Library Documentation

Context7 MCP is available in this project. When working with any external library, use it to fetch current, version-specific documentation rather than relying on training data. Invoke with the library name or a Context7 library ID (e.g. `/fastapi/fastapi`, `/pydantic/pydantic`).

## Known Constraints

- Storage: Bucket mounted locally via rclone (not gcsfuse — x86_64 macOS limitation) at `~/.mnt/berlin-lst/`. See `.opencode/skills/google-access/` for mount/access commands.
- Reproducibility: env lock (uv), Git commit hash logged per W&B run.
- Secrets via ENV, never committed.
- macOS x86_64 ceiling: `numpy<2`, `torch<2.3` for training stack.

## Notion Integration

Notion Page ID: 28c35645-1f66-8057-b647-db5aebf191a5
