# berlin-lst-downscaling

Cloud-native LST downscaling pipeline for Berlin. Uses Microsoft Planetary Computer STAC for Landsat/Sentinel-2 data access and NASA CMR (earthaccess) for ECOSTRESS data. Manifest-driven scene selection, ARD processing (COGs + STAC + ledger), and GCS-native storage.

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
- wandb — experiment tracking
- pydantic-settings — env-based config
- google-cloud-storage — bucket access
- pystac-client, odc-stac, rioxarray — PC STAC + EO data (in use)
- _planned (training stack, not yet used):_ zarr, PyTorch, Lightning, TorchGeo

## Project Type

`data-pipeline`

## Validation

- `uv run nox` — full validation gate; run before every commit
- `nox -s lint` — docs, config, comment-only changes
- `nox -s lint typecheck` — structural changes (new modules, imports, type signatures)
- No test session — tests are opt-in. Quality validated via real-data QA gates (smoke, spike scripts), not unit tests.

## Data Safety

- The bucket holds immutable canonical products, retained QA evidence, and ephemeral run outputs. Never delete or rewrite canonical products outside a planned task; never modify retained evidence; remove ephemeral outputs in the task that created them.
- Logs live only at `<output_root>/logs/<pipeline>/`; never at the repo root or outside the run's output root.
- Remove temporary/one-off scripts before closing a task.
- The VM stays stopped and protected (deletion protection on, boot disk not auto-delete) when not actively running the Dynamic pipeline.
- Secrets via ENV, never committed.

## Compute Placement

- Compute-heavy production work runs on the GCP VM (On-Demand `berlin-lst-vm`); managed model training runs on Vertex AI.
- VM lifecycle orchestration (start/stop/ssh/status, run launchers) lives **only** in `.opencode/skills/google-access/scripts/` — never in application source under `src/` or repo `scripts/`.
- Application code stays VM-agnostic: it consumes GCS and takes config via Hydra; only the skill launchers know the VM.

## Conventions

- follow existing patterns before introducing new ones
- keep the README honest and presentable — this is portfolio work
- **No tests unless explicitly requested** — QA is validated through real-data smoke/spike scripts, not unit tests
- **Build order:** Spike → Core → Framework (not the reverse — no premature scaffolding)
- Pipeline telemetry goes through `log_event` (`data/io/run_logging.py`), never raw `print()`; `print()` is allowed only for validators, spikes, and human-oriented CLI summaries.

## Library Documentation

Context7 MCP is available in this project. When working with any external library, use it to fetch current, version-specific documentation rather than relying on training data. Invoke with the library name or a Context7 library ID (e.g. `/fastapi/fastapi`, `/pydantic/pydantic`).

## Known Constraints

- Storage: Bucket mounted locally via rclone (not gcsfuse — x86_64 macOS limitation) at `~/.mnt/berlin-lst/`. See `.opencode/skills/google-access/` for mount/access commands.
- Reproducibility: env lock (uv), Git commit hash logged per W&B run.
- macOS x86_64 ceiling: `numpy<2`, `torch<2.3` for training stack.

## Documentation

- `README.md` — public project overview: architecture, status, minimal setup.
- `docs/phase-1-delivery.md` — delivered data products and phase-2 handoff.
- `docs/data-sources-and-contracts.md` — sources, canonical grid, manifest/ledger contracts.
- `docs/phase-2-preparation.md` — phase-2 preparation state.

## Notion Integration

Notion Page ID: 28c35645-1f66-8057-b647-db5aebf191a5
