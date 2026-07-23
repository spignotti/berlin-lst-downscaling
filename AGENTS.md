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
- dvc[gs] — data versioning (GCS remote)
- wandb — experiment tracking
- pydantic-settings — env-based config
- google-cloud-storage — bucket access
- pystac-client, odc-stac, rioxarray — PC STAC + EO data (in use)
- _planned (training stack, not yet used):_ zarr, PyTorch, Lightning, TorchGeo

## Project Type

`data-pipeline`

## Structure

```
src/berlin_lst_downscaling/    # main package
    data/acquisition/          # PC STAC loaders + ECOSTRESS CMR
    data/ard/                  # ARD pipeline (COG write, masking, ledger, STAC)
    data/secondary/            # Static A + B (sources + derived geometry)
    data/dynamic/              # ERA5-Land + shadows + DWD validation
    data/selection/            # v3 manifest selection & coupling
    data/io/                   # Storage (local + GCS), run logger, ephemeral staging
    common/                    # Canonical 10 m grid, env config
configs/                       # Hydra configs (ARD + selection + static_sources
                               #   + static_derived + dynamic + dwd_validation)
scripts/                       # Entry points (run_ard.py, run_static_sources.py,
                               #   run_static_derived.py, run_dynamic.py,
                               #   run_dynamic_isolated.py, run_dwd_validation.py,
                               #   build_manifest.py, validators)
```

See `docs/delivered-implementation.md` for the canonical pipeline
graph, published roots, and operations guide. See
`docs/data-sources-and-contracts.md` for the data and ledger
contracts.

### DWD-vs-ERA5 validation

Read-only sanity check on the published ERA5-Land `t2m_scene` channel
at Landsat anchor times. Acquires DWD hourly 2 m air temperature for
stations inside the Berlin AOI via `wetterdienst` and joins it to the
published COG provenance. Never feeds DWD into training or
normalisation.

```bash
uv run python scripts/run_dwd_validation.py \
    manifest_uri=gs://berlin-lst-data/manifests/v3/<cutoff>-r2/manifest.parquet \
    dynamic_full_root=gs://berlin-lst-data/dynamic/full \
    dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026 \
    output_root=gs://berlin-lst-data/dwd_validation \
    aoi_uri=gs://berlin-lst-data/boundaries/berlin_landesgrenze.geojson
```

Layout: `<output_root>/_raw/dwd/<run_id>/station_inventory.parquet`
+ `dwd_hourly_observations.parquet`,
`<output_root>/runs/dwd/<run_id>/anchor_comparison.parquet`
+ `report.json` + `provenance.json` + `complete.json`.

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

## Runtime Logging

All productive pipeline entrypoints use the shared run logger (`data/io/run_logging.py`).

- Use `log_event(logger, level, event, **fields)` — never raw `print()` for pipeline telemetry.
- `print()` is allowed only for validators, spikes, and human-oriented CLI summaries.
- Every run emits: lifecycle start/end, config context, work-unit outcomes, duration, QA summary, and tracebacks for caught failures.
- Log levels: `DEBUG` for tile detail, `INFO` for lifecycle/progress, `WARNING` for recoverable degradation, `ERROR` for failures.
- JSONL contract: `<output_root>/logs/<pipeline>/<run_id>.jsonl`. GCS runs publish after exit.
- Ledger, QA reports, STAC, provenance, and completion markers remain authoritative domain artifacts — logs complement them, not replace them.
- No secrets, tokens, credentials, or signed URLs in logs.

## Library Documentation

Context7 MCP is available in this project. When working with any external library, use it to fetch current, version-specific documentation rather than relying on training data. Invoke with the library name or a Context7 library ID (e.g. `/fastapi/fastapi`, `/pydantic/pydantic`).

## Known Constraints

- Storage: Bucket mounted locally via rclone (not gcsfuse — x86_64 macOS limitation) at `~/.mnt/berlin-lst/`. See `.opencode/skills/google-access/` for mount/access commands.
- Reproducibility: env lock (uv), Git commit hash logged per W&B run.
- Secrets via ENV, never committed.
- macOS x86_64 ceiling: `numpy<2`, `torch<2.3` for training stack.

## Notion Integration

Notion Page ID: 28c35645-1f66-8057-b647-db5aebf191a5
