# Secondary-Data Pipeline — Operations Runbook

Shared foundation for acquiring, processing, and storing all non-satellite
data sources used in the LST downscaling ablation study.

## Path Layout

All paths are relative to `output_root` (local path or `gs://bucket/prefix`).

| Path | Purpose |
|------|---------|
| `_raw/secondary/{source}/{period}/` | Raw downloaded archives — one per source and period/vintage |
| `_staging/secondary/{source}/{run_id}/` | Ephemeral processing scratch space |
| `ard/static/{category}/{source}/{vintage}/` | Final static COGs (morphology, terrain, SVF) |
| `qa/secondary/{run_id}/` | Run-specific QA reports |
| `ledger.parquet` | Persistent item-level processing ledger |

## Modes

| Mode | Output Root | Purpose |
|------|-------------|---------|
| `fixture` | Local `data/smoke/secondary/fixture` | Validate pipeline lifecycle, no real data |
| `cloud_smoke` | `gs://.../secondary/smoke/{run_id}` | Validate GCS write path |
| `full` | Configurable | Production runs with real sources |

## Ledger Semantics

The `SecondaryLedger` tracks every item (`item_id + source + period`) through
its lifecycle:

1. **pending** — newly added, not yet processed
2. **exporting** — processing in progress (crash recovery marker)
3. **done** — output COG written, validated, and checksummed
4. **failed** — processing error, will be retried on next run
5. **skipped** — explicitly skipped, never processed

### Idempotency / Resume

`reconcile()` applies the same logic as the ARD pipeline:

- Done items with matching `config_hash` + confirmed output → skipped
- Exporting items (crashed) → retry with **interrupted** reason
- Failed items → retry with **retry** reason
- Config hash mismatch → reprocess with **config_changed** reason

## Retry / Resume Behaviour

- HTTP downloads (via `download_to_raw`) use `tenacity` with
  3 attempts and exponential backoff (up to 10 s).
- Ledger upserts happen immediately — crash consistency is built in.
- An aborted run resumes by re-running the same command.  The second run
  processes only items that are missing, failed, or interrupted.

## Required Setup

### GCS access

- Install ADC: `gcloud auth application-default login`
- Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### VM

- Instance: `berlin-lst-vm`, zone `europe-west3-a`, machine `n2-standard-2`
- OS: Debian 12, Spot preemptible
- Auth: Service account via ADC (no keys in repo)
- Bucket: `gs://berlin-lst-data` (co-located in `europe-west3`)

### Disk Budget

Each full source run **must** declare a peak scratch estimate before
execution.  The run will fail preflight if the estimate exceeds the
configured budget.

Current budget: **20 GB** (VM boot disk).
Override with `+scratch_budget_gb=50` in the Hydra command line.
Use kachelweises/tilewise scratch cleanup when processing tile-based
sources (LoD2, DGM 1 m).

## Smoke Tests

```bash
# Local fixture (no GCS needed)
uv run nox -s smoke-secondary-fixture

# Cloud fixture (requires ADC)
uv run nox -s cloud-secondary-fixture
```

## Adding a New Source

1. Add source-specific band specs in a new contract (see `contract.py`).
2. Implement a download handler (or reuse `download_to_raw` for HTTP).
3. Implement a prepare handler: read raw → process → write COG.
4. Register the source in the pipeline runner (see `pipeline.py`).

Each source module should expose a single `prepare(source_id, period, cfg)`
function that returns `list[SecondaryItem]`, one per period/vintage.
