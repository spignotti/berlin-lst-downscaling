# Run Logging

Every pipeline run produces structured JSONL logs for post-hoc debugging
and audit. Logs live alongside output data and are always machine-parseable.

## Path Layout

```
<output_root>/logs/<pipeline>/<run_id>.jsonl
```

Examples:
- `data/ard/logs/ard/abc12345.jsonl`
- `data/static/sources/logs/static-sources/def67890.jsonl`
- `gs://berlin-lst-data/static/sources/full/logs/static-derived/ghi11111.jsonl`

## Pipelines

| Pipeline | Entry point | Config prefix | Pipeline ID |
|----------|-------------|---------------|-------------|
| ARD | `scripts/run_ard.py` | `configs/ard/` | `ard` |
| Selection | `scripts/build_manifest.py` | `configs/selection/` | `selection` |
| Static Sources (A) | `scripts/run_static_sources.py` | `configs/static_sources/` | `static-sources` |
| Static Derived (B) | `scripts/run_static_derived.py` | `configs/static_derived/` | `static-derived` |
| Dynamic (C) | `scripts/run_dynamic.py` | `configs/dynamic/` | `dynamic` |
| DWD validation | `scripts/run_dwd_validation.py` | `configs/dwd_validation/` | `dwd_validation` |

## JSONL Schema

Each line is a JSON object:

```json
{
  "timestamp": "2026-07-17T14:32:01.123456+00:00",
  "level": "INFO",
  "logger": "berlin_lst_downscaling.data.ard.pipeline",
  "event": "scene_done",
  "pipeline": "ard",
  "run_id": "abc12345",
  "scene_id": "LC09_L2SP_193024_20240629_02_T1",
  "source": "landsat-c2-l2",
  "attempts": 1,
  "elapsed_s": 12.34,
  "exception": "..."
}
```

- `timestamp` — ISO 8601 UTC
- `level` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `logger` — Python logger name (e.g. `berlin_lst_downscaling.data.ard.pipeline`)
- `event` — short machine-readable event key (e.g. `scene_done`, `anchor_filter_progress`)
- `pipeline` / `run_id` — context, set by the session
- Remaining fields are event-specific structured data
- `exception` — full traceback, only present on errors

## Log Levels

| Level | Use |
|-------|-----|
| `DEBUG` | Per-tile progress, intermediate statistics, dropped anchors |
| `INFO` | Run start/end, lifecycle milestones, QA summary, per-scene done/failed |
| `WARNING` | Recoverable degradation (e.g. AOI metrics error, checkpoint load failed) |
| `ERROR` | Processing failures requiring investigation |

## GCS Runs

For GCS output roots (e.g. `gs://berlin-lst-data/...`):
1. JSONL is written locally to a temp directory during the run.
2. On normal or exceptional exit, the completed JSONL is atomically uploaded
   to `<output_root>/logs/<pipeline>/<run_id>.jsonl`.
3. The local temp directory is cleaned up.

This ensures logs are never lost even on VM preemption or crashes.

## Inspecting Logs

```bash
# List logs for a pipeline
find data/ -path "*/logs/ard/*.jsonl" 2>/dev/null

# Parse all events for a run
cat data/ard/logs/ard/abc12345.jsonl | jq .

# Filter by event type
cat data/ard/logs/ard/abc12345.jsonl | jq 'select(.event == "scene_failed")'

# Filter by level
cat data/ard/logs/ard/abc12345.jsonl | jq 'select(.level == "ERROR")'

# Count scenes processed
cat data/ard/logs/ard/abc12345.jsonl | jq 'select(.event == "scene_done") | .scene_id' | wc -l

# Check GCS logs (requires rclone mount)
ls ~/.mnt/berlin-lst/ard/logs/ard/
```

## Configuration

All base configs include `logging_level: INFO`. Override per run:

```bash
uv run python scripts/run_ard.py --config-name smoke_primary logging_level=DEBUG
```

## Architecture

```
entry point (run_ard.py, build_manifest.py, etc.)
    │ reads logging_level from Hydra config
    │ creates run_id (shared with pipeline)
    ▼
RunLogSession(output_root, pipeline, run_id, level)
    ├─ ContextFilter — injects pipeline + run_id into every LogRecord
    ├─ stderr StreamHandler — concise text, live terminal
    └─ JSONL FileHandler — structured, durable
         ├─ local: <output_root>/logs/<pipeline>/<run_id>.jsonl
         └─ GCS: local spool → atomic upload on exit
```

- **Single run ID**: entry point creates one `run_id` and passes it to the pipeline function. Log, ledger, QA report, and provenance all share the same ID.
- **ContextFilter**: `pipeline` and `run_id` are attached to every `LogRecord` via a handler filter, so all `log_event()` calls automatically include them.
- **GCS publication is mandatory**: if the final JSONL upload to GCS fails, the process exits with a `RuntimeError` (after handler cleanup).
- No root logger pollution: handlers attach temporarily during the session
- Module loggers propagate to the session root via standard `logging`
- Third-party loggers (rasterio, urllib3) suppressed to ERROR during runs
