# berlin-lst-downscaling

> Cloud-native skeleton for Berlin LST downscaling (pre-implementation). Future data source: Microsoft Planetary Computer STAC.

## Development

```bash
uv sync
uv run nox               # lint + typecheck
uv run nox -s fix        # lint and format
```

## GCP access

Mount the GCS bucket before any data work:

```bash
mount-berlin             # rclone mount
```

See `.opencode/skills/google-access/SKILL.md` for full reference (auth, troubleshooting, manual debugging).
