"""Berlin LST downscaling — cloud-native skeleton (pre-implementation)."""

import os
from pathlib import Path

# Bootstrap: load .env into os.environ before any Google Cloud imports.
# Libraries like google.cloud.storage read GOOGLE_APPLICATION_CREDENTIALS
# from the process environment at import time — this ensures it's set
# regardless of shell type (zsh, bash, uv run, IDE, CI).
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _value = _line.split("=", 1)
        _key, _value = _key.strip(), _value.strip()
        if _value:
            os.environ.setdefault(_key, _value)

__version__ = "0.2.0"
