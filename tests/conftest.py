"""Offline defaults for the unit suite.

Imported before test modules. Keeps PROJ and W&B from attempting a network
call if a test imports a module that would otherwise check those services.
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PROJ_NETWORK", "OFF")
