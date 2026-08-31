# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""WB3 modeling runner (Hydra-driven synthetic training lifecycle).

Usage
-----
    # Smoke: deterministic synthetic run, W&B offline, local ephemeral
    # output (validated by `nox -s smoke-modeling`)
    uv run python scripts/run_modeling.py --config-name smoke

    # Full: base synthetic config, W&B online (requires login)
    uv run python scripts/run_modeling.py --config-name full

Exits non-zero when the lifecycle fails (fit error, missing best
checkpoint, or checkpoint reload validation failure — fail-closed).
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.modeling.run import run_training

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/modeling", config_name="full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Dispatch to the modeling lifecycle and persist run logging."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="modeling", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            output_root=output_root,
            config_name=cfg.get("_hydra", {}).get("job", {}).get("config_name", "full"),
            data_release_id=str(cfg.get("data_release_id", "")),
            seed=int(cfg.get("seed", 0)),
            wandb_mode=str(cfg.get("wandb", {}).get("mode", "")),
        )
        result = run_training(cfg)

        print(f"Modeling — run {run_id}")
        print(f"  Best checkpoint : {result.best_checkpoint}")
        print(f"  Validation loss : {result.validation_loss:.6f}")
        print(f"  OK              : {result.run_ok}")

        # Hydra 1.3.4 discards the decorated task's return value, so
        # ``raise SystemExit(main())`` would exit 0 even on failures.
        # Raise inside the task to propagate a non-zero process exit.
        if not result.run_ok:
            raise SystemExit(1)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())