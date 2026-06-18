"""Training entrypoint with Hydra config."""

import os

import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the downscaling model.

    Usage:
        python src/train.py
        python src/train.py experiment=stage_1
        python src/train.py experiment=stage_3 trainer.max_epochs=200
    """
    # ── Reproducibility -------------------------------------------------------
    # 1a. Seed all RNGs (Python random, NumPy, PyTorch CPU, DataLoader workers)
    seed_everything(cfg.seed, workers=True)

    # 1b. Use deterministic cuDNN kernels (no auto-tuning across runs)
    torch.backends.cudnn.benchmark = False

    # 1c. Raise on non-deterministic CUDA operations
    torch.use_deterministic_algorithms(True)

    # ── Run info --------------------------------------------------------------
    print(OmegaConf.to_yaml(cfg))
    print(f"  PID: {os.getpid()}")


if __name__ == "__main__":
    main()
