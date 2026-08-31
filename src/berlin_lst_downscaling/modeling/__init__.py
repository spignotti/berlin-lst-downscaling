"""WB3 modeling — deterministic U-Net training scaffold.

Provides the fixed 2D U-Net baseline, the typed batch interface, the
Lightning regression task, and a deterministic synthetic data module so
the full training lifecycle runs before real WB3 data is integrated.
Real-data readers and the masked loss arrive through the extension
points in :mod:`~berlin_lst_downscaling.modeling.contracts`.
"""

from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    Batch,
    MaskedLoss,
    SampleMeta,
)
from berlin_lst_downscaling.modeling.synthetic import SyntheticDataModule
from berlin_lst_downscaling.modeling.task import LSTRegressionTask
from berlin_lst_downscaling.modeling.unet import UNet

__all__ = [
    "Batch",
    "LSTRegressionTask",
    "MaskedLoss",
    "N_FEATURE_CHANNELS",
    "SampleMeta",
    "SyntheticDataModule",
    "UNet",
]