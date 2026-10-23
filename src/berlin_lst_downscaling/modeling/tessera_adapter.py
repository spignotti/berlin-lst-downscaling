"""Learnable 1×1 convolution adapter for the TESSERA ablation arm.

All 128 TESSERA embedding channels go into the model through a learnable
``1×1`` convolution (no PCA, no fixed reduction). The adapter output is
concatenated with the remaining hand-built channels and fed to the SAME
U-Net the main arm uses.

Channel replacement (features/v6 roster)
----------------------------------------
Replaced by TESSERA: the 6 morphology channels plus the 2 shadow channels
(8 total). Every other v6 channel (spectral, index, ERA5) stays identical
to the main arm.

The 1×1 output width is a config value (``model.tessera_adapter_width``)
with documented default :data:`TESSERA_ADAPTER_WIDTH_DEFAULT`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.modeling.unet import UNet

TESSERA_N_CHANNELS: int = 128

# Default 1×1 adapter output width (config value ``model.tessera_adapter_width``).
TESSERA_ADAPTER_WIDTH_DEFAULT: int = 8


def tessera_replaced_channel_names() -> tuple[str, ...]:
    """Return the v6 channel names the TESSERA arm replaces (morphology + shadows)."""
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    return schema.names_in_family("morphology") + schema.names_in_family("shadow")


def tessera_kept_channel_names() -> tuple[str, ...]:
    """Return the v6 channel names the TESSERA arm keeps (all others)."""
    replaced = set(tessera_replaced_channel_names())
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    return tuple(name for name in schema.names if name not in replaced)


def tessera_kept_channel_indices() -> tuple[int, ...]:
    """Return the v6 roster indices the TESSERA arm keeps, in roster order."""
    replaced = set(tessera_replaced_channel_names())
    schema = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    return tuple(i for i, name in enumerate(schema.names) if name not in replaced)


class TesseraAdapter(nn.Module):
    """Learnable 1×1 convolution over all 128 TESSERA channels.

    Parameters
    ----------
    in_channels :
        TESSERA embedding width (always 128).
    out_width :
        Adapter output width (config value, default 8).
    """

    def __init__(
        self, in_channels: int = TESSERA_N_CHANNELS, out_width: int = TESSERA_ADAPTER_WIDTH_DEFAULT
    ) -> None:
        super().__init__()
        if in_channels != TESSERA_N_CHANNELS:
            raise ValueError(
                f"TESSERA adapter expects {TESSERA_N_CHANNELS} input channels, got {in_channels}"
            )
        if not isinstance(out_width, int) or isinstance(out_width, bool) or out_width < 1:
            raise ValueError(f"adapter out_width must be a positive int, got {out_width!r}")
        self.in_channels = in_channels
        self.out_width = out_width
        self.proj = nn.Conv2d(in_channels, out_width, kernel_size=1, bias=True)

    def forward(self, embedding: Tensor) -> Tensor:
        if embedding.ndim != 4:
            raise ValueError(f"expected a 4-D embedding (B, 128, H, W), got {embedding.ndim}-D")
        if embedding.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} embedding channels, got {embedding.shape[1]}"
            )
        return self.proj(embedding)


class TesseraUNet(nn.Module):
    """SAME U-Net fronted by the TESSERA 1×1 adapter.

    Forward takes ``(kept_features, tessera_embedding)`` where
    ``kept_features`` has ``C_kept`` hand-built channels (the 14 kept v6
    channels) and ``tessera_embedding`` has 128 channels. The adapter
    compresses the embedding to ``adapter_width`` channels, the result is
    concatenated to ``C_kept + adapter_width`` channels, and the shared
    :class:`UNet` predicts from there.
    """

    def __init__(
        self,
        *,
        n_kept_channels: int,
        adapter_width: int = TESSERA_ADAPTER_WIDTH_DEFAULT,
        base_width: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if n_kept_channels < 1:
            raise ValueError(f"n_kept_channels must be >= 1, got {n_kept_channels}")
        self.adapter = TesseraAdapter(in_channels=TESSERA_N_CHANNELS, out_width=adapter_width)
        self.n_kept_channels = n_kept_channels
        self.adapter_width = adapter_width
        self.unet = UNet(
            in_channels=n_kept_channels + adapter_width,
            base_width=base_width,
            depth=depth,
        )

    @property
    def in_channels(self) -> int:
        """Total U-Net input width (kept + adapter)."""
        return self.n_kept_channels + self.adapter_width

    def forward(self, kept: Tensor, embedding: Tensor) -> Tensor:
        if kept.ndim != 4 or embedding.ndim != 4:
            raise ValueError("kept and embedding must both be 4-D (B, C, H, W)")
        if kept.shape[1] != self.n_kept_channels:
            raise ValueError(f"expected {self.n_kept_channels} kept channels, got {kept.shape[1]}")
        if kept.shape[2:] != embedding.shape[2:]:
            raise ValueError(
                f"spatial mismatch: kept {tuple(kept.shape[2:])} != "
                f"embedding {tuple(embedding.shape[2:])}"
            )
        adapted = self.adapter(embedding)
        fused = torch.cat([kept, adapted], dim=1)
        return self.unet(fused)


__all__ = [
    "TESSERA_ADAPTER_WIDTH_DEFAULT",
    "TESSERA_N_CHANNELS",
    "TesseraAdapter",
    "TesseraUNet",
    "tessera_kept_channel_indices",
    "tessera_kept_channel_names",
    "tessera_replaced_channel_names",
]
