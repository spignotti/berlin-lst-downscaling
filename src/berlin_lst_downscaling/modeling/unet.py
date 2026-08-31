"""Plain 2D U-Net — the fixed deterministic baseline.

The thesis fixes the architecture family: only input feature groups,
capacity, and (later) the Stage-5 loss vary across ablation stages. This
module provides exactly that surface — no attention variants, no
pretrained encoders, no architecture search.

Design decisions
----------------
- **Plain PyTorch only.** A custom encoder/decoder keeps every layer
  visible in the project code and avoids a configurable/pretrained
  encoder surface (TorchGeo and segmentation-models-pytorch were
  considered and rejected for that reason).
- **Shape-preserving.** Convolutions are padded, pooling halves the
  spatial extent, and the decoder upsamples by factor 2 back to the
  input extent. No patch edge length is encoded — any ``H == W`` input
  that is divisible by ``2 ** depth`` is valid.
- **Regression head.** A single output channel with a linear head; the
  interpretation of the output (LST at 100 m) is owned by the loss and
  the downstream task, not by the model.

The forward pass asserts that the output extent matches the input
extent, so a misconfigured depth fails loudly instead of producing a
silently wrong map.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ── building blocks ───────────────────────────────────────────────────


class _ConvBlock(nn.Module):
    """Two 3x3 padded convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        return x


class _EncoderBlock(nn.Module):
    """Conv block + max pooling for one downsampling stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.block(x)
        return self.pool(features), features  # pooled (next stage), skip


class _DecoderBlock(nn.Module):
    """2x upsample + concatenation with the skip + conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Skip and upsampled tensors share the same extent by construction
        # (pooled by 2**depth at the deepest stage, upsampled back).
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ── model ─────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """Fixed deterministic 2D U-Net for LST regression.

    Parameters
    ----------
    in_channels:
        Number of active input channels (first-C of the fixed V3 order).
    base_width:
        Width of the first encoder stage; each deeper stage doubles it.
    depth:
        Number of down/up stages. The input spatial extent must be
        divisible by ``2 ** depth``.
    """

    def __init__(self, in_channels: int, base_width: int = 32, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if base_width < 1:
            raise ValueError(f"base_width must be >= 1, got {base_width}")

        self.in_channels = in_channels
        self.base_width = base_width
        self.depth = depth

        widths = [base_width * (2**i) for i in range(depth + 1)]

        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(widths[0])

        self.encoders = nn.ModuleList(
            [
                _EncoderBlock(widths[i], widths[i + 1])
                for i in range(depth)
            ]
        )

        self.bottleneck = _ConvBlock(widths[depth], widths[depth] * 2)

        # Decoder stage ``i`` (deepest → shallowest) consumes the previous
        # stage's ``2 * widths[i + 1]``-channel output plus the ``widths[i + 1]``-
        # channel skip from encoder stage ``i``, and produces ``widths[i + 1]``
        # so the next decoder's input width matches. Stages are stored
        # deepest-first; ``zip(self.decoders, reversed(skips))`` pairs stage
        # ``i`` with its matching skip.
        self.decoders = nn.ModuleList(
            [
                _DecoderBlock(widths[i + 1] * 2, widths[i + 1], widths[i + 1])
                for i in range(depth - 1, -1, -1)
            ]
        )

        self.head = nn.Conv2d(widths[1], 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected a 4-D input (B, C, H, W), got {x.ndim}-D")

        input_extent = x.shape[2:]
        if input_extent[0] % (2**self.depth) != 0 or input_extent[1] % (2**self.depth) != 0:
            raise ValueError(
                f"input extent {tuple(input_extent)} must be divisible by 2 ** depth "
                f"({2 ** self.depth})"
            )

        x = F.relu(self.bn0(self.stem(x)), inplace=True)
        skips: list[Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            x = decoder(x, skip)
        out = self.head(x)

        if out.shape[2:] != input_extent:
            raise RuntimeError(
                f"decoder misalignment: output {tuple(out.shape[2:])} != input "
                f"{tuple(input_extent)}"
            )
        return out