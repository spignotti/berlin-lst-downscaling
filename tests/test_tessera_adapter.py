"""TESSERA 1x1 adapter: shape and gradient on CPU."""

from __future__ import annotations

import torch

from berlin_lst_downscaling.modeling.tessera_adapter import (
    TESSERA_ADAPTER_WIDTH_DEFAULT,
    TesseraAdapter,
    TesseraUNet,
    tessera_kept_channel_indices,
    tessera_kept_channel_names,
    tessera_replaced_channel_names,
)


def test_channel_accounting_covers_v6() -> None:
    replaced = tessera_replaced_channel_names()
    kept = tessera_kept_channel_names()
    assert len(replaced) == 8  # 6 morphology + 2 shadow
    assert len(kept) == 14
    assert len(set(replaced) & set(kept)) == 0
    assert len(tessera_kept_channel_indices()) == 14
    assert TESSERA_ADAPTER_WIDTH_DEFAULT == 8


def test_adapter_shape_and_gradient_cpu() -> None:
    torch.manual_seed(0)
    adapter = TesseraAdapter(out_width=8)
    adapter.train()
    embedding = torch.randn(2, 128, 16, 16, requires_grad=True)
    out = adapter(embedding)
    assert tuple(out.shape) == (2, 8, 16, 16)
    loss = out.pow(2).mean()
    loss.backward()
    assert embedding.grad is not None
    assert bool(torch.isfinite(embedding.grad).all())
    for param in adapter.parameters():
        assert param.grad is not None


def test_tessera_unet_fuses_and_predicts_cpu() -> None:
    torch.manual_seed(1)
    model = TesseraUNet(n_kept_channels=14, adapter_width=8, base_width=8, depth=2)
    model.train()
    kept = torch.randn(2, 14, 16, 16, requires_grad=True)
    embedding = torch.randn(2, 128, 16, 16, requires_grad=True)
    out = model(kept, embedding)
    assert tuple(out.shape) == (2, 1, 16, 16)
    assert model.in_channels == 22
    loss = out.mean()
    loss.backward()
    assert kept.grad is not None and embedding.grad is not None
