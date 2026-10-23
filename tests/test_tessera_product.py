"""TESSERA product writer + publish_guard + validator contract."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import rasterio

from berlin_lst_downscaling.data.ard.publish_guard import (
    TESSERA_PRODUCT_PREFIX,
    FrozenPrefixError,
    assert_tessera_output_root,
)
from berlin_lst_downscaling.data.embeddings.product import (
    product_cog_uri,
    product_completion_uri,
    product_provenance_uri,
    product_scale_uri,
    product_valid_uri,
    write_tessera_year_product,
)
from berlin_lst_downscaling.data.embeddings.tessera import TesseraProvenance


def _small_product(tmp_path, year: int = 2024) -> dict[str, str]:
    rng = np.random.default_rng(1)
    quantized = rng.integers(-50, 50, size=(12, 14, 128)).astype(np.int8)
    scale = np.full((12, 14), 0.5, dtype=np.float32)
    scale[0, 0] = np.nan
    valid = np.isfinite(scale)
    from affine import Affine

    transform = Affine(10.0, 0.0, 369190.0, 0.0, -10.0, 5838410.0)
    provenance = TesseraProvenance(year=year, library_version="test").as_dict()
    return write_tessera_year_product(
        quantized=quantized,
        scale=scale,
        valid=valid,
        transform=transform,
        crs="EPSG:25833",
        output_dir=str(tmp_path / str(year)),
        year=year,
        provenance=provenance,
    )


def test_writer_emits_int8_scale_valid_provenance_complete(tmp_path) -> None:
    artifacts = _small_product(tmp_path)
    for key in ("int8", "scale", "valid", "provenance", "complete"):
        assert os.path.exists(artifacts[key])
    with rasterio.open(artifacts["int8"]) as src:
        assert src.count == 128
        assert src.dtypes[0] == "int8"
    with rasterio.open(artifacts["scale"]) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
    with rasterio.open(artifacts["valid"]) as src:
        assert src.count == 1
        assert set(np.unique(src.read(1)).tolist()) <= {0, 1}
    with open(artifacts["provenance"], encoding="utf-8") as fh:
        provenance = json.load(fh)
    assert provenance["dataset_version"] == "v1.1-dclimate"
    with open(artifacts["complete"], encoding="utf-8") as fh:
        assert json.load(fh)["complete"] is True


def test_product_uri_helpers_use_canonical_prefix() -> None:
    assert TESSERA_PRODUCT_PREFIX.endswith("v1.1-dclimate")
    assert product_cog_uri(TESSERA_PRODUCT_PREFIX, 2024).endswith("tessera_2024_int8.tif")
    assert product_scale_uri(TESSERA_PRODUCT_PREFIX, 2024).endswith("tessera_2024_scale.tif")
    assert product_valid_uri(TESSERA_PRODUCT_PREFIX, 2024).endswith("tessera_2024_valid.tif")
    assert product_provenance_uri(TESSERA_PRODUCT_PREFIX, 2024).endswith("provenance.json")
    assert product_completion_uri(TESSERA_PRODUCT_PREFIX, 2024).endswith("complete.json")


def test_publish_guard_blocks_published_year_prefix(monkeypatch) -> None:
    import berlin_lst_downscaling.data.ard.publish_guard as guard

    frozen = frozenset({f"{TESSERA_PRODUCT_PREFIX}/2024"})
    monkeypatch.setattr(guard, "_FROZEN_TESSERA_ROOTS", frozen)
    with pytest.raises(FrozenPrefixError, match="published TESSERA"):
        assert_tessera_output_root(f"{TESSERA_PRODUCT_PREFIX}/2024")
    # Unpublished years stay writable for the first operator publish.
    assert_tessera_output_root(f"{TESSERA_PRODUCT_PREFIX}/2025")
