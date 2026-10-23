"""TESSERA acquisition: years, quantized path, provenance, dequantization."""

from __future__ import annotations

import numpy as np
import pytest

from berlin_lst_downscaling.data.embeddings.tessera import (
    TESSERA_DATASET_VERSION,
    TESSERA_N_CHANNELS,
    TESSERA_ZONE,
    TesseraProvenance,
    TesseraYearError,
    dequantize_tessera,
    read_tessera_region,
    tessera_invalid_mask,
    validate_tessera_year,
)


class _FakeQuantizedReader:
    """Fake preferring the quantized int8+scale path."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def read_region_quantized(self, bbox, year=2024):
        self.calls.append("quantized")
        h, w = 4, 5
        rng = np.random.default_rng(0)
        quantized = rng.integers(-128, 127, size=(h, w, 128), dtype=np.int8)
        scale = np.full((h, w), 0.5, dtype=np.float32)
        scale[0, 0] = np.nan  # one gap pixel
        transform = (10.0, 0.0, 400000.0, 0.0, -10.0, 5800000.0)
        return quantized, scale, transform, "EPSG:32633"

    def read_region(self, bbox, year=2024):  # pragma: no cover — quantized wins
        raise AssertionError("quantized path must win")

    @property
    def years_complete(self):
        return [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def test_supported_years_pass_and_2026_refused() -> None:
    for year in range(2017, 2026):
        assert validate_tessera_year(year) == year
    with pytest.raises(TesseraYearError, match="no 2026"):
        validate_tessera_year(2026)
    with pytest.raises(TesseraYearError, match="outside"):
        validate_tessera_year(2016)
    with pytest.raises(TesseraYearError, match="outside"):
        validate_tessera_year(2027)


def test_quantized_path_preferred_and_provenance_recorded() -> None:
    reader = _FakeQuantizedReader()
    read = read_tessera_region(2024, bbox=(13.08, 52.34, 13.76, 52.68), reader=reader)
    assert reader.calls == ["quantized"]
    assert read.quantized is not None and read.quantized.dtype == np.int8
    assert read.quantized.shape[2] == TESSERA_N_CHANNELS
    assert read.embedding is not None and read.embedding.shape[2] == TESSERA_N_CHANNELS
    # Gap pixel is NaN after dequantization, never zero.
    assert bool(~read.valid[0, 0])
    assert bool(np.all(np.isnan(read.embedding[0, 0])))
    assert bool(read.valid[1, 1])
    assert bool(np.all(np.isfinite(read.embedding[1, 1])))
    prov = read.provenance
    assert isinstance(prov, TesseraProvenance)
    assert prov.dataset_version == TESSERA_DATASET_VERSION
    assert prov.zone == TESSERA_ZONE
    assert prov.year == 2024
    assert prov.quantized is True
    assert prov.library_version
    assert 2024 in prov.years_complete
    doc = prov.as_dict()
    assert doc["dataset_version"] == TESSERA_DATASET_VERSION


def test_dequantize_marks_nonfinite_scale_nan() -> None:
    quantized = np.ones((2, 2, 128), dtype=np.int8)
    scale = np.ones((2, 2), dtype=np.float32)
    scale[0, 1] = np.inf
    scale[1, 0] = np.nan
    out = dequantize_tessera(quantized, scale)
    assert out.shape == (2, 2, 128)
    assert bool(np.all(np.isfinite(out[0, 0])))
    assert bool(np.all(np.isnan(out[0, 1])))
    assert bool(np.all(np.isnan(out[1, 0])))


def test_invalid_mask_threshold_and_nonfinite() -> None:
    scale = np.ones((2, 2), dtype=np.float32)
    scale[0, 0] = np.nan
    quality = np.full((2, 2), 0.9)
    quality[1, 1] = 0.1
    invalid = tessera_invalid_mask(scale, quality, quality_threshold=0.5)
    assert invalid.tolist() == [[True, False], [False, True]]
