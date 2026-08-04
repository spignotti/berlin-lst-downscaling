"""Data models for WB2c-1 profiling.

Defines the expected asset structure and profiling results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from berlin_lst_downscaling.data.ard.contract import BandSpec


@dataclass(frozen=True)
class HistogramSpec:
    """Fixed histogram bins for a specific band/channel."""

    band_name: str
    bin_edges: tuple[float, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.bin_edges) < 2:
            raise ValueError(f"HistogramSpec needs at least 2 bin edges, got {len(self.bin_edges)}")
        if not all(
            self.bin_edges[i] <= self.bin_edges[i + 1] for i in range(len(self.bin_edges) - 1)
        ):
            raise ValueError("HistogramSpec bin_edges must be non-decreasing")


@dataclass(frozen=True)
class ProfileAsset:
    """Expected COG to profile with all metadata needed for validation."""

    item_id: str
    source: str
    cog_uri: str
    stac_uri: str | None = None
    provenance_uri: str | None = None
    completion_uri: str | None = None
    partition: Literal["training", "inference", "shared_static"] = "shared_static"
    year: int | None = None
    season: str | None = None
    expected_crs: str = "EPSG:25833"
    expected_resolution: float | None = None
    expected_shape: tuple[int, int] | None = None
    expected_bands: int = 1
    expected_band_specs: tuple[str, ...] = ()
    expected_band_contracts: tuple[BandSpec, ...] = ()
    expected_histogram_specs: tuple[HistogramSpec | None, ...] = ()
    resolution_m: int | None = None

    def __post_init__(self) -> None:
        if self.partition != "shared_static" and self.year is None:
            raise ValueError(f"Non-static partition {self.partition!r} requires year")


@dataclass
class BandStatistics:
    """Descriptive statistics for a single band."""

    band_index: int
    band_name: str
    valid_count: int = 0
    missing_count: int = 0
    missing_rate: float = 0.0
    min_value: float = float("nan")
    max_value: float = float("nan")
    mean_value: float = float("nan")
    std_value: float = float("nan")
    histogram_bins: tuple[float, ...] = ()
    histogram_counts: tuple[int, ...] = ()
    p1: float = float("nan")
    p5: float = float("nan")
    p25: float = float("nan")
    p50: float = float("nan")
    p75: float = float("nan")
    p95: float = float("nan")
    p99: float = float("nan")


@dataclass
class ContractCheckResult:
    """Per-band contract validation outcomes."""

    dtype_mismatches: list[str] = field(default_factory=list)
    nodata_mismatches: list[str] = field(default_factory=list)
    band_description_mismatches: list[str] = field(default_factory=list)
    band_order_mismatches: list[str] = field(default_factory=list)
    unit_absent: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.dtype_mismatches
            or self.nodata_mismatches
            or self.band_description_mismatches
            or self.band_order_mismatches
        )


@dataclass
class CompletenessResult:
    """Manifest↔ARD-ledger completeness diff."""

    manifest_key_count: int = 0
    ledger_key_count: int = 0
    missing_in_ledger: list[str] = field(default_factory=list)
    extra_in_ledger: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_in_ledger
            or self.extra_in_ledger
            or self.duplicate_keys
        )


@dataclass
class ProfileRow:
    """Profiling result for one COG asset."""

    item_id: str
    source: str
    cog_uri: str
    partition: str
    year: int | None = None
    season: str | None = None
    resolution_m: int | None = None

    # Structural checks
    cog_exists: bool = False
    cog_valid: bool = False
    cog_errors: list[str] = field(default_factory=list)
    stac_exists: bool = False
    stac_valid: bool = False
    provenance_exists: bool = False
    completion_exists: bool = False

    # Contract validation
    contract_check: ContractCheckResult = field(default_factory=ContractCheckResult)

    # Per-band statistics
    band_stats: list[BandStatistics] = field(default_factory=list)

    # Aggregate flags
    has_hard_failure: bool = False
    failure_reasons: list[str] = field(default_factory=list)


__all__ = [
    "HistogramSpec",
    "ProfileAsset",
    "BandStatistics",
    "ContractCheckResult",
    "CompletenessResult",
    "ProfileRow",
]
