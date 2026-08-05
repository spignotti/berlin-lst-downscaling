"""Blockwise statistics computation for WB2c-1 profiling.

Reads bands through rasterio block windows and accumulates exact
valid/missing counts, min/max, mean, standard deviation, and
fixed-bin histograms without loading full scenes.
"""

from __future__ import annotations

import numpy as np
import rasterio

from berlin_lst_downscaling.data.profiling.contracts import require_histogram_spec
from berlin_lst_downscaling.data.profiling.models import BandStatistics, ProfileAsset, ProfileRow


def profile_band(
    src: rasterio.DatasetReader,
    band_index: int,
    band_name: str,
    flag: np.ndarray | None = None,
) -> BandStatistics:
    """Profile a single band using blockwise reads, optionally QA-masked.

    When *flag* is provided (aligned to *src*), pixels with ``flag != 0``
    (fill/cloud/shadow/cirrus/saturation) are excluded from ``valid``
    statistics and reported separately as ``qa_masked``. NoData pixels
    are reported as ``missing``. Statistics and histograms describe only
    QA-clean, non-nodata pixels.
    """
    stats = BandStatistics(band_index=band_index, band_name=band_name)

    try:
        spec = require_histogram_spec(band_name)
        stats.histogram_bins = spec.bin_edges
        stats.histogram_counts = tuple(0 for _ in range(len(spec.bin_edges) - 1))
    except KeyError:
        # No histogram spec for this band - skip histogram
        pass

    # Blockwise accumulation
    total_pixels = 0
    missing_pixels = 0
    qa_masked_pixels = 0
    valid_pixels = 0
    sum_value = 0.0
    sum_sq_value = 0.0
    min_value = float("inf")
    max_value = float("-inf")

    for _, window in src.block_windows(1):
        data = src.read(band_index, window=window, masked=False)

        # Build a plain boolean nodata mask (avoids MaskedArray pitfalls
        # with integer nodata on fully-masked blocks).
        nodata = src.nodata
        if nodata is None:
            nodata_mask = np.zeros(data.shape, dtype=bool)
            if np.issubdtype(data.dtype, np.floating):
                nodata_mask |= np.isnan(data)
        elif np.isnan(float(nodata)):
            nodata_mask = np.isnan(data)
        else:
            nodata_mask = data == nodata

        block = data.astype(np.float64)
        n = block.size
        total_pixels += n
        not_missing = ~nodata_mask
        missing_pixels += int(n - int(not_missing.sum()))

        if flag is not None:
            flag_block = flag[window.toslices()]
            qa_masked = not_missing & (flag_block != 0)
            valid_mask = not_missing & (flag_block == 0)
            qa_masked_pixels += int(qa_masked.sum())
        else:
            valid_mask = not_missing
            qa_masked_pixels += 0

        valid_data = block[valid_mask]
        valid_pixels += valid_data.size

        if valid_data.size > 0:
            sum_value += valid_data.sum()
            sum_sq_value += (valid_data**2).sum()
            min_value = min(min_value, float(valid_data.min()))
            max_value = max(max_value, float(valid_data.max()))

            # Update histogram
            if stats.histogram_bins:
                edges = stats.histogram_bins
                counts, _ = np.histogram(valid_data, bins=list(edges))
                stats.histogram_counts = tuple(
                    existing + int(count)
                    for existing, count in zip(stats.histogram_counts, counts, strict=True)
                )
                stats.histogram_underflow += int((valid_data < edges[0]).sum())
                stats.histogram_overflow += int((valid_data > edges[-1]).sum())

    # Final statistics — three disjoint sets
    stats.valid_count = valid_pixels
    stats.missing_count = missing_pixels
    stats.missing_rate = missing_pixels / total_pixels if total_pixels > 0 else 1.0
    stats.qa_masked_count = qa_masked_pixels
    stats.qa_masked_rate = qa_masked_pixels / total_pixels if total_pixels > 0 else 0.0

    if valid_pixels > 0:
        stats.mean_value = sum_value / valid_pixels
        variance = (sum_sq_value / valid_pixels) - (stats.mean_value**2)
        stats.std_value = max(0.0, variance) ** 0.5  # Ensure non-negative
        stats.min_value = min_value
        stats.max_value = max_value

    # Derive percentiles from histogram CDF
    if stats.histogram_bins and stats.histogram_counts and valid_pixels > 0:
        _derive_percentiles_from_histogram(stats)

    return stats


def _derive_percentiles_from_histogram(stats: BandStatistics) -> None:
    """Derive p1-p99 percentiles from histogram CDF."""
    total = sum(stats.histogram_counts)
    if total == 0:
        return

    # Compute CDF
    cdf = np.cumsum(stats.histogram_counts) / total
    bin_edges = np.array(stats.histogram_bins)

    # Find percentile values
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = []

    for p in percentiles:
        idx = np.searchsorted(cdf, p / 100.0)
        if idx >= len(bin_edges) - 1:
            percentile_values.append(float(bin_edges[-1]))
        elif idx == 0:
            percentile_values.append(float(bin_edges[0]))
        else:
            # Linear interpolation within the bin
            bin_start = bin_edges[idx - 1]
            bin_end = bin_edges[idx]
            cdf_start = cdf[idx - 1]
            cdf_end = cdf[idx]
            if cdf_end > cdf_start:
                frac = (p / 100.0 - cdf_start) / (cdf_end - cdf_start)
                percentile_values.append(float(bin_start + frac * (bin_end - bin_start)))
            else:
                percentile_values.append(float(bin_start))

    stats.p1, stats.p5, stats.p25, stats.p50, stats.p75, stats.p95, stats.p99 = percentile_values


def profile_row_statistics(row: ProfileRow, asset: ProfileAsset | None = None) -> ProfileRow:
    """Profile all bands of a COG and update the ProfileRow."""
    if not row.cog_valid or not row.cog_exists:
        return row

    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    # ARD main assets REQUIRE a QA flag COG: without it statistics would
    # include masked cloud/shadow/fill pixels. Bail out (already a hard
    # failure from inspect_asset) rather than profile unmasked.
    if asset is not None and asset.requires_qa_flag and not row.flag_valid:
        return row

    # QA flag COG (ARD main assets): load once, aligned to the main COG.
    flag = None
    if asset is not None and asset.requires_qa_flag and row.flag_valid and asset.qa_flag_uri:
        try:
            with rasterio.open(gdal_uri(asset.qa_flag_uri)) as fsrc:
                flag = fsrc.read(1)
        except Exception as exc:
            row.has_hard_failure = True
            row.failure_reasons.append(f"QA flag read failed: {exc}")
            return row

    try:
        with rasterio.open(gdal_uri(row.cog_uri)) as src:
            for band_index in range(1, src.count + 1):
                band_name = f"band_{band_index}"
                if asset and band_index <= len(asset.expected_band_specs):
                    band_name = asset.expected_band_specs[band_index - 1]

                stats = profile_band(src, band_index, band_name, flag=flag)
                row.band_stats.append(stats)

    except Exception as exc:
        row.has_hard_failure = True
        row.failure_reasons.append(f"Statistics computation failed: {exc}")

    return row


__all__ = [
    "profile_band",
    "profile_row_statistics",
]
