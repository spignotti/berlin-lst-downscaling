"""Read-only SVF radius comparison on a small DSM sample.

The published SVF uses ``svf_max_radius`` cells of 10 m (default 3, so
30 m) and 16 directions. This diagnostic computes SVF at several radii
on a sample array, reports summary statistics and the correlation
between radii, and estimates full-grid runtime from the sample timing.
It does not write a product and refuses a full-AOI array.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import xarray as xr
from xrspatial import sky_view_factor

from berlin_lst_downscaling.common.grid import canon_grid_10m

# Default comparison radii in metres. 30 m is the published default.
SVF_DIAGNOSTIC_RADII_M: tuple[float, ...] = (30.0, 50.0, 100.0, 200.0)
SVF_DEFAULT_RADIUS_M: float = 30.0
SVF_DEFAULT_CELLSIZE_M: float = 10.0
SVF_DEFAULT_N_DIRECTIONS: int = 16

# A 500 x 500 sample is 2.5 km on a side. The full Berlin grid is larger
# and is rejected so this stays a sample diagnostic.
_MAX_SAMPLE_PIXELS = 250_000


@dataclass(frozen=True)
class SvfRadiusSummary:
    """Summary of SVF at one search radius."""

    radius_m: float
    radius_cells: int
    n_finite: int
    minimum: float
    maximum: float
    mean: float
    std: float
    elapsed_s: float
    estimated_full_grid_s: float


@dataclass(frozen=True)
class SvfRadiusDiagnostic:
    """Sample SVF comparison. ``correlation`` is keyed by ``(radius_a, radius_b)``."""

    cellsize_m: float
    n_directions: int
    sample_shape: tuple[int, int]
    reference_pixels: int
    summaries: tuple[SvfRadiusSummary, ...]
    correlation: dict[tuple[float, float], float | None]
    note: str

    def format_report(self) -> str:
        """Return a plain-text report for the validator CLI."""
        lines = [
            "SVF radius diagnostic (read-only)",
            f"sample shape: {self.sample_shape[0]} x {self.sample_shape[1]}",
            f"cell size: {self.cellsize_m:g} m",
            f"directions: {self.n_directions}",
            f"reference pixels for the runtime estimate: {self.reference_pixels}",
            "",
            "radius_m  cells  n_finite  min  max  mean  std  sample_s  estimated_full_s",
        ]
        for summary in self.summaries:
            lines.append(
                f"{summary.radius_m:8.0f}  {summary.radius_cells:5d}  "
                f"{summary.n_finite:8d}  {summary.minimum:.4f}  {summary.maximum:.4f}  "
                f"{summary.mean:.4f}  {summary.std:.4f}  {summary.elapsed_s:.4f}  "
                f"{summary.estimated_full_grid_s:.1f}"
            )
        lines.append("")
        lines.append("Pearson correlation (finite pixels shared by the pair):")
        for (left, right), value in self.correlation.items():
            rendered = "undefined" if value is None else f"{value:.6f}"
            lines.append(f"  {left:g} m vs {right:g} m: {rendered}")
        lines.append("")
        lines.append(self.note)
        return "\n".join(lines)


def diagnose_svf_radii(
    dsm: np.ndarray,
    *,
    radii_m: tuple[float, ...] = SVF_DIAGNOSTIC_RADII_M,
    cellsize_m: float = SVF_DEFAULT_CELLSIZE_M,
    n_directions: int = SVF_DEFAULT_N_DIRECTIONS,
    reference_pixels: int | None = None,
) -> SvfRadiusDiagnostic:
    """Compute SVF at ``radii_m`` on a sample DSM and estimate full-grid time.

    Parameters
    ----------
    dsm :
        Elevation sample in metres, shape ``(row, col)``. Must be at most
        ``_MAX_SAMPLE_PIXELS``.
    radii_m :
        Search radii in metres. Each must be an integer number of cells.
    cellsize_m :
        DSM cell size in metres. The published grid is 10 m.
    n_directions :
        Horizon directions passed to ``xarray-spatial``.
    reference_pixels :
        Pixel count used to scale the sample timing. Defaults to the
        canonical 10 m Berlin grid.

    The runtime estimate assumes cost scales with pixel count at the
    measured radius. The neighbourhood work is inside the sample timing.
    It is not a benchmark of a full-AOI run.
    """
    elevation = np.asarray(dsm, dtype=np.float32)
    if elevation.ndim != 2:
        raise ValueError(f"dsm must be 2-D, got shape {elevation.shape}")
    n_pixels = int(elevation.size)
    if n_pixels == 0 or n_pixels > _MAX_SAMPLE_PIXELS:
        raise ValueError(
            f"SVF radius diagnostic is a sample tool (1..{_MAX_SAMPLE_PIXELS} pixels), "
            f"got {n_pixels}"
        )
    if cellsize_m <= 0:
        raise ValueError(f"cellsize_m must be positive, got {cellsize_m}")
    if isinstance(n_directions, bool) or not isinstance(n_directions, int) or n_directions < 1:
        raise ValueError(f"n_directions must be an int >= 1, got {n_directions!r}")
    if not radii_m:
        raise ValueError("radii_m must not be empty")

    radius_cells = tuple(_radius_cells(radius_m, cellsize_m) for radius_m in radii_m)
    if reference_pixels is None:
        grid = canon_grid_10m()
        reference_pixels = int(grid.shape.y) * int(grid.shape.x)
    if reference_pixels < 1:
        raise ValueError(f"reference_pixels must be >= 1, got {reference_pixels}")

    rows = np.arange(elevation.shape[0], dtype=np.float64) * cellsize_m
    cols = np.arange(elevation.shape[1], dtype=np.float64) * cellsize_m
    dsm_da = xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"y": rows, "x": cols},
        attrs={"res": (cellsize_m, cellsize_m)},
    )

    fields: list[np.ndarray] = []
    summaries: list[SvfRadiusSummary] = []
    for radius_m, cells in zip(radii_m, radius_cells, strict=True):
        started = time.perf_counter()
        svf = sky_view_factor(
            dsm_da,
            max_radius=cells,
            n_directions=n_directions,
            cellsize_x=cellsize_m,
            cellsize_y=cellsize_m,
            name="svf",
        )
        elapsed = time.perf_counter() - started
        values = np.asarray(svf.values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            minimum = maximum = mean = std = float("nan")
        else:
            minimum = float(finite.min())
            maximum = float(finite.max())
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
        summaries.append(
            SvfRadiusSummary(
                radius_m=float(radius_m),
                radius_cells=cells,
                n_finite=int(finite.size),
                minimum=minimum,
                maximum=maximum,
                mean=mean,
                std=std,
                elapsed_s=elapsed,
                estimated_full_grid_s=elapsed * reference_pixels / n_pixels,
            )
        )
        fields.append(values)

    correlation: dict[tuple[float, float], float | None] = {}
    for i, left in enumerate(summaries):
        for right, other in zip(summaries[i + 1 :], fields[i + 1 :], strict=True):
            correlation[(left.radius_m, right.radius_m)] = _pearson(fields[i], other)

    note = (
        "Runtime estimate scales sample seconds by reference_pixels / sample_pixels "
        "at the same radius and direction count. It includes neighbourhood work "
        "measured on the sample and is not a full-AOI benchmark. "
        f"The published default remains {SVF_DEFAULT_RADIUS_M:g} m "
        f"({int(SVF_DEFAULT_RADIUS_M / SVF_DEFAULT_CELLSIZE_M)} cells) until a config change."
    )
    return SvfRadiusDiagnostic(
        cellsize_m=cellsize_m,
        n_directions=n_directions,
        sample_shape=(int(elevation.shape[0]), int(elevation.shape[1])),
        reference_pixels=reference_pixels,
        summaries=tuple(summaries),
        correlation=correlation,
        note=note,
    )


def synthetic_dsm(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Return a small DSM with two building blocks on flat ground.

    Flat terrain makes SVF identically 1 and the correlation undefined.
    The blocks give the radii something to disagree about.
    """
    if shape[0] < 32 or shape[1] < 32:
        raise ValueError(f"synthetic DSM must be at least 32 x 32, got {shape}")
    dsm = np.zeros(shape, dtype=np.float32)
    dsm[8:20, 8:20] = 24.0
    dsm[36:48, 28:44] = 10.0
    return dsm


def _radius_cells(radius_m: float, cellsize_m: float) -> int:
    cells = radius_m / cellsize_m
    if cells < 1 or not np.isclose(cells, round(cells)):
        raise ValueError(f"radius {radius_m} m is not an integer number of {cellsize_m:g} m cells")
    return int(round(cells))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    a = left[mask].astype(np.float64)
    b = right[mask].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


__all__ = [
    "SVF_DEFAULT_CELLSIZE_M",
    "SVF_DEFAULT_N_DIRECTIONS",
    "SVF_DEFAULT_RADIUS_M",
    "SVF_DIAGNOSTIC_RADII_M",
    "SvfRadiusDiagnostic",
    "SvfRadiusSummary",
    "diagnose_svf_radii",
    "synthetic_dsm",
]
