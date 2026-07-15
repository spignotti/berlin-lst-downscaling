"""Spike: benchmark horizon kernel parameters on a small DSM subset.

Usage:
    uv run python scripts/spikes/secondary_geometry.py

Compares:
- Pure-NumPy horizon kernel at 16 vs 36 directions
- max_radius 100m vs 200m
- Peak RSS and runtime on representative data

Reports:
- Time per direction count × radius combination
- Peak memory usage
- Output quality check (angle range, nodata fraction)
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from berlin_lst_downscaling.data.secondary.horizon import _compute_horizon_cube


def make_synthetic_dsm(
    size: int = 200,
    building_height: float = 30.0,
    building_fraction: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic DSM with buildings for testing."""
    rng = np.random.default_rng(seed)
    terrain = np.full((size, size), 50.0, dtype=np.float32)

    # Add random buildings
    n_buildings = int(size * size * building_fraction / 25)  # ~5x5 pixel buildings
    for _ in range(n_buildings):
        bx = rng.integers(0, size - 5)
        by = rng.integers(0, size - 5)
        bw = rng.integers(3, 6)
        bh = rng.integers(3, 6)
        h = rng.uniform(building_height * 0.5, building_height * 1.5)
        terrain[by:by + bh, bx:bx + bw] = h

    return terrain


def benchmark_horizon(
    dsm: np.ndarray,
    n_directions: int,
    max_radius_m: float,
    cell_size: float = 10.0,
) -> dict:
    """Run the horizon kernel and report timing + memory."""
    tracemalloc.start()

    t0 = time.perf_counter()
    cube = _compute_horizon_cube(dsm, cell_size, max_radius_m, n_directions)
    elapsed = time.perf_counter() - t0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    nodata = 65535
    valid = cube[cube != nodata]

    return {
        "n_directions": n_directions,
        "max_radius_m": max_radius_m,
        "elapsed_s": round(elapsed, 2),
        "peak_mb": round(peak_bytes / 1024 / 1024, 1),
        "output_shape": list(cube.shape),
        "valid_pct": round(100.0 * len(valid) / cube.size, 1) if cube.size > 0 else 0,
        "min_cd": int(valid.min()) if len(valid) > 0 else None,
        "max_cd": int(valid.max()) if len(valid) > 0 else None,
    }


def main() -> None:
    """Run benchmarks on synthetic DSM."""
    print("Creating synthetic DSM (200×200)...")
    dsm = make_synthetic_dsm(size=200)
    print(f"  DSM range: {dsm.min():.1f} – {dsm.max():.1f} m")
    print()

    configs = [
        (16, 100.0),
        (16, 200.0),
        (36, 100.0),
        (36, 200.0),
    ]

    results = []
    for n_dir, radius in configs:
        print(f"Benchmarking: {n_dir} directions, {radius}m radius...")
        r = benchmark_horizon(dsm, n_dir, radius)
        results.append(r)
        print(f"  {r['elapsed_s']}s, peak {r['peak_mb']} MB, "
              f"angles {r['min_cd']}–{r['max_cd']} cd")
        print()

    print("=" * 60)
    print("Summary:")
    print(f"  {'Dirs':>5} {'Radius':>7} {'Time':>8} {'Peak':>8} {'Valid%':>7}")
    for r in results:
        print(
            f"  {r['n_directions']:>5} {r['max_radius_m']:>6.0f}m "
            f"{r['elapsed_s']:>7.1f}s {r['peak_mb']:>7.1f}MB "
            f"{r['valid_pct']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
